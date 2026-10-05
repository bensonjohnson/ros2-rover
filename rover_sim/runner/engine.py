"""Evolution arena: P individuals x G unseen houses, frozen rollouts.

Moved verbatim out of evo/arena.py (stage 2, SIM_PLATFORM): `Arena` is
renamed `RolloutEngine` (alias `Arena` kept) — the same class, byte-for-byte
the same behavior. evo/arena.py is now a re-export shim.

Reuses pnn_sim.batched.env verbatim (physics, lidar raycast, safety gate)
but feeds it external controllers instead of the PC brain. Everything that
has to happen per tick stays on GPU — including the GROUND-TRUTH room
counter: partition lines become padded [B, maxP] tensors and a room visit
is one bit in a per-env int64 bitmask, so fitness needs zero CPU syncs.

Same-G houses are shared across individuals (paired comparison); worlds
come from a fixed eval seed the evolution never sees elsewhere.

GB10 fast path (RolloutEngine(fp16=True)) — three optimizations for the Spark:
  1. fp16 raycast: the per-tick scan is six [B, 360, M] intermediates —
     bandwidth-bound, and the GB10 pays ~2x on half-precision traffic.
     Dummy segments (1e6) round to inf in fp16 and drop out via the
     isfinite hit-mask, same exclusion as fp32. Everything stateful stays
     fp32: pose integration (0.01 m steps would quantize), the gate's
     1e6 sentinels (fp16 max 65504), noise.
  2. NoSyncGate: the stock BatchedGate does `int(block.sum())` every tick
     — a forced GPU->CPU sync that serializes the whole pipeline. stops is
     never used for fitness, so accumulate it as a device tensor instead.
  3. Batch size is the free lunch: 128 GB unified means thousands more
     envs than the fp32 working set suggests.
Selection never compares across worlds/precisions: a run uses one arena
config for train and holdout.
"""

from __future__ import annotations

import numpy as np
import torch

from rover_sim.core.world import make_house
from rover_sim.core.rover import RoverConfig
from rover_sim.core.gate import GateConfig
from rover_sim.core.batched import batched_preprocess
from rover_sim.core.fast import Fp32Env, Fp16Env, NoSyncGate
from rover_sim.core.buildings import _PaddedWorld, world_caps, cell_ceiling
from rover_sim.core.render import CameraConfig, CameraRenderer

from .obs import (NUM_BINS, MAX_RANGE, _proprio,
                  DEFAULT_OBS_SPEC, ObsSpec, channel_views)

CONTROL_HZ = 15.0


class RolloutEngine:
    """P*G envs laid out as individual-major rows p*G+g; house g is shared
    by every individual (paired fitness across identical worlds)."""

    # one CUDA graph memory pool shared by every RolloutEngine in the process
    # (set on first capture); None = private pools
    _pool = None

    def __init__(self, P: int, G: int, seed: int = 777_000,
                 device: str = "cuda", rover_cfg: RoverConfig | None = None,
                 gate_cfg: GateConfig | None = None, fp16: bool = False,
                 noise_seed: int | None = None, merged_houses: int = 1,
                 door_w_range: tuple = (0.7, 1.0), cells: bool = True,
                 every_cover: int = 24, fused: bool = False,
                 compile: bool = False, gate_obs: bool = False,
                 worlds: list | None = None, caps: tuple | None = None,
                 gate_symmetric: bool = False, trim_rand: bool = False,
                 extent: float = 16.0, every_door: int = 6,
                 camera: CameraConfig | None = None):
        """merged_houses=K: play each individual in K*G houses (one set per
        seed offset) inside ONE arena — lets the whole run need a single
        CUDA graph (separate live graphs fault on this stack; see
        gmbisect) and gives fitness over K x more worlds for free.
        cells=False disables the novelty accumulator entirely (bisect).
        every_cover: coverage sample stride (ticks). At v_max 0.2 m/s /
        15 Hz a 24-tick gap moves the rover 0.32 m — under half a 0.8 m
        cell, so no cell can be skipped — and it keeps the graph ~8x
        smaller than sampling with the rooms cadence (node-count pressure
        is the run-5 Xid-43 fault class on this stack).

        worlds: G*K pnn_sim.buildings.Building objects -> BUILDING MODE
        (house k*G+g at column g of set k). Rooms come from explicit room
        rects, door thresholds add door_crossings/doors_used metrics, the
        coverage lattice spans `extent` m, and load_worlds() swaps in new
        houses by copy_ into fixed-capacity buffers (caps = (segments,
        rects, doors)) — legal between CUDA-graph replays, so the train
        set can be resampled every generation without recapture. Legacy
        mode (worlds=None) is byte-for-byte the runs 1-9 arena.
        camera=CameraConfig() opts into the stage-4b egocentric camera
        channel: a static [B, H, W, C] uint8 buffer rendered every tick and
        presented only to policies whose ObsSpec requests it. None (default)
        leaves every legacy path — buffers, numerics, channel_views keys —
        byte-identical."""
        self.cells_on = bool(cells)
        self.every_cover = int(every_cover)
        self.P, self.G = P, G
        self.G_sets = merged_houses
        self.B = P * G * merged_houses
        self.device = torch.device(device)
        self.dt = 1.0 / CONTROL_HZ
        self.fp16 = bool(fp16 and str(device).startswith("cuda"))
        self._gate_cfg = gate_cfg or GateConfig()
        self._noise_seed = noise_seed   # set = every game replays ONE
                                        # shared noise stream (paired eval;
                                        # also the CUDA-graph capture mode)

        self.building_mode = worlds is not None
        self.every_door = int(every_door)
        HG = G * merged_houses
        if self.building_mode:
            houses = list(worlds)
            assert len(houses) == HG, f"need {HG} worlds, got {len(houses)}"
            self.caps = caps or world_caps(houses)
            houses_env = [_PaddedWorld(w, self.caps[0]) for w in houses]
        else:
            houses = []
            for k in range(merged_houses):
                rng_k = np.random.default_rng(seed + 1_000_003 * k)
                houses += [make_house(rng_k, door_w_range=door_w_range)
                           for _ in range(G)]
            houses_env = houses
        # env layout: b = p*(G*K) + k*G + g  (individual-major, then set)
        env_cls = Fp16Env if self.fp16 else Fp32Env
        if fused:
            if not self.fp16:
                raise SystemExit("--fused requires fp16 raycast config")
            env_cls = type("FusedEnv", (Fp16Env,), {"fused": True})
        self.env = env_cls(self.B, rover_cfg, seed=seed, device=device)
        self.env._worlds = [houses_env[b % (G * merged_houses)]
                            for b in range(self.B)]
        self.env._build_segments()

        # Cache start poses + partition geometry as device tensors.
        pose = np.array([w.start_pose for w in self.env._worlds],
                        dtype=np.float32)
        self._pose = torch.as_tensor(pose, device=self.device)
        maxp = max(1, max(len(w.partitions) for w in houses))
        kind = np.zeros((self.B, maxp))
        pos = np.full((self.B, maxp), np.inf)   # pads: kind 0 masks them out
        for b in range(self.B):
            for k, (kd, p) in enumerate(houses[b % HG].partitions):
                kind[b, k] = 1.0 if kd == "v" else 2.0
                pos[b, k] = p
        self._pk = torch.as_tensor(kind, device=self.device)
        self._pp = torch.as_tensor(pos, device=self.device)
        self.maxp = maxp

        # Reachable-room ceiling per world (same clearance-grid trick as
        # eval_checkpoints), tiled to envs.
        totals = []
        for w in houses:
            if self.building_mode:
                totals.append(max(1, len(w.rooms_reachable)))
                continue
            xmin, ymin, xmax, ymax = w.bounds
            gx = np.linspace(xmin + 0.4, xmax - 0.4, 24)
            gy = np.linspace(ymin + 0.4, ymax - 0.4, 24)
            rr = {w.room_id(float(x), float(y)) for x in gx for y in gy
                  if w.clearance(float(x), float(y)) > 0.2}
            totals.append(max(1, len(rr)))
        self.rooms_total = torch.tensor(
            [totals[b % HG] for b in range(self.B)], device=self.device,
            dtype=torch.float32)

        # --- cell-coverage novelty geometry (run 5) ----------------------
        # The novelty the rooms term CANNOT give: rooms is a sparse cliff
        # (pays only on the tick you cross), so nothing pulls the policy
        # toward the unknown space behind a wall. Coverage of ~0.8 m cells
        # pays per new cell from the very first tick and SATURATES (a bit
        # is set once per game — count per place, never per tick, so no
        # infinite novelty fountain). Self-computable — no room_id() — so
        # the deployed analogue is place memory on real lidar. Accumulator
        # is [B, nx*ny] set by in-place scatter_ (alloc-free, graph-safe;
        # duplicate indices all write the same 1 so order doesn't matter).
        # ATTRIBUTION: nearest-centre on this lattice == floor bucketing
        # (boundaries land on 0.0/0.8/1.6...), so a cell is credited only
        # when the rover's centre physically enters its box — a wall-clamp
        # can never credit a box across a wall. Boxes that straddle a
        # partition leak a BOUNDED amount of through-wall credit (a hugger
        # at the 0.2 m gate standoff stands inside the box whose centre is
        # across the wall), but deep-room cells still require a real
        # crossing, so coverage cannot be farmed by wall-hugging.
        top = (extent - 0.39) if self.building_mode else 11.61
        gx_all = np.arange(0.4, top, 0.8)        # cell CENTRES, 0.8 m boxes
        gy_all = np.arange(0.4, top, 0.8)
        self.cell_nx, self.cell_ny = len(gx_all), len(gy_all)
        self._cell = 0.8
        # GB10/torch2.11 CUDA-graph fault (run5): argmin+scatter_ inside the
        # captured rollout reproducible illegal-memory-access on gen-1
        # replay at production size; --no-cells control (identical runs 1-4
        # kernels) passes clean. Coverage therefore uses ONLY the proven
        # elementwise family: floor bucketing + broadcast ==/&/|= (same
        # in-place bitwise op as the room counter). No index kernels.
        self._gx_all, self._gy_all = gx_all, gy_all
        cell_totals = []
        for w in houses:
            if self.building_mode:
                cell_totals.append(cell_ceiling(w, gx_all, gy_all))
                continue
            xmin, ymin, xmax, ymax = w.bounds
            n = sum(1 for xg in gx_all for yg in gy_all
                    if xmin + 0.2 <= xg <= xmax - 0.2
                    and ymin + 0.2 <= yg <= ymax - 0.2
                    and w.clearance(float(xg), float(yg)) > 0.2)
            cell_totals.append(max(1, n))        # >=1: start cell always
        self.cells_total = torch.tensor(
            [cell_totals[b % HG] for b in range(self.B)],
            device=self.device, dtype=torch.float32)
        self._cx = torch.as_tensor(gx_all, device=self.device)
        self._cy = torch.as_tensor(gy_all, device=self.device)
        self._ncell_words = max(1, -(-self.cell_nx * self.cell_ny // 63))
        # ZERO-ALLOC scratch for _cells_or: every intermediate is a static
        # buffer driven by out=/in-place ops. Fresh allocations inside the
        # captured rollout are pooled by the CUDA graph allocator and come
        # LIVE (retained) for every replay — the run-5 fault (Xid 43,
        # illegal memory access at production size, both the scatter_ and
        # the elementwise rewrite, while --no-cells passed) is exactly the
        # graph-pool-pressure failure class this stack is known for
        # (torch 2.11 allocator asserts/GB10 illegal-access notes). The
        # proven room counter allocates ~5/sampled-tick; the W-word
        # coverage version must add ZERO.
        if self.cells_on:
            z = dict(device=self.device)
            self._tf1 = torch.zeros(self.B, **z)
            self._tf2 = torch.zeros(self.B, **z)
            self._tf3 = torch.zeros(self.B, **z)
            self._cidx = torch.zeros(self.B, dtype=torch.int64, **z)
            self._cinr = torch.zeros(self.B, dtype=torch.bool, **z)
            self._clt = torch.zeros(self.B, dtype=torch.bool, **z)
            self._cbits = torch.zeros(self.B, dtype=torch.int64, **z)
            self._cbcast = torch.zeros(self.B, dtype=torch.int64, **z)
            self._cwhere = torch.zeros(self.B, dtype=torch.int64, **z)
            self._los = torch.arange(self._ncell_words, dtype=torch.int64,
                                     device=self.device) * 63

        self.gate = NoSyncGate(self._gate_cfg, self.B,
                               lambda: self._t * self.dt,
                               device=str(self.device))
        if gate_symmetric:
            self.gate.enable_symmetric()
        if trim_rand:
            # per-env effective track-speed factors (domain randomization);
            # static buffers so set_trims() works between graph replays
            rc = self.env.cfg
            self.env._trim_l = torch.full((self.B,), float(rc.left_trim),
                                          device=self.device)
            self.env._trim_r = torch.full((self.B,), float(rc.right_trim),
                                          device=self.device)
        self._preprocess = batched_preprocess
        self.gate_obs = bool(gate_obs)
        self.obs_spec = ObsSpec(proprio_version="v2" if gate_obs else "v1")
        if compile:
            self._compile_tick()
        self._t = 0
        # static buffers so the whole rollout is CUDA-graph capturable
        self._shift = torch.arange(self.maxp, device=self.device,
                                   dtype=torch.int64)
        self._one64 = torch.ones((), dtype=torch.int64, device=self.device)
        self._zero64 = torch.zeros((), dtype=torch.int64, device=self.device)
        self._acc = None
        # forward / reverse travel accumulators (static: graph-safe)
        self._fwd_m = torch.zeros(self.B, device=self.device)
        self._back_m = torch.zeros(self.B, device=self.device)
        # orbit guard (run 23): spin ticks + start pose, for net-displacement
        # and spin-fraction metrics. dist_m counts ARC LENGTH, which an
        # in-place orbit earns freely (live runs 5-8: arc 23 m, net 3 m,
        # real rover zero-turn circling). These pay for straight-line travel
        # and price opposite-wheel pivots.
        self._spin_t = torch.zeros(self.B, device=self.device)
        self._x0 = torch.zeros(self.B, device=self.device)
        self._y0 = torch.zeros(self.B, device=self.device)
        self._range = torch.zeros(self.B, device=self.device)
        if self.building_mode:
            self._init_building_buffers(houses)
        # Stage-4b opt-in camera channel: allocate the [B, H, W, C] uint8
        # frame buffer and the graph-safe renderer HERE (bind/init time,
        # never mid-tick). camera=None (default) keeps every legacy path
        # byte-identical: no buffer, no render, no channel.
        self.camera = camera
        self._cam_renderer = None
        self._camera_buf = None
        if camera is not None:
            self._camera_buf = torch.zeros(
                self.B, camera.height, camera.width, camera.channels,
                dtype=torch.uint8, device=self.device)
            self._cam_renderer = CameraRenderer(camera, self.B, self.device)
        self._reset_hooks = []      # e.g. zero the policy's hidden state
        self._rec = None            # set to dict to record (obs, cmd) streams

    # ------------------------------------------------ building mode

    def _building_arrays(self, houses):
        """numpy [HG, cap, ...] geometry of the house list (padded)."""
        Mc, Rc, Dc = self.caps
        HG = len(houses)
        segs = np.tile(np.asarray([1e6, 1e6, 1e6 + 0.1, 1e6], np.float32),
                       (HG, Mc, 1))
        rects = np.zeros((HG, Rc, 5), np.float32)
        rects[:, :, 0] = rects[:, :, 1] = np.inf    # never inside
        rects[:, :, 2] = rects[:, :, 3] = -np.inf
        doors = np.zeros((HG, Dc, 6), np.float32)
        dvalid = np.zeros((HG, Dc), bool)
        pose = np.zeros((HG, 3), np.float32)
        for h, w in enumerate(houses):
            segs[h, :w.segments.shape[0]] = w.segments
            rects[h, :w.room_rects.shape[0]] = w.room_rects
            d = w.doors[(w.doors[:, 4] >= 0) & (w.doors[:, 5] >= 0)]
            doors[h, :d.shape[0]] = d
            dvalid[h, :d.shape[0]] = True
            pose[h] = w.start_pose
        return segs, rects, doors, dvalid, pose

    def _init_building_buffers(self, houses):
        z = dict(device=self.device)
        _, rects, doors, dvalid, _ = self._building_arrays(houses)
        B = self.B
        tile = lambda a: torch.as_tensor(
            a[np.arange(B) % len(houses)], **z)
        self._rx0, self._ry0 = tile(rects[:, :, 0]), tile(rects[:, :, 1])
        self._rx1, self._ry1 = tile(rects[:, :, 2]), tile(rects[:, :, 3])
        self._rid1 = tile(rects[:, :, 4]).to(torch.int64) + 1
        ax, ay = doors[:, :, 0], doors[:, :, 1]
        dx, dy = doors[:, :, 2] - ax, doors[:, :, 3] - ay
        L = np.maximum(np.hypot(dx, dy), 1e-6)
        self._dax, self._day = tile(ax), tile(ay)
        self._ddx, self._ddy = tile(dx), tile(dy)
        self._dlen2 = tile(L * L)
        # along-threshold window with a 0.12 m margin past each jamb
        self._dtlo, self._dthi = tile(-0.12 / L), tile(1.0 + 0.12 / L)
        self._dvalid = tile(dvalid)
        self.doors_total = self._dvalid.sum(dim=1).to(torch.float32)
        Dc = doors.shape[1]
        self._dprev = torch.zeros(B, Dc, **z)
        self._dused = torch.zeros(B, Dc, dtype=torch.bool, **z)
        self._dcross = torch.zeros(B, **z)

    def load_worlds(self, houses) -> None:
        """Swap in a new G*K house list IN PLACE (copy_ only: every tensor
        keeps its address, so a captured rollout graph replays on the new
        geometry). Caps must cover the new houses."""
        assert self.building_mode, "load_worlds needs building mode"
        HG = self.G * self.G_sets
        assert len(houses) == HG
        caps = world_caps(houses)
        assert all(a <= b for a, b in zip(caps, self.caps)), \
            f"houses need caps {caps} > arena caps {self.caps}"
        segs, rects, doors, dvalid, pose = self._building_arrays(houses)
        idx = np.arange(self.B) % HG
        dev = self.device

        def put(dst, src, dtype=None):
            t = torch.as_tensor(src[idx], device=dev)
            dst.copy_(t.to(dst.dtype) if dtype is None else t.to(dtype))

        e = self.env
        a = segs[:, :, 0:2]
        ee = segs[:, :, 2:4] - segs[:, :, 0:2]
        put(e._a, a)
        put(e._e, ee)
        put(e._ee, (ee * ee).sum(axis=2))
        if hasattr(e, "_a_h"):
            e._a_h.copy_(e._a.half())
            e._e_h.copy_(e._e.half())
        put(self._pose, pose)
        put(self._rx0, rects[:, :, 0]); put(self._ry0, rects[:, :, 1])
        put(self._rx1, rects[:, :, 2]); put(self._ry1, rects[:, :, 3])
        put(self._rid1, rects[:, :, 4] + 1)
        ax, ay = doors[:, :, 0], doors[:, :, 1]
        dx, dy = doors[:, :, 2] - ax, doors[:, :, 3] - ay
        L = np.maximum(np.hypot(dx, dy), 1e-6)
        put(self._dax, ax); put(self._day, ay)
        put(self._ddx, dx); put(self._ddy, dy)
        put(self._dlen2, L * L)
        put(self._dtlo, -0.12 / L); put(self._dthi, 1.0 + 0.12 / L)
        put(self._dvalid, dvalid)
        self.doors_total.copy_(self._dvalid.sum(dim=1).to(torch.float32))
        put(self.rooms_total, np.array(
            [max(1, len(w.rooms_reachable)) for w in houses], np.float32))
        put(self.cells_total, np.array(
            [cell_ceiling(w, self._gx_all, self._gy_all) for w in houses],
            np.float32))
        self.env._worlds = [_PaddedWorld(houses[i], self.caps[0])
                            for i in idx]

    def set_trims(self, left, right) -> None:
        """Per-HOUSE-COLUMN track factors (len G*K each), tiled over
        individuals so a generation stays a paired comparison. copy_ only:
        legal between CUDA-graph replays. Needs RolloutEngine(trim_rand=True)."""
        HG = self.G * self.G_sets
        idx = np.arange(self.B) % HG
        self.env._trim_l.copy_(torch.as_tensor(
            np.asarray(left, np.float32)[idx], device=self.device))
        self.env._trim_r.copy_(torch.as_tensor(
            np.asarray(right, np.float32)[idx], device=self.device))

    def _room_code_rects(self) -> torch.Tensor:
        """Room index (-1 outside every rect) by elementwise containment
        over the padded rect list — no gather/index kernels."""
        e = self.env
        x, y = e.x.unsqueeze(1), e.y.unsqueeze(1)
        inside = (x >= self._rx0) & (x < self._rx1) \
            & (y >= self._ry0) & (y < self._ry1)
        return (inside.to(torch.int64) * self._rid1).sum(dim=1) - 1

    def _doors_step(self) -> None:
        """Threshold crossings: the rover centre changes side of a door
        segment while inside its along-threshold window (+/-0.12 m past the
        jambs). Sampled every `every_door` ticks (<= 0.08 m of travel at
        v_max). _dprev == 0 (reset) only initialises."""
        e = self.env
        qx = e.x.unsqueeze(1) - self._dax
        qy = e.y.unsqueeze(1) - self._day
        side = torch.sign(self._ddx * qy - self._ddy * qx)
        t = (qx * self._ddx + qy * self._ddy) / self._dlen2
        crossed = ((side * self._dprev) < 0) & (t > self._dtlo) \
            & (t < self._dthi) & self._dvalid
        self._dused |= crossed
        self._dcross += crossed.sum(dim=1).to(torch.float32)
        self._dprev.copy_(torch.where(side != 0, side, self._dprev))

    def _compile_tick(self):
        """torch.compile the per-tick [B, 360] elementwise chains (lidar
        noise/dropout, preprocess, gate scan logic, physics step): Inductor
        fuses them into a few Triton kernels instead of ~30 materialized
        [B, 360] temporaries per tick — the traffic class the fused raycast
        removed from the scan itself. fallback_random keeps eager aten RNG
        (default CUDA generator, same call order -> graph-capture-safe and
        the paired-noise semantics unchanged). dynamic=False: shapes are
        fixed for an arena's lifetime."""
        import torch._inductor.config as ic
        ic.fallback_random = True
        opts = dict(dynamic=False, fullgraph=False)
        e = self.env
        e.step = torch.compile(e.step, **opts)
        e._dropout = torch.compile(e._dropout, **opts)
        self.gate.process_scan = torch.compile(self.gate.process_scan,
                                               **opts)
        self.gate.gate = torch.compile(self.gate.gate, **opts)
        self._preprocess = torch.compile(batched_preprocess, **opts)

    def reset(self):
        """Allocation-free: copy_ / zero_ / fill_ only, so this is legal
        inside CUDA-graph capture and cheap to replay."""
        e = self.env
        e.x.copy_(self._pose[:, 0]); e.y.copy_(self._pose[:, 1])
        e.theta.copy_(self._pose[:, 2])
        self._x0.copy_(self._pose[:, 0]); self._y0.copy_(self._pose[:, 1])
        self._spin_t.zero_(); self._range.zero_()
        e.v_left.zero_(); e.v_right.zero_(); e._prev_v.zero_()
        e.collided.zero_()
        e.wheel_l.zero_(); e.wheel_r.zero_(); e.yaw_rate.zero_()
        e.accel.zero_()
        e.accel[:, 2] = e.cfg.gravity
        self.gate.reset_state()
        if (self._noise_seed is not None
                and not torch.cuda.is_current_stream_capturing()):
            torch.manual_seed(self._noise_seed)
        self._t = 0
        for hook in self._reset_hooks:
            hook()

    def _room_code(self) -> torch.Tensor:
        e = self.env
        xv = (e.x.unsqueeze(1) > self._pp) & (self._pk == 1.0)
        yh = (e.y.unsqueeze(1) > self._pp) & (self._pk == 2.0)
        bit = (xv | yh).to(torch.int64)
        return (bit << self._shift).sum(dim=1)          # [B]

    def _cells_or(self, cells) -> None:
        """Mark the cell containing each env's pose in the [W, B] int64
        visited-word matrix. floor(x/0.8) is EXACTLY the nearest-centre
        bucket for centres 0.4+0.8k (boundaries land on 0.0/0.8/1.6...);
        63 cells per signed-int64 word (bit 63 left clear).

        TWO hard constraints from the run-5 fault class (Xid 43 illegal
        access at production size, only with the accumulator enabled):
        (1) every OP is from the capture-proven elementwise family (the
        room counter's compare/shift/bitwise_or shapes — no index kernels:
        argmin+scatter_ were the first reproducible fault); (2) every
        INTERMEDIATE is a static scratch buffer via out=/in-place — zero
        fresh allocations per sampled tick, because graph-pool
        allocations stay live across replays on this torch 2.11/GB10
        stack."""
        e = self.env
        torch.div(e.x, self._cell, out=self._tf1)
        torch.floor(self._tf1, out=self._tf1)
        self._tf1.clamp_(0, self.cell_nx - 1)
        torch.div(e.y, self._cell, out=self._tf2)
        torch.floor(self._tf2, out=self._tf2)
        self._tf2.clamp_(0, self.cell_ny - 1)
        torch.mul(self._tf2, self.cell_nx, out=self._tf3)
        torch.add(self._tf3, self._tf1, out=self._tf3)
        self._cidx.copy_(self._tf3)                    # [B] int64
        for w in range(self._ncell_words):
            lo = 63 * w
            torch.ge(self._cidx, lo, out=self._cinr)
            torch.lt(self._cidx, lo + 63, out=self._clt)
            torch.bitwise_and(self._cinr, self._clt, out=self._cinr)
            torch.sub(self._cidx, lo, out=self._cbits)
            self._cbits.clamp_(0, 63)
            torch.bitwise_left_shift(self._one64, self._cbits,
                                      out=self._cbcast)
            torch.where(self._cinr, self._cbcast, self._zero64,
                        out=self._cwhere)
            cells[w].bitwise_or_(self._cwhere)

    @torch.no_grad()
    def _rollout_body(self, policy_step, ticks, every_room_sample, acc):
        """One full game from reset. Every op is device-only (no CPU sync,
        no host control flow on tensor values), so this is legal inside a
        CUDA-graph capture AND for eager replay. `now` for the gate is the
        device tick counter advanced by add_ (a Python counter would bake a
        constant into the graph)."""
        self.reset()
        prev_act, seen, dist, coll, cells = acc
        prev_act.zero_()
        seen.zero_(); dist.zero_(); coll.zero_(); cells.zero_()
        self._fwd_m.zero_(); self._back_m.zero_()
        if self.building_mode:
            self._dprev.zero_(); self._dused.zero_(); self._dcross.zero_()
        e = self.env
        for t in range(ticks):
            self.gate._tick.add_(self.dt)          # device-side clock
            ranges = e.scan()
            scan72 = self._preprocess(ranges, e.angle_min,
                                        e.angle_increment,
                                        num_bins=NUM_BINS,
                                        max_range=MAX_RANGE)
            self.gate.process_scan(ranges, e.angle_min, e.angle_increment)
            prop = _proprio(e)
            if self.gate_obs:
                # obs v2: proprio channels 2/3 were pure gyro NOISE (no
                # signal); carry the gate's latched blocks instead — the
                # policy otherwise only sees overrides through prev_act.
                prop = torch.cat([prop[:, :2],
                                  self.gate.front_blocked[:, None].float(),
                                  self.gate._rear_blocked[:, None].float(),
                                  prop[:, 4:]], dim=1)
            obs = torch.cat([scan72, prop, prev_act], dim=1)
            if self._cam_renderer is not None:
                # Stage-4b camera: pure tensor ops into the static uint8
                # buffer (graph-legal); default-off branch, so legacy
                # numerics never change.
                self._cam_renderer.render(e, self._camera_buf)

            cmd = policy_step(obs, prev_act).clamp(-1.0, 1.0)
            if self._rec is not None:
                # BC data collection: pre-gate command (student learns the
                # pre-gate map; the arena re-gates at deploy time)
                self._rec["obs"].append(obs.clone())
                self._rec["cmd"].append(cmd.clone())
            gated = self.gate.gate(cmd)
            px, py = e.x.clone(), e.y.clone()
            pth = e.theta.clone()
            e.step(gated, self.dt)
            dist += torch.hypot(e.x - px, e.y - py)
            # signed body-frame travel (heading before the step): forward
            # vs reverse distance for the w_rev fitness term
            sd = (e.x - px) * torch.cos(pth) + (e.y - py) * torch.sin(pth)
            self._fwd_m += sd.clamp(min=0.0)
            self._back_m += (-sd).clamp(min=0.0)
            # spin tick: gate passed a genuine pivot (opposite-sign wheels
            # with meaningful magnitude on both). These rotate in place in
            # the sim; on the real skid-steer they are worse than that —
            # the loaded track stalls and the rover orbits the dead wheel.
            self._spin_t += ((gated[:, 0] * gated[:, 1] < -0.01)
                             & (gated.abs().min(dim=1).values > 0.15)
                             ).to(gated.dtype)
            # furthest distance ever reached from the start pose (graph-safe
            # in-place max): pays for genuine translation, which an in-place
            # orbit never earns no matter how much arc it accumulates.
            torch.maximum(self._range,
                          torch.hypot(e.x - self._x0, e.y - self._y0),
                          out=self._range)
            coll += e.collided.to(torch.float32)
            if t % every_room_sample == 0:
                if self.building_mode:
                    rc = self._room_code_rects()
                    seen |= torch.where(rc >= 0,
                                        self._one64 << rc.clamp(min=0),
                                        self._zero64)
                else:
                    seen |= (self._one64 << self._room_code())
            if self.building_mode and t % self.every_door == 0:
                self._doors_step()
            if self.cells_on and t % self.every_cover == 0:
                self._cells_or(cells)
            prev_act = gated

    def run_games(self, policy_step, ticks: int,
                  every_room_sample: int = 3, graph: bool = False,
                  rec: dict | None = None) -> dict:
        """policy_step(obs [B, OBS_DIM], prev_act [B, 2]) -> raw cmd [B, 2]
        (anything is clamped). Returns per-env metrics as [B] tensors.

        graph=True captures the whole rollout as one CUDA graph (replay
        skips the ~ticks*50 kernel launches — we are launch-bound at these
        batch sizes). The policy closure must then read ONLY stable buffers
        (values refreshed by copy_ outside the graph) — see evolve.evaluate.
        Falls back to eager once with a warning if capture fails."""
        if graph and str(self.device).startswith("cuda"):
            key = (ticks, every_room_sample, self.every_cover,
                   self.cells_on)
            if not hasattr(self, "_graphs"):
                self._graphs = {}
            if key in self._graphs:
                self._graphs[key].replay()
            elif getattr(self, "_graph_broken", False):
                graph = False
            else:
                try:
                    self._graphs[key] = self._capture(
                        policy_step, ticks, every_room_sample)
                    self._graphs[key].replay()
                except Exception as ex:              # noqa: BLE001
                    print(f"(arena: graph capture failed -> eager: "
                          f"{type(ex).__name__}: {ex})", flush=True)
                    self._graph_broken = True
                    graph = False
        if not graph:
            self._rec = rec
            acc = self._make_acc()
            self._rollout_body(policy_step, ticks, every_room_sample, acc)
            self._rec = None
            _, seen, dist, coll, cells = acc
        else:
            _, seen, dist, coll, cells = self._acc           # static buffers
        seen = seen.clone()                           # detach from graph pool
        rooms = torch.tensor(
            [bin(int(m)).count("1") for m in seen.cpu().tolist()],
            device=self.device, dtype=torch.float32)
        cw = (cells.clone() if graph else cells).cpu().tolist()   # [W, B]
        cov = torch.tensor(
            [sum(bin(int(cw[w][b])).count("1")
                 for w in range(len(cw))) for b in range(self.B)],
            device=self.device, dtype=torch.float32)
        out = {"rooms": rooms, "rooms_total": self.rooms_total.clone(),
               "dist_m": dist.clone(), "collisions": coll.clone(),
               "cells": torch.minimum(cov, self.cells_total),
               "cells_total": self.cells_total.clone(),
               "stops": self.gate.stops_dev.to(torch.float32).clone(),
               "fwd_m": self._fwd_m.clone(), "back_m": self._back_m.clone(),
               "net_m": torch.hypot(self.env.x - self._x0,
                                    self.env.y - self._y0),
               "range_m": self._range.clone(),
               "spin_t": self._spin_t.clone(),
               # device-side clock (graph-safe; _t is frozen at capture)
               "spin_frac": self._spin_t * self.dt \
                   / self.gate._tick.clamp(min=self.dt)}
        if self.building_mode:
            out["door_crossings"] = self._dcross.clone()
            out["doors_used"] = self._dused.sum(dim=1).to(torch.float32)
            out["doors_total"] = self.doors_total.clone()
        return out

    def _make_acc(self):
        return (torch.zeros(self.B, 2, device=self.device),
                torch.zeros(self.B, dtype=torch.int64, device=self.device),
                torch.zeros(self.B, device=self.device),
                torch.zeros(self.B, device=self.device),
                torch.zeros(self._ncell_words if self.cells_on else 1,
                            self.B, dtype=torch.int64, device=self.device))

    def drop_graphs(self):
        """Explicitly reset captured graphs BEFORE letting them go: the
        default CUDA generator stays 'graph-registered' until reset(), and a
        later eager randn (arena warmup!) errors with 'Offset increment
        outside graph capture' if it isn't cleared."""
        for g in list(self._graphs.values()):
            try:
                g.reset()
            except Exception:
                pass
        self._graphs.clear()

    def _capture(self, policy_step, ticks, every_room_sample):
        import torch.cuda
        if self._acc is None:
            self._acc = self._make_acc()
        # ONE pool handle shared by every RolloutEngine's graphs: private
        # pools faulted (illegal memory access) once >1 graph was live and
        # interleaved on GB10/torch2.11; sharing a pool is the sanctioned
        # pattern for graphs replayed sequentially (one per generation).
        if RolloutEngine._pool is None:
            RolloutEngine._pool = torch.cuda.graph_pool_handle()
        # warmup on a side stream (allocs settle, cudnn picks kernels)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._rollout_body(policy_step, ticks, every_room_sample,
                                   self._acc)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=RolloutEngine._pool):
            self._rollout_body(policy_step, ticks, every_room_sample,
                               self._acc)
        return g

    # ------------------------------------------------ policy protocol

    def add_reset_hook(self, fn):
        """Public: fn() runs at the start of every game. Deduped."""
        if fn not in self._reset_hooks:
            self._reset_hooks.append(fn)

    def run(self, policy, ticks, *, graph=False, every_room_sample=3, rec=None):
        """Protocol entry point: bind -> register reset hook -> wrap step
        with ObsSpec channel views -> run_games. graph=True captures only when
        policy.graph_safe, else forces eager with one log line."""
        if graph and not bool(getattr(policy, "graph_safe", False)):
            print("(runner: policy not graph_safe -> eager)", flush=True)
            graph = False
        spec = getattr(policy, "obs_spec", None) or DEFAULT_OBS_SPEC
        if isinstance(spec, ObsSpec) and spec.proprio_version == "v2" \
                and not self.gate_obs:
            raise ValueError("policy obs spec v2 needs engine gate_obs=True")
        if isinstance(spec, ObsSpec) and spec.camera \
                and self._cam_renderer is None:
            raise ValueError(
                "policy obs spec requests the camera channel but this engine "
                "has no camera: build it with RolloutEngine(..., "
                "camera=CameraConfig())")
        if callable(getattr(policy, "bind", None)):
            policy.bind(self.P, self.G)
        if callable(getattr(policy, "reset", None)):
            self.add_reset_hook(policy.reset)

        camera_buf = self._camera_buf       # None when camera disabled

        def policy_step(obs_flat, prev_act):
            return policy.step(
                channel_views(obs_flat, spec, camera=camera_buf), prev_act)

        return self.run_games(policy_step, ticks,
                              every_room_sample=every_room_sample, graph=graph,
                              rec=rec)


# Backwards-compatible alias: pre-platform code (and the evo shim) calls it
# `Arena`.
Arena = RolloutEngine
