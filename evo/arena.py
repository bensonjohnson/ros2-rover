"""Evolution arena: P individuals x G unseen houses, frozen rollouts.

Reuses pnn_sim.batched.env verbatim (physics, lidar raycast, safety gate)
but feeds it external controllers instead of the PC brain. Everything that
has to happen per tick stays on GPU — including the GROUND-TRUTH room
counter: partition lines become padded [B, maxP] tensors and a room visit
is one bit in a per-env int64 bitmask, so fitness needs zero CPU syncs.

Same-G houses are shared across individuals (paired comparison); worlds
come from a fixed eval seed the evolution never sees elsewhere.

GB10 fast path (Arena(fp16=True)) — three optimizations for the Spark:
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

from pnn_sim.world import make_house
from pnn_sim.rover import RoverConfig
from pnn_sim.safety_gate import GateConfig
from pnn_sim.batched.env import BatchedEnv, BatchedGate, batched_preprocess

NUM_BINS = 72
MAX_RANGE = 5.0
N_PROPRIO = 8
N_LAST_ACT = 2
OBS_DIM = NUM_BINS + N_PROPRIO + N_LAST_ACT
CONTROL_HZ = 15.0


def _proprio(env: BatchedEnv) -> torch.Tensor:
    """Same 8-channel proprio and normalization the PC trainer used — the
    sensors exist, the policy just has to make something of them."""
    e = env
    return torch.stack([
        (0.5 + 0.5 * e.wheel_l / 8.0).clamp(0.0, 1.0),
        (0.5 + 0.5 * e.wheel_r / 8.0).clamp(0.0, 1.0),
        (0.5 + 0.5 * e.noise(e.cfg.gyro_noise_std) / 2.5).clamp(0.0, 1.0),
        (0.5 + 0.5 * e.noise(e.cfg.gyro_noise_std) / 2.5).clamp(0.0, 1.0),
        (0.5 + 0.5 * e.yaw_rate / 2.5).clamp(0.0, 1.0),
        (0.5 + 0.5 * e.accel[:, 0] / 19.6).clamp(0.0, 1.0),
        (0.5 + 0.5 * e.accel[:, 1] / 19.6).clamp(0.0, 1.0),
        (0.5 + 0.5 * e.accel[:, 2] / 19.6).clamp(0.0, 1.0),
    ], dim=1)


class _DefaultRNGMixin:
    """Route all randomness through the DEFAULT CUDA generator (registered
    natively for CUDA-graph capture; custom generators need
    register_generator_state and error out otherwise). Arena determinism
    comes from torch.manual_seed at reset when noise_seed is set."""

    def noise(self, std: float, *shape) -> torch.Tensor:
        sh = shape if shape else (self.B,)
        return torch.randn(*sh, device=self.device) * std

    def _dropout(self, r: torch.Tensor, p: float) -> torch.Tensor:
        drop = torch.rand(r.shape, device=self.device) < p
        return torch.where(drop, torch.full_like(r, torch.inf),
                           r.clamp(min=0.02))


class Fp32Env(_DefaultRNGMixin, BatchedEnv):
    """BatchedEnv with generator-free scan (graph-capturable)."""

    @torch.no_grad()
    def scan(self) -> torch.Tensor:
        c = self.cfg
        ang = self.theta.unsqueeze(1) + self._beam_offsets
        d = torch.stack([torch.cos(ang), torch.sin(ang)], dim=2)
        p = torch.stack([self.x, self.y], dim=1)
        q = self._a - p.unsqueeze(1)
        cross_eq = (self._e[:, :, 0] * q[:, :, 1]
                    - self._e[:, :, 1] * q[:, :, 0])
        cross_ed = (self._e[:, None, :, 0] * d[:, :, None, 1]
                    - self._e[:, None, :, 1] * d[:, :, None, 0])
        cross_dq = (d[:, :, None, 0] * q[:, None, :, 1]
                    - d[:, :, None, 1] * q[:, None, :, 0])
        t = cross_eq.unsqueeze(1) / cross_ed
        s = cross_dq / cross_ed
        hit = (t > 1e-9) & (s >= 0.0) & (s <= 1.0) & torch.isfinite(t)
        t = torch.where(hit, t, torch.full_like(t, torch.inf))
        r = t.min(dim=2).values.clamp(max=c.lidar_max_range)
        r = r + self.noise(c.lidar_noise_std, self.B, c.n_beams)
        return self._dropout(r, c.lidar_dropout_p)


class Fp16Env(_DefaultRNGMixin, BatchedEnv):
    """BatchedEnv with the raycast in fp16 (GB10: bandwidth-bound, ~2x).
    ONLY scan() runs half-precision: poses/velocities/proprio and the
    collision check (_clearance) stay fp32, so ground truth stays exact and
    fitness cannot be inflated by precision artifacts — fp16 can only make
    the *sensing* slightly wrong, identically for every individual.
    Dummy pad segments overflow to inf in fp16 and drop out through the
    isfinite hit-mask, exactly as their 1e6 peers do in fp32.

    fused=True swaps the fp16 op-chain for the single fused Triton kernel
    (evo.fused_scan): no [B, 360, M] intermediates at all — measured 31 ms
    -> 1.5 ms per tick at B=8192 on GB10, parity 99.83% of beams < 0.05 m
    (grazing-ray fp16 differences; deterministic across calls). The output
    is a STATIC buffer (out=): fresh allocations inside a captured graph
    are the run-5 Xid-43 fault class — zero-alloc is mandatory here."""

    fused = False                       # Arena sets per config

    def _build_segments(self):
        super()._build_segments()                    # fp32 truth kept
        self._a_h = self._a.half()
        self._e_h = self._e.half()
        if self.fused:
            self._scan_buf = torch.zeros(
                self.B, self.cfg.n_beams, device=self.device,
                dtype=torch.float32)

    @torch.no_grad()
    def scan(self) -> torch.Tensor:
        c = self.cfg
        if self.fused:
            from .fused_scan import fused_scan
            r = fused_scan(self, out=self._scan_buf)   # clean, static buf
            r = r + self.noise(c.lidar_noise_std, self.B, c.n_beams)
            return self._dropout(r, c.lidar_dropout_p)
        ang = self.theta.unsqueeze(1) + self._beam_offsets   # [B, nb] fp32
        d = torch.stack([torch.cos(ang), torch.sin(ang)], dim=2)
        d_h = d.half()
        p = torch.stack([self.x, self.y], dim=1).half()
        q = self._a_h - p.unsqueeze(1)                     # [B, M, 2]
        e = self._e_h

        cross_eq = e[:, :, 0] * q[:, :, 1] - e[:, :, 1] * q[:, :, 0]
        cross_ed = (e[:, None, :, 0] * d_h[:, :, None, 1]
                    - e[:, None, :, 1] * d_h[:, :, None, 0])  # [B,nb,M]
        cross_dq = (d_h[:, :, None, 0] * q[:, None, :, 1]
                    - d_h[:, :, None, 1] * q[:, None, :, 0])
        t = cross_eq.unsqueeze(1) / cross_ed
        s = cross_dq / cross_ed
        hit = (t > 1e-9) & (s >= 0.0) & (s <= 1.0) & torch.isfinite(t)
        t = torch.where(hit, t, torch.full_like(t, torch.inf))
        r = t.min(dim=2).values.float().clamp(max=c.lidar_max_range)

        r = r + self.noise(c.lidar_noise_std, self.B, c.n_beams)
        return self._dropout(r, c.lidar_dropout_p)


class NoSyncGate(BatchedGate):
    """BatchedGate minus its per-tick GPU->CPU sync: `stops +=
    int(block.sum())` forces a device-host round trip EVERY tick, draining
    the launch pipeline. stops is never used for fitness, so accumulate it
    on-device instead (reported per env in run_games for forensics)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stops_dev = torch.zeros(self.B, dtype=torch.int64,
                                     device=self.device)
        # Device-side clock (Arena._tdev): a Python counter would be baked
        # as a constant into a captured CUDA graph, freezing the hold-timer.
        self._tick = torch.zeros((), device=self.device)
        self._neg1 = torch.full((), -1.0, device=self.device)

    def _update_front_blocked(self):
        c = self.cfg
        now = self._tick
        resume = self._front_path_dist > c.stop_distance + c.hysteresis
        held_long = (self._front_blocked_time < 0) \
            | (now - self._front_blocked_time >= c.min_block_duration)
        release = self.front_blocked & resume & held_long
        self.front_blocked = self.front_blocked & ~release
        self._front_blocked_time = torch.where(
            release, self._neg1, self._front_blocked_time)
        block = (~self.front_blocked
                 & (self._front_path_dist < c.stop_distance)
                 & (self._front_streak >= c.block_scans))
        self.stops_dev += block.to(torch.int64)
        self.front_blocked = self.front_blocked | block
        self._front_blocked_time = torch.where(
            block, now, self._front_blocked_time)

    # ---------------------------------------------- symmetric gate (run 16)
    # The stock gate is FRONT-heavy: forward motion gets a predicted-path
    # corridor, a latched hold, and side equalization over the FRONT flanks
    # (30-90 deg); reverse only gets a 150-180 deg rear cone at 0.30 m and
    # no flank checks at all (90-150 deg is unwatched). Runs 11-15 evolved
    # reverse explorers exploiting exactly that (mirror test: the reverse
    # champion driven forward trips the front gate 67x and collides 862x).
    # symmetric=True mirrors every forward rule for reverse: rear corridor
    # (arc-shifted by the reverse command), same stop distance, hold and
    # hysteresis latch, and rear-flank (90-150 deg) side equalization.
    symmetric = False

    def enable_symmetric(self):
        z = dict(device=self.device)
        md = self.cfg.max_eval_distance
        self.symmetric = True
        self._rpath = torch.full((self.B,), md, **z)
        self._rstreak = torch.zeros(self.B, dtype=torch.long, **z)
        self._rblocked = torch.zeros(self.B, dtype=torch.bool, **z)
        self._rblocked_time = torch.full((self.B,), -1.0, **z)
        self._rleft = torch.full((self.B,), md, **z)
        self._rright = torch.full((self.B,), md, **z)
        self.rstops_dev = torch.zeros(self.B, dtype=torch.int64, **z)

    def _rear_geom(self, n, angle_min, angle_increment):
        key = (n, float(angle_min), float(angle_increment))
        g = getattr(self, "_rgeom", None)
        if g is None or g[0] != key:
            dev = self.device
            a = torch.arange(n, device=dev, dtype=torch.float32) \
                * angle_increment + angle_min
            a = torch.remainder(a + np.pi, 2 * np.pi) - np.pi
            g = (key, ((a > 1.57) & (a < 2.62),          # rear-left flank
                       (a < -1.57) & (a > -2.62)))       # rear-right flank
            self._rgeom = g
        return g[1]

    def process_scan(self, ranges, angle_min, angle_increment):
        super().process_scan(ranges, angle_min, angle_increment)
        if not self.symmetric:
            return
        c = self.cfg
        n = ranges.shape[1]
        cos_a, sin_a, _, _, _ = self._scan_geom(n, angle_min, angle_increment)
        rl_m, rr_m = self._rear_geom(n, angle_min, angle_increment)
        valid = (torch.isfinite(ranges) & (ranges > c.min_valid_range)
                 & (ranges <= c.max_eval_distance))
        r = torch.where(valid, ranges, torch.full_like(ranges, 1e6))
        x, y = r * cos_a, r * sin_a
        xb = -x - c.robot_front_offset              # distance behind bumper
        arc = (self._cmd_lin < -0.02) & (self._cmd_ang.abs() > 0.1)
        v = (-self._cmd_lin).clamp(min=0.05).unsqueeze(1)
        xbf = xb.clamp(min=0.0)
        # reverse arc: lateral offset of the path s behind = -w s^2 / (2|v|)
        y_shift = (-self._cmd_ang.unsqueeze(1) * xbf * xbf
                   / (2.0 * v)).clamp(-0.30, 0.30)
        y_eff = torch.where(arc.unsqueeze(1), y - y_shift, y)
        in_path = valid & (y_eff.abs() <= c.robot_half_width) & (xb > 0)
        big = torch.full_like(xb, torch.inf)
        rp = torch.where(in_path, xb, big).min(dim=1).values
        self._rpath = torch.where(torch.isfinite(rp), rp.clamp(min=0.0),
                                  torch.full_like(rp, c.max_eval_distance))
        close = (in_path & (xb < c.stop_distance)).sum(dim=1)
        self._rstreak = torch.where(close >= c.min_block_points,
                                    self._rstreak + 1,
                                    torch.zeros_like(self._rstreak))

        def sector_min(mask):
            rr = torch.where(valid & mask.unsqueeze(0), r,
                             torch.full_like(r, torch.inf)).min(dim=1).values
            return torch.where(torch.isfinite(rr), rr,
                               torch.full_like(rr, c.max_eval_distance))
        self._rleft = sector_min(rl_m)
        self._rright = sector_min(rr_m)
        # latch: mirror of _update_front_blocked
        now = self._tick
        resume = self._rpath > c.stop_distance + c.hysteresis
        held = (self._rblocked_time < 0) \
            | (now - self._rblocked_time >= c.min_block_duration)
        release = self._rblocked & resume & held
        self._rblocked = self._rblocked & ~release
        self._rblocked_time = torch.where(release, self._neg1,
                                          self._rblocked_time)
        block = (~self._rblocked & (self._rpath < c.stop_distance)
                 & (self._rstreak >= c.block_scans))
        self.rstops_dev += block.to(torch.int64)
        self._rblocked = self._rblocked | block
        self._rblocked_time = torch.where(block, now, self._rblocked_time)

    def gate(self, cmd):
        if not self.symmetric:
            return super().gate(cmd)
        c = self.cfg
        left = cmd[:, 0].clone()
        right = cmd[:, 1].clone()
        self._cmd_lin = (left + right) / 2.0 * 0.154
        self._cmd_ang = (right - left) / c.track_width
        fb = self.front_blocked
        left = torch.where(fb, left.clamp(max=0.0), left)
        right = torch.where(fb, right.clamp(max=0.0), right)
        rb = self._rblocked
        left = torch.where(rb, left.clamp(min=0.0), left)
        right = torch.where(rb, right.clamp(min=0.0), right)
        avg = (left + right) / 2.0
        zero_turn = (avg.abs() < 0.05) & ((left - right).abs() > 0.1)
        turn = right - left
        fwd = ~zero_turn & (avg > 0.01)
        eq_left = fwd & (turn > 0.1) & (self._left < c.stop_distance)
        left = torch.where(eq_left, right, left)
        eq_right = fwd & (turn < -0.1) & (self._right < c.stop_distance)
        right = torch.where(eq_right, left, right)
        # reverse + CCW turn swings the tail toward the RIGHT flank
        bwd = ~zero_turn & (avg < -0.01)
        turn = right - left
        eq_rr = bwd & (turn > 0.1) & (self._rright < c.stop_distance)
        right = torch.where(eq_rr, left, right)
        eq_rl = bwd & (turn < -0.1) & (self._rleft < c.stop_distance)
        left = torch.where(eq_rl, right, left)
        return torch.stack([left, right], dim=1)

    def reset_state(self):
        """In-place state zero for arena re-runs / CUDA-graph replay
        (allocating a fresh gate per game would be illegal inside a graph
        capture)."""
        if self.symmetric:
            md = self.cfg.max_eval_distance
            self._rpath.fill_(md); self._rstreak.zero_()
            self._rblocked.zero_(); self._rblocked_time.fill_(-1.0)
            self._rleft.fill_(md); self._rright.fill_(md)
            self.rstops_dev.zero_()
        self.front_blocked.zero_()
        self._rear_blocked.zero_()
        self._front_blocked_time.fill_(-1.0)
        self._front_streak.zero_()
        self._rear_streak.zero_()
        self._front_path_dist.fill_(self.cfg.max_eval_distance)
        for t in (self._left, self._right, self._rear):
            t.fill_(self.cfg.max_eval_distance)
        self._cmd_lin.zero_()
        self._cmd_ang.zero_()
        self.stops_dev.zero_()
        self._tick.zero_()


class _PaddedWorld:
    """Segment view of a Building padded to a fixed segment cap, so
    BatchedEnv._build_segments allocates [B, cap] once (load_worlds keeps
    that shape). Everything else delegates to the wrapped world."""

    _DUMMY = np.asarray([1e6, 1e6, 1e6 + 0.1, 1e6])

    def __init__(self, w, cap: int):
        self._w = w
        n = w.segments.shape[0]
        assert n <= cap, f"{n} segments > cap {cap}"
        self.segments = np.concatenate(
            [w.segments, np.tile(self._DUMMY, (cap - n, 1))], 0)

    def __getattr__(self, name):
        return getattr(self._w, name)


def world_caps(houses) -> tuple:
    """(segments, rects, doors) capacity covering every house."""
    return (max(w.segments.shape[0] for w in houses),
            max(max(1, w.room_rects.shape[0]) for w in houses),
            max(max(1, int(((w.doors[:, 4] >= 0) & (w.doors[:, 5] >= 0)).sum()))
                for w in houses))


def cell_ceiling(w, gx_all, gy_all, half: float = 0.4) -> int:
    """Coverage ceiling of a Building: lattice boxes the rover centre can
    actually enter (reachability raster), cached per lattice."""
    key = (len(gx_all), len(gy_all), float(gx_all[0]))
    cache = w.__dict__.setdefault("_cell_ceiling", {})
    if key not in cache:
        cache[key] = max(1, sum(
            1 for gy in gy_all for gx in gx_all
            if w.reachable_in_box(gx - half, gy - half, gx + half, gy + half)))
    return cache[key]


class Arena:
    """P*G envs laid out as individual-major rows p*G+g; house g is shared
    by every individual (paired fitness across identical worlds)."""

    # one CUDA graph memory pool shared by every Arena in the process
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
                 gate_symmetric: bool = False,
                 extent: float = 16.0, every_door: int = 6):
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
        mode (worlds=None) is byte-for-byte the runs 1-9 arena."""
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
        self._preprocess = batched_preprocess
        self.gate_obs = bool(gate_obs)
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
        if self.building_mode:
            self._init_building_buffers(houses)
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
               "fwd_m": self._fwd_m.clone(), "back_m": self._back_m.clone()}
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
        # ONE pool handle shared by every Arena's graphs: private pools
        # faulted (illegal memory access) once >1 graph was live and
        # interleaved on GB10/torch2.11; sharing a pool is the sanctioned
        # pattern for graphs replayed sequentially (one per generation).
        if Arena._pool is None:
            Arena._pool = torch.cuda.graph_pool_handle()
        # warmup on a side stream (allocs settle, cudnn picks kernels)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._rollout_body(policy_step, ticks, every_room_sample,
                                   self._acc)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=Arena._pool):
            self._rollout_body(policy_step, ticks, every_room_sample,
                               self._acc)
        return g


def rev_frac(metrics: dict) -> torch.Tensor:
    """Fraction of travelled distance driven in reverse (0 if static)."""
    tot = metrics["fwd_m"] + metrics["back_m"]
    return torch.where(tot > 1e-3, metrics["back_m"] / tot.clamp(min=1e-3),
                       torch.zeros_like(tot))


def fitness(metrics: dict, w_dist: float = 0.03,
            w_coll: float = 0.25, w_cov: float = 0.0,
            w_rev: float = 0.0) -> torch.Tensor:
    """Per-env scalar: fraction of reachable rooms visited (the honest
    target), distance as a small tiebreaker (stops are zero-effort rooms),
    collisions penalized (the gate is there; brute-forcing it shouldn't pay).

    w_cov > 0 adds the novelty term (run 5): fraction of reachable 0.8 m
    cells visited. Rooms is a sparse cliff that pays only on the crossing
    tick and gives the wall-hugger plateau nothing to steer on; coverage
    pays densely from tick one and saturates at the house ceiling (no
    novelty fountain). At w_cov ~0.3 coverage can buy one room crossing'
    worth of fitness — enough to pull a policy toward a door, not enough
    to replace the rooms term.

    w_rev > 0 (run 15) subtracts w_rev x the fraction of distance driven in
    reverse. The runs 11-14 champions explore mostly BACKWARDS (sim mean
    command -0.72; real rover 75% reversing) because nothing priced
    direction; the real rover's bumper geometry and camera face forward.
    Scale-free, so a short reverse to escape a front block costs little."""
    frac = metrics["rooms"] / metrics["rooms_total"].clamp(min=1.0)
    out = (frac
           + w_dist * metrics["dist_m"] / 10.0
           - w_coll * metrics["collisions"] / 100.0)
    if w_cov:
        out = out + w_cov * metrics["cells"] / metrics["cells_total"].clamp(min=1.0)
    if w_rev:
        out = out - w_rev * rev_frac(metrics)
    return out
