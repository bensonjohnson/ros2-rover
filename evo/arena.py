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
    isfinite hit-mask, exactly as their 1e6 peers do in fp32."""

    def _build_segments(self):
        super()._build_segments()                    # fp32 truth kept
        self._a_h = self._a.half()
        self._e_h = self._e.half()

    @torch.no_grad()
    def scan(self) -> torch.Tensor:
        c = self.cfg
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

    def reset_state(self):
        """In-place state zero for arena re-runs / CUDA-graph replay
        (allocating a fresh gate per game would be illegal inside a graph
        capture)."""
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
                 door_w_range: tuple = (0.7, 1.0)):
        """merged_houses=K: play each individual in K*G houses (one set per
        seed offset) inside ONE arena — lets the whole run need a single
        CUDA graph (separate live graphs fault on this stack; see
        gmbisect) and gives fitness over K x more worlds for free."""
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

        rng = np.random.default_rng(seed)
        houses = []
        for k in range(merged_houses):
            rng_k = np.random.default_rng(seed + 1_000_003 * k)
            houses += [make_house(rng_k, door_w_range=door_w_range)
                       for _ in range(G)]
        # env layout: b = p*(G*K) + k*G + g  (individual-major, then set)
        env_cls = Fp16Env if self.fp16 else Fp32Env
        self.env = env_cls(self.B, rover_cfg, seed=seed, device=device)
        self.env._worlds = [houses[b % (G * merged_houses)]
                            for b in range(self.B)]
        self.env._build_segments()

        # Cache start poses + partition geometry as device tensors.
        pose = np.array([w.start_pose for w in self.env._worlds],
                        dtype=np.float32)
        self._pose = torch.as_tensor(pose, device=self.device)
        maxp = max(1, max(len(w.partitions) for w in self.env._worlds))
        kind = np.zeros((self.B, maxp))
        pos = np.full((self.B, maxp), np.inf)   # pads: kind 0 masks them out
        for b, w in enumerate(self.env._worlds):
            for k, (kd, p) in enumerate(w.partitions):
                kind[b, k] = 1.0 if kd == "v" else 2.0
                pos[b, k] = p
        self._pk = torch.as_tensor(kind, device=self.device)
        self._pp = torch.as_tensor(pos, device=self.device)
        self.maxp = maxp

        # Reachable-room ceiling per world (same clearance-grid trick as
        # eval_checkpoints), tiled to envs.
        totals = []
        for w in houses:
            xmin, ymin, xmax, ymax = w.bounds
            gx = np.linspace(xmin + 0.4, xmax - 0.4, 24)
            gy = np.linspace(ymin + 0.4, ymax - 0.4, 24)
            rr = {w.room_id(float(x), float(y)) for x in gx for y in gy
                  if w.clearance(float(x), float(y)) > 0.2}
            totals.append(max(1, len(rr)))
        HG = G * merged_houses
        self.rooms_total = torch.tensor(
            [totals[b % HG] for b in range(self.B)], device=self.device,
            dtype=torch.float32)

        self.gate = NoSyncGate(self._gate_cfg, self.B,
                               lambda: self._t * self.dt,
                               device=str(self.device))
        self._t = 0
        # static buffers so the whole rollout is CUDA-graph capturable
        self._shift = torch.arange(self.maxp, device=self.device,
                                   dtype=torch.int64)
        self._one64 = torch.ones((), dtype=torch.int64, device=self.device)
        self._acc = None
        self._reset_hooks = []      # e.g. zero the policy's hidden state

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

    @torch.no_grad()
    def _rollout_body(self, policy_step, ticks, every_room_sample, acc):
        """One full game from reset. Every op is device-only (no CPU sync,
        no host control flow on tensor values), so this is legal inside a
        CUDA-graph capture AND for eager replay. `now` for the gate is the
        device tick counter advanced by add_ (a Python counter would bake a
        constant into the graph)."""
        self.reset()
        prev_act, seen, dist, coll = acc
        prev_act.zero_()
        seen.zero_(); dist.zero_(); coll.zero_()
        e = self.env
        for t in range(ticks):
            self.gate._tick.add_(self.dt)          # device-side clock
            ranges = e.scan()
            scan72 = batched_preprocess(ranges, e.angle_min,
                                        e.angle_increment,
                                        num_bins=NUM_BINS,
                                        max_range=MAX_RANGE)
            self.gate.process_scan(ranges, e.angle_min, e.angle_increment)
            obs = torch.cat([scan72, _proprio(e), prev_act], dim=1)

            cmd = policy_step(obs, prev_act).clamp(-1.0, 1.0)
            gated = self.gate.gate(cmd)
            px, py = e.x.clone(), e.y.clone()
            e.step(gated, self.dt)
            dist += torch.hypot(e.x - px, e.y - py)
            coll += e.collided.to(torch.float32)
            if t % every_room_sample == 0:
                seen |= (self._one64 << self._room_code())
            prev_act = gated

    def run_games(self, policy_step, ticks: int,
                  every_room_sample: int = 3, graph: bool = False) -> dict:
        """policy_step(obs [B, OBS_DIM], prev_act [B, 2]) -> raw cmd [B, 2]
        (anything is clamped). Returns per-env metrics as [B] tensors.

        graph=True captures the whole rollout as one CUDA graph (replay
        skips the ~ticks*50 kernel launches — we are launch-bound at these
        batch sizes). The policy closure must then read ONLY stable buffers
        (values refreshed by copy_ outside the graph) — see evolve.evaluate.
        Falls back to eager once with a warning if capture fails."""
        if graph and str(self.device).startswith("cuda"):
            key = (ticks, every_room_sample)
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
            acc = self._make_acc()
            self._rollout_body(policy_step, ticks, every_room_sample, acc)
            _, seen, dist, coll = acc
        else:
            _, seen, dist, coll = self._acc           # static buffers
        seen = seen.clone()                           # detach from graph pool
        rooms = torch.tensor(
            [bin(int(m)).count("1") for m in seen.cpu().tolist()],
            device=self.device, dtype=torch.float32)
        return {"rooms": rooms, "rooms_total": self.rooms_total,
                "dist_m": dist.clone(), "collisions": coll.clone(),
                "stops": self.gate.stops_dev.to(torch.float32).clone()}

    def _make_acc(self):
        return (torch.zeros(self.B, 2, device=self.device),
                torch.zeros(self.B, dtype=torch.int64, device=self.device),
                torch.zeros(self.B, device=self.device),
                torch.zeros(self.B, device=self.device))

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


def fitness(metrics: dict, w_dist: float = 0.03,
            w_coll: float = 0.25) -> torch.Tensor:
    """Per-env scalar: fraction of reachable rooms visited (the honest
    target), distance as a small tiebreaker (stops are zero-effort rooms),
    collisions penalized (the gate is there; brute-forcing it shouldn't pay).
    """
    frac = metrics["rooms"] / metrics["rooms_total"].clamp(min=1.0)
    return (frac
            + w_dist * metrics["dist_m"] / 10.0
            - w_coll * metrics["collisions"] / 100.0)
