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


class Fp16Env(BatchedEnv):
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

        # noise/dropout in fp32 — same RNG draw order as the fp32 env
        r = r + self.noise(c.lidar_noise_std, self.B, c.n_beams)
        drop = torch.rand(self.B, c.n_beams, generator=self._g,
                          device=self.device) < c.lidar_dropout_p
        return torch.where(drop, torch.full_like(r, torch.inf),
                           r.clamp(min=0.02))


class NoSyncGate(BatchedGate):
    """BatchedGate minus its per-tick GPU->CPU sync: `stops +=
    int(block.sum())` forces a device-host round trip EVERY tick, draining
    the launch pipeline. stops is never used for fitness, so accumulate it
    on-device instead (reported per env in run_games for forensics)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stops_dev = torch.zeros(self.B, dtype=torch.int64,
                                     device=self.device)

    def _update_front_blocked(self):
        c = self.cfg
        now = self._now()
        resume = self._front_path_dist > c.stop_distance + c.hysteresis
        held_long = (self._front_blocked_time < 0) \
            | (now - self._front_blocked_time >= c.min_block_duration)
        release = self.front_blocked & resume & held_long
        self.front_blocked = self.front_blocked & ~release
        self._front_blocked_time = torch.where(
            release, torch.full_like(self._front_blocked_time, -1.0),
            self._front_blocked_time)
        block = (~self.front_blocked
                 & (self._front_path_dist < c.stop_distance)
                 & (self._front_streak >= c.block_scans))
        self.stops_dev += block.to(torch.int64)
        self.front_blocked = self.front_blocked | block
        self._front_blocked_time = torch.where(
            block, torch.full_like(self._front_blocked_time, now),
            self._front_blocked_time)


class Arena:
    """P*G envs laid out as individual-major rows p*G+g; house g is shared
    by every individual (paired fitness across identical worlds)."""

    def __init__(self, P: int, G: int, seed: int = 777_000,
                 device: str = "cuda", rover_cfg: RoverConfig | None = None,
                 gate_cfg: GateConfig | None = None, fp16: bool = False):
        self.P, self.G = P, G
        self.B = P * G
        self.device = torch.device(device)
        self.dt = 1.0 / CONTROL_HZ
        self.fp16 = bool(fp16 and str(device).startswith("cuda"))
        self._gate_cfg = gate_cfg or GateConfig()

        rng = np.random.default_rng(seed)
        houses = [make_house(rng) for _ in range(G)]
        env_cls = Fp16Env if self.fp16 else BatchedEnv
        self.env = env_cls(self.B, rover_cfg, seed=seed, device=device)
        self.env._worlds = [houses[(b) % G] for b in range(self.B)]
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
        self.rooms_total = torch.tensor(
            [totals[b % G] for b in range(self.B)], device=self.device,
            dtype=torch.float32)

    def reset(self):
        e = self.env
        e.x, e.y, e.theta = self._pose[:, 0].clone(), \
            self._pose[:, 1].clone(), self._pose[:, 2].clone()
        e.v_left.zero_(); e.v_right.zero_(); e._prev_v.zero_()
        e.collided.zero_()
        e.wheel_l.zero_(); e.wheel_r.zero_(); e.yaw_rate.zero_()
        e.accel = torch.zeros(self.B, 3, device=self.device)
        e.accel[:, 2] = e.cfg.gravity
        self.gate = NoSyncGate(self._gate_cfg, self.B,
                               lambda: self._t * self.dt,
                               device=str(self.device))
        self._t = 0

    def _room_code(self) -> torch.Tensor:
        e = self.env
        xv = (e.x.unsqueeze(1) > self._pp) & (self._pk == 1.0)
        yh = (e.y.unsqueeze(1) > self._pp) & (self._pk == 2.0)
        bit = (xv | yh).to(torch.int64)
        shift = torch.arange(self.maxp, device=self.device,
                             dtype=torch.int64)
        return (bit << shift).sum(dim=1)          # [B]

    @torch.no_grad()
    def run_games(self, policy_step, ticks: int,
                  every_room_sample: int = 3) -> dict:
        """policy_step(obs [B, OBS_DIM], prev_act [B, 2]) -> raw cmd [B, 2]
        (anything is clamped). Returns per-env metrics as [B] tensors."""
        self.reset()
        e = self.env
        prev_act = torch.zeros(self.B, 2, device=self.device)
        seen = torch.zeros(self.B, dtype=torch.int64, device=self.device)
        one64 = torch.ones((), dtype=torch.int64, device=self.device)
        dist = torch.zeros(self.B, device=self.device)
        coll = torch.zeros(self.B, device=self.device)

        for t in range(ticks):
            self._t = t
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
                seen |= (one64 << self._room_code())
            prev_act = gated

        rooms = torch.tensor(
            [bin(int(m)).count("1") for m in seen.cpu().tolist()],
            device=self.device, dtype=torch.float32)
        return {"rooms": rooms, "rooms_total": self.rooms_total,
                "dist_m": dist, "collisions": coll,
                "stops": self.gate.stops_dev.to(torch.float32)}


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
