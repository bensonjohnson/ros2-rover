"""B parallel worlds, rovers, and safety gates — torch, GPU-resident.

Same physics, sensors, and gate logic as the single-env modules
(pnn_sim/rover.py, pnn_sim/safety_gate.py), with every scalar state
promoted to a [B] tensor and the whole pipeline living on `device`. The
raycast is the reason: at thousands of envs it's hundreds of millions of
ray-segment intersections per tick — brute-force tensor math that a GPU
eats and a single CPU core chokes on. (No RT cores involved: those need
OptiX/BVH pipelines; this is plain batched arithmetic, which is plenty.)

Worlds have different segment counts, so segment tensors are padded to the
max with far-away dummies (they never intersect a ray inside a house).
"""

from __future__ import annotations

import numpy as np
import torch

from ..world import make_house
from ..rover import RoverConfig
from ..safety_gate import GateConfig, _SIDE_MIN, _SIDE_MAX, _REAR_MIN

_DUMMY_SEG = [1e6, 1e6, 1e6 + 0.1, 1e6]


def batched_preprocess(ranges: torch.Tensor, angle_min: float,
                       angle_increment: float, num_bins: int = 72,
                       max_range: float = 5.0,
                       min_range: float = 0.05) -> torch.Tensor:
    """preprocess_scan over a batch: ranges [B, n] -> [B, num_bins], same
    device. Beam->bin layout is shared (same lidar geometry every env), so
    the min-pool is one scatter_reduce."""
    B, n = ranges.shape
    dev = ranges.device

    clean = ranges.clone()
    invalid = ~torch.isfinite(clean) | (clean <= 0.0) | (clean < min_range)
    clean[invalid] = max_range
    clean.clamp_(min_range, max_range)

    angles = angle_min + torch.arange(n, device=dev,
                                      dtype=torch.float32) * angle_increment
    frac = torch.remainder(angles, 2.0 * np.pi) / (2.0 * np.pi)
    bins = (frac * num_bins).long().clamp(max=num_bins - 1)   # [n]

    out = torch.full((B, num_bins), max_range, device=dev)
    out.scatter_reduce_(1, bins.unsqueeze(0).expand(B, n), clean,
                        reduce="amin")
    return out / max_range


class BatchedEnv:
    """B rovers in B houses. All state is [B] torch on `device`."""

    def __init__(self, batch: int, rover_cfg: RoverConfig | None = None,
                 seed: int = 0, device: str = "cpu"):
        self.B = batch
        self.cfg = rover_cfg or RoverConfig()
        self.device = torch.device(device)
        self.world_rng = np.random.default_rng(seed)
        self._g = torch.Generator(device=self.device)
        self._g.manual_seed(seed + 1)

        c = self.cfg
        dev = self.device
        z = lambda: torch.zeros(batch, device=dev)
        self.x, self.y, self.theta = z(), z(), z()
        self.v_left, self.v_right, self._prev_v = z(), z(), z()
        self.collided = torch.zeros(batch, dtype=torch.bool, device=dev)
        self.wheel_l, self.wheel_r, self.yaw_rate = z(), z(), z()
        self.accel = torch.zeros(batch, 3, device=dev)
        self.accel[:, 2] = c.gravity

        self._worlds = [make_house(self.world_rng) for _ in range(batch)]
        self._build_segments()
        pose = np.array([w.start_pose for w in self._worlds])
        self.x = torch.as_tensor(pose[:, 0], dtype=torch.float32, device=dev)
        self.y = torch.as_tensor(pose[:, 1], dtype=torch.float32, device=dev)
        self.theta = torch.as_tensor(pose[:, 2], dtype=torch.float32, device=dev)

        inc = 2.0 * np.pi / c.n_beams
        self._beam_offsets = (torch.arange(c.n_beams, device=dev,
                                           dtype=torch.float32) * inc)
        self.angle_min = 0.0
        self.angle_increment = inc

    def noise(self, std: float, *shape) -> torch.Tensor:
        sh = shape if shape else (self.B,)
        return torch.randn(*sh, generator=self._g, device=self.device) * std

    def _build_segments(self):
        Mmax = max(w.segments.shape[0] for w in self._worlds)
        segs = np.tile(np.asarray(_DUMMY_SEG, dtype=np.float32),
                       (self.B, Mmax, 1))
        for b, w in enumerate(self._worlds):
            segs[b, :w.segments.shape[0]] = w.segments.astype(np.float32)
        t = torch.as_tensor(segs, device=self.device)
        self._a = t[:, :, 0:2].contiguous()                # [B, M, 2]
        self._e = (t[:, :, 2:4] - t[:, :, 0:2]).contiguous()
        self._ee = (self._e * self._e).sum(dim=2)          # [B, M]

    def switch_world(self, due: np.ndarray):
        """Replace the houses of envs flagged in bool mask `due`."""
        ids = np.flatnonzero(due)
        for b in ids:
            self._worlds[b] = make_house(self.world_rng)
        self._build_segments()
        idx = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        pose = torch.as_tensor(
            np.array([self._worlds[b].start_pose for b in ids]),
            dtype=torch.float32, device=self.device)
        self.x[idx], self.y[idx], self.theta[idx] = \
            pose[:, 0], pose[:, 1], pose[:, 2]
        self.v_left[idx] = self.v_right[idx] = self._prev_v[idx] = 0.0

    # ------------------------------------------------------------------

    def _clearance(self, px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
        """Min distance from points [B] to each env's own walls -> [B]."""
        p = torch.stack([px, py], dim=1).unsqueeze(1)       # [B, 1, 2]
        ap = p - self._a                                    # [B, M, 2]
        tt = ((ap * self._e).sum(dim=2)
              / self._ee.clamp(min=1e-12)).clamp(0.0, 1.0)
        closest = self._a + tt.unsqueeze(2) * self._e
        d2 = ((p - closest) ** 2).sum(dim=2)                # [B, M]
        return d2.min(dim=1).values.sqrt()

    @torch.no_grad()
    def step(self, cmd: torch.Tensor, dt: float):
        """cmd [B, 2] on device, in [-1, 1] — SimRover.step semantics."""
        c = self.cfg
        cl = cmd[:, 0].clamp(-1.0, 1.0)
        cr = cmd[:, 1].clamp(-1.0, 1.0)
        tl = torch.where(cl.abs() < c.deadband,
                         torch.zeros_like(cl), cl * (c.v_max * c.left_trim))
        tr = torch.where(cr.abs() < c.deadband,
                         torch.zeros_like(cr), cr * (c.v_max * c.right_trim))

        k = 1.0 - float(np.exp(-dt / c.motor_tau))
        self.v_left += (tl - self.v_left) * k
        self.v_right += (tr - self.v_right) * k

        vl = self.v_left * (1.0 + self.noise(c.slip_std))
        vr = self.v_right * (1.0 + self.noise(c.slip_std))
        v = 0.5 * (vl + vr)
        w = (vr - vl) / c.track_width

        nx = self.x + v * torch.cos(self.theta) * dt
        ny = self.y + v * torch.sin(self.theta) * dt
        self.collided = self._clearance(nx, ny) < c.robot_radius
        ok = ~self.collided
        self.x = torch.where(ok, nx, self.x)
        self.y = torch.where(ok, ny, self.y)
        v = torch.where(ok, v, torch.zeros_like(v))
        self.theta = torch.remainder(self.theta + w * dt + np.pi,
                                     2 * np.pi) - np.pi

        self.wheel_l = self.v_left / c.wheel_radius + self.noise(0.05)
        self.wheel_r = self.v_right / c.wheel_radius + self.noise(0.05)
        self.yaw_rate = w + self.noise(c.gyro_noise_std)
        ax = (v - self._prev_v) / dt + self.noise(c.accel_noise_std)
        ay = v * w + self.noise(c.accel_noise_std)
        az = c.gravity + self.noise(c.accel_noise_std)
        self.accel = torch.stack([ax, ay, az], dim=1)
        self._prev_v = v

    # ------------------------------------------------------------------

    @torch.no_grad()
    def scan(self) -> torch.Tensor:
        """One lidar rev per env -> ranges [B, n_beams] on device
        (beam 0 = forward)."""
        c = self.cfg
        ang = self.theta.unsqueeze(1) + self._beam_offsets   # [B, nb]
        d = torch.stack([torch.cos(ang), torch.sin(ang)], dim=2)
        p = torch.stack([self.x, self.y], dim=1)
        q = self._a - p.unsqueeze(1)                         # [B, M, 2]

        cross_eq = (self._e[:, :, 0] * q[:, :, 1]
                    - self._e[:, :, 1] * q[:, :, 0])         # [B, M]
        cross_ed = (self._e[:, None, :, 0] * d[:, :, None, 1]
                    - self._e[:, None, :, 1] * d[:, :, None, 0])  # [B,nb,M]
        cross_dq = (d[:, :, None, 0] * q[:, None, :, 1]
                    - d[:, :, None, 1] * q[:, None, :, 0])        # [B,nb,M]
        t = cross_eq.unsqueeze(1) / cross_ed
        s = cross_dq / cross_ed
        hit = (t > 1e-9) & (s >= 0.0) & (s <= 1.0) & torch.isfinite(t)
        t = torch.where(hit, t, torch.full_like(t, torch.inf))
        r = t.min(dim=2).values.clamp(max=c.lidar_max_range)

        r = r + self.noise(c.lidar_noise_std, self.B, c.n_beams)
        drop = torch.rand(self.B, c.n_beams, generator=self._g,
                          device=self.device) < c.lidar_dropout_p
        return torch.where(drop, torch.full_like(r, torch.inf),
                           r.clamp(min=0.02))


class BatchedGate:
    """Vectorized port of SimSafetyGate, on device. All state machines are
    [B] tensors; the logic follows the scalar implementation branch for
    branch."""

    def __init__(self, cfg: GateConfig, batch: int, time_fn,
                 device: str = "cpu"):
        self.cfg = cfg
        self.B = batch
        self._now = time_fn
        self.device = torch.device(device)
        dev = self.device
        md = cfg.max_eval_distance
        self.front_blocked = torch.zeros(batch, dtype=torch.bool, device=dev)
        self._rear_blocked = torch.zeros(batch, dtype=torch.bool, device=dev)
        self._front_blocked_time = torch.full((batch,), -1.0, device=dev)
        self._front_streak = torch.zeros(batch, dtype=torch.long, device=dev)
        self._rear_streak = torch.zeros(batch, dtype=torch.long, device=dev)
        self._front_path_dist = torch.full((batch,), md, device=dev)
        self._left = torch.full((batch,), md, device=dev)
        self._right = torch.full((batch,), md, device=dev)
        self._rear = torch.full((batch,), md, device=dev)
        self._cmd_lin = torch.zeros(batch, device=dev)
        self._cmd_ang = torch.zeros(batch, device=dev)
        self.stops = 0

    @torch.no_grad()
    def process_scan(self, ranges: torch.Tensor, angle_min: float,
                     angle_increment: float):
        c = self.cfg
        B, n = ranges.shape
        dev = self.device
        angles = torch.arange(n, device=dev, dtype=torch.float32) \
            * angle_increment + angle_min
        angles = torch.remainder(angles + np.pi, 2 * np.pi) - np.pi

        valid = (torch.isfinite(ranges)
                 & (ranges > c.min_valid_range)
                 & (ranges <= c.max_eval_distance))           # [B, n]
        # Large-but-finite for invalid beams: every consumer masks on
        # `valid`, and inf would 0*inf -> nan in the arc-shift math.
        r = torch.where(valid, ranges, torch.full_like(ranges, 1e6))
        x = r * torch.cos(angles).unsqueeze(0)
        y = r * torch.sin(angles).unsqueeze(0)
        x_bumper = x - c.robot_front_offset

        # Front corridor with per-env arc shift.
        arc = (self._cmd_lin > 0.02) & (self._cmd_ang.abs() > 0.1)   # [B]
        v = self._cmd_lin.clamp(min=0.05).unsqueeze(1)
        x_fwd = x_bumper.clamp(min=0.0)
        y_shift = (self._cmd_ang.unsqueeze(1) * x_fwd * x_fwd
                   / (2.0 * v)).clamp(-0.30, 0.30)
        y_eff = torch.where(arc.unsqueeze(1), y - y_shift, y)
        in_path = valid & (y_eff.abs() <= c.robot_half_width) & (x_bumper > 0)

        big = torch.full_like(x_bumper, torch.inf)
        front = torch.where(in_path, x_bumper, big).min(dim=1).values
        self._front_path_dist = torch.where(
            torch.isfinite(front), front.clamp(min=0.0),
            torch.full_like(front, c.max_eval_distance))

        close_pts = (in_path & (x_bumper < c.stop_distance)).sum(dim=1)
        qual = close_pts >= c.min_block_points
        self._front_streak = torch.where(qual, self._front_streak + 1,
                                         torch.zeros_like(self._front_streak))
        self._update_front_blocked()

        left_m = (angles > _SIDE_MIN) & (angles < _SIDE_MAX)
        right_m = (angles < -_SIDE_MIN) & (angles > -_SIDE_MAX)
        rear_m = angles.abs() >= _REAR_MIN

        def sector_min(mask):
            rr = torch.where(valid & mask.unsqueeze(0), r,
                             torch.full_like(r, torch.inf)).min(dim=1).values
            return torch.where(torch.isfinite(rr), rr,
                               torch.full_like(rr, c.max_eval_distance))

        self._left = sector_min(left_m)
        self._right = sector_min(right_m)
        self._rear = sector_min(rear_m)

        rear_close = (valid & rear_m.unsqueeze(0)
                      & (r < c.stop_distance_rear)).sum(dim=1)
        rqual = rear_close >= c.min_block_points
        self._rear_streak = torch.where(rqual, self._rear_streak + 1,
                                        torch.zeros_like(self._rear_streak))

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
        self.stops += int(block.sum())
        self.front_blocked = self.front_blocked | block
        self._front_blocked_time = torch.where(
            block, torch.full_like(self._front_blocked_time, now),
            self._front_blocked_time)

    @torch.no_grad()
    def gate(self, cmd: torch.Tensor) -> torch.Tensor:
        """cmd [B, 2] on device -> gated [B, 2]."""
        c = self.cfg
        left = cmd[:, 0].clone()
        right = cmd[:, 1].clone()

        self._cmd_lin = (left + right) / 2.0 * 0.154
        self._cmd_ang = (right - left) / c.track_width

        # FRONT (slow zone disabled in our config: slow == stop).
        fb = self.front_blocked
        left = torch.where(fb, left.clamp(max=0.0), left)
        right = torch.where(fb, right.clamp(max=0.0), right)

        # REAR: latched block of backward motion.
        backing = torch.minimum(left, right) < -0.01
        clear = self._rear_blocked & (self._rear > c.stop_distance_rear
                                      + c.hysteresis)
        self._rear_blocked = self._rear_blocked & ~(backing & clear)
        new_block = (backing & ~self._rear_blocked
                     & (self._rear < c.stop_distance_rear)
                     & (self._rear_streak >= c.block_scans))
        self._rear_blocked = self._rear_blocked | new_block
        stop_back = backing & self._rear_blocked
        left = torch.where(stop_back, left.clamp(min=0.0), left)
        right = torch.where(stop_back, right.clamp(min=0.0), right)

        # SIDES: forward+turn toward a close obstacle -> equalize.
        avg_fwd = (left + right) / 2.0
        zero_turn = (avg_fwd.abs() < 0.05) & ((left - right).abs() > 0.1)
        fwd = ~zero_turn & (avg_fwd > 0.01)
        turn = right - left
        eq_left = fwd & (turn > 0.1) & (self._left < c.stop_distance)
        left = torch.where(eq_left, right, left)
        eq_right = fwd & (turn < -0.1) & (self._right < c.stop_distance)
        right = torch.where(eq_right, left, right)

        return torch.stack([left, right], dim=1)
