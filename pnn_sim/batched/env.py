"""B parallel worlds, rovers, and safety gates — vectorized numpy.

Same physics, sensors, and gate logic as the single-env modules
(pnn_sim/rover.py, pnn_sim/safety_gate.py), with every scalar state
promoted to a [B] array. Worlds have different segment counts, so segment
arrays are padded to the max with far-away dummies (they never intersect a
ray inside a house).
"""

from __future__ import annotations

import numpy as np

from ..world import make_house
from ..rover import RoverConfig
from ..safety_gate import GateConfig, _SIDE_MIN, _SIDE_MAX, _REAR_MIN

_DUMMY_SEG = [1e6, 1e6, 1e6 + 0.1, 1e6]


def batched_preprocess(ranges: np.ndarray, angle_min: float,
                       angle_increment: float, num_bins: int = 72,
                       max_range: float = 5.0,
                       min_range: float = 0.05) -> np.ndarray:
    """preprocess_scan over a batch: ranges [B, n] -> [B, num_bins].

    Beam->bin layout is identical for every env (same lidar geometry), so
    the bin indices are computed once and the per-sector min-pool runs as a
    single flattened minimum.at.
    """
    ranges = np.asarray(ranges, dtype=np.float32)
    B, n = ranges.shape

    clean = ranges.copy()
    invalid = ~np.isfinite(clean) | (clean <= 0.0) | (clean < min_range)
    clean[invalid] = max_range
    np.clip(clean, min_range, max_range, out=clean)

    idx = np.arange(n, dtype=np.float32)
    angles = angle_min + idx * angle_increment
    frac = np.mod(angles, 2.0 * np.pi) / (2.0 * np.pi)
    bins = np.minimum((frac * num_bins).astype(np.int64), num_bins - 1)

    out = np.full((B, num_bins), max_range, dtype=np.float32)
    flat_idx = (np.arange(B)[:, None] * num_bins + bins[None, :]).ravel()
    np.minimum.at(out.ravel(), flat_idx, clean.ravel())
    out /= max_range
    return out


class BatchedEnv:
    """B rovers in B houses. All state is [B]-shaped numpy."""

    def __init__(self, batch: int, rover_cfg: RoverConfig | None = None,
                 seed: int = 0):
        self.B = batch
        self.cfg = rover_cfg or RoverConfig()
        self.world_rng = np.random.default_rng(seed)
        self.rng = np.random.default_rng(seed + 1)

        c = self.cfg
        self.x = np.zeros(batch)
        self.y = np.zeros(batch)
        self.theta = np.zeros(batch)
        self.v_left = np.zeros(batch)
        self.v_right = np.zeros(batch)
        self._prev_v = np.zeros(batch)
        self.collided = np.zeros(batch, dtype=bool)
        self.wheel_l = np.zeros(batch)
        self.wheel_r = np.zeros(batch)
        self.yaw_rate = np.zeros(batch)
        self.accel = np.tile([0.0, 0.0, c.gravity], (batch, 1))

        self._worlds = [make_house(self.world_rng) for _ in range(batch)]
        self._build_segments()
        for b in range(batch):
            self.x[b], self.y[b], self.theta[b] = self._worlds[b].start_pose

        inc = 2.0 * np.pi / c.n_beams
        self._beam_offsets = np.arange(c.n_beams) * inc
        self.angle_min = 0.0
        self.angle_increment = inc

    def _build_segments(self):
        Mmax = max(w.segments.shape[0] for w in self._worlds)
        segs = np.tile(np.asarray(_DUMMY_SEG, dtype=np.float32),
                       (self.B, Mmax, 1))
        for b, w in enumerate(self._worlds):
            segs[b, :w.segments.shape[0]] = w.segments.astype(np.float32)
        self._a = segs[:, :, 0:2]                          # [B, M, 2]
        self._e = segs[:, :, 2:4] - segs[:, :, 0:2]        # [B, M, 2]
        self._ee = (self._e * self._e).sum(axis=2)         # [B, M]

    def switch_world(self, idx: np.ndarray):
        """Replace the houses of envs `idx` (bool mask or int array)."""
        ids = np.flatnonzero(idx) if idx.dtype == bool else np.asarray(idx)
        for b in ids:
            self._worlds[b] = make_house(self.world_rng)
            self.x[b], self.y[b], self.theta[b] = self._worlds[b].start_pose
            self.v_left[b] = self.v_right[b] = self._prev_v[b] = 0.0
        self._build_segments()

    # ------------------------------------------------------------------

    def _clearance(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """Min distance from points [B] to each env's own walls -> [B]."""
        p = np.stack([px, py], axis=1)[:, None, :]          # [B, 1, 2]
        ap = p - self._a                                    # [B, M, 2]
        tt = np.clip((ap * self._e).sum(axis=2)
                     / np.maximum(self._ee, 1e-12), 0.0, 1.0)
        closest = self._a + tt[..., None] * self._e
        d2 = ((p - closest) ** 2).sum(axis=2)               # [B, M]
        return np.sqrt(d2.min(axis=1))

    def step(self, cmd: np.ndarray, dt: float):
        """cmd [B, 2] in [-1, 1] — same dynamics as SimRover.step."""
        c = self.cfg
        cl = np.clip(cmd[:, 0], -1.0, 1.0)
        cr = np.clip(cmd[:, 1], -1.0, 1.0)
        tl = np.where(np.abs(cl) < c.deadband, 0.0, cl * c.v_max * c.left_trim)
        tr = np.where(np.abs(cr) < c.deadband, 0.0, cr * c.v_max * c.right_trim)

        k = 1.0 - np.exp(-dt / c.motor_tau)
        self.v_left += (tl - self.v_left) * k
        self.v_right += (tr - self.v_right) * k

        vl = self.v_left * (1.0 + self.rng.normal(0.0, c.slip_std, self.B))
        vr = self.v_right * (1.0 + self.rng.normal(0.0, c.slip_std, self.B))
        v = 0.5 * (vl + vr)
        w = (vr - vl) / c.track_width

        nx = self.x + v * np.cos(self.theta) * dt
        ny = self.y + v * np.sin(self.theta) * dt
        self.collided = self._clearance(nx, ny) < c.robot_radius
        ok = ~self.collided
        self.x = np.where(ok, nx, self.x)
        self.y = np.where(ok, ny, self.y)
        v = np.where(ok, v, 0.0)
        self.theta = (self.theta + w * dt + np.pi) % (2 * np.pi) - np.pi

        self.wheel_l = self.v_left / c.wheel_radius \
            + self.rng.normal(0.0, 0.05, self.B)
        self.wheel_r = self.v_right / c.wheel_radius \
            + self.rng.normal(0.0, 0.05, self.B)
        self.yaw_rate = w + self.rng.normal(0.0, c.gyro_noise_std, self.B)
        ax = (v - self._prev_v) / dt + self.rng.normal(0.0, c.accel_noise_std, self.B)
        ay = v * w + self.rng.normal(0.0, c.accel_noise_std, self.B)
        az = c.gravity + self.rng.normal(0.0, c.accel_noise_std, self.B)
        self.accel = np.stack([ax, ay, az], axis=1)
        self._prev_v = v

    # ------------------------------------------------------------------

    def scan(self) -> np.ndarray:
        """One lidar rev per env -> ranges [B, n_beams] (beam 0 = forward)."""
        c = self.cfg
        nb = c.n_beams
        ang = self.theta[:, None] + self._beam_offsets[None, :]  # [B, nb]
        d = np.stack([np.cos(ang), np.sin(ang)], axis=2).astype(np.float32)
        p = np.stack([self.x, self.y], axis=1).astype(np.float32)
        q = self._a - p[:, None, :]                               # [B, M, 2]

        cross_eq = (self._e[:, :, 0] * q[:, :, 1]
                    - self._e[:, :, 1] * q[:, :, 0])              # [B, M]
        cross_ed = (self._e[:, None, :, 0] * d[:, :, None, 1]
                    - self._e[:, None, :, 1] * d[:, :, None, 0])  # [B, nb, M]
        cross_dq = (d[:, :, None, 0] * q[:, None, :, 1]
                    - d[:, :, None, 1] * q[:, None, :, 0])        # [B, nb, M]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = cross_eq[:, None, :] / cross_ed
            s = cross_dq / cross_ed
        hit = (t > 1e-9) & (s >= 0.0) & (s <= 1.0) & np.isfinite(t)
        t = np.where(hit, t, np.inf)
        r = np.minimum(t.min(axis=2), c.lidar_max_range)

        r = r + self.rng.normal(0.0, c.lidar_noise_std, r.shape)
        drop = self.rng.random(r.shape) < c.lidar_dropout_p
        return np.where(drop, np.inf, np.maximum(r, 0.02)).astype(np.float32)


class BatchedGate:
    """Vectorized port of SimSafetyGate (itself a port of the rover's
    lidar_safety_monitor track path). All state machines are [B] arrays;
    the logic follows the scalar implementation branch for branch."""

    def __init__(self, cfg: GateConfig, batch: int, time_fn):
        self.cfg = cfg
        self.B = batch
        self._now = time_fn
        md = cfg.max_eval_distance
        self.front_blocked = np.zeros(batch, dtype=bool)
        self._rear_blocked = np.zeros(batch, dtype=bool)
        self._front_blocked_time = np.full(batch, -1.0)
        self._front_streak = np.zeros(batch, dtype=int)
        self._rear_streak = np.zeros(batch, dtype=int)
        self._front_path_dist = np.full(batch, md)
        self._left = np.full(batch, md)
        self._right = np.full(batch, md)
        self._rear = np.full(batch, md)
        self._cmd_lin = np.zeros(batch)
        self._cmd_ang = np.zeros(batch)
        self.stops = 0

    def process_scan(self, ranges: np.ndarray, angle_min: float,
                     angle_increment: float):
        c = self.cfg
        B, n = ranges.shape
        angles = np.arange(n) * angle_increment + angle_min
        angles = (angles + np.pi) % (2 * np.pi) - np.pi          # [n]

        valid = (np.isfinite(ranges)
                 & (ranges > c.min_valid_range)
                 & (ranges <= c.max_eval_distance))               # [B, n]
        # Large-but-finite for invalid beams: every consumer masks on
        # `valid`, and inf here would 0*inf -> nan in the arc-shift math.
        r = np.where(valid, ranges, 1e6)
        x = r * np.cos(angles)[None, :]
        y = r * np.sin(angles)[None, :]
        x_bumper = x - c.robot_front_offset

        # Front corridor with per-env arc shift.
        arc = (self._cmd_lin > 0.02) & (np.abs(self._cmd_ang) > 0.1)  # [B]
        v = np.maximum(self._cmd_lin, 0.05)[:, None]
        x_fwd = np.maximum(x_bumper, 0.0)
        y_shift = np.clip(self._cmd_ang[:, None] * x_fwd * x_fwd
                          / (2.0 * v), -0.30, 0.30)
        y_eff = np.where(arc[:, None], y - y_shift, y)
        in_path = valid & (np.abs(y_eff) <= c.robot_half_width) & (x_bumper > 0)

        xb_in = np.where(in_path, x_bumper, np.inf)
        front = xb_in.min(axis=1)
        self._front_path_dist = np.where(
            np.isfinite(front), np.maximum(front, 0.0), c.max_eval_distance)

        close_pts = (in_path & (x_bumper < c.stop_distance)).sum(axis=1)
        qual = close_pts >= c.min_block_points
        self._front_streak = np.where(qual, self._front_streak + 1, 0)
        self._update_front_blocked()

        abs_a = np.abs(angles)
        left_m = (angles > _SIDE_MIN) & (angles < _SIDE_MAX)
        right_m = (angles < -_SIDE_MIN) & (angles > -_SIDE_MAX)
        rear_m = abs_a >= _REAR_MIN

        def sector_min(mask):
            rr = np.where(valid & mask[None, :], r, np.inf).min(axis=1)
            return np.where(np.isfinite(rr), rr, c.max_eval_distance)

        self._left = sector_min(left_m)
        self._right = sector_min(right_m)
        self._rear = sector_min(rear_m)

        rear_close = (valid & rear_m[None, :]
                      & (r < c.stop_distance_rear)).sum(axis=1)
        rqual = rear_close >= c.min_block_points
        self._rear_streak = np.where(rqual, self._rear_streak + 1, 0)

    def _update_front_blocked(self):
        c = self.cfg
        now = self._now()
        resume = self._front_path_dist > c.stop_distance + c.hysteresis
        held_long = (self._front_blocked_time < 0) \
            | (now - self._front_blocked_time >= c.min_block_duration)
        release = self.front_blocked & resume & held_long
        self.front_blocked[release] = False
        self._front_blocked_time[release] = -1.0

        block = (~self.front_blocked
                 & (self._front_path_dist < c.stop_distance)
                 & (self._front_streak >= c.block_scans))
        self.stops += int(block.sum())
        self.front_blocked[block] = True
        self._front_blocked_time[block] = now

    def gate(self, cmd: np.ndarray) -> np.ndarray:
        """cmd [B, 2] -> gated [B, 2]."""
        c = self.cfg
        left = cmd[:, 0].copy()
        right = cmd[:, 1].copy()

        self._cmd_lin = (left + right) / 2.0 * 0.154
        self._cmd_ang = (right - left) / c.track_width

        # FRONT (slow zone disabled in our config: slow == stop).
        fb = self.front_blocked
        left = np.where(fb, np.minimum(left, 0.0), left)
        right = np.where(fb, np.minimum(right, 0.0), right)

        # REAR: latched block of backward motion.
        backing = np.minimum(left, right) < -0.01
        clear = self._rear_blocked & (self._rear > c.stop_distance_rear
                                      + c.hysteresis)
        self._rear_blocked[backing & clear] = False
        new_block = (backing & ~self._rear_blocked
                     & (self._rear < c.stop_distance_rear)
                     & (self._rear_streak >= c.block_scans))
        self._rear_blocked[new_block] = True
        stop_back = backing & self._rear_blocked
        left = np.where(stop_back, np.maximum(left, 0.0), left)
        right = np.where(stop_back, np.maximum(right, 0.0), right)

        # SIDES: forward+turn toward a close obstacle -> equalize.
        avg_fwd = (left + right) / 2.0
        zero_turn = (np.abs(avg_fwd) < 0.05) & (np.abs(left - right) > 0.1)
        fwd = ~zero_turn & (avg_fwd > 0.01)
        turn = right - left
        eq_left = fwd & (turn > 0.1) & (self._left < c.stop_distance)
        left = np.where(eq_left, right, left)
        eq_right = fwd & (turn < -0.1) & (self._right < c.stop_distance)
        right = np.where(eq_right, left, right)

        return np.stack([left, right], axis=1)
