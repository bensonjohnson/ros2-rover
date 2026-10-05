"""Graph-capturable fast paths for the batched sim.

Fp32Env / Fp16Env (fp16 raycast, optional fused Triton kernel via
rayscan), the generator-free _DefaultRNGMixin, NoSyncGate (BatchedGate
without its per-tick GPU->CPU sync), and the sim gate presets. Moved
verbatim out of evo/arena.py; no behavior change.
"""

from __future__ import annotations

import numpy as np
import torch

from .batched import BatchedEnv, BatchedGate
from .gate import GateConfig


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


class _MountOrigin:
    """Read-only env view with x/y replaced by mount-shifted origins.

    The fused Triton scan reads the ray origin straight from ``env.x`` /
    ``env.y``; this feeds it the true lidar mount WITHOUT touching the
    kernel or the env's live state (rayscan stays origin-agnostic). Only
    constructed when ``lidar_mount_x != 0.0``; the default path calls
    ``fused_scan(env)`` directly, byte-identical.
    """

    __slots__ = ("_env", "x", "y")

    def __init__(self, env, x: torch.Tensor, y: torch.Tensor):
        self._env = env
        self.x = x
        self.y = y

    def __getattr__(self, name):
        return getattr(self._env, name)


class Fp32Env(_DefaultRNGMixin, BatchedEnv):
    """BatchedEnv with generator-free scan (graph-capturable)."""

    @torch.no_grad()
    def scan(self) -> torch.Tensor:
        c = self.cfg
        ang = self.theta.unsqueeze(1) + self._beam_offsets
        d = torch.stack([torch.cos(ang), torch.sin(ang)], dim=2)
        p = torch.stack([self.x, self.y], dim=1)
        if c.lidar_mount_x != 0.0:
            p = p + c.lidar_mount_x * torch.stack(
                [torch.cos(self.theta), torch.sin(self.theta)], dim=1)
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
            from .rayscan import fused_scan
            if c.lidar_mount_x != 0.0:
                sh = c.lidar_mount_x * torch.stack(
                    [torch.cos(self.theta), torch.sin(self.theta)], dim=1)
                r = fused_scan(_MountOrigin(self, self.x + sh[:, 0],
                                            self.y + sh[:, 1]),
                               out=self._scan_buf)
            else:
                r = fused_scan(self, out=self._scan_buf)   # clean, static buf
            r = r + self.noise(c.lidar_noise_std, self.B, c.n_beams)
            return self._dropout(r, c.lidar_dropout_p)
        ang = self.theta.unsqueeze(1) + self._beam_offsets   # [B, nb] fp32
        d = torch.stack([torch.cos(ang), torch.sin(ang)], dim=2)
        d_h = d.half()
        if c.lidar_mount_x != 0.0:
            sh = c.lidar_mount_x * torch.stack(
                [torch.cos(self.theta), torch.sin(self.theta)], dim=1)
            p = (torch.stack([self.x, self.y], dim=1) + sh).half()
        else:
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


# Sim-only forward-gate prototype for doorway passage (2026-09-18). The real
# rover's lidar_safety_monitor = GateConfig() defaults. "doorway": corridor =
# the true robot radius (0.14; the stock 0.12 is NARROWER than the robot),
# stop 0.12 m from the bumper (0.18 m from centre, 4 cm over the 0.14 radius),
# quicker release (hysteresis 0.05 m, hold 0.15 s), side equalisation only
# inside 0.10 m (door jambs sit ~0.2 m off-axis in a 0.75 m door).
GATE_PRESETS = {
    "doorway": dict(stop_distance=0.12, robot_half_width=0.14,
                    hysteresis=0.05, min_block_duration=0.15,
                    side_stop_distance=0.10),
}


def gate_config(name: str = "legacy") -> "GateConfig":
    return GateConfig(**GATE_PRESETS.get(name, {}))


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
        side = (c.stop_distance if c.side_stop_distance is None
                else c.side_stop_distance)
        eq_left = fwd & (turn > 0.1) & (self._left < side)
        left = torch.where(eq_left, right, left)
        eq_right = fwd & (turn < -0.1) & (self._right < side)
        right = torch.where(eq_right, left, right)
        # reverse + CCW turn swings the tail toward the RIGHT flank
        bwd = ~zero_turn & (avg < -0.01)
        turn = right - left
        eq_rr = bwd & (turn > 0.1) & (self._rright < side)
        right = torch.where(eq_rr, left, right)
        eq_rl = bwd & (turn < -0.1) & (self._rleft < side)
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
