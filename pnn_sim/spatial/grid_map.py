"""Batched occupancy map — the GPU substrate for the spatial brain.

This is the batched, device-resident version of the explicit frontier grid
(pnn_sim/frontier.py): a log-odds occupancy map per env, built by ray-marching
the lidar (free along each beam, occupied at the hit) at the robot's odom pose.
Unlike frontier.py it is pure tensor math — `[B, H, W]` scatter-adds — so it
runs batched on the Spark, which the grid+BFS explorer could not.

Two roles in the redesign:
  - GROUND TRUTH / TEACHER for the learned predictive-coding spatial map: what
    the PC map should end up encoding, and where the frontiers are.
  - the path-integration + map-formation mechanics, validated batched before
    any learning is added.

Frame: odom-frame, origin pinned at each env's start pose, robot tracked by its
odom (x,y) — ground truth in sim, fused wheel+IMU+rf2o EKF on the rover. The
map is sized to cover a house without recentering (recenter-on-edge is a
long-run rover concern, deferred). Pose drift is handled the same way the grid
explorer tolerates it: the map is rebuilt continuously and only nearby cells
drive behavior.
"""

from __future__ import annotations

import torch


class BatchedOccMap:
    def __init__(self, batch: int, res: float = 0.15, size: int = 200,
                 max_range: float = 5.0, device: str = "cpu",
                 l_free: float = -0.4, l_occ: float = 0.85, l_clamp: float = 4.0,
                 beam_stride: int = 1):
        self.B = batch
        self.res = res
        self.n = size
        self.max_range = max_range
        self.device = torch.device(device)
        self.l_free, self.l_occ, self.l_clamp = l_free, l_occ, l_clamp
        self.beam_stride = beam_stride
        self.lo = torch.zeros(batch, size, size, device=self.device)
        self.seen = torch.zeros(batch, size, size, dtype=torch.bool,
                                device=self.device)
        self.origin = None     # [B,2] world coord of cell (0,0)
        self._steps = (torch.arange(0, max_range, res, device=self.device))

    @torch.no_grad()
    def update(self, pos: torch.Tensor, heading: torch.Tensor,
               ranges: torch.Tensor, bearings: torch.Tensor):
        """Fold one scan into the map.

        pos [B,2] world xy, heading [B] rad, ranges [B,K], bearings [K] rad
        (robot-relative beam angles, 0 = forward)."""
        B, n = self.B, self.n
        if self.origin is None:
            # Pin cell (n//2, n//2) to each env's start pose.
            self.origin = pos - (n // 2) * self.res

        if self.beam_stride > 1:
            ranges = ranges[:, ::self.beam_stride]
            bearings = bearings[::self.beam_stride]
        K = ranges.shape[1]
        S = self._steps.shape[0]

        ang = heading[:, None] + bearings[None, :]              # [B,K]
        cs, sn = ang.cos(), ang.sin()
        rng = ranges.clamp(0, self.max_range)
        ox = self.origin[:, 0][:, None, None]
        oy = self.origin[:, 1][:, None, None]

        # Free cells along each ray (before the hit).
        px = pos[:, 0][:, None, None] + cs[:, :, None] * self._steps    # [B,K,S]
        py = pos[:, 1][:, None, None] + sn[:, :, None] * self._steps
        ci = ((px - ox) / self.res).round().long()
        cj = ((py - oy) / self.res).round().long()
        inb = (ci >= 0) & (ci < n) & (cj >= 0) & (cj < n)
        free = (self._steps[None, None, :] < (rng[:, :, None] - self.res)) & inb
        bb = torch.arange(B, device=self.device)[:, None, None].expand(B, K, S)
        lin = (bb * n * n + ci * n + cj)[free]
        self._accum(lin, self.l_free)

        # Occupied cells at the hit (only beams that hit within range).
        hit = rng < self.max_range
        hx = pos[:, 0][:, None] + cs * rng
        hy = pos[:, 1][:, None] + sn * rng
        hi = ((hx - self.origin[:, 0][:, None]) / self.res).round().long()
        hj = ((hy - self.origin[:, 1][:, None]) / self.res).round().long()
        inb2 = (hi >= 0) & (hi < n) & (hj >= 0) & (hj < n) & hit
        bb2 = torch.arange(B, device=self.device)[:, None].expand(B, K)
        linh = (bb2 * n * n + hi * n + hj)[inb2]
        self._accum(linh, self.l_occ)

        self.lo.clamp_(-self.l_clamp, self.l_clamp)

    def _accum(self, lin: torch.Tensor, val: float):
        if lin.numel() == 0:
            return
        src = torch.full((lin.numel(),), val, device=self.device)
        self.lo.view(-1).scatter_add_(0, lin, src)
        self.seen.view(-1).scatter_(
            0, lin, torch.ones_like(lin, dtype=torch.bool))

    # ---- read-outs -----------------------------------------------------

    def occupied(self) -> torch.Tensor:
        return self.seen & (self.lo > 0.0)

    def free(self) -> torch.Tensor:
        return self.seen & (self.lo <= 0.0)

    def frontier(self) -> torch.Tensor:
        """Free cells bordering unknown — the exploration target field."""
        unk = ~self.seen
        un = torch.zeros_like(unk)
        un[:, :-1] |= unk[:, 1:]; un[:, 1:] |= unk[:, :-1]
        un[:, :, :-1] |= unk[:, :, 1:]; un[:, :, 1:] |= unk[:, :, :-1]
        return self.free() & un

    def reset(self, idx=None):
        if idx is None:
            self.lo.zero_(); self.seen.zero_(); self.origin = None
        else:
            self.lo[idx] = 0.0; self.seen[idx] = False
            if self.origin is not None:
                self.origin[idx] = 0.0   # re-pinned on next update for these
