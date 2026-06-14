"""Spatial explorer — exploration as information gain over the PC map.

The predictive-coding map (pc_map.PCSpatialMap) exposes a per-cell uncertainty:
cells it can't yet explain (logit~0) are unknown, and FREE cells bordering them
are frontiers — exactly where going would most reduce the map's uncertainty.
This module turns that into motion the principled, batched way:

  1. seed a wavefront at the frontier cells,
  2. propagate geodesic distance-to-frontier through KNOWN-FREE space only
     (occupied/unknown block) — a batched min-relaxation on a downsampled grid,
  3. the robot follows the field downhill = the shortest route through explored
     space to the nearest reachable frontier (through doorways, naturally),
  4. pursue it with the committed escape-pivot control validated in frontier.py
     (a soft prior to a stochastic actor could not hold this; the explorer owns
     the steering, the safety gate owns collisions).

So "go where the map is uncertain" + a reachability field = frontier
exploration, emergent from the spatial generative model. All tensor ops, so it
batches on the Spark. Per-env Python BFS (frontier.py) is the un-batchable
thing this replaces.
"""

from __future__ import annotations

import math

import torch

from .pc_map import PCSpatialMap

_BIG = 1e6


def wavefront(passable: torch.Tensor, source: torch.Tensor,
              iters: int) -> torch.Tensor:
    """Geodesic distance from `source` cells through `passable` cells. [B,h,w]."""
    dist = torch.where(source, torch.zeros_like(passable, dtype=torch.float32),
                       torch.full_like(passable, _BIG, dtype=torch.float32))
    block = ~(passable | source)
    for _ in range(iters):
        up = torch.full_like(dist, _BIG); up[:, :-1, :] = dist[:, 1:, :]
        dn = torch.full_like(dist, _BIG); dn[:, 1:, :] = dist[:, :-1, :]
        lf = torch.full_like(dist, _BIG); lf[:, :, :-1] = dist[:, :, 1:]
        rt = torch.full_like(dist, _BIG); rt[:, :, 1:] = dist[:, :, :-1]
        nb = torch.stack([up, dn, lf, rt]).amin(0) + 1.0
        dist = torch.minimum(dist, nb)
        dist = dist.masked_fill(block, _BIG)
        dist = dist.masked_fill(source, 0.0)
    return dist


class SpatialExplorer:
    def __init__(self, batch: int, res: float = 0.15, size: int = 200,
                 max_range: float = 5.0, device: str = "cpu",
                 down: int = 4, replan_every: int = 12, wave_iters: int = 60,
                 escape_ticks: int = 14, **pc_kwargs):
        self.map = PCSpatialMap(batch, res=res, size=size, max_range=max_range,
                                device=device, **pc_kwargs)
        self.B = batch
        self.res = res
        self.n = size
        self.device = torch.device(device)
        self.down = down
        self.hc = size // down                 # coarse grid side
        self.replan_every = replan_every
        self.wave_iters = wave_iters
        self.escape_ticks = escape_ticks
        self.dist = None                       # [B, hc, hc]
        self.escape = torch.zeros(batch, dtype=torch.long, device=self.device)
        self.edir = torch.ones(batch, device=self.device)
        self.k = 0

    def _replan(self):
        d = self.down
        free = self.map.free().view(self.B, self.hc, d, self.hc, d)
        free = free.any(4).any(2)
        # Coarse unknown = a cell with NO observed sub-cell. A frontier is a
        # coarse-free cell adjacent to a coarse-unknown cell that itself opens
        # onto a SUBSTANTIAL unknown region — this rejects the trivial
        # unexplored slivers hugging walls (always the nearest frontier, a
        # geodesic trap) and keeps doorways into real unseen rooms.
        unk = ~self.map.seen().view(self.B, self.hc, d, self.hc, d).any(4).any(2)
        # how much unknown sits in a 5x5 window around each cell
        u = unk.float().unsqueeze(1)
        k = 5
        unk_mass = (torch.nn.functional.avg_pool2d(
            u, k, stride=1, padding=k // 2) * (k * k)).squeeze(1)
        big_unk = unk & (unk_mass >= 6)
        un = torch.zeros_like(big_unk)
        un[:, :-1] |= big_unk[:, 1:]; un[:, 1:] |= big_unk[:, :-1]
        un[:, :, :-1] |= big_unk[:, :, 1:]; un[:, :, 1:] |= big_unk[:, :, :-1]
        fr = free & un
        self.dist = wavefront(free, fr, self.wave_iters)

    @torch.no_grad()
    def step(self, pos, heading, ranges, bearings, blocked):
        """Batched. pos[B,2] heading[B] ranges[B,K] bearings[K] blocked[B] bool
        -> cmd[B,2] in [-1,1]."""
        self.map.update(pos, heading, ranges, bearings)
        self.k += 1
        if self.dist is None or self.k % self.replan_every == 0:
            self._replan()

        B, hc = self.B, self.hc
        origin = self.map.origin
        # robot coarse cell, clamped 1..hc-2 so 3x3 finite differences are valid
        ci = (((pos[:, 0] - origin[:, 0]) / self.res).long() // self.down)
        cj = (((pos[:, 1] - origin[:, 1]) / self.res).long() // self.down)
        ci = ci.clamp(1, hc - 2)
        cj = cj.clamp(1, hc - 2)
        bb = torch.arange(B, device=self.device)
        df = self.dist.clamp(max=_BIG)

        def at(i, j):
            return df[bb, i, j]
        # downhill (negative gradient) = toward nearest frontier
        gx = at(ci + 1, cj) - at(ci - 1, cj)
        gy = at(ci, cj + 1) - at(ci, cj - 1)
        # world bearing of decreasing distance; map axes: i<->x, j<->y
        tgt_world = torch.atan2(-gy, -gx)
        # no reachable frontier (flat/unreachable field) -> scan in place
        reachable = at(ci, cj) < _BIG / 2
        tb = (tgt_world - heading + math.pi) % (2 * math.pi) - math.pi   # [-pi,pi]

        # --- committed escape when the gate clamps forward ---
        new_esc = blocked & (self.escape <= 0)
        la = ((bearings - math.radians(55) + math.pi) % (2 * math.pi)
              - math.pi).abs() < math.radians(40)
        ra = ((bearings + math.radians(55) + math.pi) % (2 * math.pi)
              - math.pi).abs() < math.radians(40)
        left_open = (ranges * la).sum(1) / la.sum().clamp(min=1)
        right_open = (ranges * ra).sum(1) / ra.sum().clamp(min=1)
        edir_new = torch.where(left_open >= right_open,
                               torch.ones(B, device=self.device),
                               -torch.ones(B, device=self.device))
        self.edir = torch.where(new_esc, edir_new, self.edir)
        self.escape = torch.where(new_esc,
                                  torch.full_like(self.escape, self.escape_ticks),
                                  self.escape)
        escaping = (self.escape > 0) | blocked
        self.escape = torch.where(escaping, (self.escape - 1).clamp(min=0),
                                  self.escape)

        # --- pursuit toward the frontier bearing ---
        turn = (tb / 0.6).clamp(-0.8, 0.8)
        fwd = (1.0 - tb.abs() / 1.2).clamp(min=0.15) * 0.9
        pursuit = torch.stack([fwd - turn, fwd + turn], dim=1)
        pivot_s = torch.where(tb >= 0, torch.ones(B, device=self.device),
                              -torch.ones(B, device=self.device))
        pivot = torch.stack([-0.7 * pivot_s, 0.7 * pivot_s], dim=1)
        # big heading error -> pivot in place; no frontier -> slow scan spin
        scan = torch.tensor([0.3, -0.3], device=self.device).expand(B, 2)
        cmd = torch.where((tb.abs() > 0.6)[:, None], pivot, pursuit)
        cmd = torch.where(reachable[:, None], cmd, scan)
        esc = torch.stack([-0.75 * self.edir, 0.75 * self.edir], dim=1)
        cmd = torch.where(escaping[:, None], esc, cmd)
        return cmd.clamp(-1, 1)

    def reset(self, idx=None):
        self.map.reset(idx)
        if idx is None:
            self.dist = None
            self.escape.zero_()
        else:
            self.escape[idx] = 0
