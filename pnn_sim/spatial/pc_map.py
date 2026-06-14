"""Predictive-coding spatial map — the learned, pure-PNN version of the map.

Where BatchedOccMap rasterizes log-odds in closed form, this INFERS the map as
a latent by predictive coding: the cells are occupancy logits, a differentiable
soft-raycast decodes them into a predicted scan, and the cells are updated to
minimize the prediction error against the real scan. That is exactly free-
energy minimization over a spatial generative model — the map is what the brain
believes is out there because it best predicts what the lidar sees.

Why this shape:
  - The decoder is the KNOWN sensor model (a soft raycast: expected free
    distance along each beam), so there are no deep weights to backprop — the
    map-inference gradient is a single-layer, LOCAL error signal (only the
    cells a beam passes through, up to the soft hit, get updated; occluded
    cells get ~zero gradient via the reach factor). This is PC inference, not
    backprop.
  - Confidence is free: |logit| is certainty (logit≈0 = never-explained =
    unknown), so FRONTIER = free cells bordering logit≈0 cells. That is the
    uncertainty field the epistemic actor will climb — exploration as
    information gain over this map, no bolt-on frontier detector.

Validated against BatchedOccMap (the geometric ground truth): does PC inference
recover the same house occupancy from the same trajectory?

Frame/pose conventions match BatchedOccMap (odom-frame, origin pinned at start).
"""

from __future__ import annotations

import torch


class PCSpatialMap:
    def __init__(self, batch: int, res: float = 0.15, size: int = 200,
                 max_range: float = 5.0, device: str = "cpu",
                 infer_lr: float = 0.8, infer_iters: int = 3,
                 beam_stride: int = 4):
        self.B = batch
        self.res = res
        self.n = size
        self.max_range = max_range
        self.device = torch.device(device)
        self.infer_lr = infer_lr
        self.infer_iters = infer_iters
        self.beam_stride = beam_stride
        self.M = torch.zeros(batch, size, size, device=self.device)   # logits
        self.cnt = torch.zeros(batch, size, size, device=self.device)  # obs count
        # Per-env odom-frame origin (world coord of cell 0), pinned at each
        # env's first observation and re-pinnable per-env on reset — so an env
        # that switches house (or the rover after a lift) re-centres cleanly.
        self.origin = torch.zeros(batch, 2, device=self.device)
        self._pinned = torch.zeros(batch, dtype=torch.bool, device=self.device)
        self._steps = torch.arange(0, max_range, res, device=self.device)
        self.last_err = 0.0

    @torch.no_grad()
    def _ray_index(self, pos, heading, bearings):
        """Linear cell indices [B,K,S] along each beam, plus an in-bounds mask."""
        B, n = self.B, self.n
        ang = heading[:, None] + bearings[None, :]                  # [B,K]
        cs, sn = ang.cos(), ang.sin()
        px = pos[:, 0][:, None, None] + cs[:, :, None] * self._steps   # [B,K,S]
        py = pos[:, 1][:, None, None] + sn[:, :, None] * self._steps
        ci = ((px - self.origin[:, 0][:, None, None]) / self.res).round().long()
        cj = ((py - self.origin[:, 1][:, None, None]) / self.res).round().long()
        inb = (ci >= 0) & (ci < n) & (cj >= 0) & (cj < n)
        lin = (ci.clamp(0, n - 1) * n + cj.clamp(0, n - 1))         # [B,K,S]
        return lin, inb

    def update(self, pos, heading, ranges, bearings):
        """One observation: settle the visible map cells to predict this scan."""
        # Pin any not-yet-pinned env's origin at its current pose (centre cell).
        need_pin = ~self._pinned
        if bool(need_pin.any()):
            self.origin = torch.where(
                need_pin[:, None], pos - (self.n // 2) * self.res, self.origin)
            self._pinned |= need_pin
        if self.beam_stride > 1:
            ranges = ranges[:, ::self.beam_stride]
            bearings = bearings[::self.beam_stride]
        B, K = ranges.shape
        S = self._steps.shape[0]
        rng = ranges.clamp(0, self.max_range)
        lin, inb = self._ray_index(pos, heading, bearings)
        flat_lin = lin.reshape(B, -1)                                   # [B,K*S]
        # The proper beam sensor model: a hit at step s has likelihood
        # reach_s * p_s (free until s, occupied at s); "no return" has
        # likelihood reach_full (free the whole way). Inferring the map by the
        # NLL of the OBSERVED hit location sharply supervises occupancy at the
        # hit cell and free at the cells before it — unlike the diffuse L2 on
        # expected distance, which left walls unmarked.
        s_star = (rng / self.res).round().long().clamp(0, S - 1)        # [B,K]
        hitmask = ranges < self.max_range

        # The map-inference gradient is a single-layer local error signal;
        # enable grad locally so the explorer can call this under no_grad.
        cm = torch.enable_grad()
        cm.__enter__()
        for _ in range(self.infer_iters):
            M = self.M.detach().requires_grad_(True)
            p = torch.sigmoid(M).view(B, -1)
            p_ray = p.gather(1, flat_lin).view(B, K, S)
            p_ray = torch.where(inb, p_ray, torch.zeros_like(p_ray)) \
                .clamp(1e-4, 1 - 1e-4)                                   # oob = free
            free = 1.0 - p_ray
            reach = torch.cat([torch.ones(B, K, 1, device=self.device),
                               torch.cumprod(free, dim=2)[:, :, :-1]], dim=2)
            p_hit = reach * p_ray                                        # [B,K,S]
            reach_full = reach[:, :, -1] * free[:, :, -1]                # [B,K]
            p_hit_star = p_hit.gather(2, s_star[:, :, None]).squeeze(2) \
                .clamp_min(1e-6)
            nll = torch.where(hitmask, -torch.log(p_hit_star),
                              -torch.log(reach_full.clamp_min(1e-6)))
            loss = nll.sum()
            loss.backward()
            self.M = self.M - self.infer_lr * M.grad
        cm.__exit__(None, None, None)
        # Readable metric: |observed openness - expected free distance|.
        with torch.no_grad():
            p = torch.sigmoid(self.M).view(B, -1)
            p_ray = torch.where(inb, p.gather(1, flat_lin).view(B, K, S),
                                torch.zeros(B, K, S, device=self.device))
            reach = torch.cat([torch.ones(B, K, 1, device=self.device),
                               torch.cumprod(1 - p_ray, dim=2)[:, :, :-1]], dim=2)
            pred_open = (reach.sum(dim=2) * self.res / self.max_range).clamp(0, 1)
            self.last_err = float((rng / self.max_range - pred_open).abs().mean())

        # Coverage: mark every cell each beam reached, INCLUDING its hit cell
        # (step index <= s_star), so "unknown" means genuinely never sensed —
        # not merely "logit still weak". Frontiers then sit at the real edge of
        # explored space (doorways, unseen rooms), not in an under-confidence
        # halo around the robot. The learned logit classifies free vs occupied;
        # this count classifies explored vs unexplored.
        with torch.no_grad():
            step_idx = torch.arange(S, device=self.device)
            observed = inb & (step_idx[None, None, :] <= s_star[:, :, None])
            cidx = lin[observed]
            if cidx.numel():
                bb = torch.arange(B, device=self.device)[:, None, None] \
                    .expand(B, K, S)[observed]
                self.cnt.view(-1).scatter_add_(
                    0, bb * self.n * self.n + cidx,
                    torch.ones(cidx.numel(), device=self.device))

    # ---- read-outs (mirror BatchedOccMap so validation/exploration share) ---
    # Explored = sensor coverage (cnt>0); the learned logit gives occupancy
    # among explored cells (sign), unknown = never sensed.

    def seen(self):
        return self.cnt > 0

    def occupied(self):
        return (self.cnt > 0) & (self.M > 0.0)

    def free(self):
        return (self.cnt > 0) & (self.M <= 0.0)

    def confidence(self):
        return self.M.abs()

    def frontier(self):
        unk = self.cnt == 0
        un = torch.zeros_like(unk)
        un[:, :-1] |= unk[:, 1:]; un[:, 1:] |= unk[:, :-1]
        un[:, :, :-1] |= unk[:, :, 1:]; un[:, :, 1:] |= unk[:, :, :-1]
        return self.free() & un

    def reset(self, idx=None):
        if idx is None:
            self.M.zero_(); self.cnt.zero_(); self._pinned.zero_()
        else:
            self.M[idx] = 0.0; self.cnt[idx] = 0.0
            self._pinned[idx] = False     # re-pin at the next observation
