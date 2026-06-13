"""Batched topological place memory — PlaceMemory vectorized over B envs.

Same arithmetic as the reference (fingerprint = [mean openness, low-freq
|FFT| harmonics], Euclidean match, presence-weight decay/reinforce/replace)
with the per-env variable-length place lists replaced by fixed [B, max_places]
slot arrays: a slot is live while its weight > 0. Slot ORDER differs from
the reference's list compaction, but order never enters the math — matching
is argmin distance, replacement is argmin weight.

The sim ticks at a fixed period, so dt is a scalar per update (the
reference's time_fn dt is uniform here by construction).
"""

from __future__ import annotations

import torch


class BatchedPlaceMemory:
    def __init__(self, batch: int, device: str = "cpu", n_freq: int = 10,
                 match_thresh: float = 0.35, tau_s: float = 900.0,
                 max_places: int = 64, shape_weight: float = 1.0,
                 fam_scale_s: float = 20.0, fp_ema_tau_s: float = 0.6):
        self.B = batch
        self.device = torch.device(device)
        self.n_freq = n_freq
        self.match_thresh = match_thresh
        self.tau_s = tau_s
        self.P = max_places
        self.shape_weight = shape_weight    # emphasize room SHAPE over size
        self.fam_scale_s = fam_scale_s      # sustained-novelty timescale (s)
        self.fp_ema_tau_s = fp_ema_tau_s    # fingerprint denoise timescale (s)
        F = 1 + (n_freq - 1)            # mean + harmonics 1..n_freq-1
        self.F = F
        self._fps = torch.zeros(batch, max_places, F, device=self.device)
        self._w = torch.zeros(batch, max_places, device=self.device)
        self._fp_ema = torch.zeros(batch, F, device=self.device)
        self._ema_valid = torch.zeros(batch, dtype=torch.bool,
                                      device=self.device)

    def fingerprint(self, scan: torch.Tensor) -> torch.Tensor:
        """scan [B, n] -> [B, F]; reference arithmetic, batched."""
        s = scan.double()
        m = s.mean(dim=1, keepdim=True)
        harm = torch.fft.rfft(s - m, dim=1).abs()[:, 1:self.n_freq] \
            / (s.shape[1] / 2.0)
        return torch.cat([m, self.shape_weight * harm], dim=1).float()

    @torch.no_grad()
    def update(self, scan: torch.Tensor, dt: float) -> torch.Tensor:
        """Fold scans in; return place novelty [B] in [0, 1]."""
        # Decay + prune (weight 0 = free slot).
        k = float(torch.exp(torch.tensor(-dt / self.tau_s)))
        self._w *= k
        self._w[self._w <= 0.05] = 0.0

        fp = self.fingerprint(scan)                       # [B, F]
        # Temporal denoise in the rotation-invariant domain (see reference
        # PlaceMemory): EMA the fingerprint per env; fresh envs init to fp.
        if self.fp_ema_tau_s > 0.0:
            a = min(1.0, dt / self.fp_ema_tau_s)
            self._fp_ema = torch.where(
                self._ema_valid.unsqueeze(1),
                self._fp_ema + a * (fp - self._fp_ema), fp)
            self._ema_valid[:] = True
            fp = self._fp_ema
        live = self._w > 0.0                              # [B, P]
        any_live = live.any(dim=1)

        d = (self._fps - fp.unsqueeze(1)).norm(dim=2)     # [B, P]
        d = torch.where(live, d, torch.full_like(d, torch.inf))
        dmin, imin = d.min(dim=1)
        novelty = torch.where(
            any_live, (dmin / self.match_thresh).clamp(0.0, 1.0),
            torch.ones_like(dmin))

        matched = any_live & (dmin < self.match_thresh)
        w_add = max(dt, 0.1)
        ar = torch.arange(self.B, device=self.device)

        # Reinforce + blend matched slots.
        mi = imin[matched]
        mb = ar[matched]
        self._w[mb, mi] += dt
        self._fps[mb, mi] = 0.98 * self._fps[mb, mi] + 0.02 * fp[matched]
        # Sustained novelty: matched place stays novel until fam_scale_s of
        # presence accumulates there (see reference PlaceMemory.update).
        fresh = (1.0 - self._w[mb, mi] / self.fam_scale_s).clamp(0.0, 1.0)
        novelty[mb] = torch.maximum(novelty[mb], fresh)

        # New place: first free slot, else evict the lightest.
        new = ~matched
        if bool(new.any()):
            free = ~live
            has_free = free.any(dim=1)
            first_free = free.float().argmax(dim=1)
            lightest = torch.where(
                live, self._w, torch.full_like(self._w, torch.inf)
            ).argmin(dim=1)
            slot = torch.where(has_free, first_free, lightest)
            nb = ar[new]
            ns = slot[new]
            self._fps[nb, ns] = fp[new]
            self._w[nb, ns] = w_add
        return novelty

    def n_places(self) -> torch.Tensor:
        return (self._w > 0.0).sum(dim=1)

    def clear(self, idx):
        """Forget everything for envs `idx` (new building)."""
        self._w[idx] = 0.0
        self._ema_valid[idx] = False
