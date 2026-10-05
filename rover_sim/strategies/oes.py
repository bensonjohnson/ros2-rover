"""OpenAI-ES strategy (SIM_PLATFORM stage 3b).

Moved VERBATIM out of evo/evolve.py: `class OES` is byte-for-byte the
legacy class; only this import block and module docstring are new.
"""

from __future__ import annotations

import numpy as np
import torch


class OES:
    """OpenAI-ES (Salimans et al. 2017) over the flat genome, as an
    alternative to the truncation GA (--algo oes).

    Population slot 0 = the centre mu; slots 1..2n = mirrored pairs
    mu +/- sigma*scale*eps (antithetic sampling cancels the fitness
    baseline); a leftover odd slot re-scores the best genome seen. All 2n
    perturbed scores become centred ranks in [-0.5, 0.5] (fitness shaping:
    invariant to the w_cov/w_dist scale, robust to collision outliers), and
    the search-gradient estimate g = sum_i (r+_i - r-_i) eps_i / (2 n sigma)
    drives Adam on z = mu / scale (per-gene init-scale units, like the GA's
    sigma) with L2 decay. Every one of the P rollouts feeds one gradient —
    the GA keeps the top quarter and discards the rest."""

    def __init__(self, mu: torch.Tensor, scale: torch.Tensor, P: int,
                 sigma: float, lr: float, l2: float = 0.005):
        self.z = (mu / scale).clone()
        self.scale, self.P = scale, P
        self.sigma, self.lr, self.l2 = sigma, lr, l2
        self.n = (P - 1) // 2
        self.m = torch.zeros_like(self.z)
        self.v = torch.zeros_like(self.z)
        self.t = 0
        self.eps = None

    @property
    def mu(self) -> torch.Tensor:
        return self.z * self.scale

    def ask(self, best: torch.Tensor | None) -> torch.Tensor:
        N = self.z.numel()
        self.eps = torch.randn(self.n, N, device=self.z.device)
        d = self.sigma * self.eps * self.scale
        rows = [self.mu[None], self.mu + d, self.mu - d]
        if self.P - 1 - 2 * self.n:
            rows.append((best if best is not None else self.mu)[None])
        return torch.cat(rows, dim=0)

    def tell(self, fit_np: np.ndarray) -> float:
        n = self.n
        f = fit_np[1:1 + 2 * n]
        ranks = np.empty(2 * n, dtype=np.float32)
        ranks[np.argsort(f)] = np.arange(2 * n, dtype=np.float32)
        ranks = ranks / (2 * n - 1) - 0.5
        w = torch.as_tensor(ranks[:n] - ranks[n:], device=self.z.device)
        g = (w[:, None] * self.eps).sum(0) / (2 * n * self.sigma)
        g = g - self.l2 * self.z                 # ascent + decay
        self.t += 1
        b1, b2 = 0.9, 0.999
        self.m.mul_(b1).add_(g, alpha=1 - b1)
        self.v.mul_(b2).addcmul_(g, g, value=1 - b2)
        mh = self.m / (1 - b1 ** self.t)
        vh = self.v / (1 - b2 ** self.t)
        step = self.lr * mh / (vh.sqrt() + 1e-8)
        self.z.add_(step)
        return float(step.norm() / (self.z.norm() + 1e-12))
