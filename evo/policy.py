"""Genome: recurrent MLP policy, packed into a flat float32 vector.

    o_t (scan bins + proprio + last action)
      -> h_t = tanh(Wx o + Wh h_{t-1} + bh)
      -> a_t = tanh(Wo h + bo)                     (2 wheel commands)

Weights are evolved, never gradient-learned. The flat packing keeps the
whole population in one [P, N] tensor so mutation/crossover are elementwise
ops and the batched forward is a handful of einsums.
"""

from __future__ import annotations

import numpy as np
import torch


def genome_size(n_in: int, hidden: int) -> int:
    return (n_in * hidden + hidden          # Wx, bh
            + hidden * hidden               # Wh
            + hidden * 2 + 2)               # Wo, bo


def _param_shapes(n_in: int, hidden: int):
    """(name, shape, init_scale) in packing order. Biases init hot-ish:
    a fully-zero policy never moves, and an ES on frozen-at-zero dynamics
    has no gradient of progress to follow."""
    return [
        ("Wx", (n_in, hidden), 1.0 / np.sqrt(n_in)),
        ("bh", (hidden,), 0.5),
        ("Wh", (hidden, hidden), 1.0 / np.sqrt(hidden)),
        ("bo", (2,), 0.5),
        ("Wo", (hidden, 2), 1.0 / np.sqrt(hidden)),
    ]


def per_gene_scale(n_in: int, hidden: int) -> np.ndarray:
    """Per-gene init scale, reused as the mutation-size unit: a sigma of 1.0
    means one init-standard-deviation for that gene, so layers with tiny
    weights (deep, wide fan-in) are not frozen out of the search."""
    out = []
    for _, shape, scale in _param_shapes(n_in, hidden):
        out.append(np.full(int(np.prod(shape)), scale, dtype=np.float32))
    return np.concatenate(out)


def sample_population(P: int, n_in: int, hidden: int,
                      rng: np.random.Generator) -> np.ndarray:
    """P fresh nets, each drawn from the init distribution — the natural
    'random greenfield rover' the first generation must not be worse than."""
    scales = per_gene_scale(n_in, hidden)
    return (rng.standard_normal((P, scales.size)).astype(np.float32)
            * scales[None, :])


class PopulationNet:
    """[P individuals] x [G envs] batched forward. All tensors on device."""

    def __init__(self, thetas: torch.Tensor, n_in: int, hidden: int):
        """thetas [P, N] on device."""
        self.P, self.N = thetas.shape
        self.n_in, self.hidden = n_in, hidden
        self.thetas = thetas
        self._views = None
        self._split = None

    def bind(self, P: int, G: int):
        """(Re)bind views after thetas changed. P may differ from thetas'
        first dim only via expand — we always pass the exact population."""
        t = self.thetas
        off = 0
        views = {}
        for name, shape, _ in _param_shapes(self.n_in, self.hidden):
            numel = int(np.prod(shape))
            views[name] = t[:, off:off + numel].view(self.P, *shape)
            off += numel
        assert off == self.N
        self._views = views
        self._G = G

    def step(self, o: torch.Tensor, h: torch.Tensor):
        """o [P, G, n_in], h [P, G, hidden] -> (action [P,G,2], h)."""
        v = self._views
        G = self._G
        # expand params over the env dim (stride-0, no copy until einsum)
        Wx = v["Wx"].unsqueeze(1).expand(-1, G, -1, -1)
        Wh = v["Wh"].unsqueeze(1).expand(-1, G, -1, -1)
        bh = v["bh"].unsqueeze(1).expand(-1, G, -1)
        Wo = v["Wo"].unsqueeze(1).expand(-1, G, -1, -1)
        bo = v["bo"].unsqueeze(1).expand(-1, G, -1)
        pre = (torch.einsum("pgi,pgih->pgh", o, Wx)
               + torch.einsum("pgh,pghk->pgk", h, Wh) + bh)
        h_new = torch.tanh(pre)
        act = torch.tanh(torch.einsum("pgh,pgHK->pgK", h_new, Wo) + bo)
        return act, h_new
