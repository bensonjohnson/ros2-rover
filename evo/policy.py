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


OBS_MODES = ("v1", "v2")      # v2: gate-flag channels + inputs centred
ACTION_MODES = ("lr", "vw")   # vw: outputs (v, w) mixed to tracks


def genome_meta(obs: str = "v1", action: str = "lr") -> dict:
    """npz fields every saved genome/population carries (absent = v1/lr)."""
    assert obs in OBS_MODES and action in ACTION_MODES
    return {"obs_mode": obs, "action_mode": action}


def read_meta(d) -> tuple[str, str]:
    """(obs_mode, action_mode) of a loaded npz; files from runs <= 9 have
    no fields and are v1/lr."""
    obs = str(d["obs_mode"]) if "obs_mode" in d else "v1"
    act = str(d["action_mode"]) if "action_mode" in d else "lr"
    return obs, act


class PopulationNet:
    """[P individuals] x [G envs] batched forward. All tensors on device.

    obs="v2": the first n_in-2 inputs (scan bins + proprio, all in [0, 1])
    are centred to [-1, 1] before the net — raw [0, 1] scans sit near 1 in
    open space, a large DC drive into tanh units whose biases also init at
    0.5 (evolved controllers came out bang-bang, |a| ~0.97). last-action
    inputs are already in [-1, 1]. (The gate-flag channels themselves are
    produced by Arena(gate_obs=True).)
    action="vw": outputs are (forward, turn), tracks = clamp(v -/+ w): going
    straight / turning are single-output changes instead of a coordinated
    left/right pair (left_trim 0.8 makes 'straight' an asymmetric L/R)."""

    def __init__(self, thetas: torch.Tensor, n_in: int, hidden: int,
                 obs: str = "v1", action: str = "lr"):
        """thetas [P, N] on device."""
        self.P, self.N = thetas.shape
        self.n_in, self.hidden = n_in, hidden
        self.thetas = thetas
        self.obs_center = obs == "v2"
        self.action_vw = action == "vw"
        self._views = None

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
        """o [P, G, n_in], h [P, G, hidden] -> (action [P,G,2], h).
        Batched over individuals with bmm (row-vector convention, same
        maths as the old expand+einsum, without expanded weight views)."""
        v = self._views
        if self.obs_center:
            k = self.n_in - 2
            o = torch.cat([o[..., :k] * 2.0 - 1.0, o[..., k:]], dim=-1)
        pre = (torch.bmm(o, v["Wx"]) + torch.bmm(h, v["Wh"])
               + v["bh"].unsqueeze(1))
        h_new = torch.tanh(pre)
        act = torch.tanh(torch.bmm(h_new, v["Wo"]) + v["bo"].unsqueeze(1))
        if self.action_vw:
            fwd, turn = act[..., 0:1], act[..., 1:2]
            act = torch.cat([fwd - turn, fwd + turn], dim=-1).clamp(-1, 1)
        return act, h_new
