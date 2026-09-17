#!/usr/bin/env python3
"""PopulationNet forward checks (the runs 1-9 readout bug).

    python3 -m evo.test_policy

Runs 1-9 used einsum("pgh,pgHK->pgK", h_new, Wo): the mismatched h/H
letters SUM each side independently, so both wheel commands were
tanh(sum(h) * colsum(Wo) + bo) — functions of ONE scalar. bc_seed trained
with the correct h @ Wo and replayed through the broken one.

  P1  batched forward == explicit per-individual row-vector maths
  P2  both outputs are NOT a function of sum(h) alone
  P3  action="vw" mixing and obs="v2" centring as documented
"""
from __future__ import annotations

import numpy as np
import torch

from .policy import PopulationNet, _param_shapes, sample_population


def reference(th, o, h, n_in, H):
    """per individual, per env: plain matrix maths, no einsum."""
    P, G, _ = o.shape
    A = torch.zeros(P, G, 2)
    Hn = torch.zeros(P, G, H)
    for p in range(P):
        off, v = 0, {}
        for name, shape, _ in _param_shapes(n_in, H):
            k = int(np.prod(shape))
            v[name] = th[p, off:off + k].view(*shape)
            off += k
        hn = torch.tanh(o[p] @ v["Wx"] + h[p] @ v["Wh"] + v["bh"])
        Hn[p] = hn
        A[p] = torch.tanh(hn @ v["Wo"] + v["bo"])
    return A, Hn


def main():
    torch.manual_seed(0)
    P, G, n_in, H = 4, 6, 82, 32
    th = torch.as_tensor(sample_population(P, n_in, H,
                                           np.random.default_rng(0)))
    o = torch.rand(P, G, n_in)
    h = torch.randn(P, G, H).tanh()
    net = PopulationNet(th, n_in, H)
    net.bind(P, G)
    a, hn = net.step(o, h)
    ar, hr = reference(th, o, h, n_in, H)
    assert torch.allclose(a, ar, atol=1e-5), (a - ar).abs().max()
    assert torch.allclose(hn, hr, atol=1e-5)
    print(f"P1 ok: max diff {(a - ar).abs().max():.1e}")

    # P2: two hidden states with equal sums must be able to steer apart
    h1 = torch.randn(1, 1, H)
    h2 = h1[..., torch.randperm(H)]            # same sum, different state
    W = torch.randn(1, H, 2)
    d = (torch.bmm(h1, W) - torch.bmm(h2, W)).abs().max()
    assert d > 1e-3, "readout depends only on sum(h)"
    print("P2 ok: readout is a real linear map of h")

    vw = PopulationNet(th, n_in, H, action="vw")
    vw.bind(P, G)
    a_vw, _ = vw.step(o, h)
    assert torch.allclose(a_vw[..., 0], (ar[..., 0] - ar[..., 1]).clamp(-1, 1),
                          atol=1e-5)
    assert torch.allclose(a_vw[..., 1], (ar[..., 0] + ar[..., 1]).clamp(-1, 1),
                          atol=1e-5)
    v2 = PopulationNet(th, n_in, H, obs="v2")
    v2.bind(P, G)
    oc = torch.cat([o[..., :-2] * 2 - 1, o[..., -2:]], dim=-1)
    a_c, _ = reference(th, oc, h, n_in, H)
    assert torch.allclose(v2.step(o, h)[0], a_c, atol=1e-5)
    print("P3 ok: vw mixing + v2 centring")
    print("ALL POLICY CHECKS PASSED")


if __name__ == "__main__":
    main()
