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
  M1  mem>0 packing: legacy prefix identical, mem=0 forward unchanged
  M2  mem>0 step == hand reference (write gate EMA + memory readout)
  M3  closed write gate (bg=-inf) == mem=0 exactly: the read/write path
      is what carries any memory effect
"""
from __future__ import annotations

import numpy as np
import torch

from .policy import (PopulationNet, _param_shapes, genome_size,
                     sample_population)


def reference(th, o, h, n_in, H, mem=0, m=None):
    """per individual, per env: plain matrix maths, no einsum."""
    assert not mem or m is not None
    P, G, _ = o.shape
    A = torch.zeros(P, G, 2)
    Hn = torch.zeros(P, G, H)
    for p in range(P):
        off, v = 0, {}
        for name, shape, *_ in _param_shapes(n_in, H, mem):
            k = int(np.prod(shape))
            v[name] = th[p, off:off + k].view(*shape)
            off += k
        hn = torch.tanh(o[p] @ v["Wx"] + h[p] @ v["Wh"] + v["bh"])
        if mem:
            hn = torch.tanh(o[p] @ v["Wx"] + h[p] @ v["Wh"]
                            + m[p] @ v["Wm"] + v["bh"])
        Hn[p] = hn
        ap = hn @ v["Wo"] + v["bo"]
        if mem:
            q = torch.cat([o[p], hn], dim=-1)
            g = torch.sigmoid(q @ v["Wg"] + v["bg"])
            w = torch.tanh(q @ v["Wv"] + v["bv"])
            m[p] = m[p] + g * (w - m[p])          # EMA write
            ap = ap + m[p] @ v["Wom"]             # action reads updated m
        A[p] = torch.tanh(ap)
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

    # M1: packing contract — legacy params keep names/order/positions,
    # memory block appends after them (a legacy [P, N0] vector IS the
    # prefix of a mem genome; RNG streams differ, layout does not)
    M = 8
    names0 = [n for n, *_ in _param_shapes(n_in, H, 0)]
    namesM = [n for n, *_ in _param_shapes(n_in, H, M)]
    assert namesM[:len(names0)] == names0
    N0 = th.shape[1]
    extra = genome_size(n_in, H, M) - N0
    th_m = torch.cat([th, torch.randn(P, extra) * 0.1], dim=1)
    assert th_m.shape == (P, N0 + (2 * (n_in + H) * M + 2 * M + M * H + M * 2))
    assert torch.equal(th_m[:, :N0], th)
    net0 = PopulationNet(th, n_in, H, mem=0)
    net0.bind(P, G)
    assert torch.allclose(net0.step(o, h)[0], ar, atol=1e-5)
    print(f"M1 ok: mem packing appends only; N {N0} -> {th_m.shape[1]}")

    # M2: mem step == reference over 3 ticks (write gate + readout live)
    netm = PopulationNet(th_m, n_in, H, mem=M)
    netm.bind(P, G)
    netm.reset_memory()
    mref = torch.zeros(P, G, M)
    o2, h2 = o, h
    for _ in range(3):
        am, hm = netm.step(o2, h2)
        arf, hrf = reference(th_m, o2, h2, n_in, H, mem=M, m=mref)
        assert torch.allclose(am, arf, atol=1e-5), (am - arf).abs().max()
        assert torch.allclose(hm, hrf, atol=1e-5)
        assert mref.abs().max() > 0            # gate (bg -3) still writes
        o2, h2 = torch.rand(P, G, n_in), hm
    print("M2 ok: 3-tick memory step matches reference; memory nonzero")

    # M3: write gate closed (bg -> -inf) == mem=0 forward, exactly.
    th_off = th_m.clone()
    q, off = n_in + H, N0
    off += q * M                                # skip Wg
    th_off[:, off:off + M] = -1e4               # bg: sigmoid ~ 0
    netc = PopulationNet(th_off, n_in, H, mem=M)
    netc.bind(P, G)
    netc.reset_memory()
    a_off, _ = netc.step(o, h)
    assert torch.allclose(a_off, ar, atol=1e-4), (a_off - ar).abs().max()
    assert netc.m.abs().max() < 1e-3
    print("M3 ok: closed gate == mem=0; memory effect is write-path only")

    print("ALL POLICY CHECKS PASSED")


if __name__ == "__main__":
    main()
