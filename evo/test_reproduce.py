#!/usr/bin/env python3
"""Reproduction operator checks (the runs 1-8 parent-gather bug).

    python3 -m evo.test_reproduce

  R1  every child is within mutation distance of SOME scored genome
      (inheritance) — the runs 1-8 operator produced constant vectors
  R2  survivors are the top n_elite rows, bit-identical
  R3  child sigma respects sigma_max; sigma=tiny => child == a parent
"""
from __future__ import annotations

import numpy as np
import torch

from .arena import OBS_DIM
from .evolve import reproduce
from .policy import genome_size, per_gene_scale, sample_population


def main():
    H, P = 16, 32
    N = genome_size(OBS_DIM, H)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    thetas = torch.as_tensor(sample_population(P, OBS_DIM, H, rng))
    scale = torch.as_tensor(per_gene_scale(OBS_DIM, H))
    fit = rng.standard_normal(P).astype(np.float32)
    order = np.argsort(-fit)
    kw = dict(n_elite=8, n_immigrant=3, tau=1 / np.sqrt(N), sigma0=0.03,
              hidden=H)

    # R1/R3: tiny sigma -> each child equals some parent row
    sig = torch.full((P, 1), 1e-4)
    th2, s2 = reproduce(thetas, sig, fit, order, scale, rng,
                        sigma_max=0.15, **kw)
    children = th2[8:P - 3]
    d = torch.cdist(children, thetas).min(dim=1).values
    assert d.max() < 1e-2, f"R1 child not a parent copy: {d.max():.4g}"
    assert (children.std(dim=1) > 1e-3).all(), "R1 constant child genome"
    print(f"R1 ok: max child-parent dist {d.max():.2e}")

    # R2
    assert torch.equal(th2[:8], thetas[torch.as_tensor(order[:8])]), "R2"
    print("R2 ok: survivors = top-8 untouched")

    # R3: sigma clamp, both rules
    big = torch.full((P, 1), 5.0)
    for legacy, smax in ((False, 0.15), (True, 1.0)):
        _, s3 = reproduce(thetas, big, fit, order, scale, rng,
                          sigma_max=smax, legacy=legacy,
                          **{**kw, "tau": 0.4 if legacy else kw["tau"]})
        assert float(s3[8:P - 3].max()) <= smax + 1e-6, "R3 clamp"
    print("R3 ok: sigma clamped (v9 0.15, legacy 1.0)")
    print("ALL REPRODUCE CHECKS PASSED")


if __name__ == "__main__":
    main()
