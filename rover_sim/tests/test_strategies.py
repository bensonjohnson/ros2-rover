"""Strategy-package checks (docs/SIM_PLATFORM.md §6 stage 3b).

    python3 -m rover_sim.tests.test_strategies

S1  identity: the names evo.evolve re-exports ARE the strategy-package
    objects (reproduce, OES) — legacy `from evo.evolve import ...` resolves
    to the moved code.
S2  registry: STRATEGIES['ga']['reproduce'] / ['oes']['class'] are wired to
    those same objects.
S3  reproduce determinism smoke (CPU): two identically-seeded calls are
    bit-identical and preserve shapes; runs on cpu tensors.
S4  OES smoke (CPU): constructs on cpu tensors, ask() is mirrored, tell()
    moves the centre on a toy objective (no CUDA hardcoding).
"""

from __future__ import annotations

import numpy as np
import torch

from evo.evolve import OES as evo_OES
from evo.evolve import reproduce as evo_reproduce
from rover_sim.policies.es_genome import genome_size, per_gene_scale, \
    sample_population
from rover_sim.runner.obs import OBS_DIM
from rover_sim.strategies import STRATEGIES
from rover_sim.strategies.ga import reproduce
from rover_sim.strategies.oes import OES


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

    # S1: identity (evo.evolve re-exports the strategy objects) -----------
    assert evo_reproduce is reproduce, "evo.evolve.reproduce is not ga.reproduce"
    assert evo_OES is OES, "evo.evolve.OES is not oes.OES"
    print("S1 ok: evo.evolve.reproduce/OES are the strategy-package objects")

    # S2: registry wiring -------------------------------------------------
    assert STRATEGIES["ga"]["reproduce"] is reproduce, STRATEGIES
    assert STRATEGIES["oes"]["class"] is OES, STRATEGIES
    print(f"S2 ok: registry keys {sorted(STRATEGIES)} wired to the same objects")

    # S3: reproduce determinism smoke (CPU) -------------------------------
    def run_once():
        r = np.random.default_rng(4242)
        torch.manual_seed(99)
        return reproduce(thetas, torch.full((P, 1), 0.03), fit, order, scale,
                         r, sigma_max=0.15, **kw)

    a_th, a_sg = run_once()
    b_th, b_sg = run_once()
    assert thetas.device.type == "cpu", thetas.device
    assert a_th.shape == (P, N) and a_sg.shape == (P, 1), \
        (a_th.shape, a_sg.shape)
    assert torch.equal(a_th, b_th), "reproduce thetas not deterministic"
    assert torch.equal(a_sg, b_sg), "reproduce sigma not deterministic"
    assert torch.isfinite(a_th).all() and torch.isfinite(a_sg).all()
    # survivors = top n_elite untouched (inheritance sanity)
    assert torch.equal(a_th[:8],
                       thetas[torch.as_tensor(order[:8])]), "survivors differ"
    print(f"S3 ok: reproduce bit-identical across identical seeds "
          f"(thetas {tuple(a_th.shape)}, sigma {tuple(a_sg.shape)}, "
          f"|Δ|={float((a_th - b_th).abs().max()):.1e})")

    # S4: OES smoke (CPU) -------------------------------------------------
    target = torch.randn(N)
    oes = OES(torch.zeros(N), scale, 33, sigma=0.05, lr=0.05, l2=0.0)
    assert oes.mu.device.type == "cpu", oes.mu.device

    def f(th):
        return -((th / scale - target) ** 2).sum(1).numpy()

    d0 = float(((oes.mu / scale - target) ** 2).sum())
    for _ in range(40):
        pop = oes.ask(None)
        assert pop.shape == (33, N), pop.shape
        assert torch.allclose(pop[1:17] + pop[17:33], 2 * pop[0], atol=1e-4), \
            "OES pairs not mirrored"
        oes.tell(f(pop))
    d1 = float(((oes.mu / scale - target) ** 2).sum())
    assert d1 < d0, f"OES did not climb: {d0:.1f} -> {d1:.1f}"
    print(f"S4 ok: OES on cpu tensors, mirrored pairs, distance "
          f"{d0:.0f} -> {d1:.0f}")

    print("ALL STRATEGIES CHECKS PASSED")


if __name__ == "__main__":
    main()
