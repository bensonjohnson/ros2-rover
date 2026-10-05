"""Shared ES evaluation primitive (SIM_PLATFORM stage 3a).

`evaluate()` is the drop-in successor to `evo.evolve.evaluate`: same
signature, same returns, but implemented on the platform pieces — the
`RolloutEngine` protocol entry point (`engine.run`) plus the native
`ESGenomePolicy` adapter. On CPU it is BIT-EXACT with the legacy closure
(rover_sim/tests/test_es_adapter.py A1), which is what lets stage 3b rewire
evo.evolve onto it with zero behavior change.

One-shot semantics: `engine.run()` registers the policy's `reset` as a reset
hook; a one-shot eval must not leak that hook into later (legacy) runs, so it
is removed again here. The hook the engine added IS this policy's bound
method, so it is the only one removed.
"""

from __future__ import annotations

import torch

from rover_sim.adapters.es import ESGenomePolicy
from rover_sim.runner.obs import OBS_DIM
from rover_sim.scoring import fitness


def evaluate(engine, thetas: torch.Tensor, hidden: int, ticks: int,
             w_dist: float = 0.03, w_coll: float = 0.25,
             w_cov: float = 0.0, obs: str = "v1", w_rev: float = 0.0,
             w_net: float = 0.0, w_spin: float = 0.0,
             action: str = "lr", mem: int = 0) -> tuple[torch.Tensor, dict]:
    """Run every individual in every house; returns per-individual fitness
    (mu) and the raw per-env metrics, exactly like evo.evolve.evaluate."""
    P = engine.P
    G_eff = engine.G * engine.G_sets
    policy = ESGenomePolicy(P, hidden, obs=obs, action=action, mem=mem,
                            merged_houses=engine.G_sets, device=thetas.device)
    policy.set_thetas(thetas)
    metrics = engine.run(policy, ticks)
    # Drop the temporary reset hook the engine registered for this one-shot
    # policy (do not leak hooks across repeated evaluate() calls).
    try:
        engine._reset_hooks.remove(policy.reset)
    except ValueError:
        pass
    fit = fitness(metrics, w_dist=w_dist, w_coll=w_coll,
                  w_cov=w_cov, w_rev=w_rev, w_net=w_net,
                  w_spin=w_spin).view(P, G_eff).mean(dim=1)
    return fit, metrics
