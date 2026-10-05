"""Truncation GA reproduction operator (SIM_PLATFORM stage 3b).

Moved VERBATIM out of evo/evolve.py: the body of `reproduce()` is
byte-for-byte the legacy function; only this import block and module
docstring are new. Free names it needed (`sample_population`, `OBS_DIM`)
are imported from their canonical homes in rover_sim.
"""

from __future__ import annotations

import numpy as np
import torch

from rover_sim.policies.es_genome import sample_population
from rover_sim.runner.obs import OBS_DIM


def reproduce(thetas: torch.Tensor, sigma: torch.Tensor, fit_np: np.ndarray,
              order: np.ndarray, scale: torch.Tensor,
              rng: np.random.Generator, *, n_elite: int, n_immigrant: int,
              tau: float, sigma_max: float, sigma0: float, hidden: int,
              legacy: bool = False, mem: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Next population, laid out [survivors | children | immigrants].

    Survivors = top n_elite untouched. Each child copies ONE tournament
    parent's genome and gets per-gene Gaussian mutation in init-scale
    units with a log-normal self-adapted per-individual sigma.

    Runs 1-8 built children as `p1.unsqueeze(1).expand(-1, N) * mix`, i.e.
    the parent INDEX (0..P-1) broadcast as every gene: children were
    constant vectors + noise, nothing was inherited, and the ES reduced to
    elitism over gen-0 samples and random immigrants (evo.test_reproduce).
    """
    P, N = thetas.shape
    dev = thetas.device
    n_child = P - n_elite - n_immigrant
    survivors = torch.as_tensor(order[:n_elite], device=dev)

    def tournament(k=3):
        cand = rng.integers(0, P, size=(n_child, k))
        return cand[np.arange(n_child), fit_np[cand].argmax(axis=1)]

    par = torch.as_tensor(tournament(), device=dev)
    if legacy:
        # runs 1-8 step-size rule (max of two parents, tau 0.4, <= 1.0)
        par2 = torch.as_tensor(tournament(), device=dev)
        s_child = sigma[par].maximum(sigma[par2]) * torch.exp(
            tau * torch.randn(n_child, 1, device=dev)
            + 0.1 * torch.randn(n_child, 1, device=dev))
    else:
        s_child = sigma[par] * torch.exp(
            tau * torch.randn(n_child, 1, device=dev))
    s_child = s_child.clamp(1e-4, sigma_max)
    child = thetas[par] + s_child * scale.unsqueeze(0) \
        * torch.randn(n_child, N, device=dev)

    immigrants = torch.as_tensor(
        sample_population(n_immigrant, OBS_DIM, hidden, rng, mem=mem),
        device=dev)
    sig_imm = torch.full((n_immigrant, 1), sigma0, device=dev)
    return (torch.cat([thetas[survivors], child, immigrants], dim=0),
            torch.cat([sigma[survivors], s_child, sig_imm], dim=0))
