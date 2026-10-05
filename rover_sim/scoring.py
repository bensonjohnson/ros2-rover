"""Fitness and metric semantics for ES rollouts.

Moved verbatim out of evo/arena.py (stage 2, SIM_PLATFORM). See
docs/SIM_PLATFORM.md §2/§3.
"""

from __future__ import annotations

import torch


def rev_frac(metrics: dict) -> torch.Tensor:
    """Fraction of travelled distance driven in reverse (0 if static)."""
    tot = metrics["fwd_m"] + metrics["back_m"]
    return torch.where(tot > 1e-3, metrics["back_m"] / tot.clamp(min=1e-3),
                       torch.zeros_like(tot))


def fitness(metrics: dict, w_dist: float = 0.03,
            w_coll: float = 0.25, w_cov: float = 0.0,
            w_rev: float = 0.0, w_net: float = 0.0,
            w_spin: float = 0.0) -> torch.Tensor:
    """Per-env scalar: fraction of reachable rooms visited (the honest
    target), distance as a small tiebreaker (stops are zero-effort rooms),
    collisions penalized (the gate is there; brute-forcing it shouldn't pay).

    w_cov > 0 adds the novelty term (run 5): fraction of reachable 0.8 m
    cells visited. Rooms is a sparse cliff that pays only on the crossing
    tick and gives the wall-hugger plateau nothing to steer on; coverage
    pays densely from tick one and saturates at the house ceiling (no
    novelty fountain). At w_cov ~0.3 coverage can buy one room crossing'
    worth of fitness — enough to pull a policy toward a door, not enough
    to replace the rooms term.

    w_rev > 0 (run 15) subtracts w_rev x the fraction of distance driven in
    reverse. The runs 11-14 champions explore mostly BACKWARDS (sim mean
    command -0.72; real rover 75% reversing) because nothing priced
    direction; the real rover's bumper geometry and camera face forward.
    Scale-free, so a short reverse to escape a front block costs little.

    w_net > 0 (run 23) pays for genuine translation: max straight-line
    distance ever reached from the start pose, /10 m. dist_m above counts
    ARC LENGTH, which an in-place orbit earns for free — live runs 5-8
    proved the sim's arc-hungry champions (arc 23 m, net 3 m) are zero-turn
    orbiters on the real skid-steer, where a pivot loads and stalls the
    downhill track. w_spin > 0 subtracts w_spin x the fraction of ticks
    spent pivoting (opposite-sign gated wheels): honest turning is arcs,
    pirouettes are chassis abuse on this rover. Both are 0 by default —
    every historical run keeps its exact scores."""
    frac = metrics["rooms"] / metrics["rooms_total"].clamp(min=1.0)
    out = (frac
           + w_dist * metrics["dist_m"] / 10.0
           - w_coll * metrics["collisions"] / 100.0)
    if w_cov:
        out = out + w_cov * metrics["cells"] / metrics["cells_total"].clamp(min=1.0)
    if w_rev:
        out = out - w_rev * rev_frac(metrics)
    if w_net:
        out = out + w_net * metrics["range_m"] / 10.0
    if w_spin:
        out = out - w_spin * metrics["spin_frac"]
    return out
