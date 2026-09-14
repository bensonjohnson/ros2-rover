#!/usr/bin/env python3
"""Scripted baselines + champion replay on the holdout arena.

Baselines get the SAME sensors, the SAME safety gate, and the SAME houses
as evolved policies: if evolution can't beat a 20-line wall-follower, say
so in the report.

    python3 -m evo.baselines --device cuda            # baselines + best genome
    python3 -m evo.baselines --genome evo_out/best_genome.npz
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from .arena import Arena, OBS_DIM, NUM_BINS, fitness
from .policy import PopulationNet


def _scan(obs: torch.Tensor) -> torch.Tensor:
    return obs[:, :NUM_BINS]


def random_walk(obs, prev):
    B = obs.shape[0]
    a = torch.rand(B, 2, device=obs.device) * 2 - 1
    return torch.where((prev.abs().sum(1, keepdim=True) < 0.05)
                       | (torch.rand(B, 1, device=obs.device) < 0.05),
                       a, prev)


def go_forward(obs, prev):
    """Trust the gate: full stick forward, let the safety monitor deal."""
    B = obs.shape[0]
    return torch.ones(B, 2, device=obs.device)


def wall_follower(obs, prev):
    """Reactive wall-hug: steer toward the more open hemisphere, modulated
    by front clearance. ~30 lines, no state beyond persistence."""
    s = _scan(obs)                       # [B, 72], bin 0 = forward
    B = s.shape[0]
    front = torch.cat([s[:, :8], s[:, -8:]], dim=1).min(dim=1).values
    left = s[:, 9:36].mean(dim=1)        # 45..180 deg
    right = s[:, 36:63].mean(dim=1)      # 180..315 deg
    open_diff = (left - right).clamp(-1, 1)
    # steer toward open space; slow down when close
    fwd = (0.3 + 0.7 * front.clamp(0.0, 1.0)) * 0.8
    turn = 0.6 * open_diff + 0.25 * (s[:, :8].min(dim=1).values < 0.15) \
        * torch.where(prev[:, 0] >= prev[:, 1], 1.0, -1.0)
    l = (fwd - turn).clamp(-1, 1)
    r = (fwd + turn).clamp(-1, 1)
    # if wedged (front blocked AND sides blocked), back up with a turn
    wedged = (front < 0.06) & (torch.maximum(left, right) < 0.12)
    back = torch.stack([-0.6 * torch.sign(prev[:, 0] + prev[:, 1] + 0.01),
                        0.6 * torch.sign(prev[:, 0] + prev[:, 1] + 0.01)],
                       dim=1)
    return torch.where(wedged[:, None], back, torch.stack([l, r], dim=1))


BASELINES = {"random_walk": random_walk, "go_forward": go_forward,
             "wall_follower": wall_follower}


def run_named(fn, arena: Arena, ticks: int) -> dict:
    def step(obs, prev):
        return fn(obs, prev)
    m = arena.run_games(step, ticks)
    fit = fitness(m)
    return {"name": fn.__name__,
            "fitness": float(fit.mean()),
            "rooms": float(m["rooms"].mean()),
            "rooms_total": float(m["rooms_total"].mean()),
            "dist_m": float(m["dist_m"].mean()),
            "collisions": float(m["collisions"].mean())}


def run_genome(path: str, arena: Arena, ticks: int) -> dict:
    d = np.load(path)
    thetas = torch.as_tensor(d["thetas"], device=arena.device)
    if thetas.ndim == 1:                    # best_genome.npz (single row)
        thetas = thetas.unsqueeze(0)
    hidden = int(d["hidden"])
    P = thetas.shape[0]
    G = arena.B // P
    assert arena.B % P == 0
    net = PopulationNet(thetas, OBS_DIM, hidden)
    net.bind(P, G)
    state = {"h": torch.zeros(P, G, hidden, device=arena.device)}

    def step(obs, prev):
        a, state["h"] = net.step(obs.view(P, G, OBS_DIM), state["h"])
        return a.view(P * G, 2)

    m = arena.run_games(step, ticks)
    fit = fitness(m)
    return {"name": f"genome({path.split('/')[-1]})",
            "fitness": float(fit.mean()),
            "rooms": float(m["rooms"].mean()),
            "rooms_total": float(m["rooms_total"].mean()),
            "dist_m": float(m["dist_m"].mean()),
            "collisions": float(m["collisions"].mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--ticks", type=int, default=3600)
    ap.add_argument("--holdout-seed", type=int, default=777_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--genome", default="evo_out/best_genome.npz")
    ap.add_argument("--population", default="",
                    help="optional final_population.npz: score all, report best/median")
    args = ap.parse_args()

    arena = Arena(1, args.games, seed=args.holdout_seed, device=args.device,
                  fp16=args.fp16)
    rows = [run_named(fn, arena, args.ticks) for fn in BASELINES.values()]
    import os
    if os.path.exists(args.genome):
        rows.append(run_genome(args.genome, arena, args.ticks))
    if args.population and os.path.exists(args.population):
        d = np.load(args.population)
        P = d["thetas"].shape[0]
        a2 = Arena(P, args.games, seed=args.holdout_seed, device=args.device,
                   fp16=args.fp16)
        r = run_genome(args.population, a2, args.ticks)
        from .evolve import evaluate as _ev
        th = torch.as_tensor(d["thetas"], device=args.device)
        fitp, _ = _ev(a2, th, int(d["hidden"]), args.ticks)
        r["name"] = "population(best)"
        r["fitness"] = float(fitp.max())
        r2 = dict(r); r2["name"] = "population(median)"
        r2["fitness"] = float(fitp.median())
        rows += [r, r2]

    hdr = (f"{'controller':<28} {'fitness':>8} {'rooms':>6} {'/tot':>5} "
           f"{'dist_m':>7} {'coll':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:<28} {r['fitness']:>8.4f} {r['rooms']:>6.2f} "
              f"{r['rooms_total']:>5.1f} {r['dist_m']:>7.1f} "
              f"{r['collisions']:>6.1f}")


if __name__ == "__main__":
    main()
