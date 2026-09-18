#!/usr/bin/env python3
"""Trim robustness of a genome: score it on the level-3 building holdout
(32 houses, full games) under several effective track-factor settings.

    python3 -m evo.trim_eval --genome evo_runs/X/val_best_genome.npz

A genome that only works with the sim's lopsided 0.8/1.0 tracks will not
survive the real rover (whose 0.8 left trim is a driver-side correction).
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from .arena import Arena, OBS_DIM, fitness, rev_frac
from .policy import PopulationNet, read_meta
from .worlds import HOLDOUT_SEED, build_pool

TRIMS = [(0.8, 1.0), (1.0, 1.0), (1.0, 0.8), (0.9, 1.0), (1.0, 0.9)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", required=True)
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--ticks", type=int, default=5400)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gate", choices=("legacy", "symmetric"), default="legacy")
    args = ap.parse_args()
    d = np.load(args.genome)
    obs, action = read_meta(d)
    H = int(d["hidden"])
    th = torch.as_tensor(d["thetas"], device=args.device).view(1, -1)
    worlds = build_pool(3, args.games, HOLDOUT_SEED)
    fused = args.device.startswith("cuda")
    rows = []
    for lt, rt in TRIMS:
        ar = Arena(1, args.games, seed=1, device=args.device, fp16=fused,
                   fused=fused, worlds=worlds, trim_rand=True,
                   gate_obs=obs == "v2",
                   gate_symmetric=args.gate == "symmetric")
        ar.set_trims([lt] * args.games, [rt] * args.games)
        net = PopulationNet(th, OBS_DIM, H, obs=obs, action=action)
        net.bind(1, args.games)
        st = {"h": torch.zeros(1, args.games, H, device=args.device)}

        def step(o, prev):
            a, st["h"] = net.step(o.view(1, args.games, OBS_DIM), st["h"])
            return a.view(args.games, 2)
        torch.manual_seed(3)
        m = ar.run_games(step, args.ticks)
        rows.append((lt, rt, float(fitness(m).mean()), float(m["rooms"].mean()),
                     float((m["rooms"] >= 2).float().mean()),
                     float(m["collisions"].mean()), float(rev_frac(m).mean())))
    print(f"trim robustness: {args.genome} (gate {args.gate})")
    print(f"  {'L':>4} {'R':>4} {'fit':>6} {'rooms':>5} {'cross':>5} {'coll':>6} {'rev':>4}")
    for r in rows:
        print(f"  {r[0]:4.1f} {r[1]:4.1f} {r[2]:6.3f} {r[3]:5.2f} {r[4]:5.3f} "
              f"{r[5]:6.1f} {r[6]:4.2f}")
    nom = rows[0][3]
    worst = min(r[3] for r in rows)
    print(f"  worst/nominal rooms = {worst / max(nom, 1e-6):.2f}")


if __name__ == "__main__":
    main()
