#!/usr/bin/env python3
"""Can short games stand in for 6-minute games during selection?

Scores one population at --short and --long ticks on the SAME houses
(eager, same fitness weights as the run) and reports the Spearman rank
correlation of per-individual fitness and cells, plus top-quarter overlap.
If rho >= ~0.8 and the elites overlap, early generations can run at the
short length (chain phases via --seed-from; graph shapes are fixed per
process) for a ~long/short cut in generation time.

    python3 -m evo.tick_rankcorr --population evo_runs/greenfield9b/final_population.npz
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from .arena import Arena
from .evolve import evaluate
from .policy import read_meta


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--rotations", type=int, default=4)
    ap.add_argument("--train-seed", type=int, default=63000)
    ap.add_argument("--short", default="1800,2700")
    ap.add_argument("--long", type=int, default=5400)
    ap.add_argument("--w-dist", type=float, default=0.005)
    ap.add_argument("--w-cov", type=float, default=0.3)
    args = ap.parse_args()
    d = np.load(args.population)
    obs, action = read_meta(d)
    th = torch.as_tensor(d["thetas"], device=args.device)
    P, H = th.shape[0], int(d["hidden"])
    ar = Arena(P, args.games, seed=args.train_seed, device=args.device,
               fp16=True, merged_houses=args.rotations,
               gate_obs=obs == "v2")
    G_eff = args.games * args.rotations

    def score(ticks):
        torch.manual_seed(11)
        fit, m = evaluate(ar, th, H, ticks, w_dist=args.w_dist,
                          w_cov=args.w_cov, obs=obs, action=action)
        cells = (m["cells"] / m["cells_total"]).view(P, G_eff).mean(1)
        return fit.cpu().numpy(), cells.cpu().numpy()

    f_long, c_long = score(args.long)
    q = max(2, P // 4)
    top_long = set(np.argsort(-f_long)[:q])
    print(f"population {args.population}  P={P}  long={args.long}")
    for t in (int(x) for x in args.short.split(",")):
        f_s, c_s = score(t)
        top_s = set(np.argsort(-f_s)[:q])
        print(f"  short {t:>5d}: rho_fit={spearman(f_s, f_long):.3f} "
              f"rho_cells={spearman(c_s, c_long):.3f} "
              f"top-{q} overlap={len(top_s & top_long) / q:.2f}")


if __name__ == "__main__":
    main()
