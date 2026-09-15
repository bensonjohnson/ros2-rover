#!/usr/bin/env python3
"""Bisect the run-5a crash: CUDA illegal memory access on the FIRST graph
replay AFTER a live capture + interleaved eager holdout eval (gen 1).

Three cells, each in its OWN PROCESS (launch one per invocation):
  A: production 5a config, w_cov 0.3   (the crashing combo)
  B: production 5a config, w_cov 0.0   (isolates the cells accumulator)
  C: production 5a config, w_cov 0.3, holdout disabled (isolates the
     interleaved eager holdout Arena allocs)

    python3 -m evo.bisect5 --cell A --gens 3

PASS = reaches "CELL-A-DONE" (or the cell's own DONE marker) without a
CUDA error. Prints progress each gen.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from evo.arena import Arena, OBS_DIM, fitness
from evo.evolve import GraphRunner, evaluate
from evo.policy import (PopulationNet, per_gene_scale, sample_population)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices="ABC", required=True)
    ap.add_argument("--gens", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--pop", type=int, default=128)
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--ticks", type=int, default=5400)
    ap.add_argument("--rotations", type=int, default=4)
    ap.add_argument("--train-seed", type=int, default=62000)
    ap.add_argument("--holdout-seed", type=int, default=777_000)
    args = ap.parse_args()

    dev = "cuda"
    P, G, hidden = args.pop, args.games, args.hidden
    w_cov = 0.0 if args.cell == "B" else 0.3
    cells_on = args.cell != "B"
    do_holdout = args.cell != "C"

    rng = np.random.default_rng(6)
    thetas = torch.as_tensor(sample_population(P, OBS_DIM, hidden, rng),
                             device=dev)

    tr = Arena(P, G, seed=args.train_seed, device=dev, fp16=True,
               merged_houses=args.rotations, cells=cells_on)
    ho = (Arena(P, G, seed=args.holdout_seed, device=dev, fp16=True,
                cells=cells_on)
          if do_holdout else None)
    rn = GraphRunner(tr, P, hidden)

    G_eff = G * args.rotations
    for gen in range(args.gens):
        fit, metrics = rn.run(thetas, args.ticks, 0.005, 0.25, w_cov)
        print(f"gen {gen}: replay OK  best={float(fit.max()):.4f} "
              f"cells={float((metrics['cells']/metrics['cells_total']).mean()):.3f}",
              flush=True)
        if do_holdout:
            # exactly what evolve.py does at report gens: eager eval on the
            # SECOND (holdout) Arena while the train graph is live
            hf, hm = evaluate(ho, thetas, hidden, args.ticks,
                              w_dist=0.005, w_coll=0.25, w_cov=w_cov)
            print(f"gen {gen}: holdout eager OK best={float(hf.max()):.4f}",
                  flush=True)
        torch.cuda.synchronize()
        print(f"gen {gen}: sync OK", flush=True)

    print(f"CELL-{args.cell}-DONE", flush=True)


if __name__ == "__main__":
    main()
