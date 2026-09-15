#!/usr/bin/env python3
"""Production-exact crash repro: the FULL evolve.py loop (including the
between-generation ES reproduction with fresh torch.randn children) for
--gens 2 at production sizes. If evolve.py crashes on gen 1, this does.

  python3 -m evo.repro5                      (cells scatter in graph)
  python3 -m evo.repro5 --no-cells           (accumulator disabled at the
                                              rollout-body level)
"""
from __future__ import annotations

import argparse
import types

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cells", action="store_true")
    ap.add_argument("--w-cov", type=float, default=0.3)
    ap.add_argument("--gens", type=int, default=2)
    args = ap.parse_args()

    if args.no_cells:
        # neuter the accumulator: _make_acc returns a dummy 1-col buffer,
        # _cell_index pinned to column 0 (scatter becomes same-cell churn)
        from evo import arena as A

        def _make_acc(self):
            return (torch.zeros(self.B, 2, device=self.device),
                    torch.zeros(self.B, dtype=torch.int64, device=self.device),
                    torch.zeros(self.B, device=self.device),
                    torch.zeros(self.B, device=self.device),
                    torch.zeros(self.B, 1, device=self.device))

        def _cell_index(self):
            return torch.zeros(self.B, dtype=torch.int64, device=self.device)

        A.Arena._make_acc = _make_acc
        A.Arena._cell_index = _cell_index
        # run_games' cells key must still exist:
        orig_rg = A.Arena.run_games

        def rg(self, *a, **k):
            m = orig_rg(self, *a, **k)
            return m
        A.Arena.run_games = rg
        A.Arena.cells_total = torch.ones(1, device="cuda")

    from evo.evolve import evolve

    ns = argparse.Namespace(
        pop=128, games=16, hidden=64, ticks=5400, gens=args.gens,
        sigma0=0.25, train_rotations=4, train_seed=62000,
        holdout_seed=777_000, report_every=1, w_dist=0.005, w_coll=0.25,
        w_cov=args.w_cov, fp16=True, graph=True, device="cuda",
        out_dir="/tmp/evo5_repro", seed=6, seed_from="", door_w=0.0,
        anneal_gen=0, anneal_door_w=0.94)
    evolve(ns)
    print("REPRO-DONE", flush=True)


if __name__ == "__main__":
    main()
