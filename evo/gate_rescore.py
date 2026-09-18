#!/usr/bin/env python3
"""Re-score champions on the level-3 building holdout under different
safety-gate presets (arena.GATE_PRESETS), nominal trims, full games.

    python3 -m evo.gate_rescore --gates legacy,doorway GENOME.npz ...
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from .arena import Arena, OBS_DIM, fitness, gate_config, rev_frac
from .policy import PopulationNet, read_meta
from .worlds import HOLDOUT_SEED, build_pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("genomes", nargs="+")
    ap.add_argument("--gates", default="legacy,doorway")
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    worlds = build_pool(3, args.games, HOLDOUT_SEED)
    fused = args.device.startswith("cuda")
    print(f"{'genome':44s} {'gate':8s} {'fit':>6} {'rooms':>5} {'cross':>5} "
          f"{'coll':>6} {'rev':>4} {'fstops':>6}")
    for path in args.genomes:
        d = np.load(path)
        obs, action = read_meta(d)
        H = int(d["hidden"])
        th = torch.as_tensor(d["thetas"], device=args.device).view(1, -1)
        for gname in args.gates.split(","):
            ar = Arena(1, args.games, seed=1, device=args.device, fp16=fused,
                       fused=fused, worlds=worlds, gate_obs=obs == "v2",
                       gate_cfg=gate_config(gname),
                       gate_symmetric=gname == "symmetric")
            net = PopulationNet(th, OBS_DIM, H, obs=obs, action=action)
            net.bind(1, args.games)
            st = {"h": torch.zeros(1, args.games, H, device=args.device)}

            def step(o, prev):
                a, st["h"] = net.step(o.view(1, args.games, OBS_DIM), st["h"])
                return a.view(args.games, 2)
            torch.manual_seed(3)
            m = ar.run_games(step, 5400)
            short = "/".join(path.split("/")[-2:])
            print(f"{short:44s} {gname:8s} {float(fitness(m).mean()):6.3f} "
                  f"{float(m['rooms'].mean()):5.2f} "
                  f"{float((m['rooms'] >= 2).float().mean()):5.3f} "
                  f"{float(m['collisions'].mean()):6.1f} "
                  f"{float(rev_frac(m).mean()):4.2f} "
                  f"{float(m['stops'].mean()):6.1f}", flush=True)


if __name__ == "__main__":
    main()
