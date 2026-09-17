#!/usr/bin/env python3
"""V620 stability soak: N eager production-size rollouts (the mode the
graph-capture segfault forces us into), watching for the ROCm 'hang'
failure class from CONTINUOUS_MODE.md. Prints per-game times; a hang or
slowdown cliff (MIOpen-style autotune stalls) shows up immediately.

  python3 -m evo.soak620 --games 60 --ticks 5400
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from evo.arena import Arena, OBS_DIM
from evo.policy import PopulationNet, sample_population


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pop", type=int, default=128)
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--rotations", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--ticks", type=int, default=5400)
    ap.add_argument("--iters", type=int, default=60)
    args = ap.parse_args()

    dev = args.device
    P, G, K, hidden = args.pop, args.games, args.rotations, args.hidden
    B = P * G * K
    rng = np.random.default_rng(3)
    thetas = torch.as_tensor(sample_population(P, OBS_DIM, hidden, rng),
                             device=dev)
    tr = Arena(P, G, seed=62000, device=dev, fp16=True, merged_houses=K)
    net = PopulationNet(thetas, OBS_DIM, hidden)
    net.bind(P, G * K)
    h = torch.zeros(P, G * K, hidden, device=dev)
    tr._reset_hooks.append(h.zero_)

    def policy_step(obs, prev_act):
        a, hn = net.step(obs.view(P, G * K, OBS_DIM), h)
        h.copy_(hn)
        return a.view(B, 2)

    ts = []
    for it in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        m = tr.run_games(policy_step, args.ticks)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        ts.append(dt)
        print(f"iter {it:3d}  {dt:7.1f}s  ({B * args.ticks / dt:,.0f} "
              f"env-ticks/s)  cells={float(m['cells'].mean()):.2f} "
              f"rooms={float(m['rooms'].mean()):.2f}", flush=True)
    ts = np.array(ts)
    print(f"\nSOAK n={len(ts)}  min/med/max = "
          f"{ts.min():.1f}/{np.median(ts):.1f}/{ts.max():.1f}s  "
          f"spread={ts.max() - ts.min():.1f}s  "
          f"=> est per-gen {np.median(ts):.0f}s")
    print("SOAK-DONE", flush=True)


if __name__ == "__main__":
    main()
