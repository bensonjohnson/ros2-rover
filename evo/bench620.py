#!/usr/bin/env python3
"""V620 (gfx1030 / ROCm) vs GB10 benchmark for the evo fast path.

Measures the two things that decide whether the V620 is worth training on:
  1. eager rollout wall-clock (bandwidth-bound raycast: V620 GDDR6 ~3x the
     GB10's shared-LPDDR bandwidth on paper)
  2. CUDA/HIP-graph capture + replay (our whole 2x fast path assumes this
     works; ROCm's graph stack is the big unknown per CONTINUOUS_MODE.md)

Run: python3 -m evo.bench620 [--device cuda] [--ticks 150]
Compare the printed numbers against evo/prof5.py output on the Spark.
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
    ap.add_argument("--ticks", type=int, default=150)
    args = ap.parse_args()

    dev = args.device
    P, G, K, hidden = args.pop, args.games, args.rotations, args.hidden
    B = P * G * K
    print(f"device={dev}  B={B}  ticks={args.ticks}")
    if dev.startswith("cuda") and torch.version.hip:
        print(f"  torch {torch.__version__} (ROCm {torch.version.hip})")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

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

    # ---- eager (correctness anchor: fitness numbers must match Spark) ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    m = tr.run_games(policy_step, args.ticks)
    torch.cuda.synchronize()
    t_eager = time.perf_counter() - t0
    env_ticks_per_s = B * args.ticks / t_eager
    print(f"eager: {t_eager:7.2f}s   {env_ticks_per_s:,.0f} env-ticks/s"
          f"   fit_best~{float(m['cells'].mean()):.2f} cells")

    # ---- graph capture (the ROCm unknown) --------------------------------
    try:
        t0 = time.perf_counter()
        m1 = tr.run_games(policy_step, args.ticks, graph=True)
        torch.cuda.synchronize()
        t_cap = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(3):
            tr.run_games(policy_step, args.ticks, graph=True)
        torch.cuda.synchronize()
        t_replay = (time.perf_counter() - t0) / 3
        ok = torch.allclose(m1["dist_m"], m["dist_m"], rtol=0.1, atol=1.0)
        print(f"graph: capture {t_cap:.2f}s  replay {t_replay:.2f}s"
              f"   ({t_eager / t_replay:.2f}x vs eager)  "
              f"dist-parity {'OK' if ok else 'DRIFT>10%'}")
        print(f"       {B * args.ticks / t_replay:,.0f} env-ticks/s replayed")
    except Exception as ex:
        print(f"graph: FAILED -> {type(ex).__name__}: {ex}")
        tr._graph_broken = True

    print("BENCH-DONE")


if __name__ == "__main__":
    main()
