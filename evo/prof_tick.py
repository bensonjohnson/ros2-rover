#!/usr/bin/env python3
"""Per-stage tick profile of the production rollout (run on Spark, GPU idle).

Where does a tick go once the raycast is fused? Times each stage of
Arena._rollout_body with synchronize() fences (eager, so absolute ms are an
upper bound on the graph path; the SPLIT is what matters), for each
config in --configs (fp16 | fused | fused+compile).

    python3 -m evo.prof_tick --device cuda --hidden 128
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np
import torch

from .arena import Arena, OBS_DIM, NUM_BINS, MAX_RANGE, _proprio
from .policy import PopulationNet, sample_population


def profile(arena: Arena, net: PopulationNet, h: torch.Tensor, ticks: int):
    e, g = arena.env, arena.gate
    arena.reset()
    prev = torch.zeros(arena.B, 2, device=arena.device)
    dist = torch.zeros(arena.B, device=arena.device)
    cells = torch.zeros(arena._ncell_words, arena.B, dtype=torch.int64,
                        device=arena.device)
    P, G = net.P, arena.B // net.P
    acc = defaultdict(float)
    sync = (torch.cuda.synchronize if arena.device.type == "cuda"
            else (lambda: None))

    def lap(name, t0):
        sync()
        t1 = time.perf_counter()
        acc[name] += t1 - t0
        return t1

    sync()
    t = time.perf_counter()
    for k in range(ticks):
        g._tick.add_(arena.dt)
        if getattr(e, "fused", False):
            from .fused_scan import fused_scan
            r = fused_scan(e, out=e._scan_buf)
        else:
            # raycast only: reuse the class scan minus noise by timing whole
            r = type(e).scan(e)
        t = lap("raycast(+noise if unfused)", t)
        if getattr(e, "fused", False):
            r = r + e.noise(e.cfg.lidar_noise_std, e.B, e.cfg.n_beams)
            r = e._dropout(r, e.cfg.lidar_dropout_p)
            t = lap("noise+dropout", t)
        scan72 = arena._preprocess(r, e.angle_min, e.angle_increment,
                                   num_bins=NUM_BINS, max_range=MAX_RANGE)
        t = lap("preprocess", t)
        g.process_scan(r, e.angle_min, e.angle_increment)
        t = lap("gate.process_scan", t)
        obs = torch.cat([scan72, _proprio(e), prev], dim=1)
        t = lap("proprio+obs", t)
        a, hn = net.step(obs.view(P, G, OBS_DIM), h)
        h.copy_(hn)
        cmd = a.view(arena.B, 2).clamp(-1, 1)
        t = lap("policy", t)
        gated = g.gate(cmd)
        t = lap("gate.gate", t)
        px, py = e.x.clone(), e.y.clone()
        e.step(gated, arena.dt)
        dist += torch.hypot(e.x - px, e.y - py)
        t = lap("env.step(+clearance)", t)
        if k % arena.every_cover == 0:
            arena._cells_or(cells)
            t = lap("cells", t)
        prev = gated
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pop", type=int, default=128)
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--rotations", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--ticks", type=int, default=150)
    ap.add_argument("--configs", default="fp16,fused,fused+compile")
    args = ap.parse_args()
    P, G, K = args.pop, args.games, args.rotations
    rng = np.random.default_rng(3)
    th = torch.as_tensor(sample_population(P, OBS_DIM, args.hidden, rng),
                         device=args.device)
    for cfg in args.configs.split(","):
        ar = Arena(P, G, seed=62000, device=args.device, fp16=True,
                   merged_houses=K, fused="fused" in cfg,
                   compile="compile" in cfg)
        net = PopulationNet(th, OBS_DIM, args.hidden)
        net.bind(P, G * K)
        h = torch.zeros(P, G * K, args.hidden, device=args.device)
        profile(ar, net, h, 30)                  # warm (+ compile)
        acc = profile(ar, net, h, args.ticks)
        tot = sum(acc.values())
        print(f"\n== {cfg}  B={ar.B}  {1e3 * tot / args.ticks:.2f} ms/tick "
              f"({ar.B * args.ticks / tot / 1e3:.0f}k env-ticks/s)")
        for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
            print(f"  {k:28s} {1e3 * v / args.ticks:7.2f} ms  "
                  f"{100 * v / tot:5.1f}%")
        del ar, net, h
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
