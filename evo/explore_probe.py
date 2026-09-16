#!/usr/bin/env python3
"""Decisive probe: can ANY scan72-only reactive controller explore, or is
internal MEMORY structurally required?

Runs 1-6a verdict: every evolved population AND the scripted baselines
collapse to ~4 cells (cov 0.063). This probe runs N=128 RANDOMIZED script
variants (per-env behavioural biases + commitment timers, so the 128
"policies" don't share the wall-follow attractor) through the same
holdout arena with the same coverage metric.

  if max cov across variants >> 0.066 -> exploration is representable
      from scan72 alone; the ES failure is search-init -> seed/BC branch
  else -> reactive scan-only policies fundamentally cannot cover ground;
      the genome needs explicit memory (CAMEMBE-style) -> structured branch

    python3 -m evo.explore_probe --device cuda
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from .arena import Arena, NUM_BINS, fitness


def make_scripted(rng: np.random.Generator, B: int, mode: str, device: str):
    """Stateful scripted policy factory: per-env persistent personality.
    Returns policy_step(obs, prev_act) closing over device-side state."""
    dev = device
    # per-env personality (drawn once; the whole point is DIVERSITY)
    bias = torch.as_tensor(rng.uniform(-1, 1, B), device=dev)     # turn bias
    commit_T = torch.as_tensor(
        rng.integers(15, 150, B).astype(np.float32), device=dev)  # tick len
    timer = torch.zeros(B, device=dev)
    mode_i = torch.as_tensor(rng.integers(0, 3, B), device=dev) if mode == "mix" \
        else torch.full((B,), {"fwd": 0, "wall": 1, "random": 2}[mode],
                        device=dev)
    # committed direction per env (skid-steer pair), resampled on timer fire
    cmdL = torch.as_tensor(rng.uniform(-1, 1, B), device=dev)
    cmdR = torch.as_tensor(rng.uniform(-1, 1, B), device=dev)
    hold = torch.stack([cmdL, cmdR], dim=1).clone()

    def step(obs, prev):
        nonlocal timer
        s = obs[:, :NUM_BINS]
        front = s[:, :8].min(dim=1).values
        left = s[:, 9:36].mean(dim=1)
        right = s[:, 36:63].mean(dim=1)
        timer = timer + 1
        fire = timer >= commit_T
        # resample committed commands where timer fired
        newL = torch.as_tensor(rng.uniform(-1, 1, B), device=dev)
        newR = torch.as_tensor(rng.uniform(-1, 1, B), device=dev)
        # modes with steering: forward-biased + openness/bias steering
        open_diff = (left - right).clamp(-1, 1)
        steer = 0.5 * open_diff + 0.5 * bias
        f = (0.4 + 0.6 * front.clamp(0, 1))
        m0 = torch.stack([f - steer, f + steer], dim=1)            # cruise
        m1 = torch.stack([torch.full((B,), 0.15, device=dev),
                          torch.full((B,), 1.0, device=dev)], dim=1)  # arc-right
        m1b = torch.stack([torch.full((B,), 1.0, device=dev),
                           torch.full((B,), 0.15, device=dev)], dim=1)  # arc-left
        m1s = torch.where((bias > 0).unsqueeze(1), m1, m1b)
        m2 = torch.stack([newL, newR], dim=1)                      # random
        pick = torch.where(mode_i.unsqueeze(1) == 0, m0,
                           torch.where(mode_i.unsqueeze(1) == 1, m1s, m2))
        holdNew = torch.where(fire.unsqueeze(1), pick, hold)
        # wedged override: front closed AND both sides closed -> back up
        wedged = (front < 0.06) & (torch.maximum(left, right) < 0.12)
        back = torch.stack([torch.full((B,), -0.5, device=dev),
                            torch.full((B,), 0.5, device=dev)], dim=1)
        back2 = torch.stack([torch.full((B,), 0.5, device=dev),
                             torch.full((B,), -0.5, device=dev)], dim=1)
        holdNew = torch.where(wedged.unsqueeze(1),
                              torch.where((bias > 0).unsqueeze(1), back, back2),
                              holdNew)
        hold.copy_(holdNew)
        timer = torch.where(fire, torch.zeros_like(timer), timer)
        return hold.clamp(-1, 1)

    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--variants", type=int, default=128)
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--ticks", type=int, default=5400)
    ap.add_argument("--holdout-seed", type=int, default=777_000)
    ap.add_argument("--fp16", action="store_true", default=True)
    args = ap.parse_args()

    P, G = args.variants, args.games
    arena = Arena(P, G, seed=args.holdout_seed, device=args.device,
                  fp16=args.fp16)
    rng = np.random.default_rng(1)
    rows = []
    for mode in ("mix", "wall", "fwd", "random"):
        # rebuild fresh personalities per mode
        torch.manual_seed(5)
        step = make_scripted(rng, P * G, mode, args.device)
        m = arena.run_games(step, args.ticks)
        cov = float((m["cells"] / m["cells_total"]).mean())
        # per-variant mean coverage: the BEST single scripted personality
        vcov = (m["cells"] / m["cells_total"]).view(P, G).mean(1)
        fit = float((m["rooms"] / m["rooms_total"]
                     + 0.005 * m["dist_m"] / 10.0
                     - 0.25 * m["collisions"] / 100.0).mean())
        rows.append((mode, cov, float(vcov.max()), float(m["rooms"].mean()),
                     float(vcov.argmax()), fit))
        print(f"{mode:7s}  mean_cov={cov:.3f}  best-variant_cov="
              f"{float(vcov.max()):.3f}  mean_rooms={float(m['rooms'].mean()):.2f}  "
              f"fit={fit:.4f}", flush=True)
    best = max(r[2] for r in rows)
    print(f"\nPROBE: best single scripted personality coverage = {best:.3f}")
    print("  vs evolved-population 0.066, baselines ~0.07")
    print("VERDICT:", "REPRESENTABLE-from-scan (fix = search init/seed)"
          if best > 0.15 else
          "scan-only reactive CANNOT explore (fix = structured memory genome)")


if __name__ == "__main__":
    main()
