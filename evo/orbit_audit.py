"""Orbit audit: does a genome genuinely TRANSLATE in the sim, or does it
accumulate arc length while turning in place?

Motivation (2026-09-19 live run 8): on the real rover champ21f_idx103 spent
58% of ticks with opposite-sign wheels at gyro 1.68 rad/s and net linear
speed ~0.07 m/s — a zero-turn orbit (radius ~4 cm) — while every
command-sign metric called it "100% forward". The sim fitness's dist term
counts PATH ARC (`dist` accumulates |step|), which an orbit earns just as
well as travel. This script separates the two:

    arc   = dist_m as scored (path length)
    net   = straight-line distance from start pose to final pose
    ratio = arc / net   (>>1 = orbiting; ~1-3 = honest wandering)
    wander = furthest distance ever reached from start (needs per-tick
             tracking -> recorded via env position sampling)

Usage:
    python3 -m evo.orbit_audit --genome evo_runs/greenfield21f_s8/deploy_genome.npz
    python3 -m evo.orbit_audit --population evo_runs/greenfield21f_s8/final_population.npz
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from .arena import Arena, OBS_DIM, fitness
from .policy import PopulationNet, read_meta


def _run(path: str, arena: Arena, ticks: int) -> dict:
    d = np.load(path)
    obs, action = read_meta(d)
    thetas = torch.as_tensor(d["thetas"], device=arena.device)
    if thetas.ndim == 1:
        thetas = thetas.unsqueeze(0)
    hidden = int(d["hidden"])
    P = thetas.shape[0]
    G = arena.B // P
    net = PopulationNet(thetas, OBS_DIM, hidden, obs=obs, action=action)
    net.bind(P, G)
    state = {"h": torch.zeros(P, G, hidden, device=arena.device)}

    env = arena.env
    arena.reset()                      # put env at start poses...
    x0, y0 = env.x.clone(), env.y.clone()

    def step(o, prev):
        a, state["h"] = net.step(o.view(P, G, OBS_DIM), state["h"])
        return a.view(P * G, 2)

    m = arena.run_games(step, ticks)   # ...run_games resets identically
    fin = torch.hypot(env.x - x0, env.y - y0)
    P_, G_ = P, G
    row = {
        "arc_m": float(m["dist_m"].mean()),
        "net_m": float(fin.mean()),
        "arc/net": float((m["dist_m"] / fin.clamp(min=0.05)).mean()),
        "fwd_m": float(m["fwd_m"].mean()),
        "back_m": float(m["back_m"].mean()),
        "rooms": float(m["rooms"].mean()),
    }
    if "door_crossings" in m:
        row["cross"] = float(m["door_crossings"].mean())
    # per-genome net (population files): reshape B -> (P, G)
    row["_fin2"] = fin.view(P_, G_).cpu().numpy()
    row["_arc2"] = m["dist_m"].view(P_, G_).cpu().numpy()
    row["_rev2"] = (m["back_m"] / (m["fwd_m"] + m["back_m"]).clamp(min=1e-6)
                   ).view(P_, G_).cpu().numpy()
    if "door_crossings" in m:
        row["_x2"] = m["door_crossings"].view(P_, G_).cpu().numpy()
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=
                                 argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genome", default="")
    ap.add_argument("--population", default="")
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--ticks", type=int, default=5400)
    ap.add_argument("--holdout-seed", type=int, default=777_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--level", type=int, default=3)
    args = ap.parse_args()

    from .worlds import HOLDOUT_SEED, build_pool, summarize
    hw = build_pool(args.level, args.games, HOLDOUT_SEED)
    print(f"building holdout L{args.level}: {summarize(hw)}")

    which = args.genome or args.population
    if not which:
        raise SystemExit("give --genome or --population")
    d = np.load(which)
    obs, _ = read_meta(d)
    P = 1 if d["thetas"].ndim == 1 else d["thetas"].shape[0]
    arena = Arena(P, args.games, seed=args.holdout_seed, device=args.device,
                  fp16=args.fp16, gate_obs=(obs == "v2"), worlds=hw)
    r = _run(which, arena, args.ticks)
    fin, arc = r.pop("_fin2"), r.pop("_arc2")
    rev, x = r.pop("_rev2", None), r.pop("_x2", None)
    print("  ".join(f"{k}={v:.3f}" for k, v in r.items()))
    if P > 1:
        ratio = arc / np.clip(fin, 0.05, None)
        print(f"per-genome arc/net: median {np.median(ratio):.1f}  "
              f"p10 {np.percentile(ratio,10):.1f}  p90 {np.percentile(ratio,90):.1f}")
        print(f"per-genome net: median {np.median(fin):.2f} m  "
              f"max {fin.max(1).mean():.2f} m (best genome mean over games)")
        bad = (ratio > 8).mean()
        print(f"genomes orbiting (arc/net > 8): {bad:.0%}")
        if x is not None:
            gm = fin.mean(1)          # per-genome mean net displacement
            gx = x.mean(1)            # per-genome mean door crossings
            gr = rev.mean(1)          # per-genome reverse fraction
            print("\nTOP honest explorers (net displacement, not arc):")
            print(f"{'idx':>4} {'net_m':>6} {'arc_m':>6} {'a/n':>5} "
                  f"{'cross':>6} {'rev':>5}")
            order = np.argsort(-gm)
            for i in order[:12]:
                print(f"{i:>4} {gm[i]:>6.2f} {arc.mean(1)[i]:>6.1f} "
                      f"{ratio.mean(1)[i]:>5.1f} {gx[i]:>6.2f} {gr[i]:>5.2f}")
            print("\nTOP by door crossings among non-orbiters (net>=2m, rev<0.15):")
            ok = (gm >= 2.0) & (gr < 0.15)
            idxs = np.where(ok)[0]
            order2 = idxs[np.argsort(-gx[idxs])] if len(idxs) else []
            for i in order2[:12]:
                print(f"{i:>4} {gm[i]:>6.2f} {arc.mean(1)[i]:>6.1f} "
                      f"{ratio.mean(1)[i]:>5.1f} {gx[i]:>6.2f} {gr[i]:>5.2f}")


if __name__ == "__main__":
    main()
