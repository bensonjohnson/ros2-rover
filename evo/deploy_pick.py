#!/usr/bin/env python3
"""Robust deploy selection: score EVERY genome of a final population on the
level-3 building holdout across the track-trim envelope, rank by worst case.

The real rover's effective track factors are unknown (live run 2 turned the
opposite way to the sim's trim prediction), so a deployable genome must
survive the whole trim envelope — not just the sim's nominal 0.8/1.0 that
the validation champion is picked on.

    python3 -m evo.deploy_pick --population evo_runs/X/final_population.npz

Writes deploy_candidates.json (full table) and deploy_genome.npz (top-1)
next to the population. Selection: worst-trim door-cross rate primary (the
rover task is entering rooms), worst-trim collisions <= --coll-guard hard
guard, nominal (0.8/1.0) fitness as tiebreak. Reverse fraction is reported
but NOT selected on (live run 2 showed reverse driving works on hardware).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .arena import Arena, OBS_DIM, fitness, rev_frac
from .policy import PopulationNet, read_meta
from .worlds import HOLDOUT_SEED, build_pool, summarize

TRIMS = [(0.8, 1.0), (1.0, 1.0), (1.0, 0.8), (0.9, 1.0), (1.0, 0.9)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--population", required=True)
    ap.add_argument("--games", type=int, default=32)
    ap.add_argument("--ticks", type=int, default=5400)
    ap.add_argument("--w-dist", type=float, default=0.005)
    ap.add_argument("--w-coll", type=float, default=1.0)
    ap.add_argument("--w-cov", type=float, default=0.3)
    ap.add_argument("--coll-guard", type=float, default=20.0)
    ap.add_argument("--reset-draws", type=int, default=3,
                    help="independent initial-pose draws per trim; a "
                    "single draw UNDERESTIMATES collision risk (run 19: "
                    "idx109 passed at 11.8 on one draw, 62-95 colls on "
                    "others) — guard and ranking take worst across draws")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    d = np.load(args.population)
    obs, action = read_meta(d)
    hidden = int(d["hidden"])
    thetas = torch.as_tensor(d["thetas"], device=args.device)
    P = int(thetas.shape[0])
    G = args.games
    worlds = build_pool(3, G, HOLDOUT_SEED)
    fused = args.device.startswith("cuda")
    print(f"deploy_pick: {args.population}  P={P} hidden={hidden} "
          f"obs={obs} action={action}  {summarize(worlds)}", flush=True)

    arena = Arena(P, G, seed=777_000, device=args.device, fp16=fused,
                  fused=fused, gate_obs=obs == "v2", worlds=worlds,
                  trim_rand=True)
    net = PopulationNet(thetas, OBS_DIM, hidden, obs=obs, action=action)
    net.bind(P, G)

    per_trim = []
    for lt, rt in TRIMS:
        arena.set_trims([lt] * G, [rt] * G)
        draws = []
        for k in range(args.reset_draws):
            state = {"h": torch.zeros(P, G, hidden, device=args.device)}

            def step(o, prev, _s=state):
                a, _s["h"] = net.step(o.view(P, G, OBS_DIM), _s["h"])
                return a.view(P * G, 2)

            torch.manual_seed(3 + 1000 * k)
            m = arena.run_games(step, args.ticks)
            fit = fitness(m, w_dist=args.w_dist, w_coll=args.w_coll,
                          w_cov=args.w_cov).view(P, G).mean(1)
            rooms = m["rooms"].view(P, G)
            draws.append({
                "fit": fit.cpu().numpy(),
                "cross": (rooms >= 2).float().mean(1).cpu().numpy(),
                "rooms": rooms.mean(1).cpu().numpy(),
                "coll": m["collisions"].view(P, G).mean(1).cpu().numpy(),
                "rev": rev_frac(m).view(P, G).mean(1).cpu().numpy(),
            })
        per_trim.append({
            "trim": [lt, rt],
            # worst across reset draws (collision risk hides on single draws)
            "coll": np.maximum.reduce([d["coll"] for d in draws]),
            "fit": np.mean([d["fit"] for d in draws], axis=0),
            "cross": np.mean([d["cross"] for d in draws], axis=0),
            "rooms": np.mean([d["rooms"] for d in draws], axis=0),
            "rev": np.mean([d["rev"] for d in draws], axis=0),
        })
        print(f"  trim {lt:.1f}/{rt:.1f}: mean cross "
              f"{per_trim[-1]['cross'].mean():.3f}  worst-draw mean coll "
              f"{per_trim[-1]['coll'].mean():.1f}", flush=True)

    cross = np.stack([t["cross"] for t in per_trim])      # [T, P]
    coll = np.stack([t["coll"] for t in per_trim])
    rooms = np.stack([t["rooms"] for t in per_trim])
    fitn = per_trim[0]["fit"]                               # nominal 0.8/1.0
    worst_cross = cross.min(0)
    worst_coll = coll.max(0)
    worst_rooms = rooms.min(0)
    ok = worst_coll <= args.coll_guard

    order = np.lexsort((-fitn, -worst_cross))               # cross, then fit
    ranked = [int(i) for i in order if ok[int(i)]]
    print(f"\nPASS guard (worst-trim coll <= {args.coll_guard:g}): "
          f"{len(ranked)}/{P} genomes")
    hdr = (f"{'rank':>4} {'idx':>4} {'wCross':>6} {'wRooms':>6} "
           f"{'wColl':>6} {'nomFit':>6} {'nomRev':>6}")
    print(hdr)
    for r, i in enumerate(ranked[:args.top]):
        print(f"{r:>4} {i:>4} {worst_cross[i]:6.3f} {worst_rooms[i]:6.2f} "
              f"{worst_coll[i]:6.1f} {fitn[i]:6.3f} "
              f"{per_trim[0]['rev'][i]:6.2f}")

    out_dir = os.path.dirname(os.path.abspath(args.population))
    table = [{"idx": int(i), "worst_cross": float(worst_cross[i]),
              "worst_rooms": float(worst_rooms[i]),
              "worst_coll": float(worst_coll[i]), "nominal_fit": float(fitn[i]),
              "per_trim": [{"trim": t["trim"], "cross": float(t["cross"][i]),
                            "rooms": float(t["rooms"][i]),
                            "coll": float(t["coll"][i]),
                            "fit": float(t["fit"][i]),
                            "rev": float(t["rev"][i])} for t in per_trim]}
             for i in ranked[:50]]
    with open(os.path.join(out_dir, "deploy_candidates.json"), "w") as f:
        json.dump({"population": os.path.abspath(args.population),
                   "trims": TRIMS, "coll_guard": args.coll_guard,
                   "n_pass": len(ranked), "population_size": P,
                   "candidates": table}, f, indent=1)

    if ranked:
        top = ranked[0]
        np.savez(os.path.join(out_dir, "deploy_genome.npz"),
                 thetas=thetas[top].cpu().numpy(), hidden=hidden,
                 obs_mode=obs, action_mode=action,
                 worst_cross=float(worst_cross[top]),
                 worst_coll=float(worst_coll[top]),
                 nominal_fit=float(fitn[top]),
                 source=os.path.abspath(args.population))
        print(f"\nDEPLOY-PICK top genome idx={top} -> "
              f"{out_dir}/deploy_genome.npz "
              f"(worst cross {worst_cross[top]:.3f}, "
              f"worst coll {worst_coll[top]:.1f}, "
              f"nominal fit {fitn[top]:.3f})")
    else:
        print("\nDEPLOY-PICK: no genome passed the collision guard")


if __name__ == "__main__":
    main()
