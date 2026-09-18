#!/usr/bin/env python3
"""Merge guard-passing genomes from several final_population.npz files into
one founding pool for --seed-from.

Use case: one population is strong but its champions are collision
brute-forcers, another is trim/collision-honest but weaker. Warm-starting ES
from the UNION of their deploy_pick-verified genomes tests whether the two
properties co-inherit — far cheaper than a fresh 120-gen run that may find
neither.

    python3 -m evo.merge_pools --out merged.npz \
        evo_runs/A/deploy_candidates.json evo_runs/A/final_population.npz 64 \
        evo_runs/B/deploy_candidates.json evo_runs/B/final_population.npz 64

Pairs are (candidates.json, final_population.npz, take). Rank inside each
source by worst_cross (deploy_pick's own order), take the top N, and pad by
cycling if a source offers fewer. Writes thetas [P, D] + sigma [P] so
--seed-from --keep-seed-sigma works unchanged.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("sources", nargs="+",
                    help="triples: candidates.json population.npz take")
    args = ap.parse_args()
    if len(args.sources) % 3:
        raise SystemExit("sources must be triples of candidates.json "
                         "population.npz take")

    rows, sigmas, origin, meta = [], [], [], None
    for i in range(0, len(args.sources), 3):
        cand_path, pop_path, take = args.sources[i:i + 3]
        take = int(take)
        d = np.load(pop_path)
        thetas, sigma = d["thetas"], (d["sigma"] if "sigma" in d else None)
        with open(cand_path) as f:
            cand = json.load(f)
        order = [c["idx"] for c in cand["candidates"]
                 if c["worst_coll"] <= cand.get("coll_guard", 20.0)]
        if not order:
            order = [c["idx"] for c in cand["candidates"]]     # trust ranking
        pick = [order[k % len(order)] for k in range(take)]
        print(f"{pop_path}: {len(order)} pass guard, taking {take} "
              f"(top: {pick[:6]}...)")
        for j in pick:
            rows.append(thetas[j])
            if sigma is not None:
                sigmas.append(sigma[j])
            origin.append(os.path.basename(os.path.dirname(pop_path)))
        m = {k: str(d[k]) for k in ("hidden", "obs_mode", "action_mode")
             if k in d.files}
        if meta is None:
            meta = m
        elif meta != m:
            raise SystemExit(f"{pop_path}: genome modes {m} != {meta}")

    if not rows:
        raise SystemExit("no genomes collected from sources")
    th = np.stack(rows).astype(np.float32)
    out = {"thetas": th, "hidden": int(meta["hidden"]),
           "obs_mode": meta.get("obs_mode", "v1"),
           "action_mode": meta.get("action_mode", "lr"),
           "merged_from": "+".join(sorted(set(origin)))}
    if sigmas:
        out["sigma"] = np.asarray(sigmas, dtype=np.float32)
    np.savez(args.out, **out)
    print(f"merged pool {th.shape} -> {args.out} "
          f"(per source: { {o: origin.count(o) for o in sorted(set(origin))} })")


if __name__ == "__main__":
    main()
