#!/usr/bin/env python3
"""Replay a t4_place_probe npz offline: what SHOULD the place memory have
created on the exact fingerprint stream the live run saw?

  kcenter       greedy k-center (frozen refs) on the visited fp stream —
                the ceiling: places the trajectory actually contains.
  sim(blend)    replay the live match/consolidation rule with a different
                slot blend rate; tests the chase hypothesis quantitatively
                (blend 0.02 should reproduce live n_places; blend 0 should
                approach kcenter).

    python -m pnn_sim.tools.t4_place_rules --npz /tmp/pnn_probe.npz
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def kcenter(F: np.ndarray, thresh: float, cap: int = 64) -> int:
    cent = [0]
    while len(cent) < cap:
        d = np.linalg.norm(F[:, None, :] - F[None, cent, :], axis=2)
        dmin = d.min(axis=1)
        j = int(np.argmax(dmin))
        if dmin[j] < thresh:
            break
        cent.append(j)
    return len(cent)


def simulate(F: np.ndarray, thresh: float, blend: float,
             fp_ema_tau_s: float, dt: float, P: int = 64) -> int:
    """Replay BatchedPlaceMemory.update for one env on fp stream F [T,F].
    Query EMA is already baked in by the probe; slots blend at `blend`."""
    slots = np.full((P, F.shape[1]), np.nan)
    w = np.zeros(P)
    live = np.zeros(P, dtype=bool)
    for t in range(F.shape[0]):
        w[w <= 0.05] = 0.0                       # prune (tau decay baked: 900s)
        d = np.linalg.norm(slots[live] - F[t], axis=1)
        if live.any() and d.min() < thresh:      # matched -> consolidate
            i = int(np.flatnonzero(live)[np.argmin(d)])
            w[i] += dt
            slots[i] = (1 - blend) * slots[i] + blend * F[t]
        else:                                     # new place: freeze ref
            free = np.flatnonzero(~live)
            i = int(free[0]) if len(free) else int(np.argmin(w))
            slots[i] = F[t]
            w[i] = max(dt, 0.1)
            live[i] = True
    return int(live.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/tmp/pnn_probe.npz")
    ap.add_argument("--warmup-ticks", type=int, default=90)
    args = ap.parse_args()

    z = np.load(args.npz)
    fp, npl, thresh, dt = z["fp"], z["npl"], float(z["thresh"]), float(z["dt"])
    T, B, _ = fp.shape
    rows = []
    for b in range(B):
        Fb = fp[args.warmup_ticks:, b].astype(np.float64)
        rows.append({
            "env": b,
            "live_final": int(npl[-1, b]),
            "kcenter_ceiling": kcenter(Fb, thresh),
            "sim_blend0.02": simulate(Fb, thresh, 0.02, 0.6, dt),
            "sim_blend0.005": simulate(Fb, thresh, 0.005, 0.6, dt),
            "sim_blend0.0": simulate(Fb, thresh, 0.0, 0.6, dt),
        })
    print(json.dumps(rows, indent=2))
    k = max(r["kcenter_ceiling"] for r in rows)
    s0 = max(r["sim_blend0.0"] for r in rows)
    print(f"\nreplication check: sim@0.02 vs live_final per env: "
          f"{all(r['sim_blend0.02'] == r['live_final'] for r in rows)}")
    print(f"best env: ceiling {k} places, blend-0 replay reaches {s0}")


if __name__ == "__main__":
    main()
