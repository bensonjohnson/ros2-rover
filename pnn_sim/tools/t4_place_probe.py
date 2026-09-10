#!/usr/bin/env python3
"""Instrumented T4 run: log the per-tick fingerprint stream that
BatchedPlaceMemory ACTUALLY matches against (post fp-EMA), plus scan-min,
pre-update dmin and live-place count.

The scalar ablations said places=1 forever despite discriminative static
fingerprints (probe A: 27-31 distinct under rotation blur). Hypothesis:
the matched-slot blend (0.98/tick) + fp EMA make the slot reference CHASE
the agent's slowly drifting fingerprint — dmin equilibrates below
match_thresh on continuous walks, so new places can only be born on JUMPS,
which a 10 Hz rover never makes. Replay pnn_sim/tools/t4_place_rules.py
over the npz to test creation rules offline on the same stream.

    python -m pnn_sim.tools.t4_place_probe --device cuda --out /tmp/pnn_probe.npz
"""
from __future__ import annotations

import argparse
import json
import sys

sys.path[:0] = ['.', 'src/tractor_bringup']

import numpy as np
import torch

import pnn_sim.batched.trainer as T
from pnn_sim.batched.place import BatchedPlaceMemory
from pnn_sim.batched.trainer import BatchedTrainConfig, BatchedTrainer


class Traced(BatchedPlaceMemory):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rec_fp: list[np.ndarray] = []
        self.rec_smin: list[np.ndarray] = []
        self.rec_dmin: list[np.ndarray] = []
        self.rec_npl: list[np.ndarray] = []

    def update(self, scan: torch.Tensor, dt: float,
               err: torch.Tensor | None = None) -> torch.Tensor:
        # snapshot what update() is about to compare against
        live = self._w > 0.0
        d = (self._fps - self._fp_ema.unsqueeze(1)).norm(dim=2)
        d = torch.where(live, d, torch.full_like(d, float("inf")))
        dmin = d.min(dim=1).values
        npl = live.sum(dim=1)
        nov = super().update(scan, dt, err)
        self.rec_fp.append(self._fp_ema.double().cpu().numpy())
        self.rec_smin.append(scan.min(dim=1).values.double().cpu().numpy())
        self.rec_dmin.append(dmin.double().cpu().numpy())
        self.rec_npl.append(npl.cpu().numpy())
        return nov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--ticks", type=int, default=4500)
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--thresh", type=float, default=0.20)
    ap.add_argument("--shape", type=float, default=2.0)
    ap.add_argument("--frontier-weight", type=float, default=1.0)
    ap.add_argument("--wall-guard", action="store_true")
    ap.add_argument("--babble", type=float, default=0.0)
    ap.add_argument("--out", default="/tmp/pnn_probe.npz")
    args = ap.parse_args()

    T.BatchedPlaceMemory = Traced  # trainer constructs the traced subclass
    cfg = BatchedTrainConfig(
        envs=args.envs, device=args.device, seed=args.seed,
        out_dir="/tmp/pnn_probe_out", switch_world_every=0,
        snapshot_every=0, save_interval_s=10 ** 9, log_envs=0,
        babble_eps0=args.babble,
        frontier_weight=args.frontier_weight,
        place_match_thresh=args.thresh,
        place_shape_weight=args.shape,
        place_wall_guard=args.wall_guard)
    tr = BatchedTrainer(cfg)
    max_range = float(cfg.rover.lidar_max_range)

    poses: list[np.ndarray] = []
    for _ in range(args.ticks):
        tr.tick()
        poses.append(np.stack([tr.env.x.cpu().numpy(),
                               tr.env.y.cpu().numpy(),
                               tr.env.theta.cpu().numpy()], axis=1))

    pl = tr.place
    fp = np.stack(pl.rec_fp)          # [T, B, F] post-EMA fingerprints
    smin = np.stack(pl.rec_smin)      # [T, B]
    dmin = np.stack(pl.rec_dmin)      # [T, B] pre-update dist to nearest live slot
    npl = np.stack(pl.rec_npl)        # [T, B]
    P = np.stack(poses)               # [T, B, 3]
    np.savez_compressed(args.out, fp=fp, smin=smin, dmin=dmin, npl=npl,
                        poses=P, thresh=args.thresh, dt=tr.tick_period,
                        max_range=max_range)

    B = args.envs
    step = np.linalg.norm(np.diff(P[:, :, :2], axis=0), axis=2)
    print(json.dumps({
        "out": args.out, "ticks": args.ticks, "envs": B,
        "dt": tr.tick_period, "max_range": max_range,
        "thresh": args.thresh, "F": fp.shape[2],
        "final_places": [int(v) for v in tr.place.n_places().cpu()],
        "dmin_max_per_env": [round(float(v), 3) for v in dmin.max(axis=0)],
        "dmin_p99_per_env": [round(float(np.percentile(dmin[:, b], 99)), 3)
                             for b in range(B)],
        "fp_step_median": round(float(np.median(
            np.linalg.norm(np.diff(fp, axis=0), axis=2))), 5),
        "path_len_per_env": [round(float(step[:, b].sum()), 1)
                             for b in range(B)],
        "smin_min": round(float(smin.min()), 4),
        "smin_p10": round(float(np.percentile(smin, 10)), 4),
    }, indent=2))
    # equilibrium-gap prediction: chase rate 0.02/tick -> gap ~ speed/0.02
    print(f"fp drift speed median -> predicted dmin equilibrium "
          f"~ {np.median(np.linalg.norm(np.diff(fp, axis=0), axis=2)) / 0.02:.3f} "
          f"vs thresh {args.thresh}")
    tr.close()


if __name__ == "__main__":
    main()
