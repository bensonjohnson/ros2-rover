#!/usr/bin/env python3
"""Frozen-eval checkpoints in unseen houses — pick the transfer brain by data.

Each checkpoint is loaded with learning OFF and dropped into the same B
fresh houses (an eval seed the training run never saw). Per checkpoint:

    err_early   mean sensory prediction error over the first minute
                (how fast a frozen brain makes sense of a NOVEL house)
    err_late    mean error over the last half of the run (settled quality)
    stops/h     safety-gate front blocks per sim-hour
    coll/h      collisions per sim-hour (gate failures)
    rooms       distinct place-memory fingerprints found per env
    dist_m      meters traveled per env (exploration actually happening)

All checkpoints see identical worlds and identical noise streams, so the
numbers are directly comparable. The suggested pick maximizes exploration
(rooms, dist) at low cost (err_late, coll/h), each metric min-max
normalized across the evaluated checkpoints.

Usage:
    python3 -m pnn_sim.eval_checkpoints sim_out_batched/snapshots/*.pt
    python3 -m pnn_sim.eval_checkpoints --envs 64 --ticks 4500 ckpt1.pt ckpt2.pt
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_REPO, os.path.join(_REPO, "src", "tractor_bringup")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch


def make_eval_args(envs=64, ticks=4500, early_ticks=900,
                   eval_seed=777_000, device="cuda",
                   work_dir="/tmp/pnn_eval"):
    """Namespace for eval_checkpoint() — lets train_batched reuse the eval."""
    return argparse.Namespace(envs=envs, ticks=ticks, early_ticks=early_ticks,
                              eval_seed=eval_seed, device=device,
                              work_dir=work_dir)


def eval_checkpoint(path: str, args) -> dict:
    from pnn_sim.batched.trainer import BatchedTrainer, BatchedTrainConfig

    slow_path = path.replace("pnn_brain_", "pnn_brain_slow_") \
        if "pnn_brain_" in os.path.basename(path) else ""

    trainer = BatchedTrainer(BatchedTrainConfig(
        envs=args.envs, device=args.device, learn=False,
        switch_world_every=0,             # one unseen house per env
        seed=args.eval_seed,              # same worlds for every checkpoint
        load_path=path,
        out_dir=os.path.join(args.work_dir, "eval_tmp"),
        log_envs=0, save_interval_s=1e12, snapshot_every=0,
    ))
    # The trainer auto-loads <out_dir>/pnn_brain_slow.pt; for snapshots the
    # slow weights live next to the fast ones under the step-tagged name.
    if trainer.slow is not None and slow_path and os.path.exists(slow_path):
        ok, reason = trainer.slow.load(slow_path)
        if not ok:
            print(f"  (slow layer for {os.path.basename(path)}: {reason})")

    torch.manual_seed(args.eval_seed)
    B = args.envs
    early_ticks = min(args.early_ticks, args.ticks)
    err_sum_early = np.zeros(B)
    err_sum_late = np.zeros(B)
    late_from = args.ticks // 2
    dist = np.zeros(B)

    for t in range(args.ticks):
        px = trainer.env.x.cpu().numpy().copy()
        py = trainer.env.y.cpu().numpy().copy()
        trainer.tick()
        dist += np.hypot(trainer.env.x.cpu().numpy() - px,
                         trainer.env.y.cpu().numpy() - py)
        err = trainer.last_obs_err.cpu().numpy()
        if t < early_ticks:
            err_sum_early += err
        if t >= late_from:
            err_sum_late += err

    sim_hours = args.ticks / trainer.cfg.control_rate_hz / 3600.0
    return {
        "ckpt": path,
        "err_early": float(err_sum_early.mean()) / early_ticks,
        "err_late": float(err_sum_late.mean()) / (args.ticks - late_from),
        "stops_h": trainer.gate.stops / B / sim_hours,
        "coll_h": trainer._collisions / B / sim_hours,
        "rooms": float(trainer.place.n_places().float().mean()),
        "dist_m": float(dist.mean()),
    }


def suggest(results: list[dict]) -> int:
    """Composite: explore a lot, predict well, don't hit things.

    A metric only votes if its spread across checkpoints is meaningful
    (>5% of its mean magnitude) — min-max normalization would otherwise
    blow measurement noise up to a full point. Ties go to the lower
    settled error.
    """
    def norm(key, invert=False):
        vals = np.array([r[key] for r in results], dtype=float)
        span = vals.max() - vals.min()
        if span <= 0.05 * max(np.abs(vals).mean(), 1e-12):
            return np.zeros_like(vals)          # uninformative — abstain
        n = (vals - vals.min()) / span
        return 1.0 - n if invert else n

    score = (norm("rooms") + norm("dist_m")
             + norm("err_late", invert=True) + norm("coll_h", invert=True))
    for r, s in zip(results, score):
        r["score"] = float(s)
    err = np.array([r["err_late"] for r in results])
    order = np.lexsort((err, -score))           # max score, then min err
    return int(order[0])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+",
                    help="pnn_brain*.pt files (snapshots dir glob works)")
    ap.add_argument("--envs", type=int, default=64,
                    help="unseen houses per checkpoint")
    ap.add_argument("--ticks", type=int, default=4500,
                    help="eval length per house (4500 = 5 sim-min)")
    ap.add_argument("--early-ticks", type=int, default=900,
                    help="window for err_early (900 = 1 sim-min)")
    ap.add_argument("--eval-seed", type=int, default=777_000,
                    help="world seed; keep it far from training seeds")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--work-dir", default="/tmp/pnn_eval")
    args = ap.parse_args()

    ckpts = sorted(c for c in args.checkpoints
                   if "pnn_brain_slow" not in os.path.basename(c))
    if not ckpts:
        raise SystemExit("no fast-brain checkpoints given")

    results, best = run_eval(ckpts, args)


def run_eval(ckpts: list[str], args) -> tuple[list[dict], int]:
    """Evaluate checkpoints, print the table, return (results, best_idx)."""
    results = []
    for c in ckpts:
        print(f"evaluating {c} ...", flush=True)
        results.append(eval_checkpoint(c, args))

    best = suggest(results)
    hdr = (f"{'checkpoint':<42} {'err_early':>9} {'err_late':>8} "
           f"{'stops/h':>8} {'coll/h':>7} {'rooms':>6} {'dist_m':>7} {'score':>6}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for i, r in enumerate(results):
        mark = "  <-- suggested" if i == best else ""
        print(f"{os.path.basename(r['ckpt']):<42} {r['err_early']:>9.3f} "
              f"{r['err_late']:>8.3f} {r['stops_h']:>8.1f} {r['coll_h']:>7.1f} "
              f"{r['rooms']:>6.1f} {r['dist_m']:>7.1f} {r['score']:>6.2f}{mark}")
    print(f"\nsuggested transfer brain: {results[best]['ckpt']}")
    print("(score = normalized rooms + dist + low err_late + low coll/h; "
          "sanity-check the raw columns before trusting it)")
    return results, best


if __name__ == "__main__":
    main()
