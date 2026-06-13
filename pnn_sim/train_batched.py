#!/usr/bin/env python3
"""Train ONE brain in B parallel worlds (batched, GPU-friendly).

Unlike train_sim.py --workers (independent brains), this shares a single
set of weights across all envs: every tick the local PC updates are
averaged over the batch — one brain drinking from B experience streams.

Examples:
    # 256 worlds on the GPU
    python3 -m pnn_sim.train_batched --envs 256 --ticks 100000

    # CPU run (small batches are fine there too)
    python3 -m pnn_sim.train_batched --envs 32 --device cpu

    # continue from the rover's brain, hotter batched learning rate
    python3 -m pnn_sim.train_batched --load ~/.ros/pnn_brain.pt --lr-scale 2

Output: <out-dir>/pnn_brain.pt (+ slow layer + experience jsonl of env 0)
— same files, same schema as the rover and the single-stream sim.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_REPO, os.path.join(_REPO, "src", "tractor_bringup")):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--ticks", type=int, default=None,
                    help="batch ticks; sim experience = ticks * envs / 15Hz "
                    "(default 100k, or unlimited when --wall-hours is set)")
    ap.add_argument("--wall-hours", type=float, default=0.0,
                    help="stop after this much wall time (0 = tick-gated "
                    "only); the post-run eval adds a few minutes on top")
    ap.add_argument("--snapshot-every-min", type=float, default=0.0,
                    help="wall minutes between snapshots (0 = tick-based "
                    "--snapshot-every); use with --wall-hours so the eval "
                    "field doesn't depend on hardware speed")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lr-scale", type=float, default=1.0,
                    help="scale on the batch-averaged updates (the gradient "
                    "is ~sqrt(B) less noisy; 2-4 is worth trying)")
    ap.add_argument("--switch-world-every", type=int, default=27_000,
                    help="per-env ticks between house swaps, staggered "
                    "across the batch (0 = never)")
    ap.add_argument("--load", default="",
                    help="existing pnn_brain.pt to continue from")
    ap.add_argument("--snapshot-every", type=int, default=10_000,
                    help="batch ticks between numbered snapshots under "
                    "<out-dir>/snapshots/ for eval_checkpoints (0 = off)")
    ap.add_argument("--out-dir", default="sim_out_batched")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--action-scale", type=float, default=0.6)

    # --- exploration / curiosity knobs (defaults match the rover) ----------
    # These exist to fight the dark-room collapse (a brain that drives
    # prediction error to zero by not moving). See pnn_sim/README.md.
    expl = ap.add_argument_group("exploration drive")
    expl.add_argument("--epi-floor", type=float, default=0.02,
                      help="ensemble-disagreement level at which curiosity is "
                      "at full weight; LOWER (e.g. 0.005) keeps a confident "
                      "brain curious instead of coasting (dark room)")
    expl.add_argument("--forward-bias", type=float, default=0.3,
                      help="actor pragmatic_weight (0=pure curiosity, "
                      "1=pure preference)")
    expl.add_argument("--novelty-pref-weight", type=float, default=2.0,
                      help="appetite for predicted-novel places; RAISE to pull "
                      "harder toward new rooms")
    expl.add_argument("--hold-pref-weight", type=float, default=2.0,
                      help="aversion to tripping the safety gate; LOWER (e.g. "
                      "1.0) lets the brain approach doorways/walls to cross "
                      "into new rooms")
    expl.add_argument("--target-novelty", type=float, default=0.8,
                      help="preferred place-novelty level in [0,1]")
    ap.add_argument("--torch-threads", type=int, default=0,
                    help="torch CPU threads (0 = torch default; set for "
                    "--device cpu runs)")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--log-envs", type=int, default=1,
                    help="how many envs write experience jsonl")
    ap.add_argument("--report-s", type=float, default=5.0)
    ap.add_argument("--no-eval-after", action="store_true",
                    help="skip the automatic frozen-eval + best_ pick")
    ap.add_argument("--eval-envs", type=int, default=64)
    ap.add_argument("--eval-ticks", type=int, default=4500,
                    help="frozen-eval length per house (4500 = 5 sim-min)")
    ap.add_argument("--eval-seed", type=int, default=777_000)
    args = ap.parse_args()

    from pnn_sim.batched.trainer import BatchedTrainer, BatchedTrainConfig

    trainer = BatchedTrainer(BatchedTrainConfig(
        envs=args.envs, device=args.device, lr_scale=args.lr_scale,
        switch_world_every=args.switch_world_every,
        load_path=args.load,
        # Wall-based snapshots replace tick-based ones when requested.
        snapshot_every=(0 if args.snapshot_every_min > 0
                        else args.snapshot_every),
        out_dir=args.out_dir, seed=args.seed,
        action_scale=args.action_scale, torch_threads=args.torch_threads,
        learn=not args.freeze, log_envs=args.log_envs,
        epi_floor=args.epi_floor, forward_bias=args.forward_bias,
        novelty_pref_weight=args.novelty_pref_weight,
        hold_pref_weight=args.hold_pref_weight,
        target_novelty=args.target_novelty,
    ))

    max_ticks = args.ticks if args.ticks is not None \
        else (sys.maxsize if args.wall_hours > 0 else 100_000)
    deadline = time.time() + args.wall_hours * 3600 \
        if args.wall_hours > 0 else None

    t0 = time.time()
    last = t0
    last_snap = t0
    try:
        for _ in range(max_ticks):
            trainer.tick()
            now = time.time()
            if deadline is not None and now >= deadline:
                print(f"wall-time limit reached "
                      f"({args.wall_hours:.2f} h)", flush=True)
                break
            if args.snapshot_every_min > 0 \
                    and now - last_snap >= args.snapshot_every_min * 60:
                last_snap = now
                trainer.snapshot()
            if now - last >= args.report_s:
                last = now
                s = trainer.status()
                tps = s["step"] / max(now - t0, 1e-9)
                etps = tps * args.envs
                print(f"step={s['step']:>8d}  exp={s['sim_hours_total']:8.1f}h  "
                      f"{tps:6.1f} ticks/s = {etps:9.0f} env-ticks/s "
                      f"(x{etps / 15.0:7.0f} realtime)  F={s['F']:8.2f}  "
                      f"err={s['obs_err']:6.3f}  nov={s['novelty']:4.2f}  "
                      f"stops={s['gate_stops']:6d}  coll={s['collisions']:6d}",
                      flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        snapshots_on = args.snapshot_every > 0 or args.snapshot_every_min > 0
        if snapshots_on and trainer.status()["step"] > 0:
            trainer.snapshot()        # final state always joins the lineup
        trainer.close()
        s = trainer.status()
        wall = time.time() - t0
        print(f"done: {s['step']} batch ticks x {args.envs} envs = "
              f"{s['sim_hours_total']:.1f} sim-hours of experience in "
              f"{wall / 60.0:.1f} min wall; brain -> {trainer.model_path}",
              flush=True)

    if not args.no_eval_after and snapshots_on:
        _eval_and_pick_best(args)


def _eval_and_pick_best(args):
    """Frozen-eval every snapshot in unseen houses; copy the winner to
    <out-dir>/best_pnn_brain.pt (+ slow) and record the table as json."""
    import glob
    import json
    import shutil

    from pnn_sim.eval_checkpoints import make_eval_args, run_eval

    snaps = sorted(glob.glob(
        os.path.join(args.out_dir, "snapshots", "pnn_brain_step*.pt")))
    if not snaps:
        return
    print(f"\nfrozen-eval of {len(snaps)} snapshots in unseen houses "
          f"(seed {args.eval_seed}) ...", flush=True)
    results, best = run_eval(snaps, make_eval_args(
        envs=args.eval_envs, ticks=args.eval_ticks,
        eval_seed=args.eval_seed, device=args.device,
        work_dir=os.path.join(args.out_dir, "eval_tmp"),
        # Evaluate under the SAME policy the run trained with (and the one
        # you'd then set on the rover) — not the hardcoded defaults.
        epi_floor=args.epi_floor, forward_bias=args.forward_bias,
        novelty_pref_weight=args.novelty_pref_weight,
        hold_pref_weight=args.hold_pref_weight,
        target_novelty=args.target_novelty))

    results_path = os.path.join(args.out_dir, "eval_results.json")
    if best < 0:
        # Dark-room collapse: do NOT write best_ — there's nothing safe to
        # transfer. The table (with the loud warning) is already printed.
        with open(results_path, "w") as f:
            json.dump({"results": results, "best": None,
                       "verdict": "no_transferable_brain"}, f, indent=2)
        print(f"\nno best_pnn_brain.pt written; table -> eval_results.json",
              flush=True)
        return

    src = results[best]["ckpt"]
    dst = os.path.join(args.out_dir, "best_pnn_brain.pt")
    shutil.copy2(src, dst)
    slow_src = src.replace("pnn_brain_", "pnn_brain_slow_")
    if os.path.exists(slow_src):
        shutil.copy2(slow_src,
                     os.path.join(args.out_dir, "best_pnn_brain_slow.pt"))
    with open(results_path, "w") as f:
        json.dump({"results": results, "best": src}, f, indent=2)
    print(f"\nbest brain -> {dst} (from {os.path.basename(src)}); "
          f"table -> eval_results.json", flush=True)


if __name__ == "__main__":
    main()
