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
    ap.add_argument("--ticks", type=int, default=100_000,
                    help="batch ticks; sim experience = ticks * envs / 15Hz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lr-scale", type=float, default=1.0,
                    help="scale on the batch-averaged updates (the gradient "
                    "is ~sqrt(B) less noisy; 2-4 is worth trying)")
    ap.add_argument("--switch-world-every", type=int, default=27_000,
                    help="per-env ticks between house swaps, staggered "
                    "across the batch (0 = never)")
    ap.add_argument("--load", default="",
                    help="existing pnn_brain.pt to continue from")
    ap.add_argument("--out-dir", default="sim_out_batched")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--action-scale", type=float, default=0.6)
    ap.add_argument("--torch-threads", type=int, default=0,
                    help="torch CPU threads (0 = torch default; set for "
                    "--device cpu runs)")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--log-envs", type=int, default=1,
                    help="how many envs write experience jsonl")
    ap.add_argument("--report-s", type=float, default=5.0)
    args = ap.parse_args()

    from pnn_sim.batched.trainer import BatchedTrainer, BatchedTrainConfig

    trainer = BatchedTrainer(BatchedTrainConfig(
        envs=args.envs, device=args.device, lr_scale=args.lr_scale,
        switch_world_every=args.switch_world_every,
        load_path=args.load, out_dir=args.out_dir, seed=args.seed,
        action_scale=args.action_scale, torch_threads=args.torch_threads,
        learn=not args.freeze, log_envs=args.log_envs,
    ))

    t0 = time.time()
    last = t0
    try:
        for _ in range(args.ticks):
            trainer.tick()
            now = time.time()
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
        trainer.close()
        s = trainer.status()
        wall = time.time() - t0
        print(f"done: {s['step']} batch ticks x {args.envs} envs = "
              f"{s['sim_hours_total']:.1f} sim-hours of experience in "
              f"{wall / 60.0:.1f} min wall; brain -> {trainer.model_path}",
              flush=True)


if __name__ == "__main__":
    main()
