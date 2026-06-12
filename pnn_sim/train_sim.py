#!/usr/bin/env python3
"""Train the PC active-inference brain in the headless 2D sim.

Runs as fast as the CPU (or GPU) allows — no clocks, no ROS. Needs only
python3 + numpy + torch; the brain code is imported straight from
src/tractor_bringup, no colcon build.

Examples:
    # 12 sim-hours in one world, CPU
    python3 -m pnn_sim.train_sim --ticks 650000

    # 16 parallel lifetimes with different worlds/seeds (one per core)
    python3 -m pnn_sim.train_sim --ticks 650000 --workers 16

    # New house every ~10 sim-minutes, start from the rover's brain
    python3 -m pnn_sim.train_sim --switch-world-every 9000 \
        --load ~/.ros/pnn_brain.pt

The trained brain lands in <out-dir>/pnn_brain.pt (+ pnn_brain_slow.pt and
pnn_experience.jsonl). Copy the .pt files to ~/.ros/ on the rover to
transfer; the sleep consolidator runs on the experience log as-is.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Make tractor_bringup importable without a colcon install.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_REPO, os.path.join(_REPO, "src", "tractor_bringup")):
    if p not in sys.path:
        sys.path.insert(0, p)


def build_config(args, worker: int = 0):
    from pnn_sim.headless_runner import SimBrainConfig
    out_dir = (os.path.join(args.out_dir, f"worker_{worker:02d}")
               if args.workers > 1 else args.out_dir)
    return SimBrainConfig(
        seed=args.seed + worker * 1000,
        device=args.device,
        torch_threads=args.torch_threads,
        learn=not args.freeze,
        action_scale=args.action_scale,
        switch_world_every=args.switch_world_every,
        out_dir=out_dir,
        load_path=os.path.expanduser(args.load) if args.load else "",
        log_experience=not args.no_log,
    )


def run_worker(args, worker: int = 0):
    import torch
    torch.manual_seed(args.seed + worker * 1000)
    from pnn_sim.headless_runner import SimBrainRunner

    runner = SimBrainRunner(build_config(args, worker))
    t0 = time.time()
    last_report = t0
    tag = f"[w{worker:02d}] " if args.workers > 1 else ""
    try:
        for _ in range(args.ticks):
            runner.tick()
            now = time.time()
            if now - last_report >= args.report_s:
                last_report = now
                s = runner.status()
                tps = s["step"] / max(now - t0, 1e-9)
                print(f"{tag}step={s['step']:>8d}  "
                      f"sim={s['sim_hours']:6.2f}h  {tps:7.1f} ticks/s "
                      f"(x{tps / 15.0:5.1f} realtime)  F={s['F']:8.2f}  "
                      f"err={s['obs_err']:6.3f}  nov={s['novelty']:4.2f}  "
                      f"places={s['places']:3d}  stops={s['gate_stops']:4d}  "
                      f"coll={s['collisions']:4d}  dist={s['dist_m']:7.1f}m",
                      flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        runner.close()
        s = runner.status()
        wall = time.time() - t0
        print(f"{tag}done: {s['step']} ticks = {s['sim_hours']:.2f} sim-hours "
              f"in {wall / 60.0:.1f} min wall "
              f"({s['step'] / max(wall, 1e-9):.0f} ticks/s); "
              f"brain -> {runner.model_path}", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticks", type=int, default=650_000,
                    help="control ticks to run (650k = ~12 sim-hours @15Hz)")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel independent lifetimes (own world/seed/output)")
    ap.add_argument("--device", default="cpu",
                    help="torch device for the brain: cpu or cuda")
    ap.add_argument("--torch-threads", type=int, default=1,
                    help="torch CPU threads per worker (rover uses 1; the "
                    "per-tick tensors are too small for more to help)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--action-scale", type=float, default=0.6)
    ap.add_argument("--switch-world-every", type=int, default=27_000,
                    help="ticks between house swaps (27k = ~30 sim-min; "
                    "0 = one world forever)")
    ap.add_argument("--load", default="",
                    help="existing pnn_brain.pt to continue from")
    ap.add_argument("--out-dir", default="sim_out")
    ap.add_argument("--freeze", action="store_true",
                    help="run without learning (evaluate a brain)")
    ap.add_argument("--no-log", action="store_true",
                    help="skip the experience jsonl")
    ap.add_argument("--report-s", type=float, default=5.0,
                    help="seconds between progress lines")
    args = ap.parse_args()

    if args.workers <= 1:
        run_worker(args, 0)
        return

    import multiprocessing as mp
    procs = []
    for w in range(args.workers):
        p = mp.Process(target=run_worker, args=(args, w))
        p.start()
        procs.append(p)
    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
