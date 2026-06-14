"""Stage-3 smoke test: the pure-PNN spatial explorer, batched, at scale.

Runs the distilled student (PC spatial map + delta-rule map-policy + pursuit)
fully batched on the GPU for a long run with staggered house switching, to
confirm it (a) runs on the Spark's CUDA, (b) sustains throughput at large B,
(c) stays numerically stable over a long run (no NaN / map blow-up), and (d)
holds its ground-truth room coverage across many fresh houses.

The teacher (frontier.py) is needed only to TRAIN the policy (per-env CPU);
once distilled, this run is pure batched tensor ops — no teacher, no planner.

    # pretrain on CPU (teacher-bound), then long GPU run:
    python3 -m pnn_sim.spatial.spark_smoke --device cuda --envs 256 \
        --run-ticks 18000 --switch-every 6000 --save-policy policy.pt
    # reuse a saved policy:
    python3 -m pnn_sim.spatial.spark_smoke --device cuda --envs 512 \
        --load-policy policy.pt --run-ticks 30000
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (_REPO, os.path.join(_REPO, "src", "tractor_bringup")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch

from pnn_sim.batched.env import BatchedEnv, BatchedGate
from pnn_sim.rover import RoverConfig
from pnn_sim.safety_gate import GateConfig
from pnn_sim.spatial.pc_map import PCSpatialMap
from pnn_sim.spatial.policy import egocentric_patch, PCPolicy, pursuit_cmd
from pnn_sim.spatial.distill_policy import rollout, P, HALF_M, total_rooms


def smoke(args):
    dev = args.device
    B = args.envs
    in_dim = 3 * P * P
    policy = PCPolicy(in_dim, n_feat=args.n_feat, lr=args.lr,
                      device=dev, seed=0)

    # --- policy: load or pretrain (teacher is CPU per-env) ---
    if args.load_policy:
        policy.load(args.load_policy)
        print(f"loaded policy from {args.load_policy}", flush=True)
    else:
        print(f"pretraining policy ({args.epochs} epochs x {args.pre_envs} "
              f"envs x {args.pre_ticks} ticks, CPU teacher) ...", flush=True)
        cpu_pol = PCPolicy(in_dim, n_feat=args.n_feat, lr=args.lr,
                           device="cpu", seed=0)
        for ep in range(args.epochs):
            _, vis, tot = rollout(args.pre_envs, args.pre_ticks, cpu_pol,
                                  "teacher", seed=1000 + ep * 13, train=True)
            print(f"  epoch {ep}: teacher rooms✓={vis:.2f}/{tot:.2f}  "
                  f"|Wr|={cpu_pol.Wr.norm():.3f}", flush=True)
        policy.Wf = cpu_pol.Wf.to(dev)
        policy.bf = cpu_pol.bf.to(dev)
        policy.Wr = cpu_pol.Wr.to(dev)
        if args.save_policy:
            cpu_pol.save(args.save_policy)
            print(f"saved policy -> {args.save_policy}", flush=True)

    # --- long batched student-only run on the GPU ---
    print(f"\nstudent-only run: {B} envs x {args.run_ticks} ticks on {dev}, "
          f"house switch every {args.switch_every} (staggered) ...", flush=True)
    clk = {"v": 0.0}
    env = BatchedEnv(B, RoverConfig(), seed=args.seed, device=dev)
    gate = BatchedGate(GateConfig(), B, time_fn=lambda: clk["v"], device=dev)
    pcmap = PCSpatialMap(B, res=args.res, size=args.size, device=dev,
                         beam_stride=4)
    nb = env.scan().shape[1]
    bearings = torch.arange(nb, device=dev) * env.angle_increment + env.angle_min
    escape = torch.zeros(B, dtype=torch.long, device=dev)
    edir = torch.ones(B, device=dev)

    # rooms✓ accounting per house-life (reset on switch)
    visited = [set() for _ in range(B)]
    life_frac = []                       # completed house-lives' coverage frac
    def cur_total(b):
        return max(1, total_rooms(env._worlds[b]))

    t0 = time.time()
    last = t0
    for t in range(args.run_ticks):
        # staggered house switch
        if args.switch_every > 0 and t > 0:
            stride = max(1, args.switch_every // B)
            due = (t + np.arange(B) * stride) % args.switch_every == 0
            if due.any():
                idx = np.flatnonzero(due)
                for b in idx:
                    life_frac.append(len(visited[b]) / cur_total(b))
                    visited[b] = set()
                env.switch_world(due)
                ti = torch.from_numpy(idx).to(dev)
                pcmap.reset(ti)
                escape[ti] = 0

        ranges = env.scan()
        gate.process_scan(ranges, env.angle_min, env.angle_increment)
        pos = torch.stack([env.x, env.y], dim=1)
        pcmap.update(pos, env.theta, ranges, bearings)
        patch = egocentric_patch(pcmap, pos, env.theta, HALF_M, P)
        tb_pred, _ = policy.predict(patch)
        cmd = pursuit_cmd(tb_pred, ranges, bearings, gate.front_blocked,
                          (escape, edir))
        env.step(gate.gate(cmd), 1 / 15.0)
        clk["v"] += 1 / 15.0

        xs, ys = env.x.cpu().numpy(), env.y.cpu().numpy()
        for b in range(B):
            visited[b].add(env._worlds[b].room_id(float(xs[b]), float(ys[b])))

        if time.time() - last >= args.report_s:
            last = time.time()
            tps = (t + 1) / (last - t0)
            nan = bool(torch.isnan(pcmap.M).any() or torch.isnan(policy.Wr).any())
            live = np.mean([len(visited[b]) / cur_total(b) for b in range(B)])
            done = np.mean(life_frac) if life_frac else float("nan")
            print(f"  t={t+1:>6} {tps*B:>8.0f} env-tick/s (x{tps*B/15:>6.0f} rt)  "
                  f"live_frac={live:.2f} closed_frac={done:.2f} "
                  f"lives={len(life_frac)} NaN={nan} |Wr|={policy.Wr.norm():.2f}",
                  flush=True)
            if nan:
                print("  *** NaN DETECTED — aborting ***", flush=True)
                return

    # final
    for b in range(B):
        life_frac.append(len(visited[b]) / cur_total(b))
    wall = time.time() - t0
    print(f"\ndone: {args.run_ticks} ticks x {B} envs in {wall/60:.1f} min "
          f"({args.run_ticks*B/wall:.0f} env-tick/s)", flush=True)
    print(f"  house-lives completed: {len(life_frac)}", flush=True)
    print(f"  mean coverage per house-life: {np.mean(life_frac):.3f} "
          f"(frac of reachable rooms visited)", flush=True)
    print(f"  stability: NaN={bool(torch.isnan(pcmap.M).any())}  "
          f"|Wr|={policy.Wr.norm():.3f}  map mean|logit|={pcmap.M.abs().mean():.3f}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--run-ticks", type=int, default=18000)
    ap.add_argument("--switch-every", type=int, default=6000)
    ap.add_argument("--res", type=float, default=0.15)
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--report-s", type=float, default=15.0)
    # policy
    ap.add_argument("--load-policy", default="")
    ap.add_argument("--save-policy", default="")
    ap.add_argument("--n-feat", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--pre-envs", type=int, default=24)
    ap.add_argument("--pre-ticks", type=int, default=4000)
    args = ap.parse_args()
    smoke(args)


if __name__ == "__main__":
    main()
