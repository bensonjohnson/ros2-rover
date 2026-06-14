"""Distil frontier-seeking into a pure-PNN map-policy, and validate it.

Online imitation (DAgger-style): the proven frontier.py explorer (the TEACHER,
~1.4 rooms) drives B batched rovers; every tick we build the batched PC map,
read each rover's egocentric map patch, and train the PCPolicy readout (local
delta rule) to predict the teacher's target bearing from that patch. Then the
STUDENT drives alone — reading only its own PC map — and we score ground-truth
rooms✓ against the teacher and a fresh baseline.

If the student matches the teacher, exploration with spatial memory has been
distilled into a pure predictive-coding stack (PC map + delta-rule map-policy):
the redesign's thesis, validated in sim.

    python3 -m pnn_sim.spatial.distill_policy --train-envs 24 --train-ticks 4000
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (_REPO, os.path.join(_REPO, "src", "tractor_bringup")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch

from pnn_sim.batched.env import BatchedEnv, BatchedGate
from pnn_sim.rover import RoverConfig
from pnn_sim.safety_gate import GateConfig
from pnn_sim.frontier import FrontierExplorer
from pnn_sim.spatial.pc_map import PCSpatialMap
from pnn_sim.spatial.policy import egocentric_patch, PCPolicy, pursuit_cmd

P = 32
HALF_M = 4.0


def total_rooms(w):
    return len(set(w.room_id(x, y)
                   for x in np.linspace(0.4, w.bounds[2] - 0.4, 26)
                   for y in np.linspace(0.4, w.bounds[3] - 0.4, 26)
                   if w.clearance(x, y) > 0.2))


def rollout(B, ticks, policy, mode, seed, res=0.15, size=200, train=False):
    """mode: 'teacher' (frontier.py drives) or 'student' (policy drives).
    When train=True (teacher mode), update `policy` toward the teacher each
    tick. Returns mean rooms✓ fraction."""
    clk = {"v": 0.0}
    env = BatchedEnv(B, RoverConfig(), seed=seed, device="cpu")
    gate = BatchedGate(GateConfig(), B, time_fn=lambda: clk["v"], device="cpu")
    pcmap = PCSpatialMap(B, res=res, size=size, device="cpu", beam_stride=4)
    nb = env.scan().shape[1]
    bearings = torch.arange(nb) * env.angle_increment + env.angle_min
    bnp = bearings.numpy()
    teachers = [FrontierExplorer() for _ in range(B)] if mode == "teacher" else None
    escape = torch.zeros(B, dtype=torch.long)
    edir = torch.ones(B)
    visited = [set() for _ in range(B)]

    for _ in range(ticks):
        ranges = env.scan()
        gate.process_scan(ranges, env.angle_min, env.angle_increment)
        pos = torch.stack([env.x, env.y], dim=1)
        pcmap.update(pos, env.theta, ranges, bearings)
        blocked = gate.front_blocked

        if mode == "teacher":
            xs, ys, ths = env.x.numpy(), env.y.numpy(), env.theta.numpy()
            cmds = np.zeros((B, 2), np.float32)
            tb = np.zeros(B, np.float32)
            valid = np.zeros(B, bool)
            rn = ranges.numpy()
            for b in range(B):
                c, info = teachers[b].step(xs[b], ys[b], ths[b], rn[b], bnp,
                                           blocked=bool(blocked[b]))
                cmds[b] = c
                if info["target_bearing"] is not None:
                    tb[b] = info["target_bearing"]
                    valid[b] = True
            if train:
                patch = egocentric_patch(pcmap, pos, env.theta, HALF_M, P)
                _, feats = policy.predict(patch)
                policy.learn(feats, torch.from_numpy(tb),
                             torch.from_numpy(valid))
            cmd = torch.from_numpy(cmds)
        else:  # student drives from its own map
            patch = egocentric_patch(pcmap, pos, env.theta, HALF_M, P)
            tb_pred, _ = policy.predict(patch)
            cmd = pursuit_cmd(tb_pred, ranges, bearings, blocked,
                              (escape, edir))

        env.step(gate.gate(cmd), 1 / 15.0)
        clk["v"] += 1 / 15.0
        xs, ys = env.x.numpy(), env.y.numpy()
        for b in range(B):
            visited[b].add(env._worlds[b].room_id(float(xs[b]), float(ys[b])))

    vis = np.array([len(v) for v in visited])
    tot = np.array([total_rooms(env._worlds[b]) for b in range(B)])
    return float((vis / np.maximum(tot, 1)).mean()), float(vis.mean()), \
        float(tot.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-envs", type=int, default=24)
    ap.add_argument("--train-ticks", type=int, default=4000)
    ap.add_argument("--eval-envs", type=int, default=16)
    ap.add_argument("--eval-ticks", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--n-feat", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--save", default="", help="path to save the trained policy")
    ap.add_argument("--load", default="", help="skip training, load this policy")
    args = ap.parse_args()

    in_dim = 3 * P * P
    policy = PCPolicy(in_dim, n_feat=args.n_feat, lr=args.lr, seed=0)

    if args.load:
        policy.load(args.load)
        print(f"loaded policy from {args.load} (skipping training)", flush=True)
    else:
        print(f"training (teacher drives, student imitates) "
              f"{args.epochs} epochs x {args.train_envs} envs x "
              f"{args.train_ticks} ticks ...", flush=True)
        for ep in range(args.epochs):
            frac, vis, tot = rollout(args.train_envs, args.train_ticks, policy,
                                     "teacher", seed=1000 + ep * 13, train=True)
            print(f"  epoch {ep}: teacher rooms✓={vis:.2f}/{tot:.2f} "
                  f"(frac {frac:.2f})  |Wr|={policy.Wr.norm():.3f}", flush=True)
        if args.save:
            policy.save(args.save)
            print(f"saved policy -> {args.save}", flush=True)

    print("\neval in UNSEEN houses (seed 777000):", flush=True)
    tf, tv, tt = rollout(args.eval_envs, args.eval_ticks, FrontierExplorer,
                         "teacher", seed=777_000)
    sf, sv, st = rollout(args.eval_envs, args.eval_ticks, policy,
                         "student", seed=777_000)
    print(f"  TEACHER (frontier.py): rooms✓={tv:.2f}/{tt:.2f}  frac={tf:.2f}")
    print(f"  STUDENT (pure-PNN)   : rooms✓={sv:.2f}/{st:.2f}  frac={sf:.2f}")
    print(f"\n  student / teacher coverage ratio: {sf / max(tf, 1e-9):.2f}")


if __name__ == "__main__":
    main()
