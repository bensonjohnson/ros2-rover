"""Validate the predictive-coding spatial map against the geometric substrate.

Drives B rovers with the frontier explorer, feeds the SAME trajectory to both
BatchedOccMap (geometric ground truth) and PCSpatialMap (inferred by predictive
coding), and reports how well PC inference recovers occupancy + free space,
plus an ASCII render of the PC map. Run:

    python3 -m pnn_sim.spatial.validate_pc [--envs 2] [--ticks 1800]
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

from pnn_sim.world import make_house
from pnn_sim.rover import SimRover, RoverConfig
from pnn_sim.safety_gate import SimSafetyGate, GateConfig
from pnn_sim.frontier import FrontierExplorer
from pnn_sim.spatial.grid_map import BatchedOccMap
from pnn_sim.spatial.pc_map import PCSpatialMap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=2)
    ap.add_argument("--ticks", type=int, default=1800)
    ap.add_argument("--res", type=float, default=0.15)
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--beam-stride", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    B = args.envs
    worlds = [make_house(np.random.default_rng(s)) for s in range(B)]
    rovers = [SimRover(w, RoverConfig()) for w in worlds]
    clk = {"v": 0.0}
    gates = [SimSafetyGate(GateConfig(), time_fn=lambda: clk["v"])
             for _ in range(B)]
    fes = [FrontierExplorer() for _ in range(B)]
    geo = BatchedOccMap(B, res=args.res, size=args.size, device=args.device,
                        beam_stride=args.beam_stride)
    pc = PCSpatialMap(B, res=args.res, size=args.size, device=args.device,
                      beam_stride=args.beam_stride)
    nb = rovers[0].cfg.n_beams
    bearings = torch.arange(nb, device=args.device) * 2 * np.pi / nb
    bnp = bearings.cpu().numpy()

    errs = []
    for i in range(args.ticks):
        poss = np.zeros((B, 2), np.float32)
        hds = np.zeros(B, np.float32)
        rngs = np.zeros((B, nb), np.float32)
        for b in range(B):
            r, amin, ainc = rovers[b].scan()
            gates[b].process_scan(r, amin, ainc)
            cmd, _ = fes[b].step(rovers[b].x, rovers[b].y, rovers[b].theta,
                                 r, bnp, blocked=gates[b].front_blocked)
            gl, gr = gates[b].gate(float(cmd[0]), float(cmd[1]))
            rovers[b].step(gl, gr, 1 / 15.0)
            poss[b] = [rovers[b].x, rovers[b].y]
            hds[b] = rovers[b].theta
            rngs[b] = r
        clk["v"] += 1 / 15.0
        P = torch.from_numpy(poss).to(args.device)
        H = torch.from_numpy(hds).to(args.device)
        R = torch.from_numpy(rngs).to(args.device)
        geo.update(P, H, R, bearings)
        pc.update(P, H, R, bearings)
        if i % 300 == 0:
            errs.append((i, pc.last_err))

    print("PC scan-prediction error over time (lower = map predicts better):")
    for t, e in errs:
        print(f"  t{t:>5}: |err|={e:.4f}")

    go, gf, gs = (geo.occupied().cpu().numpy(), geo.free().cpu().numpy(),
                  geo.seen.cpu().numpy())
    po, pf = pc.occupied().cpu().numpy(), pc.free().cpu().numpy()
    ps = pc.seen().cpu().numpy()
    occ_iou, free_iou = [], []
    for b in range(B):
        co = gs[b] & ps[b]
        occ_iou.append((go[b] & po[b] & co).sum()
                       / max(((go[b] | po[b]) & co).sum(), 1))
        free_iou.append((gf[b] & pf[b] & co).sum()
                        / max(((gf[b] | pf[b]) & co).sum(), 1))
    print(f"\nPC-vs-geometric over co-seen cells: occ_IoU={np.mean(occ_iou):.3f} "
          f"free_IoU={np.mean(free_iou):.3f}")
    print(f"confident cells: geo={gs.reshape(B, -1).sum(1).mean():.0f} "
          f"pc={ps.reshape(B, -1).sum(1).mean():.0f}")

    b = 0
    seen = pc.seen().cpu().numpy()[b]
    ys, xs = np.where(seen)
    if len(xs):
        i0, i1, j0, j1 = ys.min(), ys.max(), xs.min(), xs.max()
        step = max(1, (max(i1 - i0, j1 - j0)) // 44)
        print(f"\n=== env 0 PC MAP ('#'=occ '.'=free ' '=unknown, step {step}) ===")
        for i in range(i0, i1 + 1, step):
            print("".join("#" if po[b, i, j] else "." if pf[b, i, j]
                          else " " for j in range(j0, j1 + 1, step)))


if __name__ == "__main__":
    main()
