"""Validate + render the batched occupancy substrate (BatchedOccMap).

Drives B rovers with the frontier explorer, builds the batched map, scores
free-space reconstruction over the explored region vs ground truth, and ASCII-
renders env 0 so you can watch the map form. Run:

    python3 -m pnn_sim.spatial.validate_map [--envs 4] [--ticks 4500]
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


def gt_free(world, origin, n, res):
    xs = origin[0] + np.arange(n) * res
    ys = origin[1] + np.arange(n) * res
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    inb = ((gx >= world.bounds[0]) & (gx <= world.bounds[2])
           & (gy >= world.bounds[1]) & (gy <= world.bounds[3]))
    free = np.zeros((n, n), bool)
    for i, j in np.argwhere(inb):
        free[i, j] = world.clearance(float(gx[i, j]), float(gy[i, j])) > 0.14
    return free


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--ticks", type=int, default=4500)
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
    omap = BatchedOccMap(B, res=args.res, size=args.size, device=args.device,
                         beam_stride=args.beam_stride)
    nb = rovers[0].cfg.n_beams
    bearings = torch.arange(nb, device=args.device) * 2 * np.pi / nb
    bear_np = bearings.cpu().numpy()

    for _ in range(args.ticks):
        poss = np.zeros((B, 2), np.float32)
        hds = np.zeros(B, np.float32)
        rngs = np.zeros((B, nb), np.float32)
        for b in range(B):
            r, amin, ainc = rovers[b].scan()
            gates[b].process_scan(r, amin, ainc)
            cmd, _ = fes[b].step(rovers[b].x, rovers[b].y, rovers[b].theta,
                                 r, bear_np, blocked=gates[b].front_blocked)
            gl, gr = gates[b].gate(float(cmd[0]), float(cmd[1]))
            rovers[b].step(gl, gr, 1 / 15.0)
            poss[b] = [rovers[b].x, rovers[b].y]
            hds[b] = rovers[b].theta
            rngs[b] = r
        clk["v"] += 1 / 15.0
        omap.update(torch.from_numpy(poss).to(args.device),
                    torch.from_numpy(hds).to(args.device),
                    torch.from_numpy(rngs).to(args.device), bearings)

    origin = omap.origin.cpu().numpy()
    mfree = omap.free().cpu().numpy()
    mocc = omap.occupied().cpu().numpy()
    mseen = omap.seen.cpu().numpy()
    n = args.size
    ious = []
    for b in range(B):
        gf = gt_free(worlds[b], origin[b], n, args.res)
        seen = mseen[b]
        inter = (mfree[b] & gf & seen).sum()
        union = ((mfree[b] | gf) & seen).sum()
        ious.append(inter / max(union, 1))
    print(f"free-space IoU over explored region (mean {B} envs): "
          f"{np.mean(ious):.3f}  per-env {[f'{x:.2f}' for x in ious]}")
    print(f"mean cells seen: {mseen.reshape(B, -1).sum(1).mean():.0f}/{n * n}  "
          f"frontier cells: {omap.frontier().cpu().numpy().reshape(B, -1).sum(1).mean():.0f}")

    b = 0
    seen = mseen[b]
    ys, xs = np.where(seen)
    if len(xs):
        i0, i1, j0, j1 = ys.min(), ys.max(), xs.min(), xs.max()
        step = max(1, (max(i1 - i0, j1 - j0)) // 44)
        print(f"\n=== env 0 batched occupancy map "
              f"('#'=occ '.'=free ' '=unknown, step {step}) ===")
        for i in range(i0, i1 + 1, step):
            row = "".join("#" if mocc[b, i, j] else "." if mfree[b, i, j]
                          else " " for j in range(j0, j1 + 1, step))
            print(row)


if __name__ == "__main__":
    main()
