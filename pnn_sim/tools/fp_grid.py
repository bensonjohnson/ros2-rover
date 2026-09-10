#!/usr/bin/env python3
"""Fingerprint discriminability grid: how do (shape_weight, match_thresh)
trade OFF whole-house place separation against lidar-jitter robustness?

Vectorized: BatchedEnv with one env per (x, y, heading) sample; scan all
poses in one pass; fingerprint once. Clean pass uses noiseless lidar; the
jitter pass re-scans with the real sensor noise model (std 0.01 m, dropout
2%). Good region = clean greedy place-count HIGH (separates rooms) AND
jitter inflation LOW (same place must not spawn phantom places).
"""
import sys, json
sys.path[:0] = ['.', 'src/tractor_bringup']
import numpy as np
import torch
from pnn_sim.batched.env import BatchedEnv, batched_preprocess
from pnn_sim.batched.place import BatchedPlaceMemory
from pnn_sim.rover import RoverConfig


def sample_fingerprints(sw, jitter_seed=0):
    """Whole-house clean scan samples -> fingerprints [K, F] (device)."""
    w_rng = np.random.default_rng(44)
    probe = BatchedEnv(1, RoverConfig(), seed=44, device="cuda")
    w = probe._worlds[0]
    xs = np.linspace(0.5, float(w.segments[:, 0].max()) - 0.5, 10)
    ys = np.linspace(0.5, float(w.segments[:, 1].max()) - 0.5, 10)
    poses = []
    for x in xs:
        for y in ys:
            if float(probe._clearance(
                    torch.tensor([float(x)], device="cuda"),
                    torch.tensor([float(y)], device="cuda"))) < 0.45:
                continue
            for th in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                poses.append((x, y, th))
    K = len(poses)
    cfg = RoverConfig(lidar_noise_std=0.01 if jitter_seed else 0.0,
                      lidar_dropout_p=0.02 if jitter_seed else 0.0)
    env = BatchedEnv(K, cfg, seed=jitter_seed or 44, device="cuda")
    P = np.array(poses)
    env.x = torch.tensor(P[:, 0], device="cuda", dtype=torch.float32)
    env.y = torch.tensor(P[:, 1], device="cuda", dtype=torch.float32)
    env.theta = torch.tensor(P[:, 2], device="cuda", dtype=torch.float32)
    ranges = env.scan()
    s72 = batched_preprocess(ranges, env.angle_min, env.angle_increment)
    pm = BatchedPlaceMemory(1, device="cuda", shape_weight=sw)
    return pm.fingerprint(s72)                       # [K, F]


def greedy_count(fp, thresh):
    d = torch.cdist(fp.double(), fp.double()).cpu().numpy()
    cent = [0]
    while True:
        dmin = d[:, cent].min(axis=1)
        j = int(np.argmax(dmin))
        if dmin[j] < thresh:
            break
        cent.append(j)
        if len(cent) > 30:
            break
    return len(cent)


if __name__ == "__main__":
    out = []
    for sw in (1.0, 2.0, 4.0):
        fp_clean = sample_fingerprints(sw, 0)
        fp_jit = sample_fingerprints(sw, 7)
        for thr in (0.35, 0.20, 0.15, 0.10, 0.06):
            c = greedy_count(fp_clean, thr)
            j = greedy_count(fp_jit, thr)
            out.append({"shape_weight": sw, "match_thresh": thr,
                        "rooms_clean": c, "rooms_jitter": j,
                        "inflation": round(j / max(c, 1), 2)})
            print(out[-1], flush=True)
    with open("/tmp/pnn_fp_grid.json", "w") as f:
        json.dump(out, f, indent=2)
    print("fp grid done")
