#!/usr/bin/env python3
"""Trajectory forensics for the places-stuck-at-1 collapse.

Distinguishes the two failure hypotheses the scalar metrics can't separate:
  H1 stuck-at-wall: agent parks against a wall (low path length, high
     blocked fraction, velocity pinned at zero despite nonzero command)
  H2 roams-but-exits-never: agent genuinely explores its spawn room
     (high path length, diverse headings) yet never crosses a doorway

Also answers the geometry question: is there a doorway within reachable
distance of the spawn pose at all, and how far is it?

Usage: python3 -m pnn_sim.tools.t4_trace [--device cuda] [--config C]
"""
import sys, json, argparse
sys.path[:0] = ['.', 'src/tractor_bringup']
import numpy as np
import torch

from pnn_sim.batched.trainer import BatchedTrainer, BatchedTrainConfig

CONFIGS = {
    "A_defaults":       dict(),
    "C_epifloor_appet": dict(epi_floor=0.005, novelty_pref_weight=3.0,
                             hold_pref_weight=1.0),
    "D_early_babble_off": dict(epi_floor=0.005, novelty_pref_weight=3.0,
                               hold_pref_weight=1.0, babble_decay_ticks=500),
}


def doorway_analysis(world):
    """Find doorways: wall endpoints (corners) that are unconnected gaps.
    Heuristic from world.py: internal walls have doorway GAPS — detect them
    as pairs of segment endpoints that face each other across < 1.2 m with no
    segment between (a gap in the wall line). Returns list of (x, y, dist_to_spawn)."""
    segs = world.segments
    ends = np.concatenate([segs[:, 0:2], segs[:, 2:4]], axis=0)
    doors = []
    for i in range(len(ends)):
        for j in range(i + 1, len(ends)):
            d = np.linalg.norm(ends[i] - ends[j])
            if 0.4 < d < 1.3:      # doorway-width gap between wall ends
                mid = (ends[i] + ends[j]) / 2.0
                # dedupe: skip if midpoint near an existing door
                if all(np.linalg.norm(mid - dd[:2]) > 0.5 for dd in doors):
                    doors.append((mid[0], mid[1], d))
    sx, sy = world.start_pose[0], world.start_pose[1]
    for k, (dx, dy, w) in enumerate(doors):
        doors[k] = (dx, dy, float(np.hypot(dx - sx, dy - sy)))
    return sorted(doors, key=lambda t: t[2])


def trace(name, over, device, envs=8, ticks=4500, env_of_interest=0):
    cfg = BatchedTrainConfig(envs=envs, device=device, seed=44,
                             out_dir=f"/tmp/pnn_trace_{name}",
                             switch_world_every=0, snapshot_every=0,
                             save_interval_s=1e9, log_envs=0, **over)
    tr = BatchedTrainer(cfg)
    world = tr.env._worlds[env_of_interest]
    doors = doorway_analysis(world)
    spawn = (float(tr.env.x[env_of_interest]),
             float(tr.env.y[env_of_interest]))

    pos = []
    cmd_mag, v_mag, blocked = [], [], []
    for _ in range(ticks):
        tr.tick()
        i = env_of_interest
        pos.append((float(tr.env.x[i]), float(tr.env.y[i])))
        cmd_mag.append(float(np.abs(tr.exec_action[i]).mean()))
        v_mag.append(float(0.5 * (abs(tr.env.v_left[i]) + abs(tr.env.v_right[i]))))
        blocked.append(bool(tr.gate.front_blocked[i]))
    P = np.asarray(pos)
    steps = np.linalg.norm(np.diff(P, axis=0), axis=1)
    path_len = float(steps.sum())
    max_r = float(np.linalg.norm(P - np.array(spawn), axis=1).max())
    radius = np.linalg.norm(P - np.array(spawn), axis=1)
    # stuck window: last 40% of the trace
    w = P[int(len(P) * .6):]
    wr = np.linalg.norm(w - np.array(spawn), axis=1)
    late_path = float(steps[int(len(steps) * .6):].sum())
    r = {
        "config": name, "ticks": ticks, "env": env_of_interest,
        "spawn": [round(spawn[0], 2), round(spawn[1], 2)],
        "house_extent": [round(float(world.segments.min()), 1),
                         round(float(world.segments.max(0).max()), 1)],
        "doorways": [[round(a, 1), round(b, 1), round(d, 1)]
                     for a, b, d in doors[:5]],
        "nearest_door_m": round(doors[0][2], 2) if doors else None,
        "path_len_m": round(path_len, 2),
        "late_path_m": round(late_path, 2),
        "max_radius_m": round(max_r, 2),
        "late_radius_spread_m": round(float(wr.max() - wr.min()), 2),
        "blocked_frac": round(float(np.mean(blocked)), 3),
        "mean_cmd": round(float(np.mean(cmd_mag)), 3),
        "mean_v_mps": round(float(np.mean(v_mag)), 3),
        "places": int(tr.place.n_places()[env_of_interest]),
    }
    # verdict
    if r["late_path_m"] < 3.0 and r["blocked_frac"] > 0.5:
        r["verdict"] = "H1 stuck-at-wall"
    elif r["late_path_m"] > 8.0:
        r["verdict"] = "H2 roams-but-exits-never"
    else:
        r["verdict"] = "ambiguous"
    tr.close()
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--config", default="")
    ap.add_argument("--ticks", type=int, default=4500)
    a = ap.parse_args()
    names = [a.config] if a.config else list(CONFIGS)
    for n in names:
        r = trace(n, dict(CONFIGS[n]), a.device, ticks=a.ticks)
        print(json.dumps(r, indent=2), flush=True)
    print("trace done")
