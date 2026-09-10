#!/usr/bin/env python3
"""Signal-chain forensics: actor pick -> persist/hold -> smoothing -> gate ->
track velocities, per tick, for one env. Pins the degenerate basin behind
the places-stuck-at-1 collapse (spin-loop vs command cancellation vs gate).

Usage: python3 -m pnn_sim.tools.t4_signal [--device cuda] [--config A]
"""
import sys, json, argparse
sys.path[:0] = ['.', 'src/tractor_bringup']
import numpy as np
import torch
from collections import Counter

from pnn_sim.batched.trainer import BatchedTrainer, BatchedTrainConfig

CONFIGS = {
    "A":  dict(),
    "D":  dict(epi_floor=0.005, novelty_pref_weight=3.0,
               hold_pref_weight=1.0, babble_decay_ticks=500),
}


def run(name, over, device="cuda", envs=8, ticks=4500, e=0):
    cfg = BatchedTrainConfig(envs=envs, device=device, seed=44,
                             out_dir=f"/tmp/pnn_sig_{name}",
                             switch_world_every=0, snapshot_every=0,
                             save_interval_s=1e9, log_envs=0, **over)
    tr = BatchedTrainer(cfg)
    rows = []
    for t in range(ticks):
        tr.tick()
        rows.append(dict(
            t=t,
            held=tr.held_raw[e].copy(),
            act=tr.exec_action[e].copy(),
            vl=float(tr.env.v_left[e]), vr=float(tr.env.v_right[e]),
            yaw=float(tr.env.yaw_rate[e]),
            theta=float(tr.env.theta[e]),
            hold=bool(tr.gate.front_blocked[e]),
        ))
    H = np.asarray([r["held"] for r in rows])
    A = np.asarray([r["act"] for r in rows])
    vl = np.asarray([r["vl"] for r in rows])
    vr = np.asarray([r["vr"] for r in rows])
    yaw = np.asarray([r["yaw"] for r in rows])
    th = np.asarray([r["theta"] for r in rows])
    hold = np.asarray([r["hold"] for r in rows], dtype=bool)

    # command quadrant mix (both fwd / both back / spin L / spin R / mixed)
    def quad(l, r):
        if abs(l) < 0.1 and abs(r) < 0.1: return "idle"
        if l > 0.1 and r > 0.1: return "fwd"
        if l < -0.1 and r < -0.1: return "back"
        if np.sign(l) != np.sign(r): return "spin"
        return "mixed"
    q = Counter(quad(*a) for a in A)
    dtheta = np.abs(np.diff(np.unwrap(th)))
    # per-tick |heading change| vs |translation| ratio -> spin vs drive
    translate = np.hypot(vl, vr) * (1/15.0)
    res = {
        "config": name, "ticks": ticks,
        "cmd_mix": dict(q),
        "frac_spin_cmds": round(q.get("spin", 0) / ticks, 3),
        "frac_fwd_cmds": round(q.get("fwd", 0) / ticks, 3),
        "mean_abs_yaw_rps": round(float(np.abs(yaw).mean()), 3),
        "mean_heading_rate_rps": round(float(dtheta.mean() * 15), 3),
        "mean_track_v": round(float(0.5 * (vl.mean() + vr.mean())), 4),
        "fwd_ticks_mean_v": (round(
            float((0.5 * (vl + vr))[(H[:, 0] > 0.2) & (H[:, 1] > 0.2)].mean()),
            4) if ((H[:, 0] > 0.2) & (H[:, 1] > 0.2)).any() else None),
        "hold_frac": round(float(hold.mean()), 3),
        "novelty": round(float(tr.nov_ema[e]), 4),
        "places": int(tr.place.n_places()[e]),
    }
    # average run length of an unchanged held action (persist behaviour)
    runs, cur = [], 1
    for i in range(1, len(H)):
        if np.allclose(H[i], H[i - 1], atol=1e-6):
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    res["mean_hold_ticks"] = round(float(np.mean(runs)), 2)
    # simple spin detector: |yaw| high while |track speed| low
    spin_mask = (np.abs(yaw) > 0.8) & (np.abs(vl) + np.abs(vr) < 0.1)
    res["frac_ticks_spinning"] = round(float(spin_mask.mean()), 3)
    tr.close()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--config", default="")
    a = ap.parse_args()
    names = [a.config] if a.config else list(CONFIGS)
    for n in names:
        print(json.dumps(run(n, dict(CONFIGS[n]), device=a.device),
                         indent=2), flush=True)
    print("signal done")
