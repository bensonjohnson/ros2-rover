#!/usr/bin/env python3
"""Replay a /scan rosbag2 through the REFERENCE PlaceMemory on the rover.

Stationary test (rover on dock): the historical d3c78a7 failure mode — real
lidar fingerprint jitter shattering one room into phantom places — shows up
as growing n_places / non-decaying novelty while NOTHING MOVED. Also reports
the REAL per-tick fingerprint drift rate: the slot-blend chase equilibrium
is drift/blend, which is what made the sim match_thresh=0.35 collapse to one
place (t4_place_probe). Sweeps (match_thresh, shape_weight, slot_blend) so
the batched fix ports to the reference path on REAL data, not sim guesses.

On rover:
  cd ~/ros2-rover && source /opt/ros/jazzy/setup.bash
  python3 pnn_sim/tools/bag_place_replay.py --bag /tmp/scan_stationary
"""
from __future__ import annotations

import argparse
import sys

sys.path[:0] = ['.', 'src/tractor_bringup']

import numpy as np
from rosbag2_py import (ConverterOptions, SequentialReader, StorageOptions,
                        StorageFilter)
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan

from tractor_bringup.active_inference.place_memory import PlaceMemory
from tractor_bringup.active_inference.scan_preprocess import preprocess_scan


def load_scans(bag: str):
    """-> (t [S], fps_raw list, meta dict) from a /scan mcap bag (jazzy)."""
    reader = SequentialReader()
    reader.open(StorageOptions(uri=bag, storage_id="mcap"),
                ConverterOptions())
    reader.reset_filter()   # bag recorded /scan only
    ts, fps_raw, metas = [], [], None
    while reader.has_next():
        _topic, data, _seq = reader.read_next()
        msg = deserialize_message(data, LaserScan)
        if metas is None:
            metas = dict(angle_min=float(msg.angle_min),
                         angle_increment=float(msg.angle_increment),
                         range_max=float(msg.range_max),
                         n=len(msg.ranges))
        ts.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        fps_raw.append(np.asarray(msg.ranges, dtype=np.float64))
    t0 = ts[0]
    ts = np.asarray(ts) - t0
    return ts, fps_raw, metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--bins", type=int, default=72)
    ap.add_argument("--max-range", type=float, default=12.0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--tau", type=float, nargs="*", default=(1.5, 3.0, 6.0))
    ap.add_argument("--shape", type=float, nargs="*", default=(1.0, 2.0))
    ap.add_argument("--thresh", type=float, nargs="*",
                    default=(0.08, 0.12, 0.18, 0.35))
    ap.add_argument("--blend", type=float, nargs="*", default=(0.02, 0.0))
    ap.add_argument("--gate", type=float, nargs="*", default=(0.0,))
    args = ap.parse_args()

    ts, raw, meta = load_scans(args.bag)
    assert meta is not None and len(raw) > 20, "bag has too few /scan msgs"
    S = len(raw)
    dt_med = float(np.median(np.diff(ts)))
    s72 = [preprocess_scan(r, meta["angle_min"], meta["angle_increment"],
                           num_bins=args.bins, max_range=args.max_range)
           for r in raw]
    print(f"{S} scans, {meta['n']} beams, dt median {dt_med:.3f} s "
          f"({1/dt_med:.1f} Hz), range_max {meta['range_max']:.1f} m")

    # real-world fingerprint drift (no EMA): the chase-calibration number
    probe = PlaceMemory(n_freq=10, fp_ema_tau_s=0.0)
    fps = np.stack([probe.fingerprint(s) for s in s72])
    step = np.linalg.norm(np.diff(fps, axis=0), axis=1)
    print(f"raw fp drift/tick: median {np.median(step):.5f} "
          f"p90 {np.percentile(step, 90):.5f}  "
          f"-> chase equilibrium @blend0.02 ~ {np.median(step)/0.02:.3f}, "
          f"@blend0.005 ~ {np.median(step)/0.005:.3f}")

    # replay with a virtual clock fed by bag timestamps
    class Clock:
        def __init__(self): self.t = 0.0
        def __call__(self): return self.t

    # real-world fingerprint drift, raw and post-EMA (calibration numbers)
    for tau_probe in (0.0, 1.5, 3.0):
        clk = Clock()
        probe = PlaceMemory(fp_ema_tau_s=tau_probe, time_fn=clk)
        fpp = []
        for t, s in zip(ts, s72):
            clk.t = float(t)
            probe.update(s)
            fpp.append(probe._fp_ema.copy() if probe._fp_ema is not None
                       else probe.fingerprint(s))
        fpp = np.stack(fpp)
        step = np.linalg.norm(np.diff(fpp, axis=0), axis=1)
        span = np.linalg.norm(fpp.max(0) - fpp.min(0))
        print(f"fp drift/tick @tau={tau_probe}: median {np.median(step):.5f} "
              f"p90 {np.percentile(step, 90):.5f} max {step.max():.4f} "
              f"(whole-bag span {span:.3f})")

    rows = []
    for thresh in args.thresh:
        for tau in args.tau:
            for shape in args.shape:
                for blend in args.blend:
                    for gate in args.gate:
                        clk = Clock()
                        pm = PlaceMemory(match_thresh=thresh,
                                         shape_weight=shape,
                                         fp_ema_tau_s=tau, slot_blend=blend,
                                         create_drift_gate=gate, time_fn=clk)
                        novs = []
                        for t, s in zip(ts, s72):
                            clk.t = float(t)
                            novs.append(pm.update(s))
                        novs = np.asarray(novs)
                        rows.append({
                            "thresh": thresh, "tau": tau, "shape": shape,
                            "blend": blend, "gate": gate,
                            "places": pm.n_places(),
                            "nov_mean_late": round(float(novs[60:].mean()), 3),
                            "nov_max_late": round(float(novs[60:].max()), 3),
                            "nov_last10": round(float(novs[-10:].mean()), 3)})
    hdr = (f"{'thresh':>6} {'tau':>5} {'shape':>5} {'blend':>5} {'gate':>6} "
           f"{'places':>6} {'nov_ml':>6} {'nov_mx':>6} {'nov_l10':>7}")
    print(f"\n{args.bag} (stationary wants places<=1 & nov_ml<0.1; "
          f"tour wants places~rooms):")
    print(hdr)
    for r in sorted(rows, key=lambda r: (r["places"], r["nov_mean_late"])):
        print(f"{r['thresh']:>6} {r['tau']:>5} {r['shape']:>5} "
              f"{r['blend']:>5} {r['gate']:>6} {r['places']:>6} "
              f"{r['nov_mean_late']:>6} {r['nov_max_late']:>6} "
              f"{r['nov_last10']:>7}")
    if "stationary" in args.bag:
        print("\nSTATIONARY VERDICT: " + (
            "PASS" if all(r["places"] <= 1 and r["nov_last10"] < 0.05
                          for r in rows) else
            "config-dependent — pick rows with places==1 and nov_l10<0.05"))


if __name__ == "__main__":
    main()
