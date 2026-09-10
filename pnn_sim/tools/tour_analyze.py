#!/usr/bin/env python3
"""Tour-bag analysis: segment a carried/teleoped house tour into room dwells
and measure REAL inter-room fingerprint distances.

The decision-critical quantity: place formation happens on the ROLLING
MEDIAN fingerprint (transient walking drift spikes dmin for ~1-2 s then
relaxes; a room-to-room jump is permanent). This tool computes post-EMA
fingerprints, smooths them (median window), clusters with temporal
contiguity, and reports the distance spectrum: stationary spread (lower
bound on match_thresh) vs minimum inter-room distance (upper bound).

    python3 pnn_sim/tools/tour_analyze.py --bag /tmp/scan_tour
"""
from __future__ import annotations

import argparse
import sys

sys.path[:0] = ['.', 'src/tractor_bringup']

import numpy as np
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan

from tractor_bringup.active_inference.place_memory import PlaceMemory
from tractor_bringup.active_inference.scan_preprocess import preprocess_scan


def load(bag: str):
    reader = SequentialReader()
    reader.open(StorageOptions(uri=bag, storage_id="mcap"), ConverterOptions())
    reader.reset_filter()
    ts, raw = [], []
    while reader.has_next():
        _t, data, _s = reader.read_next()
        m = deserialize_message(data, LaserScan)
        ts.append(m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
        raw.append(np.asarray(m.ranges, dtype=np.float64))
    meta = dict(angle_min=m.angle_min, angle_increment=m.angle_increment)
    return np.asarray(ts) - ts[0], raw, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--shape", type=float, default=1.0)
    ap.add_argument("--win", type=int, default=31, help="median window, ticks")
    ap.add_argument("--max-range", type=float, default=12.0)
    args = ap.parse_args()

    ts, raw, meta = load(args.bag)
    S = len(raw)
    print(f"{S} scans, span {ts[-1]:.0f} s")
    s72 = [preprocess_scan(r, meta["angle_min"], meta["angle_increment"],
                           num_bins=72, max_range=args.max_range)
           for r in raw]

    # post-EMA fingerprints (tau s), then rolling-median to kill walking
    # transients (carry jitter persists ~1-2 s; a room jump is permanent)
    class Clock:
        t = 0.0
        def __call__(self): return self.t
    clk = Clock()
    pm = PlaceMemory(fp_ema_tau_s=args.tau, shape_weight=args.shape,
                     time_fn=clk)
    print(f"--- tau={args.tau} shape={args.shape} ---")
    fps = []
    for t, s in zip(ts, s72):
        clk.t = float(t)
        pm.update(s)
        fps.append(pm._fp_ema.copy())
    fp = np.stack(fps)                                   # [S, 10]

    w = args.win | 1
    half = w // 2
    med = np.stack([np.median(fp[max(0, i-half):i+half+1], axis=0)
                    for i in range(S)])                   # rolling median

    step = np.linalg.norm(np.diff(med, axis=0), axis=1)
    print(f"post-EMA+median step/tick: median {np.median(step):.5f} "
          f"p90 {np.percentile(step,90):.5f} max {step.max():.4f}")

    # stationary/dwell spread: per-tick deviation from the rolling median
    dev = np.linalg.norm(fp - med, axis=1)
    print(f"deviation fp vs rolling median: p50 {np.percentile(dev,50):.4f} "
          f"p95 {np.percentile(dev,95):.4f} p99 {np.percentile(dev,99):.4f}")

    # temporal-contiguity k-center on the median-smoothed stream:
    # walk forward; new place when > thresh from ALL accepted refs
    def kcenter_temporal(F, thresh):
        refs = [0]
        for i in range(1, len(F)):
            d = np.linalg.norm(F[i] - F[np.array(refs)], axis=1).min()
            if d > thresh:
                refs.append(i)
        return refs

    # distance spectrum: for the sequence of first-creation gaps at a fine
    # ladder of thresholds, find the jump (noise floor -> room gaps)
    print("\nthresh -> places created (temporal k-center on median fp):")
    counts = {}
    for t_ in (0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30,
              0.35, 0.45, 0.60):
        refs = kcenter_temporal(med, t_)
        counts[t_] = len(refs)
        print(f"  {t_:>5} -> {len(refs)}")

    # representative refs at a mid threshold: pairwise inter-room distances
    refs = kcenter_temporal(med, 0.15)
    R = med[np.array(refs)]
    D = np.linalg.norm(R[:, None] - R[None, :], axis=2)
    iu = np.triu_indices(len(refs), 1)
    if len(refs) > 1:
        print(f"\nrefs@0.15: {len(refs)}   pairwise inter-ref distances: "
              f"min {D[iu].min():.3f} median {np.median(D[iu]):.3f} "
              f"max {D[iu].max():.3f}")
    # revisit check: distance of stream-end segment to first room
    d_first = np.linalg.norm(med[-20:] - med[refs[0]], axis=1)
    print(f"last 20 ticks vs FIRST room ref: median "
          f"{np.median(d_first):.3f} (low => revisit recognized)")


if __name__ == "__main__":
    main()
