#!/usr/bin/env python3
"""Replay a /scan + /camera/image_raw rosbag2 through the FUSED place memory.

The visual twin of bag_place_replay.py: it answers, on REAL house data, the
three questions synthetic rooms cannot:

  1. STABILITY — how much does the visual fingerprint drift on a STILL view
     (AE hunting, sensor noise)? A vis fp whose stationary jitter approaches
     the lidar room-to-room gap (~0.15-0.25) must never steer place matching.
  2. SEPARATION — do visually-different rooms separate BEYOND that jitter?
     (stationary bag vs tour bag, plus per-segment distances)
  3. WEIGHT — at what place_vis_weight does the fused replay give the right
     place counts? Row vis_weight=0.0 must reproduce the lidar-only baseline
     EXACTLY (regression guard on the fusion arithmetic).

The validated lidar temperament (thresh 0.15 / fp_ema_tau 6 / blend 0 /
drift gate 0.004) is pinned; only the visual weight and vis EMA tau sweep.

Record bags on the rover (NO --max-rate in jazzy; images are ~0.6 MB @ 30 fps
over USB — cap with rclpy QoS depth 1 subscribers or record at low res):
  ros2 bag record -s mcap -o /tmp/vis_stationary /scan /camera/image_raw
  ros2 bag record -s mcap -o /tmp/vis_tour        /scan /camera/image_raw

On rover:
  cd ~/ros2-rover && source /opt/ros/jazzy/setup.bash
  python3 pnn_sim/tools/bag_vis_replay.py --bag /tmp/vis_stationary
"""
from __future__ import annotations

import argparse
import sys

sys.path[:0] = ['.', 'src/tractor_bringup']

import numpy as np
from rosbag2_py import (ConverterOptions, SequentialReader, StorageOptions)
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, LaserScan

from tractor_bringup.active_inference.place_memory import PlaceMemory
from tractor_bringup.active_inference.scan_preprocess import preprocess_scan
from tractor_bringup.active_inference.vis_fingerprint import (
    VisFingerprintEMA, visual_fingerprint)

# field-validated lidar temperament (2026-09 joint acceptance) — pinned here;
# this tool calibrates ONLY the visual channel on top of it.
LIDAR = dict(match_thresh=0.15, shape_weight=2.0, fp_ema_tau_s=6.0,
             slot_blend=0.0, create_drift_gate=0.004)


def load_bag(bag: str):
    """-> (scans [(t, LaserScan)], frames [(t, Image)]) ordered by bag time."""
    reader = SequentialReader()
    reader.open(StorageOptions(uri=bag, storage_id="mcap"), ConverterOptions())
    reader.reset_filter()
    scans, frames = [], []
    while reader.has_next():
        topic, data, _seq = reader.read_next()
        if topic.endswith("/scan"):
            m = deserialize_message(data, LaserScan)
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            scans.append((t, m))
        elif topic.endswith("/image_raw"):
            m = deserialize_message(data, Image)
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            frames.append((t, m))
    assert len(scans) > 20, "bag has too few /scan msgs"
    assert len(frames) > 20, "bag has too few /camera/image_raw msgs"
    t0 = scans[0][0]
    scans = [(t - t0, m) for t, m in scans]
    frames = [(t - t0, m) for t, m in frames]
    return scans, frames


def pair_frames(scans, frames, max_age_s: float):
    """For each scan tick, the freshest frame not older than max_age_s
    (exactly the runner's staleness rule), or None."""
    out = []
    j = 0
    for t, _ in scans:
        while j + 1 < len(frames) and frames[j + 1][0] <= t:
            j += 1
        ft, fm = frames[j] if frames and frames[j][0] <= t else (None, None)
        out.append(fm if (fm is not None and t - ft <= max_age_s) else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--bins", type=int, default=72)
    ap.add_argument("--max-range", type=float, default=5.0)
    ap.add_argument("--camera-max-age", type=float, default=2.0)
    ap.add_argument("--vis-tau", type=float, nargs="*", default=(0.0, 1.0, 2.0))
    ap.add_argument("--vis-weight", type=float, nargs="*",
                    default=(0.0, 0.1, 0.2, 0.4, 0.8))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    scans, frames = load_bag(args.bag)
    S = len(scans)
    dt_med = float(np.median(np.diff([t for t, _ in scans])))
    meta = dict(angle_min=scans[0][1].angle_min,
                angle_increment=scans[0][1].angle_increment)
    print(f"{S} scans ({1/dt_med:.1f} Hz), {len(frames)} frames "
          f"({len(frames)/max(scans[-1][0],1e-6):.1f} fps avg), "
          f"{frames[0][1].width}x{frames[0][1].height} {frames[0][1].encoding}")

    paired = pair_frames(scans, frames, args.camera_max_age)
    n_paired = sum(p is not None for p in paired)
    print(f"paired a frame to {n_paired}/{S} scan ticks "
          f"(>{100*(1-n_paired/S):.0f}% unpaired => camera too slow/stale)")

    raw_fps = [preprocess_scan(np.asarray(m.ranges, dtype=np.float32),
                               meta["angle_min"], meta["angle_increment"],
                               num_bins=args.bins, max_range=args.max_range)
               for _, m in scans]

    # --- 1) stationarity of the visual fingerprint, raw and post-EMA --------
    for tau in args.vis_tau:
        ema = VisFingerprintEMA(tau_s=tau)
        fps = []
        t = 0.0
        for (st, _), fm in zip(scans, paired):
            t = st
            if fm is None:
                continue
            fps.append(ema.update(visual_fingerprint(fm), t))
        fps = np.stack(fps)
        step = np.linalg.norm(np.diff(fps, axis=0), axis=1)
        print(f"vis fp drift/tick @tau={tau}: median {np.median(step):.5f} "
              f"p90 {np.percentile(step, 90):.5f} max {step.max():.4f} "
              f"(span {np.linalg.norm(fps.max(0)-fps.min(0)):.3f})")

    # --- 2) fused replay sweep ---------------------------------------------
    class Clock:
        def __init__(self): self.t = 0.0

    rows = []
    base_places = None
    for vis_tau in args.vis_tau:
        for w in args.vis_weight:
            clk = Clock()
            ema = VisFingerprintEMA(tau_s=vis_tau) if w > 0 else None
            pm = PlaceMemory(vis_weight=w, time_fn=clk, **LIDAR)
            novs = []
            for (st, _), fm, s in zip(scans, paired, raw_fps):
                clk.t = st
                vf = None
                if ema is not None and fm is not None:
                    vf = ema.update(visual_fingerprint(fm), st)
                novs.append(pm.update(s, vis_fp=vf))
            novs = np.asarray(novs)
            rows.append({"vis_tau": vis_tau, "w": w,
                         "places": pm.n_places(),
                         "nov_mean_late": round(float(novs[60:].mean()), 3),
                         "nov_last10": round(float(novs[-10:].mean()), 3)})
            if w == 0.0:
                if base_places is None:
                    base_places = (pm.n_places(), rows[-1]["nov_mean_late"])
                elif (rows[-1]["places"], rows[-1]["nov_mean_late"]) != base_places:
                    # lidar-only rows must all agree (vis channel inert)
                    print("  NOTE: vis_tau affects w=0 rows — vis EMA leaked "
                          "into the lidar path (bug)")

    hdr = (f"{'vis_tau':>7} {'w':>5} {'places':>6} {'nov_ml':>6} {'nov_l10':>7}")
    print(f"\n{args.bag}:")
    print(hdr)
    for r in sorted(rows, key=lambda r: (r["vis_tau"], r["w"])):
        mark = "  <- lidar-only baseline" if r["w"] == 0.0 else ""
        print(f"{r['vis_tau']:>7} {r['w']:>5} {r['places']:>6} "
              f"{r['nov_mean_late']:>6} {r['nov_last10']:>7}{mark}")

    name = args.bag.lower()
    base_p = base_places[0] if base_places else 0
    if "stationary" in name:
        good = [r for r in rows if r["places"] <= 1 and r["nov_last10"] < 0.05]
        best_w = max((r["w"] for r in good), default=None)
        print("\nSTATIONARY VERDICT: visual weight tolerable up to "
              + (f"w={best_w} (largest w keeping places<=1 & nov_l10<0.05)"
                 if best_w else "NONE — vis channel destabilizes places; "
                                 "raise vis tau or keep w=0"))
        print(f"baseline (lidar-only) places={base_p}")
    elif "tour" in name:
        gains = [(r["w"], r["places"]) for r in rows if r["w"] > 0
                 and r["places"] > base_p]
        print("\nTOUR: lidar-only sees", base_p, "places;",
              "weights that ADD places:", gains if gains else
              "none — vision adds no resolution on this bag (do not enable)")
        print("Compare against tour_analyze.py's k-center ceiling: added "
              "places only count if they approach it WITHOUT phantoms "
              "(check nov_l10 stays low at dwell stops).")


if __name__ == "__main__":
    main()
