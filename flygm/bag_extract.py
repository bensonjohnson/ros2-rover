#!/usr/bin/env python3
"""Extract the FlyGM observation stream from a recorded rover bag -> npz.

Runs ON THE ROVER (needs rosbag2_py + rclpy + installed tractor_bringup).
Stores RAW descriptors + timestamps; EMA/tick-alignment happens in the
Spark-side gate, replicating the runner's exact query-side pipeline:
  - lidar fingerprint s72 per scan (preprocess_scan)
  - visual fingerprint 23-d per image (visual_fingerprint on the ROS msg)
  - the gate then: freshest frame per lidar tick (2 s stale -> lidar-only),
    VisFingerprintEMA(tau=6.0), fused d = lidar_d + vis_weight * vis_d

  cd ~/ros2-rover && source /opt/ros/jazzy/setup.bash && source install/setup.bash
  python3 flygm/bag_extract.py --bag /tmp/scan_stationary_r2 --out /tmp/flygm_obs.npz
"""
import argparse
import sys
import time

sys.path[:0] = [".", "src/tractor_bringup"]

import numpy as np
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, LaserScan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bins", type=int, default=72)
    ap.add_argument("--max-range", type=float, default=12.0)
    args = ap.parse_args()

    from tractor_bringup.active_inference.scan_preprocess import preprocess_scan
    from tractor_bringup.active_inference.vis_fingerprint import visual_fingerprint

    reader = SequentialReader()
    reader.open(StorageOptions(uri=args.bag, storage_id="mcap"),
                ConverterOptions("", ""))
    topics = {t.name: t.type for t in reader.get_all_topics_and_types()}
    print("bag topics:", topics)
    want_scan, want_img = "/scan" in topics, "/camera/image_raw" in topics

    meta, ts_scan, raw_scans = None, [], []
    ts_img, vis_fps, img_enc = [], [], []
    t0 = time.time()
    while reader.has_next():
        topic, data, t = reader.read_next()
        if want_scan and topic == "/scan":
            msg = deserialize_message(data, LaserScan)
            if meta is None:
                meta = dict(angle_min=float(msg.angle_min),
                            angle_increment=float(msg.angle_increment),
                            range_max=float(msg.range_max),
                            n=len(msg.ranges))
            ts_scan.append(t * 1e-9)
            raw_scans.append(np.asarray(msg.ranges, dtype=np.float64))
        elif want_img and topic == "/camera/image_raw":
            msg = deserialize_message(data, Image)
            ts_img.append(t * 1e-9)
            img_enc.append(msg.encoding)
            vis_fps.append(visual_fingerprint(msg))
    print(f"[{time.time()-t0:4.1f}s] read: {len(raw_scans)} scans, "
          f"{len(vis_fps)} images ({set(img_enc) if img_enc else '-'})")

    t_base = ts_scan[0] if ts_scan else ts_img[0]
    ts_scan = np.asarray(ts_scan) - t_base
    ts_img = np.asarray(ts_img) - t_base
    fps = np.stack([preprocess_scan(r, meta["angle_min"],
                                    meta["angle_increment"],
                                    num_bins=args.bins,
                                    max_range=args.max_range)
                    for r in raw_scans]).astype(np.float32)
    vis = (np.stack(vis_fps).astype(np.float32) if vis_fps
           else np.zeros((0, 1), np.float32))
    np.savez(args.out, ts_scan=ts_scan, ts_img=ts_img, fps=fps, vis_fps=vis,
             meta=np.array([(meta or {}).get("range_max", 0.0)]))
    hz_s = 1 / np.median(np.diff(ts_scan)) if len(ts_scan) > 2 else 0.0
    hz_i = (1 / np.median(np.diff(ts_img)) if len(ts_img) > 2 else 0.0)
    print(f"[{time.time()-t0:4.1f}s] saved {args.out}: scan {fps.shape} "
          f"@ {hz_s:.2f} Hz, vis {vis.shape} @ {hz_i:.2f} Hz")


if __name__ == "__main__":
    main()
