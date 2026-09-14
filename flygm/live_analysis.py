#!/usr/bin/env python3
"""Analyze a flygm_livetest bag: what did the connectome actually command?

Run on the rover:
  cd ~/ros2-rover && source /opt/ros/jazzy/setup.bash && source install/setup.bash
  python3 flygm/live_analysis.py --bag /tmp/flygm_livetest
"""
import argparse
import sys

sys.path[:0] = [".", "src/tractor_bringup"]

import numpy as np
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    args = ap.parse_args()

    reader = SequentialReader()
    reader.open(StorageOptions(uri=args.bag, storage_id="mcap"),
                ConverterOptions("", ""))
    t0 = None
    scans, cmds, diags = {}, [], []
    while reader.has_next():
        topic, data, t = reader.read_next()
        t = t * 1e-9
        if t0 is None:
            t0 = t
        t -= t0
        if topic == "/scan":
            m = deserialize_message(data, LaserScan)
            r = np.asarray(m.ranges, dtype=np.float64)
            valid = r[(r > 0.05) & (r < 12.0)]
            scans[t] = float(valid.min()) if len(valid) else 12.0
        elif topic == "/track_cmd_ai":
            m = deserialize_message(data, Float32MultiArray)
            cmds.append((t, float(m.data[0]), float(m.data[1])))
        elif topic == "/flygm/diagnostics":
            m = deserialize_message(data, Float32MultiArray)
            diags.append((t, list(m.data)))

    if not cmds:
        print("no /track_cmd_ai in bag")
        return
    c = np.asarray([(x[1], x[2]) for x in cmds])
    ct = np.asarray([x[0] for x in cmds])
    modes = {}
    for _, d in diags:
        modes[int(d[3])] = modes.get(int(d[3]), 0) + 1

    print(f"bag: {len(scans)} scans, {len(cmds)} cmd msgs, "
          f"diag modes {modes} (0=fly,1=teleop,2=stale,3=hardstop)")
    live = c[np.abs(c).sum(axis=1) > 1e-6]
    print(f"nonzero commands: {len(live)}/{len(c)}")
    if len(live):
        print(f"  L: mean {live[:,0].mean():+.3f} std {live[:,0].std():.3f} "
              f"min {live[:,0].min():+.3f} max {live[:,0].max():+.3f}")
        print(f"  R: mean {live[:,1].mean():+.3f} std {live[:,1].std():.3f} "
              f"min {live[:,1].min():+.3f} max {live[:,1].max():+.3f}")
        v = 0.5 * (c[:, 0] + c[:, 1])
        w = (c[:, 1] - c[:, 0]) / 2
        print(f"  v: mean {v.mean():+.3f} | w: mean {w.mean():+.3f} "
              f"std {w.std():.3f}")
        flip = int((np.abs(np.diff(np.sign(v[~np.isnan(v)]))) > 1.5).sum())
        print(f"  forward/reverse flips: {flip}")

    # reactivity: |dcmd| vs |dscan-min| (world-lockedness, live edition)
    ts = np.asarray(sorted(scans))
    mn = np.asarray([scans[t] for t in ts])
    dmn = np.abs(np.diff(mn))
    dc = np.linalg.norm(np.diff(c, axis=0), axis=1)
    n = min(len(dc), len(dmn))
    if n > 30:
        cc = np.corrcoef(dc[:n], dmn[:n])[0, 1]
        print(f"corr(|dcmd|, |dmin_scan|): {cc:+.3f} over {n} samples")

    # proximity-stop exposure: how often was the rover within stop_distance?
    near = float((mn < 0.15).mean()) if len(mn) else 0.0
    print(f"fraction of scans with min range < 0.15 m: {near:.1%}")


if __name__ == "__main__":
    main()
