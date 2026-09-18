#!/usr/bin/env python3
"""Run an evolved genome (evo/ recurrent MLP) on the real rover.

    python3 evo_runner.py --genome champ.npz            # DRY RUN (default)
    python3 evo_runner.py --genome champ.npz --drive --action-scale 0.6

Observation = exactly the sim's (evo.arena._rollout_body):
    72 scan bins   preprocess_scan(ranges, angle_min, angle_increment)
                   (min-pooled, /5 m, 1.0 = open) — same function family the
                   PC brain uses on this rover
    8 proprio      [wheel_l, wheel_r, gyro_x, gyro_y, yaw_rate, ax, ay, az],
                   0.5 + 0.5 * v / scale: wheels /8 rad/s, rates /2.5 rad/s,
                   accel /19.6 m/s^2 (sim channels 2/3 are noise ~0.5; the
                   real roll/pitch rates sit near 0.5 too).
                   Wheels default to the SIM's definition (--wheel-source
                   model): the gated track command through the sim motor
                   model (v_max 0.2 m/s, left trim 0.8, 0.15 s first-order
                   lag) / 0.025 m radius. First live test: the right
                   encoder read 0.0 while the gyro proved the track was
                   reversing, and the left pinned at 8 rad/s.
    2 last action  the GATED command in policy space ([-1, 1], i.e. the
                   post-safety-monitor /track_cmd divided by action_scale)

Output: tanh [L, R] in [-1, 1] -> x action_scale -> Float32MultiArray on
/track_cmd_ai (only with --drive) -> lidar_safety_monitor (same parameters as
the sim gate) -> /track_cmd -> hiwonder_motor_driver. Without --drive the
command goes to /evo/track_cmd_dry and nothing moves.

Control rate 15 Hz = the sim's CONTROL_HZ; the latest scan is reused between
lidar revolutions (~10 Hz on the LD19), as the PC brain does.
Stop: Ctrl-C (publishes zeros), the rover-control HARDSTOP, or the motor
driver's 0.5 s command watchdog if this process dies.
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np

NUM_BINS, MAX_RANGE = 72, 5.0
OBS_DIM = NUM_BINS + 8 + 2


def preprocess_scan(ranges, angle_min, angle_increment, num_bins=NUM_BINS,
                    max_range=MAX_RANGE, min_range=0.05):
    r = np.asarray(ranges, dtype=np.float32)
    clean = r.copy()
    bad = ~np.isfinite(clean) | (clean <= 0.0) | (clean < min_range)
    clean[bad] = max_range
    np.clip(clean, min_range, max_range, out=clean)
    ang = angle_min + np.arange(r.shape[0], dtype=np.float32) * angle_increment
    frac = np.mod(ang, 2.0 * np.pi) / (2.0 * np.pi)
    bins = np.minimum((frac * num_bins).astype(np.int64), num_bins - 1)
    out = np.full(num_bins, max_range, dtype=np.float32)
    np.minimum.at(out, bins, clean)
    return out / max_range


class Genome:
    """numpy twin of evo.policy.PopulationNet for ONE individual
    (row-vector convention, same packing order Wx, bh, Wh, bo, Wo)."""

    def __init__(self, path):
        d = np.load(path)
        th = np.asarray(d["thetas"], dtype=np.float32).reshape(-1)
        self.hidden = H = int(d["hidden"])
        self.obs_mode = str(d["obs_mode"]) if "obs_mode" in d else "v1"
        self.action_mode = str(d["action_mode"]) if "action_mode" in d else "lr"
        if self.obs_mode != "v1" or self.action_mode != "lr":
            raise SystemExit("runner supports obs v1 / action lr genomes only")
        shapes = [("Wx", (OBS_DIM, H)), ("bh", (H,)), ("Wh", (H, H)),
                  ("bo", (2,)), ("Wo", (H, 2))]
        off, self.w = 0, {}
        for name, shp in shapes:
            n = int(np.prod(shp))
            self.w[name] = th[off:off + n].reshape(shp)
            off += n
        assert off == th.size, f"genome size {th.size} != expected {off}"
        self.h = np.zeros(H, dtype=np.float32)

    def reset(self):
        self.h[:] = 0.0

    def step(self, obs):
        w = self.w
        self.h = np.tanh(obs @ w["Wx"] + self.h @ w["Wh"] + w["bh"])
        return np.tanh(self.h @ w["Wo"] + w["bo"])


def norm01(v, scale):
    return float(np.clip(0.5 + 0.5 * v / scale, 0.0, 1.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genome", required=True)
    ap.add_argument("--drive", action="store_true",
                    help="publish to /track_cmd_ai (moves the rover); "
                         "default is a dry run on /evo/track_cmd_dry")
    ap.add_argument("--action-scale", type=float, default=0.6,
                    help="track command multiplier (sim trained at 1.0)")
    ap.add_argument("--rate", type=float, default=15.0)
    ap.add_argument("--duration", type=float, default=0.0,
                    help="stop after N seconds (0 = until Ctrl-C)")
    ap.add_argument("--wheel-source", choices=("model", "encoder"),
                    default="model")
    ap.add_argument("--log", default="",
                    help="npz path to save (obs, action) per tick")
    args = ap.parse_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Imu, JointState, LaserScan
    from std_msgs.msg import Bool, Float32MultiArray

    net = Genome(args.genome)
    scale = max(1e-3, float(args.action_scale))

    class Runner(Node):
        def __init__(self):
            super().__init__("evo_runner")
            self.scan = None
            self.scan_t = 0.0
            self.wl = self.wr = 0.0
            self.gyro = (0.0, 0.0, 0.0)
            self.acc = (0.0, 0.0, 9.81)
            self.gated = np.zeros(2, np.float32)   # policy space
            self.gated_raw = np.zeros(2, np.float32)
            self.vtrack = np.zeros(2, np.float32)  # sim motor model state
            self.estop = False
            self.create_subscription(LaserScan, "/scan", self.on_scan,
                                     qos_profile_sensor_data)
            self.create_subscription(JointState, "/joint_states",
                                     self.on_joints, 10)
            self.create_subscription(Imu, "/imu/data", self.on_imu,
                                     qos_profile_sensor_data)
            self.create_subscription(Float32MultiArray, "/track_cmd",
                                     self.on_gated, 10)
            self.create_subscription(Bool, "/emergency_stop",
                                     self.on_estop, 10)
            topic = "/track_cmd_ai" if args.drive else "/evo/track_cmd_dry"
            self.pub = self.create_publisher(Float32MultiArray, topic, 10)
            self.t0 = time.monotonic()
            self.ticks = 0
            self.rec_obs, self.rec_act, self.rec_enc = [], [], []
            self.last_cmd_raw = np.zeros(2, np.float32)
            self.timer = self.create_timer(1.0 / args.rate, self.tick)
            self.get_logger().info(
                f"evo runner: genome {args.genome} h{net.hidden} -> {topic} "
                f"(scale {scale}, {'DRIVE' if args.drive else 'DRY RUN'})")

        def on_scan(self, m):
            self.scan = preprocess_scan(m.ranges, m.angle_min,
                                        m.angle_increment)
            self.scan_t = time.monotonic()

        def on_joints(self, m):
            try:
                li = m.name.index("left_viz_wheel_joint")
                ri = m.name.index("right_viz_wheel_joint")
            except ValueError:
                return
            if len(m.velocity) > max(li, ri):
                self.wl, self.wr = float(m.velocity[li]), float(m.velocity[ri])

        def on_imu(self, m):
            w, a = m.angular_velocity, m.linear_acceleration
            self.gyro = (float(w.x), float(w.y), float(w.z))
            self.acc = (float(a.x), float(a.y), float(a.z))

        def on_gated(self, m):
            if len(m.data) >= 2:
                self.gated_raw = np.asarray(m.data[:2], np.float32)
                self.gated = np.clip(self.gated_raw / scale, -1.0, 1.0)

        def on_estop(self, m):
            self.estop = bool(m.data)

        def publish(self, lr):
            msg = Float32MultiArray()
            msg.data = [float(lr[0]), float(lr[1])]
            self.pub.publish(msg)

        def tick(self):
            now = time.monotonic()
            if args.duration and now - self.t0 > args.duration:
                self.publish((0.0, 0.0))
                raise SystemExit
            if self.scan is None or now - self.scan_t > 0.5:
                self.publish((0.0, 0.0))          # no fresh lidar: hold still
                if self.ticks % 15 == 0:
                    self.get_logger().warn("waiting for fresh /scan")
                self.ticks += 1
                return
            prev = self.gated if args.drive else getattr(
                self, "last_cmd", np.zeros(2, np.float32))
            if args.wheel_source == "model":
                # sim BatchedEnv.step: deadband 0.05, v_max 0.2, trims
                # (0.8, 1.0), first-order lag tau 0.15 s at dt = 1/rate
                cmd = self.gated_raw if args.drive else self.last_cmd_raw
                tgt = np.where(np.abs(cmd) < 0.05, 0.0,
                               cmd * 0.2 * np.array([0.8, 1.0], np.float32))
                k = 1.0 - math.exp(-(1.0 / args.rate) / 0.15)
                self.vtrack += (tgt - self.vtrack) * k
                wl, wr = self.vtrack / 0.025
            else:
                wl, wr = self.wl, self.wr
            prop = np.array([
                norm01(wl, 8.0), norm01(wr, 8.0),
                norm01(self.gyro[0], 2.5), norm01(self.gyro[1], 2.5),
                norm01(self.gyro[2], 2.5),
                norm01(self.acc[0], 19.6), norm01(self.acc[1], 19.6),
                norm01(self.acc[2], 19.6)], np.float32)
            obs = np.concatenate([self.scan, prop, prev]).astype(np.float32)
            a = np.clip(net.step(obs), -1.0, 1.0)
            self.last_cmd = a.astype(np.float32)
            out = a * scale
            self.last_cmd_raw = out.astype(np.float32)
            # /emergency_stop = FRONT BLOCKED only; the safety monitor and
            # motor driver clamp forward motion on it and still allow
            # reverse — exactly the sim gate. Do NOT zero here (first live
            # test: zeroing on estop blocked the genome's reverse escape).
            self.publish(out)
            if args.log:
                self.rec_obs.append(obs)
                self.rec_act.append(a)
                self.rec_enc.append((self.wl, self.wr, *self.gated_raw,
                                     float(self.estop)))
            if self.ticks % 15 == 0:
                s = self.scan
                front = float(min(s[:4].min(), s[-4:].min())) * MAX_RANGE
                self.get_logger().info(
                    f"t={now - self.t0:5.1f}s front={front:4.2f}m "
                    f"L={s[9:27].mean() * MAX_RANGE:4.2f} "
                    f"R={s[45:63].mean() * MAX_RANGE:4.2f} "
                    f"cmd=[{a[0]:+.2f},{a[1]:+.2f}] gated=[{self.gated[0]:+.2f},"
                    f"{self.gated[1]:+.2f}] wheels=[{self.wl:+.1f},{self.wr:+.1f}]"
                    f" yaw={self.gyro[2]:+.2f} az={self.acc[2]:+.1f}"
                    f"{' ESTOP' if self.estop else ''}")
            self.ticks += 1

    rclpy.init()
    node = Runner()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        for _ in range(5):
            node.publish((0.0, 0.0))
            time.sleep(0.02)
        if args.log and node.rec_obs:
            np.savez(args.log, obs=np.stack(node.rec_obs),
                     act=np.stack(node.rec_act),
                     enc=np.asarray(node.rec_enc, np.float32))
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
