#!/usr/bin/env python3
"""Spatial explorer — local occupancy memory + frontier seeking on the rover.

    /scan (LaserScan) ─┐
                       ├─> driver ─> track command
    /odometry/filtered ┘     │
                             v
                  /track_cmd_ai (Float32MultiArray [L, R])
                             │  lidar_safety_monitor gates it
                             v
                       /track_cmd  -> motor driver

Two drivers, selected by the `driver` parameter:

  "frontier" (DEFAULT) — FrontierExplorer: a rolling local occupancy grid in
      the odom frame, frontier = free cell bordering unknown, BFS to the
      nearest one, pure-pursuit with a committed escape pivot when the safety
      gate blocks. Validated in sim against ground-truth map rooms: ~1.40
      rooms/house and 14 m travelled, versus 1.00 rooms and ZERO doorway
      crossings for every reactive-on-scan controller tried. Drift-tolerant
      (1.40 -> 1.30 at 5 deg/s yaw drift) because the target is always a
      nearby frontier reached in seconds.

  "pnn" — the research path: PCSpatialMap (predictive-coding occupancy map)
      read by a delta-rule PCPolicy distilled from the frontier teacher.
      DO NOT DEPLOY without passing `python3 -m pnn_sim.spatial.check_policy`.
      As of 2026-07-25 no checkpoint passes: the RFF-over-egocentric-patch
      representation does not generalise across houses. Least-squares — the
      ceiling for any delta rule on those features — scores MAE ratio 0.93 to
      1.03 against a best-constant baseline on unseen houses, while fitting
      in-distribution at 0.67. The distilled policies were statistically
      indistinguishable from a constant turn, and the coverage metric that
      "validated" them (student 0.94 of teacher) cannot tell the difference,
      because a constant turn plus the escape law wall-follows about as far as
      the weak teacher walks.

Either driver needs odometry: the memory is a metric grid in the odom frame.
Drift is tolerated by design; the map re-pins on an odom jump.
"""

from __future__ import annotations

import math
import os

import numpy as np
import rclpy
import torch
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32MultiArray

from tractor_bringup.spatial.frontier import FrontierExplorer
from tractor_bringup.spatial.pc_map import PCSpatialMap
from tractor_bringup.spatial.policy import PCPolicy, egocentric_patch, pursuit_cmd
from tractor_bringup.spatial.scan_beams import scan_to_beams


def _yaw_from_quat(q) -> float:
    """Yaw (rad) from a geometry_msgs Quaternion."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class SpatialExplorerRunner(Node):
    def __init__(self):
        super().__init__("spatial_explorer_runner")

        def p(name, default):
            self.declare_parameter(name, default)

        p("driver", "frontier")           # "frontier" (works) or "pnn"
        p("scan_topic", "/scan")
        p("odom_topic", "/odometry/filtered")
        p("track_cmd_topic", "/track_cmd_ai")
        p("estop_topic", "/emergency_stop")
        p("policy_path", os.path.expanduser("~/.ros/spatial_policy.pt"))
        p("control_rate_hz", 15.0)
        # 1.0 = the envelope the control law was tuned and validated under.
        # 0.6 (inherited from the old active-inference brain) cuts speed 40% on
        # a rover whose coverage is already speed-bound at ~0.126 m/s. The
        # safety monitor, not this scale, is what bounds risk.
        p("action_scale", 1.0)
        # PC map (must match the values the policy was distilled under)
        p("map_res", 0.15)
        p("map_size", 200)
        p("map_max_range", 5.0)
        p("beam_stride", 4)
        p("infer_iters", 3)
        p("infer_lr", 0.8)
        # Egocentric patch + policy (must match training)
        p("patch_half_m", 4.0)
        p("patch_cells", 32)
        p("n_feat", 1024)
        p("escape_ticks", 14)
        # Guards
        p("odom_jump_m", 1.0)
        p("sensor_timeout_s", 0.5)
        p("torch_threads", 2)

        g = self.get_parameter
        self.scan_topic = str(g("scan_topic").value)
        self.odom_topic = str(g("odom_topic").value)
        self.control_rate = float(g("control_rate_hz").value)
        self.action_scale = float(g("action_scale").value)
        self.map_max_range = float(g("map_max_range").value)
        self.map_res = float(g("map_res").value)
        self.patch_half_m = float(g("patch_half_m").value)
        self.patch_p = int(g("patch_cells").value)
        self.escape_ticks = int(g("escape_ticks").value)
        self.odom_jump_m = float(g("odom_jump_m").value)
        self.sensor_timeout = float(g("sensor_timeout_s").value)

        torch.set_num_threads(max(1, int(g("torch_threads").value)))
        torch.set_grad_enabled(False)

        # --- the driver ---------------------------------------------------
        self.driver = str(g("driver").value).lower()
        if self.driver not in ("frontier", "pnn"):
            raise ValueError(f"driver must be 'frontier' or 'pnn', got "
                             f"{self.driver!r}")
        self.pcmap = None
        self.policy = None
        self.frontier = None

        if self.driver == "frontier":
            self.frontier = FrontierExplorer(
                res=float(g("map_res").value),
                max_range=self.map_max_range,
                escape_ticks=self.escape_ticks)
            self.get_logger().info(
                "driver=frontier — local occupancy grid + BFS to nearest "
                "frontier (the validated explorer)")
        else:
            self.pcmap = PCSpatialMap(
                1,
                res=float(g("map_res").value),
                size=int(g("map_size").value),
                max_range=self.map_max_range,
                device="cpu",
                infer_lr=float(g("infer_lr").value),
                infer_iters=int(g("infer_iters").value),
                beam_stride=int(g("beam_stride").value),
            )
            in_dim = 3 * self.patch_p * self.patch_p
            self.policy = PCPolicy(in_dim, n_feat=int(g("n_feat").value),
                                   device="cpu")
            policy_path = str(g("policy_path").value)
            if not os.path.exists(policy_path):
                raise FileNotFoundError(
                    f"No distilled policy at {policy_path}. Train one with:\n"
                    "  python3 -m pnn_sim.spatial.distill_policy "
                    "--save spatial_policy.pt\n"
                    "then copy it to the rover's ~/.ros/.")
            self.policy.load(policy_path)
            self.get_logger().warning(
                f"driver=pnn — loaded {policy_path} "
                f"(|Wr|={self.policy.Wr.norm():.3f}). This is the RESEARCH "
                "path; verify it with pnn_sim.spatial.check_policy before "
                "trusting its behaviour.")

        # pursuit_cmd's committed-escape state (mutated in place each tick)
        self.escape = torch.zeros(1, dtype=torch.long)
        self.edir = torch.ones(1)

        # --- state -------------------------------------------------------
        self._ranges: torch.Tensor | None = None
        self._bearings: torch.Tensor | None = None
        self._scan_stamp = 0.0
        self._pose: tuple[float, float, float] | None = None
        self._odom_stamp = 0.0
        self._blocked = False
        self._step = 0
        self._bearing_cache: float | None = None

        # --- ROS ---------------------------------------------------------
        self.create_subscription(
            LaserScan, self.scan_topic, self._scan_cb, qos_profile_sensor_data)
        self.create_subscription(
            Odometry, self.odom_topic, self._odom_cb, qos_profile_sensor_data)
        self.create_subscription(
            Bool, str(g("estop_topic").value), self._estop_cb, 10)
        self.track_pub = self.create_publisher(
            Float32MultiArray, str(g("track_cmd_topic").value), 10)
        self.diag_pub = self.create_publisher(
            Float32MultiArray, "/pnn/spatial_diag", 10)

        self.create_timer(1.0 / self.control_rate, self._control_step)
        self.get_logger().info(
            f"spatial explorer up: {self.scan_topic} + {self.odom_topic} "
            f"-> {g('track_cmd_topic').value} @ {self.control_rate:.0f} Hz")

    # ---- callbacks -------------------------------------------------------
    def _scan_cb(self, msg: LaserScan):
        n = len(msg.ranges)
        if n == 0:
            return
        # Log the geometry ONCE, then only on a material change. The LD19
        # returns a different beam count AND a correspondingly different
        # angle_increment every revolution (502-505 beams measured), so any
        # exact-match cache — on the count or on the angles — re-fires on
        # every single scan and floods the log at 10 Hz.
        inc = msg.angle_increment
        prev = self._bearing_cache
        if prev is None or abs(inc - prev) > 0.05 * max(abs(prev), 1e-9):
            self._bearing_cache = inc
            self.get_logger().info(
                f"scan geometry: ~{n} beams, "
                f"[{math.degrees(msg.angle_min):.1f}deg, "
                f"{math.degrees(msg.angle_min + n * msg.angle_increment):.1f}deg], "
                f"increment {math.degrees(msg.angle_increment):.2f}deg")

        r, b = scan_to_beams(
            msg.ranges, msg.angle_min, msg.angle_increment,
            msg.range_min, msg.range_max, self.map_max_range)
        if r is None:
            return
        self._ranges = torch.from_numpy(r).view(1, -1)
        self._bearings = torch.from_numpy(b)
        self._scan_stamp = self._now()

    def _odom_cb(self, msg: Odometry):
        pp = msg.pose.pose
        self._pose = (pp.position.x, pp.position.y, _yaw_from_quat(pp.orientation))
        self._odom_stamp = self._now()

    def _estop_cb(self, msg: Bool):
        self._blocked = bool(msg.data)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ---- control ---------------------------------------------------------
    def _publish(self, left: float, right: float):
        m = Float32MultiArray()
        m.data = [float(left), float(right)]
        self.track_pub.publish(m)

    def _control_step(self):
        t0 = self._now()
        stale = (self._ranges is None or self._pose is None
                 or t0 - self._scan_stamp > self.sensor_timeout
                 or t0 - self._odom_stamp > self.sensor_timeout)
        if stale:
            self._publish(0.0, 0.0)
            if self._step % 30 == 0:
                self.get_logger().warning(
                    "waiting for /scan + odom (holding stop)",
                    throttle_duration_sec=5.0)
            self._step += 1
            return

        x, y, th = self._pose
        pos = torch.tensor([[x, y]], dtype=torch.float32)
        heading = torch.tensor([th], dtype=torch.float32)

        # An EKF reset or a lift teleports the pose; the memory is in the odom
        # frame, so fusing across a jump smears the house. Re-pin instead.
        jumped = False
        if getattr(self, "_last_pos", None) is not None:
            if float(torch.linalg.norm(pos - self._last_pos)) > self.odom_jump_m:
                jumped = True
                self.get_logger().warning("odom jump — resetting spatial memory")
                if self.pcmap is not None:
                    self.pcmap.reset(idx=torch.zeros(1, dtype=torch.long))
                else:
                    self.frontier = FrontierExplorer(
                        res=self.map_res, max_range=self.map_max_range,
                        escape_ticks=self.escape_ticks)
        self._last_pos = pos

        ranges, bearings = self._ranges, self._bearings
        rnp = ranges[0].numpy()
        bnp = bearings.numpy()

        if self.driver == "frontier":
            cmd_np, info = self.frontier.step(
                x, y, th, rnp.astype(np.float64), bnp.astype(np.float64),
                blocked=self._blocked)
            tb_val = info.get("target_bearing")
            exploring = bool(info.get("exploring", True))
            left, right = float(cmd_np[0]), float(cmd_np[1])
            # Grid telemetry: how much of the house it has mapped, and how
            # much unexplored edge is still reachable. Watching frontier fall
            # to 0 is how you see it decide the space is finished.
            fr, _ = self.frontier._masks()
            map_err = 0.0
            n_front = float(int(fr.sum()))
            n_seen = float(int(self.frontier.seen.sum()))
        else:
            self.pcmap.update(pos, heading, ranges, bearings)
            patch = egocentric_patch(
                self.pcmap, pos, heading, self.patch_half_m, self.patch_p)
            tb, _ = self.policy.predict(patch)
            blocked = torch.tensor([self._blocked], dtype=torch.bool)
            cmd = pursuit_cmd(tb, ranges, bearings, blocked,
                              (self.escape, self.edir), self.escape_ticks)
            tb_val = float(tb[0])
            exploring = True
            left, right = float(cmd[0, 0]), float(cmd[0, 1])
            map_err = float(self.pcmap.last_err)
            n_front = float(int(self.pcmap.frontier().sum()))
            n_seen = float(int(self.pcmap.seen().sum()))

        out_l = float(np.clip(left * self.action_scale, -1.0, 1.0))
        out_r = float(np.clip(right * self.action_scale, -1.0, 1.0))
        self._publish(out_l, out_r)

        tick_ms = (self._now() - t0) * 1e3
        diag = Float32MultiArray()
        diag.data = [float(tb_val if tb_val is not None else 0.0),
                     out_l, out_r, map_err, n_front, n_seen,
                     float(self._blocked), float(tick_ms),
                     float(exploring), float(jumped)]
        self.diag_pub.publish(diag)

        budget_ms = 1000.0 / self.control_rate
        if tick_ms > budget_ms:
            self.get_logger().warning(
                f"tick overran: {tick_ms:.0f}ms > {budget_ms:.0f}ms",
                throttle_duration_sec=5.0)
        if self._step % 30 == 0:
            bearing_s = ("--" if tb_val is None
                         else f"{math.degrees(tb_val):+.0f}deg")
            extra = (f"map_err={map_err:.3f} seen={int(n_seen)} "
                     f"frontier={int(n_front)} " if self.driver == "pnn"
                     else f"{'exploring' if exploring else 'SCANNING'} ")
            self.get_logger().info(
                f"step={self._step} bearing={bearing_s} "
                f"L={out_l:+.2f} R={out_r:+.2f} {extra}"
                f"{'[BLOCKED] ' if self._blocked else ''}{tick_ms:.0f}ms")
        self._step += 1


def main(args=None):
    rclpy.init(args=args)
    node = SpatialExplorerRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._publish(0.0, 0.0)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
