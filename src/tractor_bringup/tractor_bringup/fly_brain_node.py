#!/usr/bin/env python3
"""Live fly-connectome brain node for the rover (FlyGM tier-2 spike, live test).

Drop-in alternative brain publishing the SAME interface as the PCN runner:
/track_cmd_ai Float32MultiArray [L,R] -> lidar_safety_monitor -> /track_cmd.
Runs the pruned MaleCNS graph (75k nodes, 13.8M edges) one step per lidar
tick with fixed random encoder/decoder (stage-2 training replaces those).

  ros2 run tractor_bringup fly_brain_node --ros-args \\
      -p graph_npz:=/home/ubuntu/flygm/flygm_live.npz -p action_scale:=0.6

SAFETY (mirror of the PCN runner's hard rules):
  - publishes ZERO [0,0] until first scan is fresh (stale_timeout 0.2 s)
  - stops publishing on >0.2 s stale scans (motor driver watchdog + safety
    monitor handle the rest)
  - HARDSTOP latch (~/.ros/pc_brain_hardstop.latch) is honored: no output
  - action envelope: out = clip(raw, -1, 1) * action_scale (0.6), identical
    to the PCN runner's cap
"""
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path[:0] = [".", "src/tractor_bringup"]

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image

from tractor_bringup.active_inference.scan_preprocess import preprocess_scan
from tractor_bringup.active_inference.vis_fingerprint import (
    visual_fingerprint, VisFingerprintEMA)

C_DEF = 8
TAU_H = 0.5
DT = 0.1


def load_graph_npz(path):
    z = np.load(path, allow_pickle=False)
    from scipy import sparse
    W = sparse.csr_matrix((z["W_data"], z["W_indices"], z["W_indptr"]),
                          shape=tuple(z["W_shape"]))
    return W, z


class FlyBrainNode(Node):
    def __init__(self):
        super().__init__("fly_brain_node")
        p = self.declare_parameter
        p("graph_npz", "/home/ubuntu/flygm/flygm_live.npz")
        p("scan_topic", "/scan")
        p("camera_topic", "/camera/image_raw")
        p("track_cmd_topic", "/track_cmd_ai")
        p("action_scale", 0.6)
        p("vis_weight", 0.0)          # camera influence on obs (0 = lidar only)
        p("max_wheel_vel", 8.0)
        p("max_yaw_rate", 2.5)
        p("kin_track_width", 0.154)
        p("kin_v_max", 0.2)
        p("stale_timeout", 0.2)
        p("control_rate_hz", 10.0)
        p("hardstop_latch", "/home/ubuntu/.ros/pc_brain_hardstop.latch")
        p("diagnostics_topic", "/flygm/diagnostics")
        p("teleop_cmd_topic", "/cmd_vel_teleop")
        p("action_persist", 5)

        g = self.get_parameter
        self.graph_npz = str(g("graph_npz").value)
        self.action_scale = float(g("action_scale").value)
        self.vis_weight = float(g("vis_weight").value)
        self.stale_timeout = float(g("stale_timeout").value)
        self.hardstop_path = str(g("hardstop_latch").value)
        self.kin_track_width = float(g("kin_track_width").value)
        self.kin_v_max = float(g("kin_v_max").value)

        os.makedirs(os.path.dirname(self.hardstop_path), exist_ok=True)

        W, z = load_graph_npz(self.graph_npz)
        self.N = W.shape[0]
        self.aff_idx = z["aff_idx"]
        self.eff_idx = z["eff_idx"]
        self.C = int(z["C"][0])
        self.enc_P = z["enc_P"]
        self.R = z["R"]
        rng = np.random.default_rng(0)
        self.eta = (0.5 * rng.standard_normal((self.N, self.C))).astype(np.float32)
        self.psi1_w = (0.2 * rng.standard_normal((2 * self.C, self.C))).astype(np.float32)
        self.psi2_w = (0.2 * rng.standard_normal((self.C, self.C))).astype(np.float32)
        self.Wg = (0.5 * rng.standard_normal((self.C, self.C))).astype(np.float32)
        self.W = W
        self.H = np.zeros((self.N, self.C), dtype=np.float32)
        self.vis_ema = VisFingerprintEMA(tau_s=6.0)
        self._vis_fp = None
        self._vis_t = None

        self.scan = None
        self.scan_t = 0.0
        self.tick = 0
        self._last_cmd = (0.0, 0.0)

        self.track_pub = self.create_publisher(
            Float32MultiArray, str(g("track_cmd_topic").value), 10)
        self.diag_pub = self.create_publisher(
            Float32MultiArray, str(g("diagnostics_topic").value), 10)
        # teleop override: any nonzero twist on /cmd_vel_teleop (xbox deadman)
        # takes over for action_persist ticks (same rule as the PCN runner)
        self._teleop_left = np.zeros(2, dtype=np.float32)
        self._persist = 0
        self._persist_n = int(g("action_persist").value)
        self._kin_w = float(g("kin_track_width").value)
        self._kin_vmax = float(g("kin_v_max").value)
        self.create_subscription(
            Twist, str(g("teleop_cmd_topic").value), self._on_teleop, 10)
        self.create_subscription(LaserScan, str(g("scan_topic").value),
                                 self._on_scan, 10)
        self.create_subscription(Image, str(g("camera_topic").value),
                                 self._on_img, 10)
        self.timer = self.create_timer(1.0 / float(g("control_rate_hz").value),
                                       self._tick)
        self.get_logger().info(
            f"fly_brain ready: {self.N} nodes, W nnz {self.W.nnz:,}, "
            f"aff {len(self.aff_idx)}, eff {len(self.eff_idx)}, "
            f"vis_weight {self.vis_weight}, action_scale {self.action_scale}")

    # ------------------------------------------------------------ sensors --
    def _on_scan(self, msg: LaserScan):
        s72 = preprocess_scan(np.asarray(msg.ranges, dtype=np.float64),
                              float(msg.angle_min),
                              float(msg.angle_increment),
                              num_bins=72, max_range=12.0)
        self.scan = s72
        self.scan_t = time.monotonic()

    def _on_img(self, msg: Image):
        # decode only at tick time; here just stash the freshest raw msg
        self._vis_fp = visual_fingerprint(msg)
        self._vis_t = time.monotonic()

    def _on_teleop(self, msg: Twist):
        """Human override: map twist -> [L,R] exactly like the PCN runner
        (v ± w/2 * track_width, normalized by v_max, rescaled by
        action_scale envelope); holds for action_persist ticks."""
        v, w = float(msg.linear.x), float(msg.angular.z)
        vl = v - 0.5 * w * self._kin_w
        vr = v + 0.5 * w * self._kin_w
        post = np.array([vl, vr], dtype=np.float32) / max(self._kin_vmax, 1e-6)
        self._teleop_left = np.clip(post, -1.0, 1.0)
        self._persist = self._persist_n

    # ------------------------------------------------------------- graph --
    def _step_graph(self, obs95: np.ndarray) -> np.ndarray:
        M = self.W @ self.H
        inj = np.tanh(obs95 @ self.enc_P @ self.Wg)
        gate = np.zeros((self.N, self.C), dtype=np.float32)
        gate[self.aff_idx] = inj
        Z = np.tanh(np.hstack([M, gate]) @ self.psi1_w)
        Hn = np.tanh(Z @ self.psi2_w + self.eta)
        a = float(np.clip(DT / TAU_H, 0.0, 1.0))
        self.H = (self.H + a * (Hn - self.H)).astype(np.float32)
        return self.H[self.eff_idx].mean(axis=1)

    # --------------------------------------------------------------- tick --
    def _tick(self):
        now = time.monotonic()
        if os.path.exists(self.hardstop_path):
            self._publish(0.0, 0.0, mode=3)
            return
        if self._persist > 0:
            self._persist -= 1
            l, r = self._teleop_left
            self._publish(float(l), float(r), mode=1)
            self._step_graph_obs_only()   # keep the graph observing
            return
        if (self.scan is None or
                now - self.scan_t > self.stale_timeout):
            self._publish(0.0, 0.0, mode=2)
            return
        obs = np.zeros(95, dtype=np.float32)
        obs[:72] = self.scan
        if self.vis_weight > 0 and self._vis_fp is not None \
                and now - (self._vis_t or 0.0) <= 2.0:
            ema = self.vis_ema.update(self._vis_fp, now)
            obs[72:] = ema * self.vis_weight

        eff_vec = self._step_graph(obs)
        raw = self.R @ eff_vec
        raw = np.clip(raw, -1.0, 1.0) * self.action_scale
        self._publish(float(raw[0]), float(raw[1]), mode=0)

    def _step_graph_obs_only(self):
        """During teleop override: fold the latest obs in so the graph stays
        calibrated to the world but publish the HUMAN's command."""
        obs = np.zeros(95, dtype=np.float32)
        obs[:72] = self.scan if self.scan is not None else 0.0
        if self.vis_weight > 0 and self._vis_fp is not None \
                and time.monotonic() - (self._vis_t or 0.0) <= 2.0:
            ema = self.vis_ema.update(self._vis_fp, time.monotonic())
            obs[72:] = ema * self.vis_weight
        self._step_graph(obs)

    def _publish(self, l: float, r: float, mode: int = 0):
        msg = Float32MultiArray()
        msg.data = [float(l), float(r)]
        self.track_pub.publish(msg)
        self._last_cmd = (l, r)
        self.tick += 1
        d = Float32MultiArray()
        # mode: 0=fly, 1=teleop, 2=stale-zero, 3=hardstop
        d.data = [float(l), float(r), float(self.tick), float(mode),
                  float(mode == 3)]
        self.diag_pub.publish(d)


def main():
    rclpy.init()
    node = FlyBrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
