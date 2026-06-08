#!/usr/bin/env python3
"""Cognitive-map active-inference brain — ROS 2 node.

Extends the predictive-coding rover brain with an allocentric growing latent
map (active_inference/latent_field.py). Lidar gives the local view, rf2o gives
the pose; the field accumulates a sparse latent map that fills in unseen cells
via a learned spatial prior. The actor is pure-epistemic over the MAP: it steers
toward frontier / low-confidence space (expected information gain about the
layout). Everything learns online with local PC rules — no backprop, no
pretraining, single house.

    /scan + /odom  ->  field.observe (infer cell latent + learn)  ->
        frontier-seeking action  ->  /track_cmd_ai  ->  safety  ->  motors

A web dashboard (pc_map_dashboard) shows the map extending and filling in.
"""

import math
import os
import time

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray

from tractor_bringup.active_inference.scan_preprocess import preprocess_scan
from tractor_bringup.active_inference.latent_field import LatentField, FieldConfig
from tractor_bringup.active_inference.pc_map_dashboard import (
    MapDashboardState, start_dashboard_server)


_STRUCTURED = [[0.0, 0.0], [1.0, 1.0], [0.6, 0.6], [-0.6, -0.6],
               [1.0, -1.0], [-1.0, 1.0], [1.0, 0.4], [0.4, 1.0],
               [-0.6, 0.6], [0.6, -0.6]]


class PCCognitiveMapRunner(Node):
    def __init__(self):
        super().__init__("pc_cognitive_map_runner")

        p = self.declare_parameter
        p("scan_topic", "/scan")
        p("odom_topic", "/odom_rf2o")
        p("track_cmd_topic", "/track_cmd_ai")
        p("control_rate_hz", 15.0)
        p("num_bins", 72)
        p("max_range", 5.0)
        p("latent_dim", 24)
        p("cell_size", 0.5)
        p("action_scale", 0.6)
        p("action_smoothing", 0.4)
        p("max_lin", 0.3)             # rough fwd speed for lookahead (m/s)
        p("yaw_gain", 2.0)            # rough yaw rate scale for lookahead
        p("lookahead_s", [0.6, 1.2, 2.0])  # times to sample novelty along arc
        p("n_random", 40)
        p("temperature", 0.3)
        p("done_novelty", 0.15)       # below this = locally explored
        p("done_hold_s", 30.0)        # sustained low novelty -> "complete"
        p("torch_threads", 4)
        p("model_path", os.path.expanduser("~/.ros/pnn_cogmap.pt"))
        p("save_interval_s", 60.0)
        p("learn", True)
        p("dashboard_port", 8083)

        g = self.get_parameter
        self.num_bins = int(g("num_bins").value)
        self.max_range = float(g("max_range").value)
        self.action_scale = float(g("action_scale").value)
        self.action_smoothing = float(g("action_smoothing").value)
        self.max_lin = float(g("max_lin").value)
        self.yaw_gain = float(g("yaw_gain").value)
        self.lookahead = [float(t) for t in g("lookahead_s").value]
        self.n_random = int(g("n_random").value)
        self.temperature = float(g("temperature").value)
        self.done_novelty = float(g("done_novelty").value)
        self.done_hold_s = float(g("done_hold_s").value)
        self.do_learn = bool(g("learn").value)
        self.model_path = g("model_path").value
        self.save_interval_s = float(g("save_interval_s").value)
        self._bin_w = 2.0 * math.pi / self.num_bins

        torch.set_num_threads(int(g("torch_threads").value))
        self._rng = np.random.default_rng(0)

        self.field = LatentField(FieldConfig(
            obs_dim=self.num_bins, latent_dim=int(g("latent_dim").value),
            cell_size=float(g("cell_size").value)))
        self._maybe_load()

        # state
        self.latest_scan = None
        self.x = self.y = self.theta = 0.0
        self.have_odom = False
        self.exec_action = np.zeros(2)
        self._last_save = time.time()
        self._step = 0
        self._low_nov_since = None
        self._done = False

        # I/O
        self.create_subscription(LaserScan, g("scan_topic").value,
                                 self._scan_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, g("odom_topic").value,
                                 self._odom_cb, 10)
        self.track_pub = self.create_publisher(
            Float32MultiArray, g("track_cmd_topic").value, 10)
        self.diag_pub = self.create_publisher(Float32MultiArray, "/pnn/map_diagnostics", 10)

        self.dash = None
        port = int(g("dashboard_port").value)
        if port > 0:
            try:
                self.dash = MapDashboardState()
                start_dashboard_server(self.dash, port=port)
                self.get_logger().info(f"Map dashboard: http://0.0.0.0:{port}")
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"Dashboard disabled: {e}")

        rate = float(g("control_rate_hz").value)
        self.create_timer(1.0 / rate, self._control_step)
        self.get_logger().info(
            f"Cognitive-map brain up: cell={g('cell_size').value}m "
            f"latent={int(g('latent_dim').value)} @ {rate:.0f} Hz, learn={self.do_learn}")

    # ----------------------------------------------------------------------

    def _scan_cb(self, msg: LaserScan):
        self.latest_scan = preprocess_scan(
            np.asarray(msg.ranges, dtype=np.float32),
            msg.angle_min, msg.angle_increment,
            num_bins=self.num_bins, max_range=self.max_range)

    def _odom_cb(self, msg: Odometry):
        q = msg.pose.pose.orientation
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.theta = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.have_odom = True

    def _candidates(self) -> np.ndarray:
        rand = self._rng.uniform(-1.0, 1.0, size=(self.n_random, 2)).astype(np.float32)
        return np.concatenate([np.array(_STRUCTURED, dtype=np.float32), rand], axis=0)

    def _score(self, action) -> float:
        """Max map-novelty sampled along the arc this action would produce."""
        L, R = float(action[0]), float(action[1])
        v = (L + R) / 2.0 * self.max_lin
        w = (R - L) / 2.0 * self.yaw_gain
        best = 0.0
        for t in self.lookahead:
            th = self.theta + 0.5 * w * t
            xt = self.x + v * t * math.cos(th)
            yt = self.y + v * t * math.sin(th)
            best = max(best, self.field.novelty_at(xt, yt))
        return best

    def _control_step(self):
        if self.latest_scan is None or not self.have_odom:
            self._publish_track(0.0, 0.0)
            return

        obs = torch.from_numpy(self.latest_scan)

        # 1. Fold the current view into the map (infer cell latent + learn).
        if self.do_learn:
            err = self.field.observe(self.x, self.y, self.theta, obs)
        else:
            cell = self.field.world_to_cell(self.x, self.y)
            err = 0.0

        # 2. Pick the most informative move (toward frontier / unknown).
        cands = self._candidates()
        scores = np.array([self._score(a) for a in cands], dtype=np.float32)
        logits = scores / max(self.temperature, 1e-6)
        logits -= logits.max()
        probs = np.exp(logits); probs /= probs.sum()
        idx = int(self._rng.choice(len(cands), p=probs))
        raw = cands[idx]
        chosen_nov = float(scores[idx])

        # 3. Smooth + scale + publish.
        a = self.action_smoothing * raw + (1.0 - self.action_smoothing) * self.exec_action
        self.exec_action = a
        out = np.clip(a * self.action_scale, -1.0, 1.0)
        self._publish_track(float(out[0]), float(out[1]))

        # 4. Termination heuristic: sustained low local novelty.
        self._update_done(chosen_nov)

        # 5. Diagnostics + dashboard.
        self._step += 1
        diag = Float32MultiArray()
        diag.data = [float(err), float(chosen_nov), float(len(self.field.cells)),
                     float(out[0]), float(out[1]), 1.0 if self._done else 0.0]
        self.diag_pub.publish(diag)
        self._push_dashboard(obs, err, chosen_nov, out)

        if self._step % 30 == 0:
            self.get_logger().info(
                f"step={self._step} cells={len(self.field.cells)} "
                f"err={err:.3f} nov={chosen_nov:.3f} pose=({self.x:.2f},{self.y:.2f}) "
                f"L={out[0]:+.2f} R={out[1]:+.2f}{' DONE' if self._done else ''}")
        self._maybe_save()

    def _update_done(self, nov: float):
        if nov < self.done_novelty:
            if self._low_nov_since is None:
                self._low_nov_since = time.time()
            elif time.time() - self._low_nov_since > self.done_hold_s and not self._done:
                self._done = True
                self.get_logger().info("Exploration appears COMPLETE (sustained low novelty).")
        else:
            self._low_nov_since = None
            self._done = False

    def _push_dashboard(self, obs, err, nov, out):
        if self.dash is None:
            return
        # Decoded prediction for the current cell, re-rotated to ego frame.
        cell = self.field.world_to_cell(self.x, self.y)
        if cell in self.field.cells:
            o_hat, _ = self.field._decode(self.field.cells[cell]["m"])
            shift = int(round(self.theta / self._bin_w))
            pred = torch.roll(o_hat, shifts=-shift, dims=0).numpy()
        else:
            pred = np.full(self.num_bins, 0.5, dtype=np.float32)
        self.dash.update(
            obs=self.latest_scan, pred=pred, err=err, nov=nov,
            cells=self.field.map_snapshot(), frontiers=self.field.frontier_cells(),
            x=self.x, y=self.y, theta=self.theta,
            cell_size=self.field.cfg.cell_size,
            L=float(out[0]), R=float(out[1]), step=self._step, done=self._done)

    def _publish_track(self, left, right):
        msg = Float32MultiArray(); msg.data = [float(left), float(right)]
        self.track_pub.publish(msg)

    # ----------------------------------------------------------------------

    def _maybe_load(self):
        if os.path.exists(self.model_path):
            try:
                sd = torch.load(self.model_path, map_location="cpu", weights_only=False)
                self.field.load_state_dict(sd)
                self.get_logger().info(
                    f"Loaded map from {self.model_path} ({len(self.field.cells)} cells)")
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"Could not load map: {e}")

    def _maybe_save(self, force=False):
        if not force and time.time() - self._last_save < self.save_interval_s:
            return
        self._last_save = time.time()
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            torch.save(self.field.state_dict(), self.model_path)
            self.get_logger().info(f"Saved map to {self.model_path}")
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"Could not save map: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = PCCognitiveMapRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._maybe_save(force=True)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
