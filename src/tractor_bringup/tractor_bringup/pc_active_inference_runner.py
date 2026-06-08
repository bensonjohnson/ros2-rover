#!/usr/bin/env python3
"""Active-inference rover brain — ROS 2 node.

Wires the predictive-coding world model and the epistemic actor into the
existing rover stack:

    /scan (LaserScan)  ->  preprocess  ->  PC world model (infer + learn online)
                                              |
                                        epistemic actor
                                              |
                                   /track_cmd_ai (Float32MultiArray [L, R])
                                              |
                                   lidar_safety_monitor  ->  /track_cmd  -> motors

Everything runs on the rover CPU. Learning is pure predictive coding (local
updates, no backprop) and happens every control tick. Action is chosen by
maximizing expected information gain (pure epistemic). The lidar safety monitor
downstream hard-stops the tracks near obstacles, so erratic early behavior is
physically bounded.

Diagnostics (free energy, sensory error, epistemic value) are published on
/pnn/diagnostics and logged, so the brain's settling is observable — important
since pure PC has no conventional loss curve.
"""

import os
import time

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray

from tractor_bringup.active_inference.scan_preprocess import preprocess_scan
from tractor_bringup.active_inference.pc_world_model import PCWorldModel, PCConfig
from tractor_bringup.active_inference.efe_actor import EFEActor, ActorConfig
from tractor_bringup.active_inference.replay import SequenceReplay
from tractor_bringup.active_inference.pc_dashboard import (
    PCDashboardState, start_dashboard_server)


class PCActiveInferenceRunner(Node):
    def __init__(self):
        super().__init__("pc_active_inference_runner")

        # --- parameters ---
        p = self.declare_parameter
        p("scan_topic", "/scan")
        p("track_cmd_topic", "/track_cmd_ai")
        p("control_rate_hz", 15.0)
        p("num_bins", 72)
        p("max_range", 5.0)
        p("latent_dim", 64)
        p("ensemble_size", 5)
        p("n_infer_iters", 24)
        # Sequence replay: extra learning passes squeezed from past experience.
        p("replay_capacity", 4000)    # ~4-5 min of stream at 15 Hz
        p("replay_passes", 1)         # replayed windows per control tick (0 = off)
        p("replay_seq_len", 16)       # window length
        p("replay_burn_in", 4)        # lead-in steps that settle but don't learn
        p("replay_min", 256)          # min buffer fill before replay starts
        p("action_scale", 0.6)        # scales [-1,1] output (gentler early on)
        p("action_smoothing", 0.4)    # low-pass on executed action [0=frozen,1=raw]
        p("torch_threads", 4)
        p("model_path", os.path.expanduser("~/.ros/pnn_brain.pt"))
        p("save_interval_s", 60.0)
        p("learn", True)              # set False to freeze the brain (eval)
        p("dashboard_port", 8082)     # 0 disables the web dashboard

        g = self.get_parameter
        self.scan_topic = g("scan_topic").value
        self.num_bins = int(g("num_bins").value)
        self.max_range = float(g("max_range").value)
        self.action_scale = float(g("action_scale").value)
        self.action_smoothing = float(g("action_smoothing").value)
        self.do_learn = bool(g("learn").value)
        self.model_path = g("model_path").value
        self.save_interval_s = float(g("save_interval_s").value)

        torch.set_num_threads(int(g("torch_threads").value))

        # --- brain ---
        self.model = PCWorldModel(PCConfig(
            obs_dim=self.num_bins,
            latent_dim=int(g("latent_dim").value),
            ensemble_size=int(g("ensemble_size").value),
            n_infer_iters=int(g("n_infer_iters").value),
        ))
        self.actor = EFEActor(ActorConfig(action_dim=2))
        self._maybe_load()

        # --- replay ---
        self.replay = SequenceReplay(
            capacity=int(g("replay_capacity").value),
            obs_dim=self.num_bins, action_dim=2)
        self.replay_passes = int(g("replay_passes").value)
        self.replay_seq_len = int(g("replay_seq_len").value)
        self.replay_burn_in = int(g("replay_burn_in").value)
        self.replay_min = int(g("replay_min").value)

        # --- state ---
        self.latest_scan = None       # preprocessed obs vector (np)
        self.last_action = torch.zeros(2)   # executed (smoothed) action
        self.exec_action = np.zeros(2)
        self._last_save = time.time()
        self._step = 0

        # --- ROS I/O ---
        self.scan_sub = self.create_subscription(
            LaserScan, self.scan_topic, self._scan_cb, qos_profile_sensor_data)
        self.track_pub = self.create_publisher(
            Float32MultiArray, g("track_cmd_topic").value, 10)
        self.diag_pub = self.create_publisher(
            Float32MultiArray, "/pnn/diagnostics", 10)

        # --- dashboard ---
        self.dash = None
        port = int(g("dashboard_port").value)
        if port > 0:
            try:
                self.dash = PCDashboardState()
                start_dashboard_server(self.dash, port=port)
                self.get_logger().info(f"Dashboard: http://0.0.0.0:{port}")
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"Dashboard disabled: {e}")

        rate = float(g("control_rate_hz").value)
        self.timer = self.create_timer(1.0 / rate, self._control_step)

        self.get_logger().info(
            f"PC active-inference brain up: obs={self.num_bins} "
            f"latent={int(g('latent_dim').value)} ensemble={int(g('ensemble_size').value)} "
            f"@ {rate:.0f} Hz, learn={self.do_learn}")

    # ----------------------------------------------------------------------

    def _scan_cb(self, msg: LaserScan):
        self.latest_scan = preprocess_scan(
            np.asarray(msg.ranges, dtype=np.float32),
            msg.angle_min, msg.angle_increment,
            num_bins=self.num_bins, max_range=self.max_range)

    def _control_step(self):
        if self.latest_scan is None:
            self._publish_track(0.0, 0.0)
            return

        o_t = torch.from_numpy(self.latest_scan)
        action_prev_np = self.last_action.numpy().copy()   # led into this o_t

        # 1. Infer current latent from observation + previous action.
        z, free_energy, obs_err = self.model.infer(o_t, self.last_action)

        # 2. Learn (pure PC local update) before moving on.
        if self.do_learn:
            self.model.learn(z, self.last_action, o_t)

        # 3. Choose the most informative next action (pure epistemic).
        action, info = self.actor.select(self.model, z)
        raw = action.numpy()

        # 4. Smooth + scale, then publish track command.
        a = self.action_smoothing * raw + (1.0 - self.action_smoothing) * self.exec_action
        self.exec_action = a
        out = np.clip(a * self.action_scale, -1.0, 1.0)
        self._publish_track(float(out[0]), float(out[1]))

        # Remember the action actually executed for the next temporal prior.
        self.last_action = torch.from_numpy(a.astype(np.float32))

        # Store this transition, then squeeze extra learning from past windows.
        # Done after publishing so replay compute never delays the motor command.
        self.replay.append(self.latest_scan, action_prev_np)
        if self.do_learn and self.replay_passes > 0 and len(self.replay) >= self.replay_min:
            self._replay_step()

        # 5. Diagnostics.
        self._step += 1
        diag = Float32MultiArray()
        diag.data = [float(free_energy), float(obs_err),
                     float(info["epistemic"]), float(info["epistemic_max"]),
                     float(out[0]), float(out[1])]
        self.diag_pub.publish(diag)

        if self.dash is not None:
            self.dash.update(
                obs=self.latest_scan,
                pred=self.model.reconstruct(z).numpy(),
                F=free_energy, err=obs_err,
                epi=info["epistemic"], epi_max=info["epistemic_max"],
                L=float(out[0]), R=float(out[1]), step=self._step)
        if self._step % 20 == 0:
            self.get_logger().info(
                f"step={self._step} F={free_energy:.3f} obs_err={obs_err:.3f} "
                f"epi={info['epistemic']:.4f}/{info['epistemic_max']:.4f} "
                f"L={out[0]:+.2f} R={out[1]:+.2f}")

        self._maybe_save()

    def _replay_step(self):
        """Replay contiguous windows, re-deriving latents with current weights.

        A short burn-in settles the recurrent state before any learning so the
        cold start (zero initial latent) doesn't poison the updates. The live
        recurrent state (model.z_prev) is left untouched (advance=False).
        """
        L = self.replay_seq_len
        D = self.model.cfg.latent_dim
        for _ in range(self.replay_passes):
            sample = self.replay.sample_sequence(L)
            if sample is None:
                return
            obs_seq, act_seq = sample
            zp = torch.zeros(D)
            for t in range(L):
                o = torch.from_numpy(obs_seq[t])
                a = torch.from_numpy(act_seq[t])
                z, _, _ = self.model.infer(o, a, z_prev=zp)
                if t >= self.replay_burn_in:
                    self.model.learn(z, a, o, z_prev=zp, advance=False)
                zp = z

    def _publish_track(self, left: float, right: float):
        msg = Float32MultiArray()
        msg.data = [left, right]
        self.track_pub.publish(msg)

    # ----------------------------------------------------------------------

    def _maybe_load(self):
        if os.path.exists(self.model_path):
            try:
                sd = torch.load(self.model_path, map_location="cpu", weights_only=False)
                self.model.load_state_dict(sd)
                self.get_logger().info(f"Loaded brain from {self.model_path}")
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"Could not load brain: {e}")

    def _maybe_save(self, force: bool = False):
        if not force and time.time() - self._last_save < self.save_interval_s:
            return
        self._last_save = time.time()
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            torch.save(self.model.state_dict(), self.model_path)
            self.get_logger().info(f"Saved brain to {self.model_path}")
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"Could not save brain: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = PCActiveInferenceRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._maybe_save(force=True)   # always persist the brain on shutdown
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
