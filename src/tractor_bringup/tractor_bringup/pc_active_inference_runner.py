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
import queue
import threading

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, JointState, Imu
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
        # Proprioception: the rover senses its own motion (fixes the self-motion
        # blind spot — one scan can't tell you if you're moving, and the
        # commanded action != actual motion under slip / safety clamping).
        p("use_proprio", True)
        p("joint_states_topic", "/joint_states")
        p("imu_topic", "/imu/data")
        p("imu_yaw_axis", "z")        # x|y|z — the chip axis that is VERTICAL on
                                      # this (strangely mounted) IMU. VERIFY first.
        p("imu_yaw_sign", 1.0)        # flip if a left turn reads negative
        p("max_wheel_vel", 8.0)       # rad/s, for normalizing wheel velocity
        p("max_yaw_rate", 2.5)        # rad/s, for normalizing IMU yaw rate
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
        p("forward_bias", 0.3)        # 0 = pure epistemic, 1 = pure forward translation
        p("pragmatic_weight", 0.4)
        p("target_wl", 0.65)
        p("target_wr", 0.65)
        p("target_yaw", 0.5)
        p("horizon", 8)
        p("action_persist", 5)        # hold a chosen action this many ticks (anti-twitch)
        p("torch_threads", 4)
        p("model_path", os.path.expanduser("~/.ros/pnn_brain.pt"))
        p("save_interval_s", 60.0)
        p("learn", True)              # set False to freeze the brain (eval)
        p("dashboard_port", 8082)     # 0 disables the web dashboard
        p("log_experience", True)
        p("experience_log_path", os.path.expanduser("~/.ros/pnn_experience.jsonl"))

        g = self.get_parameter
        self.scan_topic = g("scan_topic").value
        self.num_bins = int(g("num_bins").value)
        self.max_range = float(g("max_range").value)
        self.action_scale = float(g("action_scale").value)
        self.action_smoothing = float(g("action_smoothing").value)
        self.forward_bias = float(g("forward_bias").value)
        self.pragmatic_weight = float(g("pragmatic_weight").value)
        self.target_wl = float(g("target_wl").value)
        self.target_wr = float(g("target_wr").value)
        self.target_yaw = float(g("target_yaw").value)
        self.horizon = int(g("horizon").value)
        self.action_persist = int(g("action_persist").value)
        self.do_learn = bool(g("learn").value)
        self.model_path = g("model_path").value
        self.save_interval_s = float(g("save_interval_s").value)
        self.log_experience = bool(g("log_experience").value)
        self.experience_log_path = g("experience_log_path").value

        # Proprio config
        self.use_proprio = bool(g("use_proprio").value)
        self.imu_yaw_axis = str(g("imu_yaw_axis").value).lower()
        self.imu_yaw_sign = float(g("imu_yaw_sign").value)
        self.max_wheel_vel = float(g("max_wheel_vel").value)
        self.max_yaw_rate = float(g("max_yaw_rate").value)
        self.n_proprio = 3 if self.use_proprio else 0   # [wheel_L, wheel_R, yaw_rate]
        self.obs_dim = self.num_bins + self.n_proprio
        self._wheel_l = self._wheel_r = self._yaw_rate = 0.0

        torch.set_num_threads(int(g("torch_threads").value))

        # --- brain ---
        self.model = PCWorldModel(PCConfig(
            obs_dim=self.obs_dim,
            latent_dim=int(g("latent_dim").value),
            ensemble_size=int(g("ensemble_size").value),
            n_infer_iters=int(g("n_infer_iters").value),
        ))
        # Use forward_bias as the default pragmatic_weight for backward compatibility with launch files.
        self.actor = EFEActor(ActorConfig(
            action_dim=2,
            pragmatic_weight=self.forward_bias if self.forward_bias > 0.0 else self.pragmatic_weight,
            target_wl=self.target_wl,
            target_wr=self.target_wr,
            target_yaw=self.target_yaw,
            horizon=self.horizon
        ))
        self._maybe_load()
        self._persist_ctr = 0
        self._held_raw = np.zeros(2, dtype=np.float32)
        self._held_info = {"epistemic": 0.0, "epistemic_max": 0.0}

        # --- replay ---
        self.replay = SequenceReplay(
            capacity=int(g("replay_capacity").value),
            obs_dim=self.obs_dim, action_dim=2)
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

        # --- background logger ---
        self.log_queue = None
        self.logger_thread = None
        self.logger_active = False
        if self.log_experience:
            self.log_queue = queue.Queue()
            self.logger_active = True
            self.logger_thread = threading.Thread(target=self._experience_logger_loop, daemon=True)
            self.logger_thread.start()

        # --- ROS I/O ---
        self.scan_sub = self.create_subscription(
            LaserScan, self.scan_topic, self._scan_cb, qos_profile_sensor_data)
        if self.use_proprio:
            self.create_subscription(
                JointState, g("joint_states_topic").value, self._joint_cb, 10)
            self.create_subscription(
                Imu, g("imu_topic").value, self._imu_cb, qos_profile_sensor_data)
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

    def _joint_cb(self, msg: JointState):
        # Wheel velocities live at indices 2,3 (see hiwonder_motor_driver).
        if len(msg.velocity) >= 4:
            self._wheel_l = float(msg.velocity[2])
            self._wheel_r = float(msg.velocity[3])

    def _imu_cb(self, msg: Imu):
        w = msg.angular_velocity
        comp = {"x": w.x, "y": w.y, "z": w.z}.get(self.imu_yaw_axis, w.z)
        self._yaw_rate = self.imu_yaw_sign * float(comp)

    def _build_obs(self) -> np.ndarray:
        """Openness vector + proprio channel, all in [0,1] for the sigmoid decoder."""
        if not self.use_proprio:
            return self.latest_scan
        # Map signed velocities to [0,1] (0.5 = stationary).
        wl = np.clip(0.5 + 0.5 * self._wheel_l / self.max_wheel_vel, 0.0, 1.0)
        wr = np.clip(0.5 + 0.5 * self._wheel_r / self.max_wheel_vel, 0.0, 1.0)
        yaw = np.clip(0.5 + 0.5 * self._yaw_rate / self.max_yaw_rate, 0.0, 1.0)
        proprio = np.array([wl, wr, yaw], dtype=np.float32)
        return np.concatenate([self.latest_scan, proprio])

    def _control_step(self):
        if self.latest_scan is None:
            self._publish_track(0.0, 0.0)
            return

        obs_np = self._build_obs()
        o_t = torch.from_numpy(obs_np)
        action_prev_np = self.last_action.numpy().copy()   # led into this o_t

        # Queue for background logging
        if self.logger_active and self.log_queue is not None:
            self.log_queue.put((obs_np, action_prev_np))

        # 1. Infer current latent from observation + previous action.
        z, free_energy, obs_err = self.model.infer(o_t, self.last_action)

        # 2. Learn (pure PC local update) before moving on.
        if self.do_learn:
            self.model.learn(z, self.last_action, o_t)

        # 3. Choose an action, but HOLD it for action_persist ticks so the rover
        #    commits to a move instead of re-deciding (and twitching) every tick.
        #    Inference + learning above still run every tick regardless.
        if self._persist_ctr <= 0:
            action, info = self.actor.select(self.model, z)
            self._held_raw = action.numpy()
            self._held_info = info
            self._persist_ctr = max(1, self.action_persist)
        self._persist_ctr -= 1
        raw = self._held_raw
        info = self._held_info

        # 4. Smooth + scale, then publish track command.
        a = self.action_smoothing * raw + (1.0 - self.action_smoothing) * self.exec_action
        self.exec_action = a
        out = np.clip(a * self.action_scale, -1.0, 1.0)
        self._publish_track(float(out[0]), float(out[1]))

        # Remember the action actually executed for the next temporal prior.
        self.last_action = torch.from_numpy(a.astype(np.float32))

        # Store this transition, then squeeze extra learning from past windows.
        # Done after publishing so replay compute never delays the motor command.
        self.replay.append(obs_np, action_prev_np)
        if self.do_learn and self.replay_passes > 0 and len(self.replay) >= self.replay_min:
            self._replay_step()

        # 5. Diagnostics.
        self._step += 1
        diag = Float32MultiArray()
        diag.data = [float(free_energy), float(obs_err),
                     float(info["epistemic"]), float(info["epistemic_max"]),
                     float(out[0]), float(out[1])]
        self.diag_pub.publish(diag)

        # Capture network internals for the brain visualizer.
        o_hat, s_latent = self.model._decode(z)
        z_prev_in = self.model._trans_input(self.model.z_prev, self.last_action)
        e_o = (o_t - o_hat).numpy()
        e_z = (z - self.model._prior_mean(z_prev_in)).numpy()
        # Per-ensemble-member prediction errors for diversity display.
        trans_errors = np.array([
            float((z - self.model._predict_member(m, z_prev_in)).pow(2).mean())
            for m in range(self.model.cfg.ensemble_size)
        ])

        if self.dash is not None:
            # Abs-activation of each latent node (for node size in the diagram).
            z_abs = np.abs(s_latent.numpy())
            e_z_abs = np.abs(e_z)
            self.dash.update(
                obs=self.latest_scan,
                pred=self.model.reconstruct(z).numpy()[:self.num_bins],
                F=free_energy, err=obs_err,
                epi=info["epistemic"], epi_max=info["epistemic_max"],
                prag=info["pragmatic"],
                L=float(out[0]), R=float(out[1]), step=self._step,
                # Neural net internals for the brain visualizer.
                z=z.numpy(),                    # raw latent (64,)
                s=s_latent.numpy(),             # tanh-activated latent (64,)
                e_o=e_o,                        # sensory prediction error (72,)
                e_z=e_z,                        # state prediction error (64,)
                W_o=self.model.W_o.detach().cpu().numpy(),  # decoder weights (72,64)
                trans_errors=trans_errors,       # per-ensemble-member error (5,)
                z_abs=z_abs,                    # |activation| per latent node
                e_z_abs=e_z_abs,                # |state error| per latent node
            )
        if self._step % 20 == 0:
            self.get_logger().info(
                f"step={self._step} F={free_energy:.3f} obs_err={obs_err:.3f} "
                f"epi={info['epistemic']:.4f}/{info['epistemic_max']:.4f} "
                f"prag={info['pragmatic']:.4f} "
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

    def destroy_node(self):
        if self.logger_active:
            self.get_logger().info("Stopping experience logger and flushing queue...")
            self.logger_active = False
            if self.log_queue is not None:
                self.log_queue.put(None)
            if self.logger_thread is not None:
                self.logger_thread.join(timeout=2.0)
            self.get_logger().info("Experience logger stopped")
        super().destroy_node()

    def _experience_logger_loop(self):
        import json
        try:
            os.makedirs(os.path.dirname(self.experience_log_path), exist_ok=True)
        except Exception as e:
            self.get_logger().error(f"Could not create experience log directory: {e}")
            self.logger_active = False
            return

        self.get_logger().info(f"Experience logger thread active. Logging to {self.experience_log_path}")
        
        while self.logger_active or (self.log_queue is not None and not self.log_queue.empty()):
            try:
                item = self.log_queue.get(timeout=0.5)
                if item is None:
                    break
                
                obs_np, action_prev_np = item
                line_dict = {
                    "obs": [round(float(x), 5) for x in obs_np.tolist()],
                    "act": [round(float(x), 5) for x in action_prev_np.tolist()]
                }
                
                with open(self.experience_log_path, "a") as f:
                    f.write(json.dumps(line_dict) + "\n")
                    
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Error writing to experience log: {e}")
                time.sleep(1.0)


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
