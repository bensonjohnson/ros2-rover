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

import math
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
from std_msgs.msg import Float32MultiArray, Float32, Bool
from geometry_msgs.msg import Twist

from tractor_bringup.active_inference.scan_preprocess import preprocess_scan
from tractor_bringup.active_inference.pc_world_model import PCWorldModel, PCConfig
from tractor_bringup.active_inference.efe_actor import EFEActor, ActorConfig
from tractor_bringup.active_inference.replay import SequenceReplay
from tractor_bringup.active_inference.place_memory import PlaceMemory
from tractor_bringup.active_inference.slow_layer import SlowLayer, SlowLayerConfig
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
        p("max_accel", 19.6)          # m/s^2, scaling limit (approx 2g) for accelerometer
        p("latent_dim", 64)
        p("ensemble_size", 5)
        p("n_infer_iters", 24)
        p("replay_infer_iters", 10)   # cheaper settling during replay passes
        p("proprio_precision", 4.0)   # error weight boost for the 8 proprio dims
                                      # (vs 72 lidar dims they'd otherwise drown in)
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
        # Interoceptive novelty: place novelty (pose-free room fingerprints) is
        # an OBSERVATION the brain must predict, with a prior preference for it
        # being high — curiosity about new PLACES becomes ordinary goal-seeking
        # under EFE, alongside the ensemble's curiosity about new DYNAMICS.
        p("novelty_precision", 6.0)   # error-weight boost for the 1 novelty dim
        p("target_novelty", 0.8)      # preferred place novelty (the appetite)
        p("novelty_pref_weight", 2.0) # weight of that preference in the EFE
        p("novelty_ema_tau_s", 1.0)   # smoothing so the channel is predictable
        p("lift_accel_dev", 3.0)      # m/s^2 deviation from gravity = "picked up"
        # Hierarchy: a slow contextual layer (same PC machine, one level up)
        # ticking ~1 Hz. Its prediction of the fast latent is a top-down prior
        # in fast settling; its EFE macro action is a policy prior in the fast
        # actor. Own checkpoint — see slow_layer.py for why it is never fused.
        p("slow_enabled", True)
        p("slow_latent_dim", 16)
        p("slow_period_ticks", 15)    # fast ticks per slow tick (~1 s @ 15 Hz)
        p("slow_horizon", 8)          # slow planning steps (~10-30 s lookahead)
        p("slow_warmup_ticks", 30)    # slow ticks before it earns influence
        p("td_precision", 0.2)        # weight of the top-down prior in settling
        p("slow_prior_weight", 0.25)  # weight of the macro-action policy prior
        p("slow_model_path", os.path.expanduser("~/.ros/pnn_brain_slow.pt"))
        # Shadow teleop: a controller twist on this topic overrides the actor
        # (the human drives), but inference/learning/logging continue exactly
        # as in autonomy — the brain gains experience from human trajectories.
        # Release the deadman and the actor resumes after teleop_timeout_s.
        p("teleop_topic", "/cmd_vel_teleop")
        p("teleop_timeout_s", 0.5)
        p("kin_v_max", 0.2)           # m/s of one track at full post-scale command
        p("kin_track_width", 0.154)   # m between track centers (twist -> tracks)
        p("torch_threads", 4)
        p("model_path", os.path.expanduser("~/.ros/pnn_brain.pt"))
        p("save_interval_s", 60.0)
        p("learn", True)              # set False to freeze the brain (eval)
        p("dashboard_port", 8082)     # 0 disables the web dashboard
        p("log_experience", True)
        p("experience_log_path", os.path.expanduser("~/.ros/pnn_experience.jsonl"))
        p("experience_log_max_mb", 256.0)  # rotate to *_part_<ts>.jsonl past this

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
        self.target_novelty = float(g("target_novelty").value)
        self.do_learn = bool(g("learn").value)
        self.model_path = g("model_path").value
        self.save_interval_s = float(g("save_interval_s").value)
        self.log_experience = bool(g("log_experience").value)
        self.experience_log_path = g("experience_log_path").value
        self.experience_log_max_bytes = int(
            float(g("experience_log_max_mb").value) * 1024 * 1024)

        # Proprio config
        self.use_proprio = bool(g("use_proprio").value)
        self.imu_yaw_axis = str(g("imu_yaw_axis").value).lower()
        self.imu_yaw_sign = float(g("imu_yaw_sign").value)
        self.max_wheel_vel = float(g("max_wheel_vel").value)
        self.max_yaw_rate = float(g("max_yaw_rate").value)
        self.max_accel = float(g("max_accel").value)
        self.n_proprio = 8 if self.use_proprio else 0   # [wheel_L, wheel_R, roll_rate, pitch_rate, yaw_rate, accel_x, accel_y, accel_z]
        self.n_intero = 1             # place novelty, appended after proprio
        self.obs_dim = self.num_bins + self.n_proprio + self.n_intero
        self._wheel_l = self._wheel_r = 0.0
        self._roll_rate = self._pitch_rate = self._yaw_rate = 0.0
        self._accel_x = self._accel_y = self._accel_z = 0.0
        self._battery_voltage = 0.0
        self._battery_percentage = 0.0

        torch.set_num_threads(int(g("torch_threads").value))

        # --- brain ---
        self.model = PCWorldModel(PCConfig(
            obs_dim=self.obs_dim,
            latent_dim=int(g("latent_dim").value),
            ensemble_size=int(g("ensemble_size").value),
            n_infer_iters=int(g("n_infer_iters").value),
            n_proprio=self.n_proprio,
            precision_proprio=float(g("proprio_precision").value),
            n_intero=self.n_intero,
            precision_intero=float(g("novelty_precision").value),
        ))
        self.replay_infer_iters = int(g("replay_infer_iters").value)
        # Use forward_bias as the default pragmatic_weight for backward compatibility with launch files.
        self.actor = EFEActor(ActorConfig(
            action_dim=2,
            pragmatic_weight=self.forward_bias if self.forward_bias > 0.0 else self.pragmatic_weight,
            target_wl=self.target_wl,
            target_wr=self.target_wr,
            target_yaw=self.target_yaw,
            horizon=self.horizon,
            num_bins=self.num_bins,
            use_proprio=self.use_proprio,
            n_intero=self.n_intero,
            target_novelty=self.target_novelty,
            novelty_pref_weight=float(g("novelty_pref_weight").value),
            slow_prior_weight=float(g("slow_prior_weight").value),
        ))

        # Slow contextual layer (hierarchy story two; own checkpoint).
        self.slow: SlowLayer | None = None
        self.td_precision = float(g("td_precision").value)
        self.slow_model_path = g("slow_model_path").value
        if bool(g("slow_enabled").value):
            self.slow = SlowLayer(SlowLayerConfig(
                fast_latent_dim=int(g("latent_dim").value),
                latent_dim=int(g("slow_latent_dim").value),
                period_ticks=int(g("slow_period_ticks").value),
                horizon=int(g("slow_horizon").value),
                warmup_ticks=int(g("slow_warmup_ticks").value),
                target_novelty=self.target_novelty,
                novelty_pref_weight=float(g("novelty_pref_weight").value),
            ))

        # Topological place memory (room fingerprints, pose-free, RAM-only).
        # Its novelty IS the interoceptive observation channel: smoothed by an
        # EMA so the brain sees a predictable signal, not fingerprint flicker.
        self.place_memory = PlaceMemory()
        self.novelty_ema_tau_s = float(g("novelty_ema_tau_s").value)
        self._nov_ema = 1.0
        self.lift_accel_dev = float(g("lift_accel_dev").value)
        self._lift_ticks = 0
        self._mem_clears = 0
        self.teleop_timeout_s = float(g("teleop_timeout_s").value)
        self.kin_v_max = float(g("kin_v_max").value)
        self.kin_track_width = float(g("kin_track_width").value)
        self._teleop_action: np.ndarray | None = None
        self._teleop_stamp = 0.0
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
        # Subscribe to battery voltage and percentage
        self.create_subscription(Float32, "/battery_voltage", self._battery_voltage_cb, 10)
        self.create_subscription(Float32, "/battery_percentage", self._battery_percentage_cb, 10)
        # Safety monitor's hold state, surfaced on the dashboard so a clamped
        # rover doesn't look like a broken brain.
        self._safety_hold = False
        self.create_subscription(Bool, "/emergency_stop", self._estop_cb, 10)
        # Controller twists for shadow-teleop (deadman released = topic silent).
        self.create_subscription(
            Twist, g("teleop_topic").value, self._teleop_cb, 10)

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
        self._tick_period = 1.0 / rate
        self.timer = self.create_timer(self._tick_period, self._control_step)

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
        # hiwonder_motor_driver publishes actual wheel velocities under the
        # viz wheel joints (the base wheel joints carry raw encoder counts in
        # position and zero velocity). Look up by name so a driver-side
        # reorder can't silently feed the brain the wrong channels.
        try:
            li = msg.name.index("left_viz_wheel_joint")
            ri = msg.name.index("right_viz_wheel_joint")
        except ValueError:
            return
        if len(msg.velocity) > max(li, ri):
            self._wheel_l = float(msg.velocity[li])
            self._wheel_r = float(msg.velocity[ri])

    def _imu_cb(self, msg: Imu):
        w = msg.angular_velocity
        comp = {"x": w.x, "y": w.y, "z": w.z}.get(self.imu_yaw_axis, w.z)
        self._yaw_rate = self.imu_yaw_sign * float(comp)
        
        # Extract remaining orthogonal gyroscope axes dynamically
        if self.imu_yaw_axis == "x":
            self._roll_rate = float(w.y)
            self._pitch_rate = float(w.z)
        elif self.imu_yaw_axis == "y":
            self._roll_rate = float(w.x)
            self._pitch_rate = float(w.z)
        else:  # "z" or default
            self._roll_rate = float(w.x)
            self._pitch_rate = float(w.y)

        # Accelerometer readings
        self._accel_x = float(msg.linear_acceleration.x)
        self._accel_y = float(msg.linear_acceleration.y)
        self._accel_z = float(msg.linear_acceleration.z)

    def _battery_voltage_cb(self, msg):
        self._battery_voltage = float(msg.data)

    def _battery_percentage_cb(self, msg):
        self._battery_percentage = float(msg.data)

    def _estop_cb(self, msg):
        self._safety_hold = bool(msg.data)

    def _teleop_cb(self, msg: Twist):
        """Controller twist -> pre-scale per-track action, same convention as
        the actor's output so the model sees one consistent action space.

        out = a * action_scale always holds; full stick therefore caps at the
        same envelope autonomy gets (the brain can't represent more).
        """
        v, w = float(msg.linear.x), float(msg.angular.z)
        vl = v - 0.5 * w * self.kin_track_width
        vr = v + 0.5 * w * self.kin_track_width
        post = np.array([vl, vr], dtype=np.float32) / max(self.kin_v_max, 1e-6)
        self._teleop_action = np.clip(
            post / max(self.action_scale, 1e-6), -1.0, 1.0).astype(np.float32)
        self._teleop_stamp = time.monotonic()

    def _check_lifted(self) -> bool:
        """Detect being picked up: sustained non-gravity accel with wheels idle.

        On a lift the place memory is cleared — the rover may be somewhere
        new, so everywhere should read novel again.
        """
        accel_mag = math.sqrt(self._accel_x ** 2 + self._accel_y ** 2
                              + self._accel_z ** 2)
        if accel_mag < 0.5:           # IMU not publishing yet
            self._lift_ticks = 0
            return False
        wheels_idle = abs(self._wheel_l) + abs(self._wheel_r) < 0.2
        if wheels_idle and abs(accel_mag - 9.81) > self.lift_accel_dev:
            self._lift_ticks += 1
        else:
            self._lift_ticks = 0
        if self._lift_ticks >= 4:
            self._lift_ticks = 0
            return True
        return False

    def _build_obs(self) -> np.ndarray:
        """Openness vector + proprio + interoceptive novelty, all in [0,1].

        The last channel is the EMA-smoothed place novelty: an interoceptive
        sense the model must predict like any other observation.
        """
        nov = np.array([self._nov_ema], dtype=np.float32)
        if not self.use_proprio:
            return np.concatenate([self.latest_scan, nov])
        # Map signed velocities, rates, and accelerations to [0,1] (0.5 = neutral).
        wl = np.clip(0.5 + 0.5 * self._wheel_l / self.max_wheel_vel, 0.0, 1.0)
        wr = np.clip(0.5 + 0.5 * self._wheel_r / self.max_wheel_vel, 0.0, 1.0)
        
        # 3-axis Gyroscope
        roll = np.clip(0.5 + 0.5 * self._roll_rate / self.max_yaw_rate, 0.0, 1.0)
        pitch = np.clip(0.5 + 0.5 * self._pitch_rate / self.max_yaw_rate, 0.0, 1.0)
        yaw = np.clip(0.5 + 0.5 * self._yaw_rate / self.max_yaw_rate, 0.0, 1.0)
        
        # 3-axis Accelerometer
        ax = np.clip(0.5 + 0.5 * self._accel_x / self.max_accel, 0.0, 1.0)
        ay = np.clip(0.5 + 0.5 * self._accel_y / self.max_accel, 0.0, 1.0)
        az = np.clip(0.5 + 0.5 * self._accel_z / self.max_accel, 0.0, 1.0)
        
        proprio = np.array([wl, wr, roll, pitch, yaw, ax, ay, az], dtype=np.float32)
        return np.concatenate([self.latest_scan, proprio, nov])

    def _control_step(self):
        if self.latest_scan is None:
            self._publish_track(0.0, 0.0)
            return

        tick_start = time.monotonic()

        # Interoceptive novelty: fold the scan into place memory EVERY tick,
        # EMA-smoothed so the channel is a clean signal the one-step model can
        # latch onto rather than per-scan fingerprint flicker. Lift detection
        # resets it — the rover may be somewhere new.
        if self.use_proprio and self._check_lifted():
            self.place_memory.clear()
            self._nov_ema = 1.0
            self._mem_clears += 1
            if self.slow is not None:
                self.slow.reset_state()
            self.get_logger().info(
                "Lift detected — place memory cleared, everywhere is new again")
        nov_raw = self.place_memory.update(self.latest_scan)
        alpha = min(1.0, self._tick_period
                    / max(self.novelty_ema_tau_s, self._tick_period))
        self._nov_ema += alpha * (nov_raw - self._nov_ema)

        obs_np = self._build_obs()
        o_t = torch.from_numpy(obs_np)
        action_prev_np = self.last_action.numpy().copy()   # led into this o_t
        # Snapshot the context infer() is about to use, BEFORE learn() advances
        # z_prev and the new action overwrites last_action — the dashboard's
        # state-error display needs the real settling context.
        z_prev_pre = self.model.z_prev.clone()
        action_into_t = self.last_action.clone()

        # Queue for background logging
        if self.logger_active and self.log_queue is not None:
            self.log_queue.put((obs_np, action_prev_np))

        # 1. Infer current latent from observation + previous action, under
        #    the slow layer's top-down prior when it has one (hierarchical
        #    settling: context biases perception).
        z, free_energy, obs_err = self.model.infer(
            o_t, self.last_action,
            td_target=self.slow.td_target if self.slow is not None else None,
            td_precision=self.td_precision)

        # 2. Learn (pure PC local update) before moving on.
        if self.do_learn:
            self.model.learn(z, self.last_action, o_t)

        # 3. Choose an action. A live controller (shadow teleop) overrides the
        #    actor — the human drives, the brain watches and learns from the
        #    exact same (obs, action) stream. Otherwise the actor decides and
        #    HOLDs its choice for action_persist ticks (anti-twitch).
        #    Inference + learning above still run every tick regardless.
        teleop = (self._teleop_action is not None
                  and time.monotonic() - self._teleop_stamp
                  < self.teleop_timeout_s)
        if teleop:
            self._held_raw = self._teleop_action.copy()
            self._held_info = {"epistemic": 0.0, "epistemic_max": 0.0,
                               "pragmatic": 0.0, "epi_gate": 0.0}
            self._persist_ctr = 0   # actor re-decides as soon as human lets go
        elif self._persist_ctr <= 0:
            action, info = self.actor.select(
                self.model, z, prev_action=self._held_raw,
                slow_action=(self.slow.macro_action
                             if self.slow is not None else None))
            self._held_raw = action.numpy()
            self._held_info = info
            self._persist_ctr = max(1, self.action_persist)
            self._persist_ctr -= 1
        else:
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

        # Slow layer: fold this tick into the context window; close the window
        # every period_ticks. (After publishing — the slow tick's few ms must
        # never delay the motor command.)
        if self.slow is not None:
            self.slow.accumulate(
                torch.tanh(z),
                float(np.mean(self.latest_scan)),
                min(1.0, obs_err / math.sqrt(self.obs_dim)),  # ~RMS error
                self._nov_ema,
                a)
            if self.slow.ready():
                self.slow.tick(learn=self.do_learn)

        # Store this transition, then squeeze extra learning from past windows.
        # Done after publishing so replay compute never delays the motor command.
        self.replay.append(obs_np, action_prev_np)
        if self.do_learn and self.replay_passes > 0 and len(self.replay) >= self.replay_min:
            self._replay_step()

        # 5. Diagnostics (incl. tick wall time so overruns are observable).
        self._step += 1
        tick_time = time.monotonic() - tick_start
        if tick_time > self._tick_period:
            self.get_logger().warning(
                f"Control tick overran: {tick_time*1000:.1f} ms > "
                f"{self._tick_period*1000:.1f} ms budget (consider fewer "
                f"replay passes or replay_infer_iters)",
                throttle_duration_sec=5.0)
        diag = Float32MultiArray()
        diag.data = [float(free_energy), float(obs_err),
                     float(info["epistemic"]), float(info["epistemic_max"]),
                     float(out[0]), float(out[1]), float(tick_time)]
        self.diag_pub.publish(diag)

        # Capture network internals for the brain visualizer — only when a
        # browser has polled recently, so the brain keeps its CPU when nobody
        # is watching. (Uses the pre-learn context infer() settled against.)
        if self.dash is not None and self.dash.active():
            o_hat, s_latent = self.model._decode(z)
            z_prev_in = self.model._trans_input(z_prev_pre, action_into_t)
            e_o = (o_t - o_hat).numpy()
            e_z = (z - self.model._prior_mean(z_prev_in)).numpy()
            # Per-ensemble-member prediction errors for diversity display.
            trans_errors = np.array([
                float((z - self.model._predict_member(m, z_prev_in)).pow(2).mean())
                for m in range(self.model.cfg.ensemble_size)
            ])
            # Top-down agreement: how close the fast settle landed to the slow
            # layer's expectation (RMS gap over fast latent dims).
            td = self.slow.td_target if self.slow is not None else None
            td_gap = (float((torch.tanh(z) - td).norm() / math.sqrt(td.numel()))
                      if td is not None else None)
            self.dash.update(
                obs=self.latest_scan,
                pred=self.model.reconstruct(z).numpy()[:self.num_bins],
                F=free_energy, err=obs_err,
                epi=info["epistemic"], epi_max=info["epistemic_max"],
                prag=info["pragmatic"],
                L=float(out[0]), R=float(out[1]), step=self._step,
                # Neural net internals for the brain visualizer.
                s=s_latent.numpy(),             # tanh-activated latent (64,)
                e_o=e_o,                        # sensory prediction error (72,)
                W_o=self.model.W_o.detach().cpu().numpy(),  # -> top-K flows
                trans_errors=trans_errors,       # per-ensemble-member error (5,)
                z_abs=np.abs(s_latent.numpy()),  # |activation| per latent node
                e_z_abs=np.abs(e_z),             # |state error| per latent node
                battery_voltage=self._battery_voltage,
                battery_percentage=self._battery_percentage,
                tick_ms=tick_time * 1000.0,
                tick_budget_ms=self._tick_period * 1000.0,
                safety_hold=self._safety_hold,
                teleop=teleop,
                epi_gate=info.get("epi_gate"),
                # Interoceptive channel observability: what the brain was fed
                # vs what it predicted — the gap closing is the brain learning
                # its own novelty dynamics.
                novelty=self._nov_ema,
                novelty_pred=float(o_hat[-1]),
                novelty_target=self.target_novelty,
                places_n=self.place_memory.n_places(),
                mem_clears=self._mem_clears,
                proprio=o_t.numpy()[self.num_bins:self.num_bins + 8],
                # Learned attention: precision multiplier per lidar bin
                # (1 = at the prior, >1 = trusted, <1 = learned-noisy).
                pi=self.model.precision_mult().numpy()[:self.num_bins],
                # Slow-layer context (None until the layer exists/warms up).
                slow_epi=(self.slow.info.get("slow_epi")
                          if self.slow is not None else None),
                slow_nov_pred=(self.slow.info.get("slow_nov_pred")
                               if self.slow is not None else None),
                slow_ticks=(self.slow.ticks if self.slow is not None else None),
                slow_action=(self.slow.macro_action
                             if self.slow is not None else None),
                # Two-story brain panel: slow latent, window progress, and
                # whether the priors are live yet.
                slow_s=(self.slow.info.get("slow_s")
                        if self.slow is not None else None),
                slow_window=([self.slow.window_fill,
                              self.slow.cfg.period_ticks]
                             if self.slow is not None else None),
                slow_warm=(self.slow.macro_action is not None
                           if self.slow is not None else None),
                slow_F=(self.slow.info.get("slow_F")
                        if self.slow is not None else None),
                slow_err=(self.slow.info.get("slow_err")
                          if self.slow is not None else None),
                td_gap=td_gap,
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
                z, _, _ = self.model.infer(o, a, z_prev=zp,
                                           n_iters=self.replay_infer_iters)
                if t >= self.replay_burn_in:
                    # update_precision=False: replay errors come from cold
                    # contexts, not the sensors — keep them out of the ledger.
                    self.model.learn(z, a, o, z_prev=zp, advance=False,
                                     update_precision=False)
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
                if sd["W_o"].shape[0] != self.obs_dim:
                    self.get_logger().warning(
                        f"Saved brain has obs_dim={sd['W_o'].shape[0]} but this "
                        f"runner builds obs_dim={self.obs_dim} (observation "
                        f"layout changed) — starting with a fresh brain")
                    return
                self.model.load_state_dict(sd)
                self.get_logger().info(f"Loaded brain from {self.model_path}")
                if self.slow is not None:
                    ok, reason = self.slow.load(self.slow_model_path)
                    if ok:
                        self.get_logger().info(
                            f"Loaded slow layer from {self.slow_model_path}")
                    elif reason != "no checkpoint":
                        self.get_logger().warning(f"Slow layer: {reason}")
                # The model learned dynamics conditioned on pre-scale actions;
                # changing action_scale redefines what those actions DO.
                saved_scale = sd.get("action_scale")
                if saved_scale is not None and abs(saved_scale - self.action_scale) > 1e-6:
                    self.get_logger().warning(
                        f"action_scale mismatch: brain was trained with "
                        f"{saved_scale:.3f}, now running with "
                        f"{self.action_scale:.3f} — learned dynamics will be "
                        f"wrong until the model re-adapts")
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"Could not load brain: {e}")

    def _maybe_save(self, force: bool = False):
        if not force and time.time() - self._last_save < self.save_interval_s:
            return
        self._last_save = time.time()
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            # Write-then-rename so a crash mid-save can't corrupt the brain.
            tmp_path = self.model_path + ".tmp"
            sd = self.model.state_dict()
            sd["action_scale"] = self.action_scale
            torch.save(sd, tmp_path)
            os.replace(tmp_path, self.model_path)
            if self.slow is not None:
                self.slow.save(self.slow_model_path)
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

        # Rotate away a log written with a different observation layout —
        # mixing obs widths in one file breaks sleep replay.
        try:
            if os.path.exists(self.experience_log_path) \
                    and os.path.getsize(self.experience_log_path) > 0:
                with open(self.experience_log_path) as old:
                    n_old = len(json.loads(old.readline())["obs"])
                if n_old != self.obs_dim:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    part_path = self.experience_log_path.replace(
                        ".jsonl", f"_obs{n_old}_{ts}.jsonl")
                    os.rename(self.experience_log_path, part_path)
                    self.get_logger().info(
                        f"Experience log used obs_dim={n_old} (now "
                        f"{self.obs_dim}) — rotated to {part_path}")
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"Experience log layout check failed: {e}")

        f = None
        bytes_written = 0
        lines_since_flush = 0
        try:
            f = open(self.experience_log_path, "a")
            bytes_written = f.tell()

            while self.logger_active or (self.log_queue is not None and not self.log_queue.empty()):
                try:
                    item = self.log_queue.get(timeout=0.5)
                    if item is None:
                        break

                    obs_np, action_prev_np = item
                    line = json.dumps({
                        "obs": [round(float(x), 5) for x in obs_np.tolist()],
                        "act": [round(float(x), 5) for x in action_prev_np.tolist()],
                    }) + "\n"
                    bytes_written += f.write(line)
                    lines_since_flush += 1
                    if lines_since_flush >= 50:
                        f.flush()
                        lines_since_flush = 0

                    # Rotate past the size cap. The sleep consolidator picks up
                    # *_part_*.jsonl siblings, so no experience is lost.
                    if bytes_written >= self.experience_log_max_bytes:
                        f.close()
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        part_path = self.experience_log_path.replace(
                            ".jsonl", f"_part_{ts}.jsonl")
                        os.rename(self.experience_log_path, part_path)
                        self.get_logger().info(
                            f"Experience log rotated to {part_path}")
                        f = open(self.experience_log_path, "a")
                        bytes_written = 0
                        lines_since_flush = 0

                    self.log_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    self.get_logger().error(f"Error writing to experience log: {e}")
                    time.sleep(1.0)
        finally:
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass


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
