#!/usr/bin/env python3
"""
PPO Manager Node (Live Training - Option B)
- Subscribes to /bev/image, /cmd_vel_ai, /odom, /emergency_stop, /min_forward_distance
- Accumulates short on-policy rollouts and runs bounded PPO updates in background
- Exports actor to RKNN upon improvement or min interval and triggers reload in NPU node
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import ExternalShutdownException

from sensor_msgs.msg import Image, Imu, JointState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import Trigger

import numpy as np
import math
import time
from cv_bridge import CvBridge
import threading
import os
import glob
import torch
import shutil
from collections import deque

from .ppo_trainer_bev import PPOTrainerBEV
from .rknn_trainer_bev import RKNNTrainerBEV


class PPOManagerNode(Node):
    def __init__(self):
        super().__init__('ppo_manager')

        # Parameters
        self.declare_parameter('bev_image_topic', '/bev/image')
        self.declare_parameter('update_interval_sec', 15.0)
        self.declare_parameter('min_export_interval_sec', 120.0)
        self.declare_parameter('rollout_capacity', 4096)
        self.declare_parameter('minibatch_size', 128)
        self.declare_parameter('update_epochs', 2)
        self.declare_parameter('ppo_clip', 0.2)
        self.declare_parameter('entropy_coef', 0.01)
        self.declare_parameter('value_coef', 0.5)
        self.declare_parameter('gamma', 0.99)
        self.declare_parameter('gae_lambda', 0.95)
        self.declare_parameter('bev_channels', 4)
        self.declare_parameter('bev_size', [200, 200])
        # Proprio matches NPU runtime (21 dims):
        # [lin, ang, phase, last_a0, last_a1, wheel_diff, min_d, mean_d,
        #  near_collision, emergency, left_free, right_free, center_free,
        #  bev_gx, bev_gy, bev_gf, yaw_rate, roll, pitch, accel_forward, accel_mag]
        self.declare_parameter('proprio_dim', 21)
        self.declare_parameter('imu_topic', '/lsm9ds1_imu_publisher/imu/data')
        self.declare_parameter('reward_forward_scale', 5.0)
        self.declare_parameter('reward_block_penalty', -1.0)
        self.declare_parameter('reward_emergency_penalty', -5.0)
        self.declare_parameter('cpu_guard_skip_updates', True)
        self.declare_parameter('encoder_freeze_step', 0)
        self.declare_parameter('validation_margin', 0.05)
        self.declare_parameter('low_activity_linear_threshold', 0.03)
        self.declare_parameter('low_activity_angular_threshold', 0.1)
        self.declare_parameter('export_wait_timeout_sec', 15.0)
        self.declare_parameter('rknn_drift_tolerance', 0.15)
        self.declare_parameter('min_effective_linear', 0.03)
        self.declare_parameter('min_effective_angular', 0.05)
        self.declare_parameter('small_action_penalty', -0.3)
        self.declare_parameter('small_action_patience', 3)

        self.bev_topic = self.get_parameter('bev_image_topic').value
        self.update_interval = float(self.get_parameter('update_interval_sec').value)
        self.min_export_interval = float(self.get_parameter('min_export_interval_sec').value)
        self.reward_forward_scale = float(self.get_parameter('reward_forward_scale').value)
        self.reward_block_penalty = float(self.get_parameter('reward_block_penalty').value)
        self.reward_emergency_penalty = float(self.get_parameter('reward_emergency_penalty').value)
        self.validation_margin = float(self.get_parameter('validation_margin').value)
        self.low_activity_linear_threshold = float(self.get_parameter('low_activity_linear_threshold').value)
        self.low_activity_angular_threshold = float(self.get_parameter('low_activity_angular_threshold').value)
        self.export_wait_timeout = float(self.get_parameter('export_wait_timeout_sec').value)
        self.rknn_drift_tolerance = float(self.get_parameter('rknn_drift_tolerance').value)
        self.min_effective_linear = float(self.get_parameter('min_effective_linear').value)
        self.min_effective_angular = float(self.get_parameter('min_effective_angular').value)
        self.small_action_penalty = float(self.get_parameter('small_action_penalty').value)
        self.small_action_patience = max(1, int(self.get_parameter('small_action_patience').value))

        C = int(self.get_parameter('bev_channels').value)
        bev_size = self.get_parameter('bev_size').value
        H = int(bev_size[0])
        W = int(bev_size[1])
        proprio_dim = int(self.get_parameter('proprio_dim').value)

        self.trainer = PPOTrainerBEV(
            bev_channels=C, bev_size=(H, W), proprio_dim=proprio_dim,
            rollout_capacity=int(self.get_parameter('rollout_capacity').value),
            minibatch_size=int(self.get_parameter('minibatch_size').value),
            update_epochs=int(self.get_parameter('update_epochs').value),
            ppo_clip=float(self.get_parameter('ppo_clip').value),
            entropy_coef=float(self.get_parameter('entropy_coef').value),
            value_coef=float(self.get_parameter('value_coef').value),
            gamma=float(self.get_parameter('gamma').value),
            gae_lambda=float(self.get_parameter('gae_lambda').value),
            encoder_freeze_step=int(self.get_parameter('encoder_freeze_step').value)
        )

        # Export helper reuses existing RKNN pipeline
        self.export_helper = RKNNTrainerBEV(
            bev_channels=C,
            enable_debug=False,
            extra_proprio=proprio_dim - 3,
            encoder_freeze_step=int(self.get_parameter('encoder_freeze_step').value)
        )
        self.last_export_time = 0.0

        # State
        self.bridge = CvBridge()
        self.latest_bev = None
        self.last_bev_ts = 0.0
        self.last_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.have_odom = False
        self.current_linear = 0.0
        self.current_angular = 0.0
        self.emergency = False
        self.min_forward = 10.0
        self.last_action = np.array([0.0, 0.0], dtype=np.float32)
        self.last_step_time = time.time()
        # IMU state
        self.yaw_rate = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.accel_forward = 0.0
        self.accel_mag = 0.0
        # Wheels
        self.wheel_velocities = (0.0, 0.0)
        # Rolling metrics
        self.win_size = 600
        self.forward_attempts = 0
        self.forward_blocked = 0
        self.forward_safety_sum = 0.0
        self.forward_safety_count = 0
        self.progress_sum = 0.0
        self.time_sum = 0.0
        self.emergency_count = 0
        self.total_steps = 0
        self.gate_baseline = None
        self.validation_buffer = deque(maxlen=512)
        self.deployed_actor_state = None
        self.pending_export = None
        self.defer_start_time = None
        self.last_validation_report = None
        self.small_action_streak = 0

        # Subscriptions
        self.create_subscription(Image, self.bev_topic, self.bev_cb, qos_profile_sensor_data)
        self.create_subscription(Twist, 'cmd_vel_ai', self.act_cb, 10)
        self.create_subscription(Odometry, 'odom', self.odom_cb, 10)
        self.create_subscription(Imu, self.get_parameter('imu_topic').value, self.imu_cb, 50)
        self.create_subscription(JointState, 'joint_states', self.joint_state_cb, 10)
        self.create_subscription(Bool, 'emergency_stop', self.emerg_cb, 10)
        self.create_subscription(Float32, 'min_forward_distance', self.minf_cb, 10)

        # Publishers
        self.status_pub = self.create_publisher(String, '/ppo_status', 10)

        # Timers
        self.create_timer(self.update_interval, self.update_timer)
        self.create_timer(1.0, self.status_timer)

        # Service client to reload RKNN in NPU node
        self.reload_cli = self.create_client(Trigger, '/reload_rknn')

        self.get_logger().info("PPO Manager initialized (live training, bounded updates)")
        # Try to load latest PPO checkpoint to resume training
        self.load_ppo_checkpoint()
        # Control flags
        self.training_paused = False

    def bev_cb(self, msg: Image):
        try:
            bev = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if isinstance(bev, np.ndarray) and bev.ndim == 3:
                self.latest_bev = bev.astype(np.float32, copy=False)
                self.last_bev_ts = time.time()
                # On BEV tick, try to add transition if odom/action available
                self.maybe_add_transition()
        except Exception as e:
            self.get_logger().debug(f"BEV conversion failed: {e}")

    def act_cb(self, msg: Twist):
        self.last_action = np.array([msg.linear.x, msg.angular.z], dtype=np.float32)

    def odom_cb(self, msg: Odometry):
        self.have_odom = True
        self.position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y], dtype=np.float32)
        self.current_linear = float(msg.twist.twist.linear.x)
        self.current_angular = float(msg.twist.twist.angular.z)

    def emerg_cb(self, msg: Bool):
        self.emergency = bool(msg.data)
        if self.emergency:
            self.emergency_count += 1

    def minf_cb(self, msg: Float32):
        self.min_forward = float(msg.data)

    def compute_reward(self, dt: float) -> float:
        # Progress-based reward with safety penalties
        progress = 0.0
        if self.have_odom and hasattr(self, 'last_pos'):
            progress = float(np.linalg.norm(self.position - self.last_pos))
        r = self.reward_forward_scale * progress
        if self.emergency:
            r += self.reward_emergency_penalty
        # Penalty if trying to go forward while blocked very near
        if self.last_action[0] > 0.0 and self.min_forward <= 0.25:
            r += self.reward_block_penalty
        # Penalize ineffective tiny commands that fail to move the robot
        if (abs(self.last_action[0]) < self.min_effective_linear and
                abs(self.last_action[1]) < self.min_effective_angular and
                abs(self.current_linear) < self.min_effective_linear * 0.5 and
                abs(self.current_angular) < self.min_effective_angular and
                self.min_forward > self.min_effective_linear):
            self.small_action_streak = min(self.small_action_streak + 1, self.small_action_patience)
        else:
            self.small_action_streak = 0
        if self.small_action_streak >= self.small_action_patience:
            r += self.small_action_penalty
            self.small_action_streak = 0
        return r

    def maybe_add_transition(self):
        if self.training_paused:
            return
        if self.latest_bev is None or not self.have_odom:
            return
        now = time.time()
        dt = now - self.last_step_time
        if dt < 0.08:  # roughly limit to ~12.5 Hz max
            return
        self.last_step_time = now
        # Observation
        bev = self.latest_bev
        # Derive BEV metrics to match NPU runtime
        min_d_global, mean_d_global, near_collision = self._bev_distance_stats(bev)
        left_free, center_free, right_free = self._bev_free_bands(bev)
        bev_gx, bev_gy, bev_gf = self._bev_gradient(bev)
        # Fuse min distance with safety monitor min_forward (conservative)
        # Match NPU runtime: use BEV-derived min_d_global for proprio; emergency is separate
        min_d = float(min_d_global)
        wheel_diff = float(self.wheel_velocities[0] - self.wheel_velocities[1])
        phase = float(self.total_steps % 100) / 100.0
        emergency_numeric = 1.0 if self.emergency else 0.0
        sens = np.array([
            float(self.current_linear),
            float(self.current_angular),
            phase,
            float(self.last_action[0]),
            float(self.last_action[1]),
            wheel_diff,
            min_d,
            float(mean_d_global),
            near_collision,
            emergency_numeric,
            float(left_free),
            float(right_free),
            float(center_free),
            float(bev_gx),
            float(bev_gy),
            float(bev_gf),
            float(self.yaw_rate),
            float(self.roll),
            float(self.pitch),
            float(self.accel_forward),
            float(self.accel_mag),
        ], dtype=np.float32)
        # Map action to [-1, 1] policy domain based on expected scaling in runtime
        # Here we assume commands are already normalized in [-1,1] by the NPU; if not, clamp
        act = np.clip(self.last_action.copy(), -1.0, 1.0)
        # Compute reward and done flag
        reward = self.compute_reward(dt)
        done = False  # keep continuous episodes
        # Add transition
        self.trainer.add_transition(bev, sens, act, reward, done)
        self.validation_buffer.append({
            'bev': bev.copy() if isinstance(bev, np.ndarray) else bev,
            'sensor': sens.copy(),
            'reward': float(reward),
            'min_forward': float(self.min_forward),
            'emergency': bool(self.emergency),
            'action': act.copy(),
        })
        # Update rolling metrics and last_pos
        step_progress = 0.0
        if self.have_odom and hasattr(self, 'last_pos'):
            step_progress = float(np.linalg.norm(self.position - self.last_pos))
            self.last_pos = self.position.copy()
        self.total_steps = min(self.total_steps + 1, self.win_size)
        if act[0] > 0.0:
            self.forward_attempts = min(self.forward_attempts + 1, self.win_size)
            if self.min_forward <= 0.25:
                self.forward_blocked = min(self.forward_blocked + 1, self.win_size)
            self.forward_safety_sum += float(self.min_forward)
            self.forward_safety_count = min(self.forward_safety_count + 1, self.win_size)
        self.progress_sum += step_progress
        self.time_sum += dt

    def update_timer(self):
        now = time.time()
        if self.pending_export and self.training_paused:
            if self._ready_for_deferred_export(now):
                self._perform_export(self.pending_export)
            return
        if self.training_paused:
            return
        stats = self.trainer.update()
        if not stats.get('updated'):
            return
        self.get_logger().info(f"PPO updated: size={stats['size']} loss={stats['avg_loss']:.3f}")
        if (now - self.last_export_time) <= self.min_export_interval:
            return
        gate_ok, metrics, reason = self.check_export_gate()
        self.publish_metrics(metrics, gate_ok, reason)
        if not gate_ok and self.gate_baseline is not None:
            self.get_logger().info(f"Export gated: {reason}")
            return
        candidate_state = self._clone_state_dict(self.trainer.actor.state_dict())
        validation_ok, val_report = self._candidate_passes_validation(candidate_state)
        self.last_validation_report = val_report
        if not validation_ok:
            self.get_logger().info("Candidate actor failed validation gate; keeping deployed model")
            return
        payload = {
            'state_dict': candidate_state,
            'metrics': metrics,
            'reason': reason,
            'validation': val_report,
            'created': now,
        }
        if not self._low_activity():
            self.training_paused = True
            self.pending_export = payload
            self.defer_start_time = now
            self.get_logger().info("Deferring RKNN export until rover motion is low")
            return
        self.training_paused = True
        self._perform_export(payload)

    def _low_activity(self) -> bool:
        return (abs(self.current_linear) < self.low_activity_linear_threshold and
                abs(self.current_angular) < self.low_activity_angular_threshold)

    def _ready_for_deferred_export(self, now: float) -> bool:
        if not self.pending_export:
            return False
        if self._low_activity():
            return True
        if self.export_wait_timeout > 0 and self.defer_start_time is not None:
            if now - self.defer_start_time > self.export_wait_timeout:
                self.get_logger().info("Low-activity window not found; forcing export after timeout")
                return True
        return False

    def _clone_state_dict(self, state_dict):
        return {k: v.detach().cpu().clone() for k, v in state_dict.items()}

    def _candidate_passes_validation(self, candidate_state):
        candidate_score, candidate_metrics = self._evaluate_actor_state(candidate_state)
        report = {
            'candidate': {
                **candidate_metrics,
                'score': candidate_score,
            }
        }
        if self.deployed_actor_state is None:
            return True, report
        baseline_score, baseline_metrics = self._evaluate_actor_state(self.deployed_actor_state)
        report['baseline'] = {**baseline_metrics, 'score': baseline_score}
        improvement = candidate_score - baseline_score
        report['delta'] = improvement
        return improvement >= self.validation_margin, report

    def _evaluate_actor_state(self, state_dict):
        if not self.validation_buffer:
            return 0.0, {
                'forward_clear_mean': 0.0,
                'blocked_forward_rate': 0.0,
                'emergency_forward_rate': 0.0,
                'samples': 0,
            }
        actor = self.trainer.actor
        backup = actor.state_dict()
        actor.load_state_dict(state_dict, strict=False)
        device = next(actor.parameters()).device
        forward_clear = []
        blocked_forward = 0
        emergency_forward = 0
        total = 0
        for sample in list(self.validation_buffer)[-128:]:
            bev = sample['bev']
            sensor = sample['sensor']
            bev_t = torch.from_numpy(np.transpose(bev, (2, 0, 1))).unsqueeze(0).to(device)
            sens_t = torch.from_numpy(sensor.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = actor.encode(bev_t)
                out = actor.forward_from_features(feat, sens_t)[0]
                action = torch.tanh(out[:2]).cpu().numpy()
            if sample['min_forward'] > 0.5:
                forward_clear.append(action[0])
            if sample['min_forward'] <= 0.25 and action[0] > 0.0:
                blocked_forward += 1
            if sample['emergency'] and action[0] > 0.0:
                emergency_forward += 1
            total += 1
        actor.load_state_dict(backup, strict=False)
        forward_clear_mean = float(np.mean(forward_clear)) if forward_clear else 0.0
        blocked_rate = float(blocked_forward) / max(1, total)
        emergency_rate = float(emergency_forward) / max(1, total)
        score = forward_clear_mean - (blocked_rate * 0.5) - (emergency_rate * 0.5)
        return score, {
            'forward_clear_mean': forward_clear_mean,
            'blocked_forward_rate': blocked_rate,
            'emergency_forward_rate': emergency_rate,
            'samples': total,
        }

    def _perform_export(self, payload):
        cand_state = payload['state_dict']
        val_delta = payload.get('validation', {}).get('delta', float('nan'))
        self.get_logger().info(f"Starting RKNN export (validation Δ={val_delta:.3f})")
        try:
            self.export_helper.model.load_state_dict(cand_state, strict=False)
        except Exception as exc:
            self.get_logger().warn(f"Failed to load candidate weights into export helper: {exc}")
            self.training_paused = False
            self.pending_export = None
            return
        self.save_ppo_checkpoint()
        backup_path = self._backup_rknn_file()
        exp_t0 = time.time()
        try:
            self.export_helper.convert_to_rknn()
        except Exception as exc:
            self.get_logger().warn(f"RKNN conversion failed: {exc}")
            self._restore_rknn_backup(backup_path)
            self.training_paused = False
            self.pending_export = None
            return
        exp_t1 = time.time()
        self.get_logger().info(f"RKNN export completed in {exp_t1-exp_t0:.2f}s")
        drift_ok, drift_value = self._check_rknn_drift()
        if not drift_ok:
            self.get_logger().warn(f"RKNN drift {drift_value:.3f} exceeds tolerance; reverting to previous model")
            self._restore_rknn_backup(backup_path)
            self.training_paused = False
            self.pending_export = None
            return
        self.last_export_time = time.time()
        self.gate_baseline = payload['metrics']
        self.pending_export = None
        self.defer_start_time = None
        reloaded = self.reload_rknn()
        if reloaded:
            self.deployed_actor_state = cand_state
            self._log_drift_metric(drift_value)
            time.sleep(3.0)
            if backup_path and os.path.exists(backup_path):
                try:
                    os.remove(backup_path)
                except Exception:
                    pass
        else:
            self._restore_rknn_backup(backup_path)
        self.training_paused = False
        self.get_logger().info("Resumed training after export cycle")

    def _backup_rknn_file(self):
        model_dir = getattr(self.export_helper, 'model_dir', 'models')
        rknn_path = os.path.join(model_dir, "exploration_model_bev.rknn")
        if not os.path.exists(rknn_path):
            return None
        backup_path = os.path.join(model_dir, "exploration_model_bev_prev.rknn")
        try:
            shutil.copy2(rknn_path, backup_path)
            return backup_path
        except Exception as exc:
            self.get_logger().warn(f"Failed to backup RKNN model: {exc}")
            return None

    def _restore_rknn_backup(self, backup_path):
        if not backup_path or not os.path.exists(backup_path):
            return
        model_dir = getattr(self.export_helper, 'model_dir', 'models')
        rknn_path = os.path.join(model_dir, "exploration_model_bev.rknn")
        try:
            shutil.copy2(backup_path, rknn_path)
        except Exception as exc:
            self.get_logger().warn(f"Failed to restore RKNN backup: {exc}")

    def _check_rknn_drift(self):
        if not self.validation_buffer:
            return True, 0.0
        try:
            self.export_helper.enable_rknn_inference()
        except Exception as exc:
            self.get_logger().warn(f"Unable to initialize RKNN runtime for drift check: {exc}")
            return False, float('inf')
        diffs = []
        for sample in list(self.validation_buffer)[-32:]:
            bev = sample['bev']
            sensor = sample['sensor']
            bev_chw = np.transpose(bev, (2, 0, 1)).astype(np.float32)
            bev_t = torch.from_numpy(bev_chw).unsqueeze(0).to(self.trainer.device)
            sens_t = torch.from_numpy(sensor.astype(np.float32)).unsqueeze(0).to(self.trainer.device)
            with torch.no_grad():
                feat = self.trainer.actor.encode(bev_t)
                out = self.trainer.actor.forward_from_features(feat, sens_t)[0]
                pytorch_action = torch.tanh(out[:2]).cpu().numpy()
            rknn_action, _ = self.export_helper.inference(bev, sensor)
            if np.isnan(rknn_action).any():
                return False, float('inf')
            diffs.append(np.max(np.abs(pytorch_action - rknn_action)))
        max_diff = float(np.max(diffs)) if diffs else 0.0
        return max_diff <= self.rknn_drift_tolerance, max_diff

    def _log_drift_metric(self, drift_value: float):
        self.get_logger().info(f"RKNN vs PyTorch max action diff: {drift_value:.3f}")

    def reload_rknn(self):
        if not self.reload_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("/reload_rknn service not available")
            return False
        req = Trigger.Request()
        future = self.reload_cli.call_async(req)
        # Allow longer time for the NPU node to reload the runtime
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if future.done():
            resp = future.result()
            if resp and resp.success:
                self.get_logger().info(f"RKNN reloaded via NPU node: {resp.message}")
                return True
            else:
                self.get_logger().warn(f"RKNN reload service call failed: {getattr(resp, 'message', 'no response')}" )
        else:
            self.get_logger().warn("RKNN reload service call timed out")
        return False

    def save_ppo_checkpoint(self):
        try:
            model_dir = getattr(self.export_helper, 'model_dir', 'models')
            os.makedirs(model_dir, exist_ok=True)
            ts = int(time.time())
            ckpt_path = os.path.join(model_dir, f"ppo_actor_critic_{ts}.pth")
            payload = {
                'actor_state_dict': self.trainer.actor.state_dict(),
                'critic_state_dict': self.trainer.critic.state_dict(),
                'log_std': self.trainer.log_std.data.clone().cpu(),
                'optimizer_state_dict': self.trainer.optimizer.state_dict(),
                'bev_channels': self.trainer.bev_channels,
                'bev_size': self.trainer.bev_size,
                'proprio_dim': self.trainer.proprio_dim,
                'timestamp': ts,
            }
            torch.save(payload, ckpt_path)
            # Update latest symlink
            latest = os.path.join(model_dir, "ppo_actor_critic_latest.pth")
            try:
                if os.path.islink(latest) or os.path.exists(latest):
                    os.remove(latest)
            except Exception:
                pass
            try:
                os.symlink(os.path.basename(ckpt_path), latest)
            except Exception:
                # Symlink might fail on some FS; ignore
                pass
            self.get_logger().info(f"Saved PPO checkpoint: {ckpt_path}")
        except Exception as e:
            self.get_logger().warn(f"Failed to save PPO checkpoint: {e}")

    def load_ppo_checkpoint(self):
        try:
            model_dir = getattr(self.export_helper, 'model_dir', 'models')
            latest = os.path.join(model_dir, "ppo_actor_critic_latest.pth")
            ckpt_path = None
            if os.path.exists(latest):
                ckpt_path = latest
            else:
                cand = sorted(glob.glob(os.path.join(model_dir, "ppo_actor_critic_*.pth")))
                if cand:
                    ckpt_path = cand[-1]
            if not ckpt_path:
                self.get_logger().info("No PPO checkpoint found; starting fresh")
                return
            payload = torch.load(ckpt_path, map_location='cpu')
            # Basic shape checks (best-effort)
            actor_state = payload.get('actor_state_dict', {})
            critic_state = payload.get('critic_state_dict', {})
            try:
                if actor_state:
                    self.trainer.actor.load_state_dict(actor_state, strict=False)
                if critic_state:
                    self.trainer.critic.load_state_dict(critic_state, strict=False)
            except RuntimeError as err:
                self.get_logger().warn(f"Checkpoint load skipped mismatched layers: {err}")
            log_std = payload.get('log_std')
            if log_std is not None and hasattr(self.trainer, 'log_std'):
                with torch.no_grad():
                    self.trainer.log_std.copy_(log_std.to(self.trainer.log_std.device))
            opt_state = payload.get('optimizer_state_dict')
            if opt_state:
                try:
                    self.trainer.optimizer.load_state_dict(opt_state)
                except Exception:
                    # Optimizer shapes may change; skip silently
                    pass
            ts = payload.get('timestamp', 0)
            if ts:
                self.last_export_time = float(ts)
            self.get_logger().info(f"Loaded PPO checkpoint: {ckpt_path}")
            self.deployed_actor_state = self._clone_state_dict(self.trainer.actor.state_dict())
        except Exception as e:
            self.get_logger().warn(f"Failed to load PPO checkpoint: {e}")

    def status_timer(self):
        msg = String()
        msg.data = (
            f"PPO | buf={self.trainer.buffer.size} last_export={int(time.time()-self.last_export_time)}s "
            f"small_action_streak={self.small_action_streak}"
        )
        self.status_pub.publish(msg)

    def check_export_gate(self):
        forward_steps = max(1, self.forward_safety_count)
        blocked_ratio = (self.forward_blocked / max(1, self.forward_attempts)) if self.forward_attempts > 0 else 1.0
        avg_safety_margin = self.forward_safety_sum / forward_steps
        progress_rate = self.progress_sum / max(1e-3, self.time_sum)
        emergency_rate = self.emergency_count / max(1, self.total_steps)
        metrics = {
            'blocked_ratio': float(blocked_ratio),
            'avg_safety_margin': float(avg_safety_margin),
            'progress_rate': float(progress_rate),
            'emergency_rate': float(emergency_rate),
            'window_steps': int(self.total_steps),
        }
        if self.gate_baseline is None:
            return True, metrics, 'baseline_init'
        thr = 0.10
        improvements = []
        if metrics['blocked_ratio'] < self.gate_baseline.get('blocked_ratio', 1.0) * (1.0 - thr):
            improvements.append('blocked_ratio')
        if metrics['avg_safety_margin'] > self.gate_baseline.get('avg_safety_margin', 0.0) * (1.0 + thr/2):
            improvements.append('avg_safety_margin')
        if metrics['progress_rate'] > self.gate_baseline.get('progress_rate', 0.0) * (1.0 + thr/2):
            improvements.append('progress_rate')
        if metrics['emergency_rate'] < self.gate_baseline.get('emergency_rate', 1.0) * (1.0 - thr):
            improvements.append('emergency_rate')
        if improvements:
            return True, metrics, f"improved: {','.join(improvements)}"
        return False, metrics, 'no_improvement'

    def publish_metrics(self, metrics: dict, gate_ok: bool, reason: str):
        try:
            msg = String()
            msg.data = (
                f"PPO metrics | gate={'OK' if gate_ok else 'SKIP'} ({reason}) "
                f"blocked={metrics['blocked_ratio']:.2f} safety={metrics['avg_safety_margin']:.2f}m "
                f"progress={metrics['progress_rate']:.3f}m/s emerg={metrics['emergency_rate']:.3f} steps={metrics['window_steps']}"
            )
            self.status_pub.publish(msg)
        except Exception:
            pass

    def imu_cb(self, msg: Imu):
        try:
            self.yaw_rate = float(msg.angular_velocity.z)
            # Quaternion to roll/pitch
            x, y, z, w = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            self.roll = math.atan2(sinr_cosp, cosr_cosp)
            sinp = 2 * (w * y - z * x)
            self.pitch = math.asin(max(-1.0, min(1.0, sinp)))
            ax, ay, az = float(msg.linear_acceleration.x), float(msg.linear_acceleration.y), float(msg.linear_acceleration.z)
            self.accel_forward = ax
            self.accel_mag = float((ax*ax + ay*ay + az*az) ** 0.5)
        except Exception:
            pass

    def _bev_distance_stats(self, bev: np.ndarray):
        try:
            if bev is None or bev.size == 0:
                return 0.0, 0.0, 0.0
            max_h = bev[:, :, 0]
            valid = max_h[max_h > 0.05]
            if valid.size == 0:
                return 0.0, 0.0, 0.0
            min_d = float(np.min(valid))
            mean_d = float(np.mean(valid))
            near_flag = 1.0 if (np.percentile(valid, 5) < 0.25) else 0.0
            return min_d, mean_d, near_flag
        except Exception:
            return 0.0, 0.0, 0.0

    def _bev_free_bands(self, bev: np.ndarray):
        """Compute left/center/right free metrics normalized to [0,1], matching NPU logic."""
        try:
            if bev is None or bev.size == 0:
                return 0.0, 0.0, 0.0
            h, w, c = bev.shape
            conf = bev[:, :, 3]
            low = bev[:, :, 2]
            occ = (conf > 0.25) | (low > 0.1)
            # Regions similar to NPU guardian
            start_x = 0.05
            x_range = 6.0  # conservative; training normalization only
            near_start = int(((start_x + x_range) / (2.0 * x_range)) * h)
            near_end = int(h * 0.75)
            front_rows = slice(near_start, near_end)
            left_cols = slice(0, int(w / 3))
            center_cols = slice(int(w / 3), int(2 * w / 3))
            right_cols = slice(int(2 * w / 3), w)

            def nearest_forward_distance(mask_slice):
                mask = occ[front_rows, mask_slice]
                ys, xs = np.where(mask)
                if ys.size == 0:
                    return x_range
                px = ys.astype(np.float32) + near_start
                x_m = (px / float(h)) * (2.0 * x_range) - x_range
                x_m = x_m[x_m >= 0.0]
                return float(np.min(x_m)) if x_m.size else x_range

            left_d = nearest_forward_distance(left_cols)
            right_d = nearest_forward_distance(right_cols)
            center_d = nearest_forward_distance(center_cols)
            left_free = float(np.clip(left_d / x_range, 0.0, 1.0))
            right_free = float(np.clip(right_d / x_range, 0.0, 1.0))
            center_free = float(np.clip(center_d / x_range, 0.0, 1.0))
            return left_free, center_free, right_free
        except Exception:
            return 0.0, 0.0, 0.0

    def _bev_gradient(self, bev: np.ndarray):
        try:
            if bev is None or bev.size == 0:
                return 0.0, 0.0, 0.0
            h, w, c = bev.shape
            height_channel = bev[:, :, 0]
            left_region = height_channel[:2*h//3, :w//3]
            center_region = height_channel[:2*h//3, w//3:2*w//3]
            right_region = height_channel[:2*h//3, 2*w//3:]

            def calc_grad(region):
                valid = region[region > 0.1]
                if valid.size < 10:
                    return 0.0
                s = np.sort(valid)
                near = s[:len(s)//3]
                far = s[-len(s)//3:]
                if near.size and far.size:
                    return float(np.mean(far) - np.mean(near))
                return 0.0
            return calc_grad(left_region), calc_grad(center_region), calc_grad(right_region)
        except Exception:
            return 0.0, 0.0, 0.0

    def joint_state_cb(self, msg: JointState):
        try:
            if 'left_viz_wheel_joint' in msg.name and 'right_viz_wheel_joint' in msg.name:
                li = msg.name.index('left_viz_wheel_joint')
                ri = msg.name.index('right_viz_wheel_joint')
                lv = float(msg.velocity[li]) if li < len(msg.velocity) else 0.0
                rv = float(msg.velocity[ri]) if ri < len(msg.velocity) else 0.0
                self.wheel_velocities = (lv, rv)
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = PPOManagerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
