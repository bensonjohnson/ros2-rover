#!/usr/bin/env python3
"""PPO manager for RTAB-based exploration observations.

Collects rollouts using the new observation tensor, computes rewards centred on
coverage progress and safety events, and periodically updates the PPO trainer.
Exports status information and retains compatibility with the RKNN reload
service used by the runtime node.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from itertools import islice
from pathlib import Path
from typing import Optional, Deque

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool, Float32, Float32MultiArray, String
from std_srvs.srv import Trigger

from .ppo_trainer_rtab import PPOTrainerRTAB


class PPOManagerRTAB(Node):
    def __init__(self) -> None:
        super().__init__('ppo_manager_rtab')

        # Parameters
        self.declare_parameter('observation_topic', '/exploration/observation')
        self.declare_parameter('cmd_topic', 'cmd_vel_ai')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('imu_topic', '/lsm9ds1_imu_publisher/imu/data')
        self.declare_parameter('joint_topic', 'joint_states')
        self.declare_parameter('update_interval_sec', 20.0)
        self.declare_parameter('rollout_capacity', 4096)
        self.declare_parameter('minibatch_size', 128)
        self.declare_parameter('update_epochs', 3)
        self.declare_parameter('min_minibatch_size', 32)
        self.declare_parameter('max_minibatch_size', 256)
        self.declare_parameter('min_update_epochs', 1)
        self.declare_parameter('max_update_epochs', 4)
        self.declare_parameter('export_idle_linear_threshold', 0.05)
        self.declare_parameter('export_idle_angular_threshold', 0.1)
        self.declare_parameter('export_idle_timeout_sec', 15.0)
        self.declare_parameter('replay_thin_threshold', 0.02)
        self.declare_parameter('replay_thin_decay', 0.9)
        self.declare_parameter('ppo_clip', 0.2)
        self.declare_parameter('entropy_coef', 0.01)
        self.declare_parameter('value_coef', 0.5)
        self.declare_parameter('reward_forward_scale', 4.0)
        self.declare_parameter('reward_emergency_penalty', -5.0)
        self.declare_parameter('reward_block_penalty', -1.0)
        self.declare_parameter('reward_coverage_scale', 2.0)
        self.declare_parameter('max_speed', 0.20)
        self.declare_parameter('max_angular_speed', 0.8)
        self.declare_parameter('min_forward_threshold', 0.25)
        self.declare_parameter('status_window', 256)
        self.declare_parameter('export_directory', 'models')

        self.obs_topic = str(self.get_parameter('observation_topic').value)
        self.cmd_topic = str(self.get_parameter('cmd_topic').value)
        self.odom_topic = str(self.get_parameter('odom_topic').value)
        self.imu_topic = str(self.get_parameter('imu_topic').value)
        self.joint_topic = str(self.get_parameter('joint_topic').value)
        self.update_interval = float(self.get_parameter('update_interval_sec').value)
        self.rollout_capacity = int(self.get_parameter('rollout_capacity').value)
        self.minibatch_size = int(self.get_parameter('minibatch_size').value)
        self.update_epochs = int(self.get_parameter('update_epochs').value)
        self.min_minibatch = max(1, int(self.get_parameter('min_minibatch_size').value))
        self.max_minibatch = max(self.minibatch_size, int(self.get_parameter('max_minibatch_size').value))
        self.min_epochs = max(1, int(self.get_parameter('min_update_epochs').value))
        self.max_epochs = max(self.update_epochs, int(self.get_parameter('max_update_epochs').value))
        self.export_idle_linear = float(self.get_parameter('export_idle_linear_threshold').value)
        self.export_idle_angular = float(self.get_parameter('export_idle_angular_threshold').value)
        self.export_idle_timeout = float(self.get_parameter('export_idle_timeout_sec').value)
        self.replay_thin_threshold = float(self.get_parameter('replay_thin_threshold').value)
        self.replay_thin_decay = float(self.get_parameter('replay_thin_decay').value)
        self.ppo_clip = float(self.get_parameter('ppo_clip').value)
        self.entropy_coef = float(self.get_parameter('entropy_coef').value)
        self.value_coef = float(self.get_parameter('value_coef').value)
        self.reward_forward_scale = float(self.get_parameter('reward_forward_scale').value)
        self.reward_emergency_penalty = float(self.get_parameter('reward_emergency_penalty').value)
        self.reward_block_penalty = float(self.get_parameter('reward_block_penalty').value)
        self.reward_coverage_scale = float(self.get_parameter('reward_coverage_scale').value)
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.max_angular_speed = float(self.get_parameter('max_angular_speed').value)
        self.min_forward_threshold = float(self.get_parameter('min_forward_threshold').value)
        self.status_window = int(self.get_parameter('status_window').value)
        self.export_dir = str(self.get_parameter('export_directory').value)

        os.makedirs(self.export_dir, exist_ok=True)

        # Runtime state
        self.trainer: Optional[PPOTrainerRTAB] = None
        self.obs_shape: Optional[tuple[int, int, int]] = None
        self.proprio_dim = 8  # matches runtime proprio vector
        self.latest_observation: Optional[np.ndarray] = None
        self.last_obs_time = 0.0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.have_last_action = False
        self.last_action_time = 0.0
        self.current_position = np.zeros(2, dtype=np.float32)
        self.last_position = None
        self.current_linear = 0.0
        self.current_angular = 0.0
        self.wheel_velocities = np.zeros(2, dtype=np.float32)
        self.roll = 0.0
        self.pitch = 0.0
        self.accel_mag = 0.0
        self.emergency_flag = False
        self.min_forward = 10.0
        self.last_coverage = 0.0
        self.last_step_time = time.time()
        self.training_paused = False
        self.pending_export = None
        self.idle_start: Optional[float] = None
        self.calibration_buffer: Deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=256)

        # Rolling metrics
        self.reward_history: Deque[float] = deque(maxlen=self.status_window)
        self.coverage_history: Deque[float] = deque(maxlen=self.status_window)
        self.progress_history: Deque[float] = deque(maxlen=self.status_window)
        self.blocked_count = 0
        self.forward_attempts = 0

        # Publishers
        self.status_pub = self.create_publisher(String, '/ppo_status', 10)

        # Subscriptions
        self.create_subscription(Float32MultiArray, self.obs_topic, self.observation_callback, qos_profile_sensor_data)
        self.create_subscription(Twist, self.cmd_topic, self.cmd_callback, 10)
        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 20)
        self.create_subscription(Imu, self.imu_topic, self.imu_callback, qos_profile_sensor_data)
        self.create_subscription(JointState, self.joint_topic, self.joint_callback, 10)
        self.create_subscription(Bool, 'emergency_stop', self.emergency_callback, 10)
        self.create_subscription(Float32, 'min_forward_distance', self.min_forward_callback, 10)

        # Services
        self.reload_cli = self.create_client(Trigger, '/reload_rknn')

        # Timers
        self.create_timer(self.update_interval, self.update_timer)
        self.create_timer(1.0, self.status_timer)

        self.get_logger().info('PPO Manager (RTAB) initialised.')

    # ------------------------------------------------------------------
    def observation_callback(self, msg: Float32MultiArray) -> None:
        try:
            dims = msg.layout.dim
            if len(dims) < 3:
                self.get_logger().warn_throttle(5.0, 'Observation layout missing dims')
                return
            channels = dims[0].size
            height = dims[1].size
            width = dims[2].size
            data = np.array(msg.data, dtype=np.float32)
            expected = channels * height * width
            if data.size != expected:
                self.get_logger().warn_throttle(5.0, f'Observation size mismatch {data.size} vs {expected}')
                return
            obs = data.reshape((channels, height, width))
            self.latest_observation = obs
            self.last_obs_time = time.time()
            if self.trainer is None:
                self._init_trainer(obs.shape)
            self._maybe_add_transition()
        except Exception as exc:
            self.get_logger().warn_throttle(5.0, f'Observation processing failed: {exc}')

    def cmd_callback(self, msg: Twist) -> None:
        self.last_action = np.array([msg.linear.x / max(self.max_speed, 1e-6), msg.angular.z / max(self.max_angular_speed, 1e-6)], dtype=np.float32)
        self.last_action = np.clip(self.last_action, -1.0, 1.0)
        self.have_last_action = True
        self.last_action_time = time.time()

    def odom_callback(self, msg: Odometry) -> None:
        self.current_position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y], dtype=np.float32)
        self.current_linear = float(msg.twist.twist.linear.x)
        self.current_angular = float(msg.twist.twist.angular.z)

    def imu_callback(self, msg: Imu) -> None:
        roll, pitch = self._roll_pitch_from_quaternion(
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        )
        ax, ay, az = msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
        self.roll = float(roll)
        self.pitch = float(pitch)
        self.accel_mag = float(math.sqrt(ax * ax + ay * ay + az * az))

    def joint_callback(self, msg: JointState) -> None:
        if len(msg.velocity) >= 2:
            self.wheel_velocities[0] = float(msg.velocity[0])
            self.wheel_velocities[1] = float(msg.velocity[1])

    def emergency_callback(self, msg: Bool) -> None:
        self.emergency_flag = bool(msg.data)

    def min_forward_callback(self, msg: Float32) -> None:
        self.min_forward = float(msg.data)

    # ------------------------------------------------------------------
    def _init_trainer(self, obs_shape: tuple[int, int, int]) -> None:
        self.obs_shape = obs_shape
        self.trainer = PPOTrainerRTAB(
            obs_shape=obs_shape,
            proprio_dim=self.proprio_dim,
            rollout_capacity=self.rollout_capacity,
            minibatch_size=self.minibatch_size,
            update_epochs=self.update_epochs,
            ppo_clip=self.ppo_clip,
            entropy_coef=self.entropy_coef,
            value_coef=self.value_coef,
        )
        self.get_logger().info(f'Trainer initialised with obs_shape={obs_shape}')

    def _compose_proprio(self) -> np.ndarray:
        return np.array([
            self.current_linear,
            self.current_angular,
            self.roll,
            self.pitch,
            self.accel_mag,
            self.wheel_velocities[0],
            self.wheel_velocities[1],
            self.min_forward,
        ], dtype=np.float32)

    def _maybe_add_transition(self) -> None:
        if self.trainer is None or self.latest_observation is None or not self.have_last_action:
            return
        # Ensure action is recent relative to observation
        if (time.time() - self.last_action_time) > 0.5:
            return
        if self.last_position is None:
            self.last_position = self.current_position.copy()
            self.last_coverage = self._coverage_ratio(self.latest_observation)
            return

        obs = self.latest_observation
        proprio = self._compose_proprio()
        dt = max(time.time() - self.last_step_time, 1e-3)
        progress = float(np.linalg.norm(self.current_position - self.last_position))
        coverage = self._coverage_ratio(obs)
        coverage_delta = coverage - self.last_coverage

        reward = self.reward_forward_scale * progress
        reward += self.reward_coverage_scale * coverage_delta
        if self.emergency_flag:
            reward += self.reward_emergency_penalty
        if self.last_action[0] > 0.0 and self.min_forward <= self.min_forward_threshold:
            reward += self.reward_block_penalty
            self.blocked_count += 1
        if self.last_action[0] > 0.0:
            self.forward_attempts += 1

        self.trainer.add_transition(obs, proprio, self.last_action, reward, False)
        self.calibration_buffer.append((obs.copy(), proprio.copy()))
        self.reward_history.append(reward)
        self.coverage_history.append(coverage)
        self.progress_history.append(progress / dt)

        if abs(self.current_linear) < self.export_idle_linear and abs(self.current_angular) < self.export_idle_angular:
            if self.idle_start is None:
                self.idle_start = time.time()
        else:
            self.idle_start = None

        self.last_position = self.current_position.copy()
        self.last_coverage = coverage
        self.last_step_time = time.time()

    def _coverage_ratio(self, obs: np.ndarray) -> float:
        if obs.shape[0] == 0:
            return 0.0
        occ = obs[0]
        # Unknown is -1, free ~0, occupied 1. Count non-unknown as explored
        explored = np.mean(occ > -0.8)
        return float(explored)

    # ------------------------------------------------------------------
    def update_timer(self) -> None:
        if self.training_paused or self.trainer is None:
            return
        self._adapt_training_schedule()
        stats = self.trainer.update()
        if not stats.get('updated', False):
            return
        self.get_logger().info(f"PPO update #{stats['updates']} size={stats['buffer_size']}")
        self._maybe_export_policy()

    def status_timer(self) -> None:
        if self.status_pub.get_subscription_count() == 0:
            return
        avg_reward = float(np.mean(self.reward_history)) if self.reward_history else 0.0
        avg_progress = float(np.mean(self.progress_history)) if self.progress_history else 0.0
        coverage = float(self.coverage_history[-1]) if self.coverage_history else 0.0
        msg = String()
        msg.data = (
            f"PPO RTAB | reward={avg_reward:.3f} progress={avg_progress:.3f}m/s coverage={coverage:.2f} "
            f"blocked={self.blocked_count}/{max(self.forward_attempts,1)}"
        )
        self.status_pub.publish(msg)

    def _adapt_training_schedule(self) -> None:
        if self.trainer is None:
            return
        buf_size = self.trainer.buffer.size
        if buf_size <= 0:
            return
        target_batch = int(np.clip(buf_size // 4, self.min_minibatch, self.max_minibatch))
        if target_batch != self.trainer.minibatch_size:
            self.trainer.set_minibatch_size(target_batch)
            self.minibatch_size = target_batch

        avg_progress = np.mean(self.progress_history) if self.progress_history else 0.0
        epochs = self.trainer.update_epochs
        if avg_progress < 0.02:
            epochs = min(self.max_epochs, epochs + 1)
        elif avg_progress > 0.05 and epochs > self.min_epochs:
            epochs = max(self.min_epochs, epochs - 1)
        if epochs != self.trainer.update_epochs:
            self.trainer.set_update_epochs(epochs)
            self.update_epochs = epochs

        if avg_progress < self.replay_thin_threshold and self.trainer.buffer.size > 0:
            self.trainer.buffer.rewards *= self.replay_thin_decay

    def _maybe_export_policy(self) -> None:
        if self.trainer is None:
            return
        if self.idle_start is None or (time.time() - self.idle_start) < self.export_idle_timeout:
            return
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        base_name = f'ppo_rtab_{timestamp}'
        base_path = os.path.join(self.export_dir, base_name)
        pth_path = base_path + '.pth'
        try:
            torch_state = self.trainer.state_dict()
        except Exception as exc:  # pragma: no cover
            self.get_logger().warn(f'Failed to serialize policy: {exc}')
            return
        tmp_path = pth_path + '.tmp'
        torch.save(torch_state, tmp_path)
        os.replace(tmp_path, pth_path)
        self.get_logger().info(f'Policy snapshot saved to {pth_path}')

        exported = self._export_rknn_policy(base_path)
        if exported:
            self._reload_rknn_runtime()
            self.idle_start = None

    @staticmethod
    def _roll_pitch_from_quaternion(x: float, y: float, z: float, w: float) -> tuple[float, float]:
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = max(min(t2, +1.0), -1.0)
        pitch = math.asin(t2)
        return roll, pitch

    def _export_rknn_policy(self, base_path: str) -> bool:
        if self.trainer is None or not self.calibration_buffer:
            self.get_logger().warn('Skipping RKNN export: insufficient calibration samples')
            return False

        try:
            self.trainer.export_onnx(base_path + '.onnx')
        except Exception as exc:
            self.get_logger().warn(f'Failed to export ONNX policy: {exc}')
            return False

        samples = list(islice(self.calibration_buffer, min(len(self.calibration_buffer), 128)))
        calib_dir = os.path.join(self.export_dir, 'calibration')
        os.makedirs(calib_dir, exist_ok=True)
        dataset_path = os.path.join(calib_dir, f'{Path(base_path).name}_dataset.txt')
        dataset_lines = []
        for idx, (obs, proprio) in enumerate(samples):
            obs_path = os.path.join(calib_dir, f'{Path(base_path).name}_obs_{idx}.npy')
            proprio_path = os.path.join(calib_dir, f'{Path(base_path).name}_pro_{idx}.npy')
            np.save(obs_path, obs.astype(np.float32))
            np.save(proprio_path, proprio.astype(np.float32))
            dataset_lines.append(f'{obs_path} {proprio_path}')
        try:
            with open(dataset_path, 'w') as dataset_file:
                dataset_file.write('\n'.join(dataset_lines))
        except OSError as exc:
            self.get_logger().warn(f'Failed to write RKNN dataset file: {exc}')
            return False

        try:
            from rknn.api import RKNN
        except ImportError:
            self.get_logger().warn('RKNN toolkit not available; skipping RKNN export')
            return False

        rknn_path = base_path + '.rknn'
        rknn = RKNN()
        try:
            mean_values = [[0.0] * self.obs_shape[0]] if self.obs_shape else None
            std_values = [[1.0] * self.obs_shape[0]] if self.obs_shape else None
            rknn.config(mean_values=mean_values, std_values=std_values, target_platform='rk3588')
            if rknn.load_onnx(onnx_model=base_path + '.onnx') != 0:
                self.get_logger().warn('RKNN load_onnx failed')
                return False
            build_ret = rknn.build(do_quantization=True, dataset=dataset_path, algorithm='hybrid')
            if build_ret != 0:
                self.get_logger().warn(f'RKNN build failed (code {build_ret})')
                return False
            if rknn.export_rknn(rknn_path) != 0:
                self.get_logger().warn('RKNN export failed')
                return False
            self.get_logger().info(f'RKNN model exported to {rknn_path}')
            return True
        except Exception as exc:
            self.get_logger().warn(f'RKNN export error: {exc}')
            return False
        finally:
            try:
                rknn.release()
            except Exception:
                pass

    def _reload_rknn_runtime(self) -> None:
        if not self.reload_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('RKNN reload service unavailable')
            return
        start = time.time()
        future = self.reload_cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        duration = time.time() - start
        if future.done():
            result = future.result()
            if result is not None and result.success:
                self.get_logger().info(f'RKNN runtime reloaded in {duration:.2f}s: {result.message}')
            else:
                message = result.message if result else 'unknown error'
                self.get_logger().warn(f'RKNN reload failed after {duration:.2f}s: {message}')
        else:
            self.get_logger().warn('RKNN reload timed out')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PPOManagerRTAB()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
