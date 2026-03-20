#!/usr/bin/env python3
"""PPO Local Runner - Fully On-Device RL Training for Rover.

Combines ROS2 sensor pipeline with local PPO training. No external server needed.
Runs inference on CPU via PyTorch, collects experience, and trains PPO on-device.

Architecture:
- Unified BEV (LiDAR + Depth) → 2×128×128 grid
- 6-dim proprioception
- PPO with GAE, clipped surrogate objective
- Direct track control: [left_speed, right_speed] in [-1, 1]
"""

import os
import math
import time
import threading
from pathlib import Path
from typing import Tuple
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import cv2
from cv_bridge import CvBridge

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ROS2 Messages
from sensor_msgs.msg import Image, Imu, JointState, LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Bool, Float32MultiArray
from std_srvs.srv import Trigger

# Reuse existing BEV processor and phase manager
from tractor_bringup.occupancy_processor import UnifiedBEVProcessor
from tractor_bringup.phase_manager import PhaseManager

# Proprioception normalization (must match SAC runner / RKNN export)
PROPRIO_MEAN = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
PROPRIO_STD = np.array([2.0, 1.0, 1.0, 0.2, 1.0, 1.0], dtype=np.float32)


def normalize_proprio(proprio: np.ndarray) -> np.ndarray:
    normalized = (proprio - PROPRIO_MEAN) / PROPRIO_STD
    return np.clip(normalized, -3.0, 3.0).astype(np.float32)


# ============================================================================
# Model Architecture (self-contained, matches remote_training_server version)
# ============================================================================

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.conv2(self.relu(self.conv1(x))))


class UnifiedBEVEncoder(nn.Module):
    def __init__(self, input_channels: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1)
        self.res1 = ResBlock(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.res2 = ResBlock(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self._output_dim = 64 * 4 * 4  # 1024

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.res1(x)
        x = self.relu(self.conv2(x))
        x = self.res2(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        return x.flatten(start_dim=1)

    @property
    def output_dim(self):
        return self._output_dim


class UnifiedBEVPPOPolicy(nn.Module):
    def __init__(self, action_dim: int = 2, proprio_dim: int = 6):
        super().__init__()
        self.bev_encoder = UnifiedBEVEncoder(input_channels=2)
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        fusion_dim = self.bev_encoder.output_dim + 32
        self.policy_net = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.value_net = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        nn.init.orthogonal_(self.policy_net[-1].weight, gain=0.1)
        nn.init.constant_(self.policy_net[-1].bias, 0.0)
        nn.init.orthogonal_(self.value_net[-1].weight, gain=0.01)
        nn.init.constant_(self.value_net[-1].bias, 0.0)

    def forward(self, bev_grid, proprio):
        bev_feats = self.bev_encoder(bev_grid)
        proprio_feats = self.proprio_encoder(proprio)
        fused = torch.cat([bev_feats, proprio_feats], dim=1)
        action_mean = self.policy_net(fused)
        value = self.value_net(fused).squeeze(-1)
        return action_mean, self.log_std, value

    def act(self, bev_grid, proprio, deterministic=False):
        action_mean, log_std, value = self.forward(bev_grid, proprio)
        std = log_std.exp().clamp(min=1e-6, max=2.0)
        dist = torch.distributions.Normal(action_mean, std)
        if deterministic:
            actions = torch.tanh(action_mean)
            log_probs = dist.log_prob(action_mean).sum(dim=-1)
        else:
            raw = dist.rsample()
            actions = torch.tanh(raw)
            log_probs = dist.log_prob(raw).sum(dim=-1)
            log_probs -= (2 * (np.log(2) - raw - F.softplus(-2 * raw))).sum(dim=-1)
        return actions, log_probs, value


# ============================================================================
# PPO Buffer
# ============================================================================

class PPOBuffer:
    """On-policy experience buffer with uint8 BEV storage for memory efficiency."""

    def __init__(self, capacity: int, proprio_dim: int = 6):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.bev = np.zeros((capacity, 2, 128, 128), dtype=np.uint8)
        self.proprio = np.zeros((capacity, proprio_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 2), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)
        self.log_probs = np.zeros((capacity,), dtype=np.float32)
        self.values = np.zeros((capacity,), dtype=np.float32)

    def add(self, bev, proprio, action, reward, done, log_prob, value):
        i = self.ptr
        self.bev[i] = (bev * 255.0).astype(np.uint8)
        self.proprio[i] = proprio
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done
        self.log_probs[i] = log_prob
        self.values[i] = value
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def compute_gae(self, last_value, gamma=0.99, lam=0.95):
        """Compute GAE advantages and returns for the current rollout."""
        n = self.size
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * (1.0 - self.dones[t]) - self.values[t]
            last_gae = delta + gamma * lam * (1.0 - self.dones[t]) * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values[:n]
        return advantages, returns

    def get_batches(self, batch_size, device):
        """Yield mini-batches for PPO update."""
        n = self.size
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            yield {
                'bev': torch.from_numpy(self.bev[idx].astype(np.float32) / 255.0).to(device),
                'proprio': torch.from_numpy(self.proprio[idx]).to(device),
                'actions': torch.from_numpy(self.actions[idx]).to(device),
                'returns': torch.from_numpy(self._returns[idx]).to(device),
                'advantages': torch.from_numpy(self._advantages[idx]).to(device),
                'old_log_probs': torch.from_numpy(self.log_probs[idx]).to(device),
            }

    def prepare_update(self, last_value, gamma=0.99, lam=0.95):
        """Compute GAE and store for mini-batch iteration."""
        self._advantages, self._returns = self.compute_gae(last_value, gamma, lam)
        # Normalize advantages
        adv_mean = self._advantages.mean()
        adv_std = self._advantages.std() + 1e-8
        self._advantages = (self._advantages - adv_mean) / adv_std

    def clear(self):
        self.ptr = 0
        self.size = 0


# ============================================================================
# Stuck Detector
# ============================================================================

class StuckDetector:
    def __init__(self, window_size=60, stuck_threshold=0.15):
        self.window_size = window_size
        self.stuck_threshold = stuck_threshold
        self.position_history = deque(maxlen=window_size)
        self.stuck_counter = 0
        self.recovery_mode = False
        self.recovery_steps = 0

    def update(self, odom_msg):
        pos = (odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y)
        self.position_history.append(pos)
        if len(self.position_history) < self.position_history.maxlen:
            return False
        positions = np.array(self.position_history)
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        total_distance = np.sum(distances)
        is_stuck = total_distance < self.stuck_threshold
        if is_stuck:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        if self.stuck_counter > 30 and not self.recovery_mode:
            self.recovery_mode = True
            self.recovery_steps = 45
        return is_stuck

    def get_recovery_action(self):
        if self.recovery_steps > 0:
            self.recovery_steps -= 1
            return np.array([-0.5, np.random.uniform(-0.8, 0.8)])
        else:
            self.recovery_mode = False
            self.stuck_counter = 0
            return None


# ============================================================================
# Main ROS2 Node
# ============================================================================

class PPOLocalRunner(Node):
    """Fully local PPO runner: inference + training on-device via ROS2."""

    def __init__(self):
        super().__init__('ppo_local_runner')

        # Parameters
        self.declare_parameter('max_linear_speed', 0.18)
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('inference_rate_hz', 30.0)
        self.declare_parameter('rollout_steps', 2048)
        self.declare_parameter('mini_batch_size', 512)
        self.declare_parameter('update_epochs', 10)
        self.declare_parameter('learning_rate', 3e-4)
        self.declare_parameter('clip_eps', 0.2)
        self.declare_parameter('gamma', 0.99)
        self.declare_parameter('gae_lambda', 0.95)
        self.declare_parameter('checkpoint_dir', './checkpoints_ppo')
        self.declare_parameter('log_dir', './logs_ppo')
        self.declare_parameter('checkpoint_interval', 5)
        self.declare_parameter('invert_linear_vel', False)

        self.max_linear = float(self.get_parameter('max_linear_speed').value)
        self.max_angular = float(self.get_parameter('max_angular_speed').value)
        self.inference_rate = float(self.get_parameter('inference_rate_hz').value)
        self.rollout_steps = int(self.get_parameter('rollout_steps').value)
        self.mini_batch_size = int(self.get_parameter('mini_batch_size').value)
        self.update_epochs = int(self.get_parameter('update_epochs').value)
        self.lr = float(self.get_parameter('learning_rate').value)
        self.clip_eps = float(self.get_parameter('clip_eps').value)
        self.gamma = float(self.get_parameter('gamma').value)
        self.gae_lambda = float(self.get_parameter('gae_lambda').value)
        self.checkpoint_dir = Path(str(self.get_parameter('checkpoint_dir').value))
        self.log_dir = Path(str(self.get_parameter('log_dir').value))
        self.checkpoint_interval = int(self.get_parameter('checkpoint_interval').value)
        self.invert_linear_vel = bool(self.get_parameter('invert_linear_vel').value)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Device
        self.device = torch.device('cpu')
        self.get_logger().info(f'Using device: {self.device}')

        # Policy + Optimizer
        self.policy = UnifiedBEVPPOPolicy(action_dim=2, proprio_dim=6).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        # Try loading latest checkpoint
        self._model_version = 0
        self._update_count = 0
        self._total_steps = 0
        self._load_latest_checkpoint()

        # PPO Buffer
        self.buffer = PPOBuffer(capacity=self.rollout_steps + 100, proprio_dim=6)

        # TensorBoard (optional)
        self._writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=str(self.log_dir))
            self.get_logger().info(f'TensorBoard logging to {self.log_dir}')
        except ImportError:
            self.get_logger().warn('TensorBoard not available, skipping logging')

        # Sensor state
        self._latest_depth_raw = None
        self._latest_scan = None
        self._latest_bev = None
        self._latest_odom = None
        self._latest_rf2o_odom = None
        self._latest_imu = None
        self._latest_wheel_vels = None
        self._min_forward_dist = 10.0
        self._safety_override = False
        self._velocity_confidence = 1.0
        self._latest_fused_yaw = 0.0

        # Curriculum
        self._curriculum_max_speed = self.max_linear

        # Action history
        self._prev_action = np.array([0.0, 0.0])
        self._prev_linear_cmds = deque(maxlen=20)
        self._prev_actions_buffer = deque(maxlen=30)

        # Gap following
        self._target_heading = 0.0
        self._bev_heading = 0.0
        self._max_depth_val = 0.0

        # Phase management
        self.phase_manager = PhaseManager(initial_phase='exploration')
        self._current_episode_length = 0
        self._current_episode_reward = 0.0
        self._episode_reward_history = deque(maxlen=50)
        self._state_visits = {}

        # Stuck/slip detection
        self.stuck_detector = StuckDetector(stuck_threshold=0.15)
        self._is_stuck = False
        self._consecutive_idle_steps = 0
        self._intent_without_motion_count = 0
        self._prev_min_clearance = 10.0
        self._steps_in_tight_space = 0

        # Wall avoidance
        self._steps_since_wall_stop = 0
        self._wall_stop_active = False
        self._wall_stop_steps = 0

        # Rotation tracking
        self._cumulative_rotation = 0.0
        self._last_yaw_for_rotation = None
        self._forward_progress_threshold = 0.3
        self._last_position_for_rotation = None
        self._revolution_penalty_triggered = False

        # Slip detection
        self._fwd_cmd_no_motion_count = 0
        self._slip_detected = False
        self._slip_recovery_active = False
        self._slip_backup_origin = None
        self._slip_backup_distance = 0.0
        self.SLIP_DETECTION_FRAMES = 15
        self.SLIP_CMD_THRESHOLD = 0.2
        self.SLIP_VEL_THRESHOLD = 0.03
        self.SLIP_BACKUP_LIMIT = 0.15

        # Sensor warmup
        self._sensor_warmup_complete = False
        self._sensor_warmup_countdown = 90

        # Training state
        self._training_in_progress = False
        self._model_ready = True
        self._warmup_active = False

        # BEV processor
        self.occupancy_processor = UnifiedBEVProcessor(grid_size=128, max_range=4.0)

        # ROS2 setup
        self.bridge = CvBridge()
        self._setup_subscribers()
        self._setup_publishers()

        # Inference timer
        self.create_timer(1.0 / self.inference_rate, self._control_loop)

        # Episode reset client
        self.reset_episode_client = self.create_client(Trigger, '/reset_episode')

        param_count = sum(p.numel() for p in self.policy.parameters())
        self.get_logger().info(
            f'PPO Local Runner initialized: {param_count:,} params, '
            f'rollout={self.rollout_steps}, lr={self.lr}'
        )

    # ========== ROS2 Setup ==========

    def _setup_subscribers(self):
        self.create_subscription(Image, '/camera/camera/depth/image_rect_raw',
                                 self._depth_cb, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odometry/filtered', self._odom_cb, 10)
        self.create_subscription(Odometry, '/odom_rf2o', self._rf2o_odom_cb, 10)
        self.create_subscription(Imu, '/imu/data', self._imu_cb, qos_profile_sensor_data)
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)
        self.create_subscription(Bool, '/emergency_stop', self._safety_cb, 10)
        self.create_subscription(Float32, '/velocity_confidence', self._vel_conf_cb, 10)

    def _setup_publishers(self):
        self.track_cmd_pub = self.create_publisher(Float32MultiArray, 'track_cmd_ai', 10)

    # ========== Sensor Callbacks ==========

    def _depth_cb(self, msg):
        self._latest_depth_raw = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

    def _scan_cb(self, msg):
        self._latest_scan = msg

    def _odom_cb(self, msg):
        self._latest_odom = (
            msg.pose.pose.position.x, msg.pose.pose.position.y,
            msg.twist.twist.linear.x, msg.twist.twist.angular.z
        )
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self._latest_fused_yaw = math.atan2(siny_cosp, cosy_cosp)
        self._is_stuck = self.stuck_detector.update(msg)

    def _rf2o_odom_cb(self, msg):
        self._latest_rf2o_odom = (msg.twist.twist.linear.x, msg.twist.twist.angular.z)

    def _imu_cb(self, msg):
        self._latest_imu = (
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z
        )

    def _joint_cb(self, msg):
        if len(msg.velocity) >= 4:
            self._latest_wheel_vels = (msg.velocity[2], msg.velocity[3])

    def _safety_cb(self, msg):
        self._safety_override = msg.data

    def _vel_conf_cb(self, msg):
        self._velocity_confidence = msg.data

    # ========== LiDAR Processing ==========

    def _find_best_gap_multiscale(self, ranges, angles, valid, scan_msg):
        if not np.any(valid):
            return 0.0, 0.0
        all_ranges = ranges.copy()
        all_ranges[~valid] = 0.0
        sort_idx = np.argsort(angles)
        sorted_angles = angles[sort_idx]
        sorted_ranges = all_ranges[sort_idx]
        if len(sorted_ranges) < 5:
            return 0.0, 0.0
        best_gap = {'angle': 0.0, 'depth': 0.0, 'score': -np.inf}
        for window_deg in [15, 25, 35]:
            window_rad = np.radians(window_deg)
            window_size = int(window_rad / scan_msg.angle_increment)
            window_size = max(3, min(window_size, len(sorted_ranges) // 3))
            if len(sorted_ranges) >= window_size:
                smoothed = np.convolve(sorted_ranges, np.ones(window_size) / window_size, mode='same')
                for i in range(window_size, len(smoothed) - window_size):
                    if smoothed[i] < 0.5:
                        continue
                    if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
                        angle = sorted_angles[i]
                        abs_angle = abs(angle)
                        if abs_angle < math.pi / 2:
                            forward_bias = 1.0 - (abs_angle / (math.pi / 2)) * 0.7
                        else:
                            forward_bias = 0.3 - ((abs_angle - math.pi / 2) / (math.pi / 2)) * 0.2
                        width_bonus = 1.0 + (window_deg / 35.0) * 0.3
                        score = smoothed[i] * forward_bias * width_bonus
                        if score > best_gap['score']:
                            best_gap = {'angle': angle, 'depth': smoothed[i], 'score': score}
        return best_gap['angle'], best_gap['depth']

    def _process_lidar_metrics(self, scan_msg):
        if not scan_msg:
            return 0.0, 0.0, 0.0
        ranges = np.array(scan_msg.ranges)
        valid = (ranges > 0.15) & (ranges < scan_msg.range_max) & np.isfinite(ranges)
        if not np.any(valid):
            return 3.0, 3.0, 0.0
        valid_ranges = ranges[valid]
        min_dist_all = np.min(valid_ranges)
        angles = scan_msg.angle_min + np.arange(len(ranges)) * scan_msg.angle_increment
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        left_mask = (angles > 0.78) & (angles < 2.35) & valid
        right_mask = (angles > -2.35) & (angles < -0.78) & valid
        l_dist = np.mean(ranges[left_mask]) if np.any(left_mask) else 3.0
        r_dist = np.mean(ranges[right_mask]) if np.any(right_mask) else 3.0
        mean_side_dist = (l_dist + r_dist) / 2.0
        best_angle, best_depth = self._find_best_gap_multiscale(ranges, angles, valid, scan_msg)
        target = np.clip(best_angle / math.pi, -1.0, 1.0)
        return min_dist_all, mean_side_dist, target

    # ========== Reward Function ==========

    def _get_current_phase(self):
        return self.phase_manager.phase

    def _compute_exploration_bonus(self, bev_grid):
        phase = self._get_current_phase()
        hash_size = 8 if phase == 'exploration' else 16
        bev_small = cv2.resize(bev_grid[0], (hash_size, hash_size))
        state_hash = tuple(bev_small.flatten().astype(np.uint8))
        if state_hash not in self._state_visits:
            self._state_visits[state_hash] = 0
        visit_count = self._state_visits[state_hash]
        magnitude = {'exploration': 0.20, 'learning': 0.12, 'refinement': 0.10}[phase]
        bonus = magnitude / (1.0 + np.sqrt(visit_count))
        self._state_visits[state_hash] += 1
        return bonus

    def _compute_action_diversity_bonus(self):
        if len(self._prev_actions_buffer) < 10:
            return 0.0
        actions = np.array(list(self._prev_actions_buffer))
        lin_std = np.std(actions[:, 0])
        ang_std = np.std(actions[:, 1])
        bonus = 0.05 * (lin_std + ang_std)
        return np.clip(bonus, 0.0, 0.15)

    def _compute_reward(self, action, linear_vel, angular_vel, min_lidar_dist,
                        side_clearance, collision, is_stuck, is_slipping=False,
                        slip_recovery_active=False, safety_blocked=False):
        VELOCITY_DEADBAND = 0.03
        if abs(linear_vel) < VELOCITY_DEADBAND:
            linear_vel = 0.0

        reward = 0.0
        target_speed = self._curriculum_max_speed
        left_track = action[0]
        right_track = action[1]
        phase = self._get_current_phase()

        if phase == 'exploration':
            forward_bonus_mult = 1.5
            spin_penalty_scale = 0.3
        elif phase == 'learning':
            forward_bonus_mult = 1.0
            spin_penalty_scale = 0.6
        else:
            forward_bonus_mult = 0.8
            spin_penalty_scale = 1.0

        cmd_fwd = (left_track + right_track) / 2.0

        # 1. Alive bonus + stagnation
        if phase == 'exploration':
            reward += 0.08
            if linear_vel < 0.08:
                self._consecutive_idle_steps += 1
            else:
                self._consecutive_idle_steps = 0
            if self._consecutive_idle_steps > 30:
                ramp = min((self._consecutive_idle_steps - 30) / 60.0, 1.0)
                reward -= 0.05 * ramp
        elif phase == 'learning':
            reward += 0.04
            if linear_vel < 0.08:
                self._consecutive_idle_steps += 1
                reward -= 0.08
            else:
                self._consecutive_idle_steps = 0
        else:
            if linear_vel < 0.08:
                self._consecutive_idle_steps += 1
                reward -= 0.1
            else:
                self._consecutive_idle_steps = 0

        # 2a. Intent reward
        intent_reward = 0.0
        if phase == 'exploration':
            if cmd_fwd > 0.05:
                if linear_vel < 0.03:
                    self._intent_without_motion_count += 1
                else:
                    self._intent_without_motion_count = 0
                decay = max(0.0, 1.0 - self._intent_without_motion_count / 60.0)
                intent_reward = 0.12 * min(cmd_fwd, 0.5) * decay
            else:
                self._intent_without_motion_count = 0
        elif phase == 'learning':
            if cmd_fwd > 0.1:
                if linear_vel < 0.03:
                    self._intent_without_motion_count += 1
                else:
                    self._intent_without_motion_count = 0
                decay = max(0.0, 1.0 - self._intent_without_motion_count / 60.0)
                intent_reward = 0.04 * min(cmd_fwd, 0.5) * decay
            else:
                self._intent_without_motion_count = 0
        reward += intent_reward

        # 2. Coupled forward reward
        meas_fwd = linear_vel / target_speed if target_speed > 0 else 0.0
        if cmd_fwd > 0 and meas_fwd > 0:
            max_abs = max(abs(left_track), abs(right_track))
            track_agreement = min(abs(left_track), abs(right_track)) / (max_abs + 1e-6) if max_abs > 0.05 else 1.0
            reward += min(cmd_fwd, meas_fwd) * forward_bonus_mult * (0.3 + 0.7 * track_agreement)

        # 2b. Speed bonus
        if phase == 'exploration':
            if linear_vel > 0.03:
                reward += 0.30 * min(linear_vel / target_speed, 1.0)
        elif phase == 'learning':
            if linear_vel > 0.05:
                reward += 0.20 * min(linear_vel / target_speed, 1.0)
        else:
            if linear_vel > 0.10:
                reward += 0.15 * min(linear_vel / target_speed, 1.0)

        # 3. Backward penalty
        if linear_vel < -0.03:
            if slip_recovery_active:
                reward += 0.1
            else:
                reward -= 0.15 + abs(linear_vel) * 0.8

        # 4. Tank steering
        if abs(cmd_fwd) < 0.1:
            symmetry_error = abs(abs(left_track) - abs(right_track))
            if symmetry_error > 0.2:
                reward -= 0.2 * symmetry_error * spin_penalty_scale

        max_abs_track = max(abs(left_track), abs(right_track))
        utilization = 0.0
        if max_abs_track > 0.1:
            utilization = min(abs(left_track), abs(right_track)) / max_abs_track
            should_penalize = utilization < 0.3
            if phase == 'exploration':
                should_penalize = should_penalize and (cmd_fwd < 0.05)
            if should_penalize:
                reward -= 0.3 * (1.0 - utilization) * spin_penalty_scale

        if cmd_fwd > 0.3:
            min_track = min(left_track, right_track)
            max_track = max(left_track, right_track)
            if max_track > 0.6 and min_track < 0.1:
                reward -= 0.1

        if utilization > 0.6 and cmd_fwd > 0.1:
            coord_bonus = 0.12 if phase == 'exploration' else 0.08
            reward += coord_bonus * utilization

        # 6. Smoothness
        action_diff = np.abs(action - self._prev_action)
        smoothness_mult = {'exploration': 0.02, 'learning': 0.03, 'refinement': 0.05}[phase]
        reward -= np.mean(action_diff) * smoothness_mult

        # 9. Exploration bonus
        if phase != 'refinement' and self._latest_bev is not None:
            reward += self._compute_exploration_bonus(self._latest_bev)

        # 10. Action diversity
        reward += self._compute_action_diversity_bonus()

        # 11. Gap-heading
        if abs(self._target_heading) > 0.1:
            intended_turn = right_track - left_track
            gap_direction = self._target_heading
            if linear_vel > 0.03:
                alignment_with_gap = intended_turn * gap_direction
                if alignment_with_gap > 0:
                    reward += 0.15 * min(alignment_with_gap, 0.4)
                elif abs(gap_direction) > 0.5:
                    reward -= 0.05

        # 12. Unstuck
        TIGHT_SPACE_THRESHOLD = 0.35
        if min_lidar_dist < TIGHT_SPACE_THRESHOLD:
            self._steps_in_tight_space += 1
        else:
            self._steps_in_tight_space = 0

        clearance_delta = min_lidar_dist - self._prev_min_clearance
        if min_lidar_dist < TIGHT_SPACE_THRESHOLD and clearance_delta > 0.02:
            tightness_factor = (TIGHT_SPACE_THRESHOLD - min_lidar_dist) / TIGHT_SPACE_THRESHOLD
            escape_bonus = clearance_delta * 2.0 * (1.0 + tightness_factor)
            reward += np.clip(escape_bonus, 0.0, 0.3)

        if min_lidar_dist < TIGHT_SPACE_THRESHOLD and self._steps_in_tight_space > 15:
            angular_action_magnitude = abs(right_track - left_track)
            if angular_action_magnitude > 0.3:
                reward += 0.1 * angular_action_magnitude

        # 13. Stuck recovery
        if is_stuck:
            fwd_effort = abs(left_track + right_track)
            reward -= 1.0 * fwd_effort
            rot_effort = abs(left_track - right_track)
            reward += 1.0 * rot_effort
            if left_track * right_track > 0:
                reward -= 0.5

        # 14. Slip
        if is_slipping:
            reward -= 0.3 * max(cmd_fwd, 0.0)

        # 15. Arc turn bonus
        if phase == 'exploration':
            arc_vel_thresh, arc_ang_thresh, arc_clear_thresh = 0.05, 0.2, 0.25
        elif phase == 'learning':
            arc_vel_thresh, arc_ang_thresh, arc_clear_thresh = 0.10, 0.3, 0.25
        else:
            arc_vel_thresh, arc_ang_thresh, arc_clear_thresh = 0.2, 0.5, 0.3
        if linear_vel > arc_vel_thresh and abs(angular_vel) > arc_ang_thresh and min_lidar_dist > arc_clear_thresh:
            reward += 0.3 * abs(linear_vel) * abs(angular_vel)

        # 16. Wall proximity
        if min_lidar_dist < 0.4 and linear_vel > 0.1:
            reward -= 0.25 * (0.4 - min_lidar_dist) * linear_vel

        # 17. Slip recovery bonus
        if slip_recovery_active and linear_vel < -0.02:
            reward += 0.2 * abs(linear_vel)

        # 18. Heading tracking
        if cmd_fwd > 0.05:
            track_diff = abs(right_track - left_track)
            straight_intent = max(0.0, 1.0 - track_diff * 5.0)
            if straight_intent > 0.1 and abs(angular_vel) > 0.1:
                reward -= 0.2 * abs(angular_vel) * straight_intent

        # 18b. Straight driving bonus
        if cmd_fwd > 0.05 and linear_vel > 0.05:
            straightness = max(0.0, 1.0 - abs(angular_vel) * 2.5)
            if straightness > 0.3:
                reward += 0.15 * straightness * min(linear_vel / target_speed, 1.0)

        # 19. Wall avoidance system
        wall_stopped = safety_blocked and linear_vel < 0.05
        if wall_stopped:
            self._wall_stop_active = True
            self._wall_stop_steps += 1
            self._steps_since_wall_stop = 0
        else:
            if self._wall_stop_active:
                self._wall_stop_active = False
                self._wall_stop_steps = 0
            self._steps_since_wall_stop += 1

        if not wall_stopped and linear_vel > 0.05:
            streak_factor = min(self._steps_since_wall_stop / 150.0, 1.0)
            vel_factor = min(linear_vel / target_speed, 1.0)
            clearance_factor = np.clip((min_lidar_dist - 0.3) / 0.5, 0.0, 1.0)
            avoidance_base = {'exploration': 0.20, 'learning': 0.25, 'refinement': 0.30}[phase]
            reward += avoidance_base * streak_factor * vel_factor * clearance_factor

        if wall_stopped:
            wall_stop_base = -0.5
            ramp = min(self._wall_stop_steps / 60.0, 1.0)
            penalty = wall_stop_base + (-0.3 * ramp)
            if phase == 'exploration':
                penalty *= 0.5
            elif phase == 'learning':
                penalty *= 0.75
            reward += penalty
            rot_effort = abs(left_track - right_track)
            if rot_effort > 0.3:
                reward += 0.15 * rot_effort
            if linear_vel < -0.02:
                reward += 0.10

        self._prev_min_clearance = min_lidar_dist
        return np.clip(reward, -1.0, 1.0)

    # ========== Control Loop ==========

    def _control_loop(self):
        if self._training_in_progress:
            stop_msg = Float32MultiArray()
            stop_msg.data = [0.0, 0.0]
            self.track_cmd_pub.publish(stop_msg)
            return

        if self._latest_depth_raw is None or self._latest_scan is None:
            return

        # Sensor warmup
        if not self._sensor_warmup_complete:
            self._sensor_warmup_countdown -= 1
            if self._sensor_warmup_countdown <= 0:
                self._sensor_warmup_complete = True
                self.get_logger().info('Sensor warmup complete')
            return

        # 1. Build BEV
        bev_grid = self.occupancy_processor.process(
            depth_img=self._latest_depth_raw,
            laser_scan=self._latest_scan
        )
        self._latest_bev = bev_grid

        # BEV gap heading
        laser_channel = bev_grid[0]
        front_half = laser_channel[64:128, :]
        free_space = 1.0 - front_half
        col_scores = np.mean(free_space, axis=0)
        forward_bias = np.zeros(128)
        forward_bias[54:74] = 0.1
        col_scores = col_scores + forward_bias
        col_scores = np.convolve(col_scores, np.ones(13) / 13, mode='same')
        best_col = np.argmax(col_scores)
        raw_heading = (64 - best_col) / 64.0
        alpha = 0.3
        self._bev_heading = alpha * raw_heading + (1 - alpha) * self._bev_heading

        # Min forward distance from BEV
        center_patch = laser_channel[118:128, 59:69]
        obstacle_density = np.mean(center_patch) if center_patch.size > 0 else 0.0
        self._min_forward_dist = (1.0 - obstacle_density) * 4.0

        # LiDAR metrics
        lidar_min, lidar_sides, gap_heading = self._process_lidar_metrics(self._latest_scan)
        self._target_heading = gap_heading

        # Velocity from rf2o or EKF fallback
        current_linear = 0.0
        current_angular = 0.0
        if self._latest_rf2o_odom:
            current_linear = self._latest_rf2o_odom[0]
            current_angular = self._latest_rf2o_odom[1]
        elif self._latest_odom:
            current_linear = self._latest_odom[2]
            current_angular = self._latest_odom[3]

        if self.invert_linear_vel:
            current_linear = -current_linear

        # Slip detection
        cmd_fwd_for_slip = (self._prev_action[0] + self._prev_action[1]) / 2.0
        if cmd_fwd_for_slip > self.SLIP_CMD_THRESHOLD and abs(current_linear) < self.SLIP_VEL_THRESHOLD:
            self._fwd_cmd_no_motion_count += 1
            if self._fwd_cmd_no_motion_count >= self.SLIP_DETECTION_FRAMES and not self._slip_detected:
                self._slip_detected = True
                if self._latest_odom and not self._slip_recovery_active:
                    self._slip_recovery_active = True
                    self._slip_backup_origin = (self._latest_odom[0], self._latest_odom[1])
                    self._slip_backup_distance = 0.0
        else:
            if self._fwd_cmd_no_motion_count > 0:
                self._fwd_cmd_no_motion_count = 0
            if self._slip_detected:
                self._slip_detected = False
                self._slip_recovery_active = False
                self._slip_backup_origin = None
                self._slip_backup_distance = 0.0

        if self._slip_recovery_active and self._slip_backup_origin and self._latest_odom:
            x, y = self._latest_odom[0], self._latest_odom[1]
            ox, oy = self._slip_backup_origin
            self._slip_backup_distance = math.sqrt((x - ox) ** 2 + (y - oy) ** 2)
            if self._slip_backup_distance >= self.SLIP_BACKUP_LIMIT:
                self._slip_recovery_active = False

        # Rotation tracking
        if self._last_yaw_for_rotation is not None:
            yaw_delta = self._latest_fused_yaw - self._last_yaw_for_rotation
            if yaw_delta > math.pi:
                yaw_delta -= 2 * math.pi
            elif yaw_delta < -math.pi:
                yaw_delta += 2 * math.pi
            self._cumulative_rotation += abs(yaw_delta)
            if self._cumulative_rotation >= 2 * math.pi:
                self._revolution_penalty_triggered = True
                self._cumulative_rotation = 0.0
        self._last_yaw_for_rotation = self._latest_fused_yaw

        if self._latest_odom and self._last_position_for_rotation is not None:
            x, y = self._latest_odom[0], self._latest_odom[1]
            last_x, last_y = self._last_position_for_rotation
            distance = math.sqrt((x - last_x) ** 2 + (y - last_y) ** 2)
            if distance > self._forward_progress_threshold:
                self._cumulative_rotation = 0.0
                self._last_position_for_rotation = (x, y)
        elif self._latest_odom:
            self._last_position_for_rotation = (self._latest_odom[0], self._latest_odom[1])

        # Proprioception
        proprio_raw = np.array([
            lidar_min,
            self._prev_action[0],
            self._prev_action[1],
            current_linear,
            current_angular,
            self._bev_heading
        ], dtype=np.float32)
        proprio = normalize_proprio(proprio_raw)

        # 2. Inference
        bev_tensor = torch.from_numpy(bev_grid[None, ...]).float().to(self.device)
        proprio_tensor = torch.from_numpy(proprio[None, ...]).to(self.device)

        # Random warmup for first model version
        if self._model_version == 0 and self._total_steps < self.rollout_steps:
            if not self._warmup_active:
                self._warmup_active = True
                self.get_logger().info('Random exploration warmup...')

            # Diverse random actions (same distribution as SAC warmup)
            rand_mode = np.random.rand()
            if rand_mode < 0.30:
                fast = np.random.uniform(0.5, 1.0)
                slow = max(fast * np.random.uniform(0.5, 0.9), 0.3)
                if np.random.rand() < 0.5:
                    action_np = np.array([fast, slow])
                else:
                    action_np = np.array([slow, fast])
            elif rand_mode < 0.55:
                base = np.random.uniform(0.6, 1.0)
                offset = np.random.uniform(0.15, 0.35)
                slow = max(base - offset, 0.3)
                if np.random.rand() < 0.5:
                    action_np = np.array([base, slow])
                else:
                    action_np = np.array([slow, base])
            elif rand_mode < 0.70:
                speed = np.random.uniform(0.5, 1.0)
                action_np = np.array([speed, speed])
            elif rand_mode < 0.80:
                spin = np.random.uniform(0.4, 0.6)
                if np.random.rand() < 0.5:
                    action_np = np.array([spin, -spin])
                else:
                    action_np = np.array([-spin, spin])
            elif rand_mode < 0.90:
                backup = np.random.uniform(0.3, 0.6)
                action_np = np.array([-backup, -backup])
            else:
                action_np = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])

            # Safety speed scaling
            clearance = lidar_min if lidar_min > 0.05 else self._min_forward_dist
            if clearance < 0.2:
                scale = 0.3
            elif clearance < 0.5:
                scale = 0.3 + 0.7 * (clearance - 0.2) / 0.3
            else:
                scale = 1.0
            action_np = np.clip(action_np * scale, -1.0, 1.0)

            # Get value estimate for buffer
            with torch.no_grad():
                _, _, value = self.policy(bev_tensor, proprio_tensor)
            value_np = value.item()
            log_prob_np = 0.0  # Random actions, no meaningful log_prob
        else:
            if self._warmup_active:
                self._warmup_active = False
                self.get_logger().info('Warmup complete, switching to learned policy')

            with torch.no_grad():
                action_t, log_prob_t, value_t = self.policy.act(bev_tensor, proprio_tensor)
            action_np = action_t[0].cpu().numpy()
            log_prob_np = log_prob_t.item()
            value_np = value_t.item()

        # 3. Execute action
        left_track = 0.0
        right_track = 0.0
        is_stuck = self._is_stuck
        monitor_blocking = self._safety_override

        effective_min_dist = self._min_forward_dist
        if self._min_forward_dist < 0.05 and lidar_min > 0.2:
            effective_min_dist = lidar_min

        if is_stuck and effective_min_dist < 0.12:
            depth_safety = effective_min_dist < 0.05
        else:
            depth_safety = not monitor_blocking and effective_min_dist < 0.12
        safety_triggered = monitor_blocking or depth_safety

        def apply_soft_deadzone(val, min_val):
            if abs(val) < 0.001:
                return 0.0
            return math.copysign(min_val + (1.0 - min_val) * abs(val), val)

        MIN_TRACK = 0.25

        if self._slip_detected and self._slip_recovery_active and not safety_triggered:
            if not hasattr(self, '_slip_recovery_turn_dir'):
                self._slip_recovery_turn_dir = 1.0 if np.random.rand() > 0.5 else -1.0
            recovery_left = -0.4 + self._slip_recovery_turn_dir * 0.15
            recovery_right = -0.4 - self._slip_recovery_turn_dir * 0.15
            left_track = float(math.copysign(MIN_TRACK + (1.0 - MIN_TRACK) * abs(recovery_left), recovery_left))
            right_track = float(math.copysign(MIN_TRACK + (1.0 - MIN_TRACK) * abs(recovery_right), recovery_right))
            actual_action = np.array([left_track, right_track])
            collision = False
        elif monitor_blocking:
            left_track = min(apply_soft_deadzone(action_np[0], MIN_TRACK), 0.0)
            right_track = min(apply_soft_deadzone(action_np[1], MIN_TRACK), 0.0)
            actual_action = np.array([left_track, right_track])
            collision = False
        elif depth_safety:
            left_track = -0.3
            right_track = -0.3
            actual_action = np.array([-0.3, -0.3])
            collision = self._sensor_warmup_complete
        else:
            left_track = apply_soft_deadzone(action_np[0], MIN_TRACK)
            right_track = apply_soft_deadzone(action_np[1], MIN_TRACK)
            actual_action = np.array([left_track, right_track])
            collision = False
            if hasattr(self, '_slip_recovery_turn_dir'):
                delattr(self, '_slip_recovery_turn_dir')

        # Track action diversity
        self._prev_actions_buffer.append(actual_action.copy())

        # Publish
        track_msg = Float32MultiArray()
        track_msg.data = [float(left_track), float(right_track)]
        self.track_cmd_pub.publish(track_msg)

        # 4. Reward
        reward = self._compute_reward(
            actual_action, current_linear, current_angular,
            lidar_min, lidar_sides, collision, is_stuck,
            is_slipping=self._slip_detected,
            slip_recovery_active=self._slip_recovery_active,
            safety_blocked=monitor_blocking
        )

        if np.isnan(reward) or np.isinf(reward):
            return

        # 5. Store transition
        self.buffer.add(bev_grid, proprio, actual_action, reward, collision, log_prob_np, value_np)
        self._total_steps += 1
        self._current_episode_reward += reward
        self._current_episode_length += 1

        # Episode boundary on collision
        if collision:
            self._trigger_episode_reset()
            self.phase_manager.record_episode(
                reward=self._current_episode_reward,
                length=self._current_episode_length,
                collision=True
            )
            self._episode_reward_history.append(self._current_episode_reward)
            self._current_episode_reward = 0.0
            self._current_episode_length = 0

        # Update state
        self._prev_action = actual_action
        self._prev_linear_cmds.append(actual_action[0])

        # Log periodically
        if self._total_steps % 300 == 0:
            phase = self._get_current_phase()
            avg_rew = np.mean(list(self._episode_reward_history)) if self._episode_reward_history else 0.0
            self.get_logger().info(
                f'Step {self._total_steps} | Phase: {phase} | '
                f'Buffer: {self.buffer.size}/{self.rollout_steps} | '
                f'AvgRew: {avg_rew:.3f} | v{self._model_version}'
            )

        # 6. Trigger training when buffer is full
        if self.buffer.size >= self.rollout_steps:
            self._run_ppo_update()

    # ========== PPO Training ==========

    def _run_ppo_update(self):
        self._training_in_progress = True
        self.get_logger().info(f'Starting PPO update #{self._update_count + 1} ({self.buffer.size} steps)...')

        # Stop robot
        stop_msg = Float32MultiArray()
        stop_msg.data = [0.0, 0.0]
        self.track_cmd_pub.publish(stop_msg)

        t0 = time.time()

        # Get last value for GAE bootstrap
        if self._latest_bev is not None:
            bev_t = torch.from_numpy(self._latest_bev[None, ...]).float().to(self.device)
            proprio_raw = np.array([
                self._min_forward_dist, self._prev_action[0], self._prev_action[1],
                0.0, 0.0, self._bev_heading
            ], dtype=np.float32)
            proprio_t = torch.from_numpy(normalize_proprio(proprio_raw)[None, ...]).to(self.device)
            with torch.no_grad():
                _, _, last_value = self.policy(bev_t, proprio_t)
            last_value = last_value.item()
        else:
            last_value = 0.0

        # Compute GAE
        self.buffer.prepare_update(last_value, self.gamma, self.gae_lambda)

        # PPO epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(self.update_epochs):
            for batch in self.buffer.get_batches(self.mini_batch_size, self.device):
                action_mean, log_std, values = self.policy(batch['bev'], batch['proprio'])
                std = log_std.exp().clamp(min=1e-6, max=2.0)
                dist = torch.distributions.Normal(action_mean, std)

                # Inverse tanh to get raw actions for log_prob calculation
                actions_clamped = batch['actions'].clamp(-0.999, 0.999)
                raw_actions = torch.atanh(actions_clamped)
                new_log_probs = dist.log_prob(raw_actions).sum(dim=-1)
                new_log_probs -= (2 * (np.log(2) - raw_actions - F.softplus(-2 * raw_actions))).sum(dim=-1)

                # PPO clipped objective
                ratio = (new_log_probs - batch['old_log_probs']).exp()
                surr1 = ratio * batch['advantages']
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch['advantages']
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * (values - batch['returns']).pow(2).mean()

                # Entropy bonus
                entropy = dist.entropy().sum(dim=-1).mean()

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        dt = time.time() - t0
        self._update_count += 1
        self._model_version += 1

        avg_pl = total_policy_loss / max(n_updates, 1)
        avg_vl = total_value_loss / max(n_updates, 1)
        avg_ent = total_entropy / max(n_updates, 1)

        self.get_logger().info(
            f'PPO update done in {dt:.1f}s | '
            f'PL: {avg_pl:.4f} | VL: {avg_vl:.4f} | Ent: {avg_ent:.4f} | '
            f'v{self._model_version}'
        )

        # TensorBoard logging
        if self._writer:
            self._writer.add_scalar('loss/policy', avg_pl, self._update_count)
            self._writer.add_scalar('loss/value', avg_vl, self._update_count)
            self._writer.add_scalar('loss/entropy', avg_ent, self._update_count)
            self._writer.add_scalar('training/model_version', self._model_version, self._update_count)
            self._writer.add_scalar('training/total_steps', self._total_steps, self._update_count)
            if self._episode_reward_history:
                self._writer.add_scalar('reward/avg_episode', np.mean(list(self._episode_reward_history)), self._update_count)

        # Save checkpoint
        if self._update_count % self.checkpoint_interval == 0:
            self._save_checkpoint()

        # Clear buffer and resume
        self.buffer.clear()
        self._training_in_progress = False

    # ========== Checkpoint Management ==========

    def _save_checkpoint(self):
        ckpt_path = self.checkpoint_dir / f'ppo_v{self._model_version}.pt'
        latest_path = self.checkpoint_dir / 'latest.pt'

        state = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_version': self._model_version,
            'update_count': self._update_count,
            'total_steps': self._total_steps,
        }
        torch.save(state, ckpt_path)
        torch.save(state, latest_path)
        self.get_logger().info(f'Checkpoint saved: {ckpt_path}')

        # Export ONNX
        try:
            self._export_onnx()
        except Exception as e:
            self.get_logger().warn(f'ONNX export failed: {e}')

    def _export_onnx(self):
        onnx_path = self.checkpoint_dir / 'latest_actor.onnx'
        self.policy.eval()
        dummy_bev = torch.zeros(1, 2, 128, 128)
        dummy_proprio = torch.zeros(1, 6)
        torch.onnx.export(
            self.policy, (dummy_bev, dummy_proprio),
            str(onnx_path),
            input_names=['bev', 'proprio'],
            output_names=['action_mean', 'log_std', 'value'],
            opset_version=12,
            dynamic_axes={'bev': {0: 'batch'}, 'proprio': {0: 'batch'}}
        )
        self.policy.train()
        self.get_logger().info(f'ONNX exported: {onnx_path}')

    def _load_latest_checkpoint(self):
        latest_path = self.checkpoint_dir / 'latest.pt'
        if latest_path.exists():
            try:
                state = torch.load(latest_path, map_location=self.device, weights_only=False)
                self.policy.load_state_dict(state['policy_state_dict'])
                self.optimizer.load_state_dict(state['optimizer_state_dict'])
                self._model_version = state.get('model_version', 0)
                self._update_count = state.get('update_count', 0)
                self._total_steps = state.get('total_steps', 0)
                self.get_logger().info(
                    f'Loaded checkpoint v{self._model_version} '
                    f'({self._total_steps} steps, {self._update_count} updates)'
                )
            except Exception as e:
                self.get_logger().warn(f'Failed to load checkpoint: {e}')

    # ========== Episode Reset ==========

    def _trigger_episode_reset(self):
        if not self.reset_episode_client.wait_for_service(timeout_sec=1.0):
            return
        request = Trigger.Request()
        future = self.reset_episode_client.call_async(request)

        def _log(fut):
            try:
                resp = fut.result()
                if resp.success:
                    self.get_logger().info(f'Episode reset: {resp.message}')
            except Exception:
                pass

        future.add_done_callback(_log)

    def destroy_node(self):
        # Save final checkpoint
        if self._update_count > 0:
            self._save_checkpoint()
        if self._writer:
            self._writer.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PPOLocalRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
