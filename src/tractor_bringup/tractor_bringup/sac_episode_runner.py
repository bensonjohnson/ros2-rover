#!/usr/bin/env python3
"""SAC Autonomous Episode Runner for Rover.

Runs continuous SAC inference on NPU, collects experience tuples,
calculates dense rewards, and asynchronously syncs with V620 server via NATS.
"""

import os
import math
import time
import threading
import queue
import tempfile
import subprocess
import asyncio
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from collections import deque
from tqdm import tqdm

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import nats
import cv2
from cv_bridge import CvBridge

# Import serialization utilities
from tractor_bringup.serialization_utils import (
    serialize_batch, deserialize_batch,
    serialize_model_update, deserialize_model_update,
    serialize_metadata, deserialize_metadata,
    serialize_status, deserialize_status
)

from tractor_bringup.occupancy_processor import DepthToOccupancy, ScanToOccupancy, LocalMapper, MultiChannelOccupancy, RawSensorProcessor, RGBDProcessor

# ROS2 Messages
from sensor_msgs.msg import Image, Imu, JointState, MagneticField, LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger

# RKNN Support
try:
    from rknnlite.api import RKNNLite
    HAS_RKNN = True
except ImportError:
    HAS_RKNN = False
    print("⚠ RKNNLite not available - cannot run on NPU")

class SACEpisodeRunner(Node):
    """Continuous SAC runner with async data collection."""

    def __init__(self) -> None:
        super().__init__('sac_episode_runner')

        # Parameters
        self.declare_parameter('nats_server', 'nats://nats.gokickrocks.org:4222')
        self.declare_parameter('algorithm', 'sac')
        self.declare_parameter('max_linear_speed', 0.18)
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('inference_rate_hz', 30.0)
        self.declare_parameter('batch_size', 64)  # Send data every N steps
        self.declare_parameter('collection_duration', 180.0) # Seconds to collect before triggering training

        self.nats_server = str(self.get_parameter('nats_server').value)
        self.algorithm = str(self.get_parameter('algorithm').value)
        self.max_linear = float(self.get_parameter('max_linear_speed').value)
        self.max_angular = float(self.get_parameter('max_angular_speed').value)
        self.inference_rate = float(self.get_parameter('inference_rate_hz').value)
        self.batch_size = int(self.get_parameter('batch_size').value)
        self.collection_duration = float(self.get_parameter('collection_duration').value)

        # TQDM Dashboard (Initialize FIRST to avoid race conditions with threads)
        print("\033[H\033[J", end="") # Clear screen
        print("==================================================")
        print("         SAC ROVER RUNNER (V620)                  ")
        print("==================================================")
        self.pbar = tqdm(total=2000, desc="⏳ Server Training", unit="step", dynamic_ncols=True)
        self.total_steps = 0
        self.episode_reward = 0.0

        # State (raw sensors)
        self._latest_rgb = None # Keep for debug/logging if needed, but not used for model
        self._latest_depth_raw = None  # Raw depth from camera (424, 240) uint16
        self._latest_scan = None
        # Processed sensor data for model
        self._latest_laser = None  # (128, 128) float32 - Binary laser occupancy
        self._latest_depth = None  # (424, 240) float32 - Processed/normalized depth
        self._latest_odom = None
        self._latest_imu = None
        self._latest_mag = None
        self._latest_wheel_vels = None
        self._min_forward_dist = 10.0
        self._safety_override = False
        self._velocity_confidence = 1.0  # Velocity estimate confidence (0.0-1.0)

        # Curriculum State (updated by server)
        self._curriculum_collision_dist = 0.5
        self._curriculum_max_speed = 0.1

        # Buffers for batching
        self._data_buffer = {
            'laser': [], 'rgbd': [], 'proprio': [],
            'actions': [], 'rewards': [], 'dones': []
        }
        self._buffer_lock = threading.Lock()

        # Model State
        self._rknn_runtime = None
        self._model_ready = True  # Allow random exploration initially
        self._temp_dir = Path(tempfile.mkdtemp(prefix='sac_rover_'))
        self._calibration_dir = Path("./calibration_data")
        self._calibration_dir.mkdir(exist_ok=True)
        self._current_model_version = -1
        self._model_update_needed = False
        
        # Warmup State
        self._warmup_start_time = 0.0
        self._warmup_active = False
        self._sensor_warmup_complete = False  # Flag for sensor stabilization
        self._sensor_warmup_countdown = 50  # ~1.7 seconds at 30Hz
        self._prev_odom_update = None
        
        # Previous action for smoothness reward
        self._prev_action = np.array([0.0, 0.0])
        self._prev_linear_cmds = deque(maxlen=20) # For oscillation detection
        self._prev_angular_actions = deque(maxlen=10) # For action smoothness tracking

        # Gap Following State
        self._target_heading = 0.0 # -1.0 (Left) to 1.0 (Right)
        self._prev_target_heading = 0.0 # For rate limiting
        self._max_depth_val = 0.0

        # IMU State for Stitching
        self._latest_imu_yaw = 0.0
        self._prev_imu_yaw = None

        # NATS Setup (will be initialized in background thread)
        self.nc = None
        self.js = None

        # Background Threads
        self._stop_event = threading.Event()
        self._initial_sync_done = threading.Event() # Wait for NATS connection
        self._last_model_update = 0.0
        self._nats_thread = threading.Thread(target=self._run_nats_loop, daemon=True)
        self._nats_thread.start()

        # ROS2 Setup
        self.bridge = CvBridge()
        # Raw sensor processor for dual-encoder SAC architecture
        # Processes laser to 128×128 binary occupancy, depth to full 424×240 resolution
        self.occupancy_processor = RawSensorProcessor(
            grid_size=128,  # Laser occupancy grid size
            max_range=4.0   # Maximum sensor range for normalization
        )
        # RGBD processor for RGB-D fusion
        self.rgbd_processor = RGBDProcessor(max_range=4.0)
        self._setup_subscribers()
        self._setup_publishers()
        
        # Inference Timer
        self.create_timer(1.0 / self.inference_rate, self._control_loop)

        # TQDM Dashboard
        print("\033[H\033[J", end="") # Clear screen
        print("==================================================")
        print("         SAC ROVER RUNNER (V620)                  ")
        print("==================================================")
        self.pbar = tqdm(total=self.batch_size, desc="🚜 Collecting", unit="step", dynamic_ncols=True)
        self.total_steps = 0
        self.episode_reward = 0.0

        # Episode reset client for encoder baseline reset
        self.reset_episode_client = self.create_client(Trigger, '/reset_episode')

        self.get_logger().info('🚀 SAC Runner Initialized')

    def _setup_subscribers(self):
        self.create_subscription(Image, '/camera/camera/color/image_raw', self._rgb_cb, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self._depth_cb, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, qos_profile_sensor_data)
        # Odometry: Use Fused EKF Output
        self.create_subscription(Odometry, '/odometry/filtered', self._odom_cb, 10)
        self.create_subscription(Imu, '/imu/data', self._imu_cb, qos_profile_sensor_data)
        # self.create_subscription(MagneticField, '/imu/mag', self._mag_cb, qos_profile_sensor_data) # Removed due to noise
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)
        # self.create_subscription(Float32, '/min_forward_distance', self._dist_cb, 10) # Calculated locally now
        self.create_subscription(Bool, '/safety_monitor_status', self._safety_cb, 10)
        self.create_subscription(Float32, '/velocity_confidence', self._vel_conf_cb, 10)
        # Initialize collection timer
        self._collection_start_time = time.time()

    def _setup_publishers(self):
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel_ai', 10)

    # Callbacks
    def _rgb_cb(self, msg):
        self._latest_rgb = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
    def _depth_cb(self, msg):
        # Use passthrough to get raw 16-bit depth (uint16 mm)
        d = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        self._latest_depth_raw = d  # Store raw for processing in control loop
    def _scan_cb(self, msg):
        self._latest_scan = msg
        
    def _odom_cb(self, msg):
        self._latest_odom = (
            msg.pose.pose.position.x, 
            msg.pose.pose.position.y, 
            msg.twist.twist.linear.x, 
            msg.twist.twist.angular.z
        )
    def _imu_cb(self, msg): 
        self._latest_imu = (
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z
        )
        
        # Extract Yaw from Quaternion
        q = msg.orientation
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self._latest_imu_yaw = math.atan2(siny_cosp, cosy_cosp)
    def _mag_cb(self, msg): self._latest_mag = (msg.magnetic_field.x, msg.magnetic_field.y, msg.magnetic_field.z)
    def _joint_cb(self, msg):
        if len(msg.velocity) >= 4: self._latest_wheel_vels = (msg.velocity[2], msg.velocity[3])
    def _safety_cb(self, msg): self._safety_override = msg.data
    def _vel_conf_cb(self, msg): self._velocity_confidence = msg.data

    def _process_lidar_metrics(self, scan_msg) -> Tuple[float, float, float]:
        """
        Extract key metrics from LiDAR scan.
        Returns:
            min_dist_all (float): Closest obstacle in 360 degrees (Safety Bubble)
            mean_side_dist (float): Average distance on left/right (for centering/clearing)
            gap_heading (float): Heading towards largest open space (-1..1)
        """
        if not scan_msg:
            return 0.0, 0.0, 0.0

        ranges = np.array(scan_msg.ranges)
        
        # 1. Strict Filtering (NaN, Inf, Range Limits)
        # Using 0.05 and range_max from message
        valid = (ranges > 0.05) & (ranges < scan_msg.range_max) & np.isfinite(ranges)
        
        if not np.any(valid):
            # No valid data? Assume wide open (safest for reward, but risky for nav)
            # Or assume blocked? If completely blind, stop.
            # Let's return safe values but valid=False signal implicitly via 0.0 heading
            return 3.0, 3.0, 0.0

        # Used for stats
        valid_ranges = ranges[valid]

        # 1. Safety Bubble
        min_dist_all = np.min(valid_ranges)

        # 2. Angle Handling
        # LD19/STL19p often outputs 0..2PI. We want -PI..PI for steering.
        # angles = angle_min + i * increment
        angles = scan_msg.angle_min + np.arange(len(ranges)) * scan_msg.angle_increment
        
        # Wrap angles to [-PI, PI]
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        
        # 3. Side Clearance
        # Left: +45 to +135 deg (+0.78 to +2.35 rad)
        # Right: -135 to -45 deg (-2.35 to -0.78 rad)
        left_mask = (angles > 0.78) & (angles < 2.35) & valid
        right_mask = (angles > -2.35) & (angles < -0.78) & valid
        
        l_dist = np.mean(ranges[left_mask]) if np.any(left_mask) else 3.0
        r_dist = np.mean(ranges[right_mask]) if np.any(right_mask) else 3.0
        mean_side_dist = (l_dist + r_dist) / 2.0

        # 4. Gap Finding
        # Find 20-degree sector with max average depth
        # We need to sort by angle to do a proper convolution or sliding window on the circle?
        # Actually, since 'angles' might be jumbled after wrapping if scan wasn't -PI..PI,
        # we should sort the data by angle first to ensure continuity.
        
        sort_idx = np.argsort(angles)
        sorted_angles = angles[sort_idx]
        sorted_ranges = ranges[sort_idx]
        sorted_valid = valid[sort_idx]
        
        # Fill invalid with 0.0 (treat as obstacle/unknown for gap finding)
        ranges_gap = sorted_ranges.copy()
        ranges_gap[~sorted_valid] = 0.0
        
        # Window size ~20 degrees
        # avg_increment = (max - min) / len? Or just use scan_msg.angle_increment
        window_size = int(np.radians(20) / scan_msg.angle_increment)
        window_size = max(1, window_size)
        
        # Circular convolution? Or just valid range?
        # Scan usually covers 360.
        # Pad array for circular continuity
        ranges_padded = np.pad(ranges_gap, (window_size//2, window_size//2), mode='wrap')
        
        # Convolve
        smoothed = np.convolve(ranges_padded, np.ones(window_size)/window_size, mode='same')
        # Remove padding
        if window_size//2 > 0:
             smoothed = smoothed[window_size//2 : -window_size//2]
        else:
             smoothed = ranges_gap
        
        # Find best index in smoothed (matches length of ranges_gap potentially, or close)
        # 'valid' mode output length = N - K + 1. 
        # Actually simplest is mode='same' on original and handle wrapping manually or ignore edge effect.
        # Let's use mode='same' on unpadded for simplicity, assuming adequate standard scan.
        # But wait, gap might be at -PI/PI boundary (behind?). 
        # Typically we want forward gaps.
        # Let's focus on [-PI/2, PI/2] (Front 180).
        
        # Filter for front sector only (±60° = ±1.05 rad) to avoid extreme turns
        # Narrower than ±90° to keep rover focused on forward motion
        front_mask = (sorted_angles > -1.05) & (sorted_angles < 1.05)
        
        # If we have front data
        if np.any(front_mask):
            # Extract front arc
            front_ranges = ranges_gap[front_mask]
            front_angles = sorted_angles[front_mask]
            
            # Smooth front arc
            if len(front_ranges) >= window_size:
                smoothed_front = np.convolve(front_ranges, np.ones(window_size)/window_size, mode='same')

                # Apply forward bias: Prefer gaps closer to straight ahead
                # Weight each gap by (1 - |angle|/max_angle) to favor forward direction
                forward_bias = 1.0 - (np.abs(front_angles) / 1.05)  # 1.0 at center, 0.0 at ±60°
                biased_scores = smoothed_front * (0.9 + 0.1 * forward_bias)  # 90% distance + 10% forward bias

                best_idx_local = np.argmax(biased_scores)
                best_angle = front_angles[best_idx_local]
            else:
                # Too few points, pick max individual with forward bias
                forward_bias = 1.0 - (np.abs(front_angles) / 1.05)
                biased_scores = front_ranges * (0.9 + 0.1 * forward_bias)
                best_idx_local = np.argmax(biased_scores)
                best_angle = front_angles[best_idx_local]
        else:
            # Fallback to 360 search if front is blocked?
            # Or just assume forward if everything fails?
            best_angle = 0.0
            
        # Map to Heading -1..1
        target = np.clip(best_angle / (math.pi/2), -1.0, 1.0)
        
        # DEBUG: Log if target is stuck at extremes
        if abs(target) > 0.9:
            self.get_logger().info(f"🔍 Gap Debug: Best Angle={best_angle:.2f} rad, Target={target:.2f}")
            self.get_logger().info(f"   Ranges: Min={min_dist_all:.2f}, Max={np.max(valid_ranges):.2f}, Mean={np.mean(valid_ranges):.2f}")
            # self.get_logger().info(f"   Gap Window Avg: {smoothed[best_idx]:.2f} (Index {best_idx}/{len(ranges)})")
        
        return min_dist_all, mean_side_dist, target

    def _compute_reward(self, action, linear_vel, angular_vel, min_lidar_dist, side_clearance, collision):
        """
        Simplified reward with 5 clear components:
        1. Forward progress scaled by safety
        2. Collision penalty
        3. Exploration bonus (implicit via exploration history in observation)
        4. Smooth control penalty
        5. Idle penalty
        """
        reward = 0.0
        target_speed = self._curriculum_max_speed

        # Extract metrics
        min_dist = min_lidar_dist

        # 1. Forward progress with safety scaling (primary objective)
        safety_factor = np.clip((min_dist - 0.2) / 0.4, 0.0, 1.0)  # 0 at 0.2m, 1 at 0.6m
        if linear_vel > 0.01:
            speed_ratio = linear_vel / target_speed
            reward += speed_ratio * safety_factor * 2.0  # Max +2.0
        else:
            reward -= 0.5  # Idle penalty

        # 2. Collision penalty (terminal signal)
        if collision or self._safety_override:
            reward -= 5.0
            return np.clip(reward, -1.0, 1.0)  # Early return

        # 3. Exploration bonus (new grid cells)
        # This is handled by exploration_map in observation
        # Implicit via Q-function learning: new areas → more future rewards

        # 4. Smooth control (anti-jerk)
        angular_change = abs(action[1] - self._prev_action[1])
        if angular_change > 0.5:  # Large steering jerk
            reward -= angular_change * 0.3

        # 5. Proximity penalty (gradual, not cliff)
        if min_dist < 0.3:
            proximity_penalty = (0.3 - min_dist) / 0.3  # 0 to 1
            reward -= proximity_penalty * 1.0

        return np.clip(reward, -1.0, 1.0)

    def _compute_reward_old(self, action, linear_vel, angular_vel, clearance, collision):
        """Aggressive reward function that DEMANDS forward movement.

        Normalized to [-1, 1] range for stable SAC training.
        Core principle: Forward movement is THE primary objective.
        """
        reward = 0.0
        target_speed = self._curriculum_max_speed

        # 1. Forward Progress - DOMINANT REWARD (up to 1.0)
        forward_vel = max(0.0, linear_vel)
        if forward_vel > 0.01:
            # Strong reward for forward motion - this should be the main signal
            speed_reward = (forward_vel / target_speed) * 1.0
            reward += speed_reward
        else:
            # IDLE PENALTY: Penalize not moving forward
            # This prevents the "safe but useless" behavior of sitting still
            reward -= 0.4

        # Backward penalty - MATCH collision penalty strength
        if linear_vel < -0.01:
            # Driving backwards is as bad as a collision!
            reward -= abs(linear_vel / target_speed) * 1.0

        # 2. Collision Penalty
        if collision or self._safety_override:
            reward -= 1.0  # Maximum negative reward

        # 3. Gap Alignment Reward - ONLY when making good forward progress
        # This prevents rewarding spinning in place to "align"
        alignment_error = abs(action[1] - self._target_heading)

        if forward_vel > 0.1:  # Increased threshold from 0.05 to 0.1
            # Only reward alignment when moving at meaningful speed
            alignment_reward = (0.5 - alignment_error) * 0.3
            reward += alignment_reward

            # Strong bonus for moving forward WHILE aligned
            if alignment_error < 0.3:
                reward += 0.3 * (forward_vel / target_speed)
        # NO reward/penalty when not moving - let idle penalty handle it

        # 4. Straightness Bonus
        # Reward driving straight (low angular velocity while moving forward)
        if forward_vel > 0.08 and abs(angular_vel) < 0.3:
            straightness_bonus = 0.3 * (forward_vel / target_speed)
            reward += straightness_bonus

        # 5. Action Smoothness
        if len(self._prev_angular_actions) > 0:
            angular_jerk = abs(action[1] - self._prev_angular_actions[-1])
            if angular_jerk > 0.4:
                reward -= min(angular_jerk * 0.3, 0.3)
        self._prev_angular_actions.append(action[1])

        # 6. Angular Velocity Penalty (prefer straight motion)
        # Strong penalty for excessive turning, especially when stationary
        if abs(angular_vel) > 0.2:
            # Base penalty for turning
            ang_penalty = abs(angular_vel) * 0.3

            # MASSIVE penalty if stationary - we want forward motion, not spinning!
            if forward_vel < 0.05:
                ang_penalty *= 5.0  # Spinning in place is nearly useless
            # Reduced penalty if we have good clearance and moving forward
            elif clearance >= 1.0 and forward_vel > 0.1:
                ang_penalty *= 0.5  # Allow exploration when safe and moving

            reward -= min(ang_penalty, 0.8)

        # Final normalization: ensure [-1, 1] range
        reward = np.clip(reward, -1.0, 1.0)

        return reward

    def _control_loop(self):
        """Main control loop running at 30Hz."""
        # 0. Wait for Initial Handshake
        if not self._initial_sync_done.is_set():
            # Publish stop command and wait
            self.cmd_pub.publish(Twist())
            return

        if not self._model_ready:
            return

        if self._latest_depth_raw is None:
            # self.get_logger().warn('Waiting for depth data...', throttle_duration_sec=5.0)
            return

        # 1. Prepare Inputs
        # Process Depth + LiDAR -> Multi-Channel Occupancy Grid
        t0 = time.time()

        # Get robot pose for exploration history
        # Process sensors with RawSensorProcessor
        # Returns: laser_grid (128, 128), depth_processed (424, 240)
        laser_grid, depth_processed = self.occupancy_processor.process(
            depth_img=self._latest_depth_raw,
            laser_scan=self._latest_scan
        )

        # Store processed sensor data
        self._latest_laser = laser_grid    # (128, 128) float32 [0, 1]
        self._latest_depth = depth_processed  # (424, 240) float32 [0, 1]

        # Gap Following Analysis using binary laser occupancy
        # laser_grid: 0.0 = free, 1.0 = occupied
        # For gap finding, invert: free space = high score
        free_space = 1.0 - laser_grid  # (128, 128), free space = 1.0

        # Simple heuristic: find column with maximum average free space
        col_scores = np.mean(free_space, axis=0)  # Average down columns

        # Smooth scores
        col_scores = np.convolve(col_scores, np.ones(7)/7, mode='same')

        best_col = np.argmax(col_scores)

        # Map col 0..127 to heading -1..1
        # Col 0 = LEFT, Col 127 = RIGHT, Col 64 = CENTER
        # Angle: Left is +1.0, Right is -1.0
        self._target_heading = (64 - best_col) / 64.0

        # Calculate min_forward_dist from laser for reward function
        # Laser: 0.0 = free, 1.0 = occupied
        # Center strip: cols 59..69 (±5 from center col 64)
        # Bottom 10 rows: 118..128 (closest to robot at row 127)
        center_patch = laser_grid[118:128, 59:69]

        # If any obstacle in patch, we're close to collision
        # Sum up occupied cells - if any are occupied (>0.5), distance is small
        obstacle_density = np.mean(center_patch) if center_patch.size > 0 else 0.0

        # Convert obstacle density to distance estimate
        # High obstacle density → low distance
        # 0.0 (all free) → 4.0m, 1.0 (all occupied) → 0.0m
        self._min_forward_dist = (1.0 - obstacle_density) * 4.0

        # DEBUG: Log forward distance stats
        # if self._min_forward_dist < 0.5:
        #     self.get_logger().info(f"📏 Safety Check: MinDist={self._min_forward_dist:.3f}m (Norm={min_normalized_dist:.3f})")
        #     self.get_logger().info(f"   Patch Mean: {np.mean(center_patch):.3f}, Min: {np.min(center_patch):.3f}")

        # Get IMU data
        if self._latest_imu:
            ax, ay, az, gx, gy, gz = self._latest_imu
        else:
            ax, ay, az, gx, gy, gz = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # LiDAR Metrics for Reward
        lidar_min, lidar_sides, gap_heading = self._process_lidar_metrics(self._latest_scan)

        # Update target heading for Gap Follower (Warmup)
        self._target_heading = gap_heading

        # Construct 10D proprio: [ax, ay, az, gx, gy, gz, min_depth, min_lidar, prev_lin, prev_ang]
        # Note: gap_heading removed - now implicit in exploration history channel
        proprio = np.array([[
            ax, ay, az, gx, gy, gz,
            self._min_forward_dist,     # Min Depth (Front)
            lidar_min,                  # Min LiDAR (360 Safety)
            self._prev_action[0],       # Previous linear action
            self._prev_action[1]        # Previous angular action
        ]], dtype=np.float32)

        # 2. Inference (RKNN)
        # Returns: [action_mean] (value head not exported in actor ONNX)
        if self._rknn_runtime:
            # Stateless inference (LSTM removed for export compatibility)
            # Input: [laser, rgbd, proprio]
            # Add Batch and Channel dimensions
            laser_input = laser_grid[None, None, ...]
            # Build RGBD if RGB is available; fallback to depth-only
            if self._latest_rgb is not None:
                rgbd_input = self.rgbd_processor.process(self._latest_rgb, self._latest_depth_raw)
                rgbd_input = rgbd_input[None, ...]  # (1, 4, 240, 424)
            else:
                # Fallback: use depth as single-channel replicated to 4 channels to avoid crashes
                # Create a grayscale "RGB" by repeating depth 3 times
                depth_normalized = (self._latest_depth_raw.astype(np.float32) / 1000.0).clip(0, 4.0) / 4.0 * 255.0
                depth_3 = np.repeat(depth_normalized[..., None], 3, axis=2).astype(np.uint8)
                rgbd_input = self.rgbd_processor.process(depth_3, self._latest_depth_raw)
                rgbd_input = rgbd_input[None, ...]  # (1, 4, 240, 424)

            outputs = self._rknn_runtime.inference(inputs=[laser_input, rgbd_input, proprio])

            # Output 0 is action (1, 2)
            action_mean = outputs[0][0] # (2,)
            action = action_mean


            # INTELLIGENT WARMUP SEQUENCE (Model 0) - MOVED OUTSIDE
            pass
            
            # Apply safety override
            if self._safety_override:
                action[0] = 0.0 # Stop linear motion
                # Allow rotation to clear obstacle if needed, or just stop
                # For now, just stop everything to be safe
                action[1] = 0.0
            # DIAGNOSTIC: Check RKNN output for NaN
            if np.isnan(action).any() or np.isinf(action).any():
                self.get_logger().error(f"❌ RKNN model output contains NaN/Inf!")
                self.get_logger().error(f"   action: {action}")
                self.get_logger().error(f"   Laser input range: [{laser_input.min():.3f}, {laser_input.max():.3f}]")
                self.get_logger().error(f"   Depth input range: [{depth_input.min():.3f}, {depth_input.max():.3f}]")
                self.get_logger().error(f"   Proprio input: {proprio}")
                # Use zeros and continue
                action_mean = np.zeros(2)

            # We don't get value from actor model, so estimate or ignore
            value = 0.0
        else:
            action_mean = np.zeros(2)
            action = np.zeros(2) # Initialize action to avoid UnboundLocalError
            value = 0.0

        # WARMUP / SEEDING
        # Use heuristic "Gap Follower" for the first model version (v0)
        # This seeds the replay buffer with "good" driving data (driving towards gaps)
        # instead of random thrashing, helping the model learn the "drive forward" objective faster.
        if self._current_model_version <= 0:
            if not self._warmup_active:
                self._warmup_active = True
                self.get_logger().info('🔥 Starting Heuristic Warmup (Gap Follower)...')
                self.pbar.set_description("🔥 Warmup: Gap Follower")

            # Heuristic Policy:
            # 1. Steer towards _target_heading (gap direction from LiDAR)
            # 2. Drive fast if aligned and clear, slow if turning or blocked

            # Rate-limit target heading to prevent oscillation
            # Max change of 0.3 per step (at 30Hz = 9 rad/s max turn rate)
            max_heading_change = 0.3
            heading_delta = self._target_heading - self._prev_target_heading
            if abs(heading_delta) > max_heading_change:
                # Clamp the change
                heading_delta = np.sign(heading_delta) * max_heading_change
            smoothed_target = self._prev_target_heading + heading_delta
            self._prev_target_heading = smoothed_target

            # Proportional control: Use gain of 0.5 to prevent bang-bang behavior
            # This gives smoother steering: small errors = small corrections
            heuristic_angular = np.clip(smoothed_target * 0.5, -1.0, 1.0)

            # Linear action based on LiDAR clearance (more reliable than depth)
            # lidar_min is the 360-degree safety bubble from actual LiDAR
            # Use LiDAR for safety distance check since it's more reliable than depth-grid computation
            clearance_dist = lidar_min if lidar_min > 0.05 else self._min_forward_dist

            # DEBUG: Log clearance values to diagnose movement issues
            if not hasattr(self, '_debug_log_count'):
                self._debug_log_count = 0
            self._debug_log_count += 1
            if self._debug_log_count % 30 == 0:  # Log every 1 second
                self.get_logger().info(f"🔍 Warmup Debug: LiDAR_min={lidar_min:.3f}m, Depth_min={self._min_forward_dist:.3f}m, clearance={clearance_dist:.3f}m, target={self._target_heading:.2f}, smooth={smoothed_target:.2f}, cmd={heuristic_angular:.2f}")

            # Determine linear speed based on alignment AND clearance
            # NOTE: These are normalized actions [-1, 1] that get scaled by max_speed later
            # Make sure values are high enough to overcome motor deadzone
            if abs(heuristic_angular) < 0.2 and clearance_dist > 0.35:
                # Well aligned and clear - go fast
                heuristic_linear = 1.0
            elif abs(heuristic_angular) < 0.3 and clearance_dist > 0.2:
                # Reasonably aligned - moderate speed
                heuristic_linear = 0.8
            elif abs(heuristic_angular) < 0.5 and clearance_dist > 0.2:
                # Turning and clear - slower
                heuristic_linear = 0.6
            elif clearance_dist > 0.15:
                # Close but some room - slow
                heuristic_linear = 0.4
            else:
                # Very close - crawl
                heuristic_linear = 0.3

            # Override model action with heuristic
            action = np.array([heuristic_linear, heuristic_angular], dtype=np.float32)

            # Add small noise ONLY to linear speed (not angular - avoid oscillation)
            # This helps the policy learn robustness without causing steering jitter
            linear_noise = np.random.normal(0, 0.1)
            action[0] = np.clip(action[0] + linear_noise, 0.0, 1.0)  # Linear only
            action[1] = np.clip(action[1], -1.0, 1.0)  # Angular unchanged
            
        elif self._warmup_active:
            self.get_logger().info('✅ Warmup Complete (Model v1+ loaded). Switching to learned policy.')
            self._warmup_active = False
            self.pbar.set_description(f"⏳ Training v{self._current_model_version}")

        # 3. Add Exploration Noise (Gaussian)
        # Only apply standard noise if NOT in warmup (Model > 0)
        if self._current_model_version > 0:
            noise = np.random.normal(0, 0.5, size=2) # 0.5 std dev
            action = np.clip(action_mean + noise, -1.0, 1.0)

            # Safety check: if action_mean has NaN, use zero and warn
            if np.isnan(action_mean).any():
                self.get_logger().error("⚠️  NaN detected in action_mean from model! Using zero action.")
                action_mean = np.zeros(2)
                action = np.clip(noise, -1.0, 1.0)  # Pure random

        # 4. Execute Action
        cmd = Twist()
        
        # SENSOR WARMUP: Count down before enabling safety-triggered resets
        # This prevents false positives from unstable sensor data during startup
        if not self._sensor_warmup_complete:
            self._sensor_warmup_countdown -= 1
            if self._sensor_warmup_countdown <= 0:
                self._sensor_warmup_complete = True
                self.get_logger().info('✅ Sensor warmup complete - safety system fully active')
            elif self._sensor_warmup_countdown % 10 == 0:
                self.get_logger().info(f'⏳ Sensor warmup: {self._sensor_warmup_countdown} cycles remaining...')
        
        # SAFETY DISTANCE: Use best available measurement
        # If depth-based distance is suspiciously low (< 0.05m), it's likely invalid data
        # Fall back to LiDAR-based minimum distance as sanity check
        effective_min_dist = self._min_forward_dist
        
        if self._min_forward_dist < 0.05 and lidar_min > 0.2:
            # Depth reports < 5cm but LiDAR sees > 20cm - depth is probably invalid
            effective_min_dist = lidar_min
            if self._sensor_warmup_complete:  # Only log after warmup
                self.get_logger().debug(f'📏 Using LiDAR fallback: Depth={self._min_forward_dist:.3f}m, LiDAR={lidar_min:.3f}m')
        
        # Safety Override Logic
        safety_triggered = self._safety_override or effective_min_dist < 0.12
        
        if safety_triggered:
            # Override: Stop and reverse slightly
            if effective_min_dist < 0.12 and self._sensor_warmup_complete:
                self.get_logger().warn(f"🛑 Safety Stop! MinDist={effective_min_dist:.3f}m (Depth={self._min_forward_dist:.3f}m, LiDAR={lidar_min:.3f}m)")
                 
            cmd.linear.x = -0.05
            cmd.angular.z = 0.0
            actual_action = np.array([-0.5, 0.0]) # Record that we stopped
            
            # Only mark as collision if sensors are warmed up
            # During warmup, still stop but don't trigger episode resets
            collision = self._sensor_warmup_complete
        else:
            # Normal execution
            # During warmup (model v0), use max_linear directly for better movement
            # After training starts, use curriculum_max_speed which may be lower
            if self._current_model_version <= 0:
                cmd.linear.x = float(action[0] * self.max_linear)
            else:
                cmd.linear.x = float(action[0] * self._curriculum_max_speed)
            cmd.angular.z = float(action[1] * self.max_angular)
            actual_action = action
            collision = False
            
        self.cmd_pub.publish(cmd)
        
        # 5. Compute Reward
        current_linear = self._latest_odom[2] if self._latest_odom else 0.0
        
        # Use IMU Gyro Z for angular velocity if available (more reliable than encoders)
        if self._latest_imu:
            current_angular = self._latest_imu[5]
        else:
            current_angular = self._latest_odom[3] if self._latest_odom else 0.0

        reward = self._compute_reward(
            actual_action, current_linear, current_angular,
            lidar_min, lidar_sides, collision
        )

        # Clip reward to prevent extreme values (already clipped in _compute_reward, but keep as safety)
        reward = np.clip(reward, -1.0, 1.0)

        # Safety check: NaN in reward
        if np.isnan(reward) or np.isinf(reward):
            self.get_logger().warn("⚠️  NaN/Inf in reward, skipping data collection")
            return

        # 6. Store Transition
        # Build RGBD for storage
        # Validate that both RGB and depth are available and have matching shapes
        if self._latest_rgb is not None and self._latest_depth_raw is not None:
            # Verify shape compatibility
            expected_depth_shape = self._latest_rgb.shape[:2]  # (height, width)
            if self._latest_depth_raw.shape[:2] != expected_depth_shape:
                self.get_logger().warn(
                    f"⚠️  Shape mismatch: RGB {self._latest_rgb.shape} vs Depth {self._latest_depth_raw.shape}, skipping"
                )
                return
            rgbd_to_store = self.rgbd_processor.process(self._latest_rgb, self._latest_depth_raw)
        elif self._latest_depth_raw is not None:
            # Fallback: grayscale depth RGB (only if depth is available)
            depth_normalized = (self._latest_depth_raw.astype(np.float32) / 1000.0).clip(0, 4.0) / 4.0 * 255.0
            depth_3 = np.repeat(depth_normalized[..., None], 3, axis=2).astype(np.uint8)
            rgbd_to_store = self.rgbd_processor.process(depth_3, self._latest_depth_raw)
        else:
            # No valid sensor data yet
            self.get_logger().warn("⚠️  No RGB or depth data available, skipping data collection")
            return
        
        with self._buffer_lock:
            self._data_buffer['laser'].append(self._latest_laser)
            self._data_buffer['rgbd'].append(rgbd_to_store)
            self._data_buffer['proprio'].append(proprio[0])
            self._data_buffer['actions'].append(actual_action)
            self._data_buffer['rewards'].append(reward)
            self._data_buffer['dones'].append(collision)
            
            # Update Dashboard
            self.total_steps += 1
            self.episode_reward += reward
            # self.pbar.update(1) # Pbar now tracks server progress
            self.pbar.set_postfix({
                'Rew': f"{reward:.2f}",
                'Vel': f"{current_linear:.2f}",
                'Tgt': f"{self._target_heading:.1f}", # Show target heading
                'Buf': f"{len(self._data_buffer['rewards'])}"
            })
            
        # Save calibration data (keep ~100 samples)
        # We save occasionally to avoid disk I/O spam
        if np.random.rand() < 0.1: # 10% chance to save sample
            calib_files = list(self._calibration_dir.glob('*.npz'))
            if len(calib_files) < 100:
                timestamp = int(time.time() * 1000)
                save_path = self._calibration_dir / f"calib_{timestamp}.npz"
                # Save RGBD for calibration
                if self._latest_rgb is not None:
                    rgbd_calib = self.rgbd_processor.process(self._latest_rgb, self._latest_depth_raw)
                else:
                    depth_normalized = (self._latest_depth_raw.astype(np.float32) / 1000.0).clip(0, 4.0) / 4.0 * 255.0
                    depth_3 = np.repeat(depth_normalized[..., None], 3, axis=2).astype(np.uint8)
                    rgbd_calib = self.rgbd_processor.process(depth_3, self._latest_depth_raw)
                
                np.savez_compressed(
                    save_path,
                    laser=self._latest_laser,
                    rgbd=rgbd_calib,
                    proprio=proprio[0]
                )
            
        # Trigger episode reset on collision
        if collision:
            self._trigger_episode_reset()

        # Update state
        self._prev_action = actual_action
        self._prev_linear_cmds.append(actual_action[0])

    def _trigger_episode_reset(self):
        """Call motor driver to reset encoder baselines."""
        if not self.reset_episode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Episode reset service unavailable')
            return

        request = Trigger.Request()
        future = self.reset_episode_client.call_async(request)

        def _log_response(fut):
            try:
                response = fut.result()
                if response.success:
                    self.pbar.write(f'Episode reset: {response.message}')
                else:
                    self.pbar.write(f'Episode reset failed: {response.message}')
            except Exception as e:
                self.get_logger().error(f'Episode reset error: {e}')

        future.add_done_callback(_log_response)

    def _run_nats_loop(self):
        """Entry point for NATS background thread."""
        asyncio.run(self._nats_main())

    async def _nats_main(self):
        """Main NATS async event loop."""
        try:
            # Connect to NATS
            await self._connect_nats()

            # Subscribe to model metadata updates
            await self.nc.subscribe("models.sac.metadata", cb=self._on_model_metadata)
            
            # Subscribe to server status
            await self.nc.subscribe("server.sac.status", cb=self._on_server_status)

            # Start publishing experience batches in background
            asyncio.create_task(self._publish_experience_loop())

            # Mark as connected
            self._initial_sync_done.set()

            # Keep running until stopped
            while not self._stop_event.is_set():
                await asyncio.sleep(0.1)

        except Exception as e:
            self.get_logger().error(f"NATS loop error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.nc:
                await self.nc.close()

    async def _connect_nats(self):
        """Connect to NATS server with auto-reconnect."""
        self.pbar.write(f"🔌 Connecting to NATS at {self.nats_server}...")

        async def on_disconnected():
            self.pbar.write("⚠ NATS disconnected")

        async def on_reconnected():
            self.pbar.write("✅ NATS reconnected")

        self.nc = await nats.connect(
            servers=[self.nats_server],
            name="rover-sac-client",
            max_reconnect_attempts=-1,  # Infinite reconnects
            reconnect_time_wait=2,       # 2s between attempts
            ping_interval=20,            # Ping every 20s
            max_outstanding_pings=3,     # Disconnect after 3 missed
            disconnected_cb=on_disconnected,
            reconnected_cb=on_reconnected,
        )

        self.js = self.nc.jetstream()
        self.pbar.write("✅ Connected to NATS")

        # Try to get latest model metadata
        try:
            msg = await self.js.get_last_msg("ROVER_MODELS", f"models.{self.algorithm}.metadata")
            metadata = deserialize_metadata(msg.data)
            server_version = metadata.get("latest_version", 0)
            self.pbar.write(f"✅ Server has model v{server_version}")

            if server_version > self._current_model_version:
                self._current_model_version = -1  # Force download
                self._model_update_needed = True
                self.get_logger().info("🚀 Triggering initial model download...")
                asyncio.create_task(self._download_model())
        except Exception as e:
            self.get_logger().info(f"ℹ No model metadata yet: {e}")

    async def _on_model_metadata(self, msg):
        """Callback when new model metadata is published."""
        try:
            metadata = deserialize_metadata(msg.data)
            server_version = metadata.get("latest_version", 0)
            # self.pbar.write(f"📨 Received metadata: Server has v{server_version}, Local has v{self._current_model_version}")

            if server_version > self._current_model_version:
                self.pbar.write(f"🔔 New model v{server_version} available (current: v{self._current_model_version})")
                self._model_update_needed = True
                # Download model in background
                asyncio.create_task(self._download_model())
            else:
                pass
                # self.pbar.write(f"✓ Local model is up to date (v{self._current_model_version})")

        except Exception as e:
            self.get_logger().error(f"Model metadata callback error: {e}")

    async def _on_server_status(self, msg):
        """Callback for server status updates."""
        try:
            status = deserialize_status(msg.data)
            total_steps = status.get("total_steps", 0)
            
            # Update progress bar to show steps towards next model (modulo 2000)
            progress = total_steps % 2000
            self.pbar.n = progress
            self.pbar.refresh()
            
            # Update description if model version changed
            server_ver = status.get("model_version", 0)
            if server_ver > self._current_model_version:
                 self.pbar.set_description(f"🚀 New Model v{server_ver} Ready!")
            else:
                 self.pbar.set_description(f"⏳ Training v{server_ver}")

        except Exception as e:
            pass # Don't spam errors on status updates

    async def _download_model(self):
        """Download and convert the latest model from JetStream."""
        if not self._model_update_needed:
            return

        try:
            self.pbar.write("📥 Downloading model from NATS...")

            # Get latest model from stream
            msg = await self.js.get_last_msg("ROVER_MODELS", f"models.{self.algorithm}.update")
            # self.pbar.write(f"📦 Received model message: {len(msg.data)} bytes")
            
            model_data = deserialize_model_update(msg.data)

            onnx_bytes = model_data["onnx_bytes"]
            model_version = model_data["version"]
            
            self.pbar.write(f"📦 Deserialized model v{model_version}, ONNX size: {len(onnx_bytes)} bytes")

            # Save ONNX to temp file
            onnx_path = self._temp_dir / "latest_model.onnx"
            with open(onnx_path, 'wb') as f:
                f.write(onnx_bytes)
                f.flush()
                os.fsync(f.fileno())

            self.pbar.write(f"💾 Saved ONNX model v{model_version} to {onnx_path}")

            # Convert to RKNN
            if HAS_RKNN:
                self.pbar.write("🔄 Converting to RKNN (this may take a minute)...")
                rknn_path = str(onnx_path).replace('.onnx', '.rknn')

                # Call conversion script
                cmd = ["./convert_onnx_to_rknn.sh", str(onnx_path), str(self._calibration_dir)]
                self.pbar.write(f"🛠 Executing: {' '.join(cmd)}")

                if not os.path.exists("convert_onnx_to_rknn.sh"):
                    self.pbar.write("⚠ convert_onnx_to_rknn.sh not found, skipping conversion")
                    if os.path.exists("/home/benson/Documents/ros2-rover/convert_onnx_to_rknn.sh"):
                         cmd[0] = "/home/benson/Documents/ros2-rover/convert_onnx_to_rknn.sh"
                         self.pbar.write(f"✓ Found script at {cmd[0]}")
                    else:
                         self.get_logger().error("❌ Conversion script missing!")
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                # Check if output file was created despite potential crash (e.g. double free on exit)
                if os.path.exists(rknn_path):
                    if result.returncode != 0:
                        self.pbar.write(f"⚠ RKNN Conversion script crashed (code {result.returncode}) but output file exists. Assuming success.")
                    else:
                        self.pbar.write("✅ RKNN Conversion successful")

                    # Load new model
                    self.pbar.write("🔄 Loading new RKNN model...")
                    new_runtime = RKNNLite()
                    ret = new_runtime.load_rknn(rknn_path)
                    if ret != 0:
                        self.pbar.write(f"Load RKNN failed: {ret}")
                    else:
                        ret = new_runtime.init_runtime()
                        if ret != 0:
                            self.pbar.write(f"Init RKNN runtime failed: {ret}")
                        else:
                            # Swap runtime
                            self._rknn_runtime = new_runtime
                            self._current_model_version = model_version
                            self._model_ready = True
                            self._model_update_needed = False
                            self.pbar.write(f"🚀 New model v{model_version} loaded and active!")
                else:
                    self.pbar.write(f"❌ RKNN Conversion failed with code {result.returncode}")
                    self.pbar.write(f"Stdout: {result.stdout}")
                    self.pbar.write(f"Stderr: {result.stderr}")
            else:
                # If no RKNN (e.g. testing on PC), just mark as updated
                self.pbar.write("⚠ RKNN not available, skipping conversion (simulating success)")
                self._current_model_version = model_version
                self._model_update_needed = False

        except Exception as e:
            self.pbar.write(f"Model download failed: {e}")
            import traceback
            traceback.print_exc()

    async def _publish_experience_loop(self):
        """Periodically publish experience batches to NATS."""
        while not self._stop_event.is_set():
            try:
                # Check if we have enough data to send
                batch_to_send = None
                with self._buffer_lock:
                    if len(self._data_buffer['rewards']) >= self.batch_size:
                        # Extract batch
                        batch_to_send = {k: np.array(v) for k, v in self._data_buffer.items()}
                        # Clear buffer
                        for k in self._data_buffer:
                            self._data_buffer[k] = []

                if batch_to_send:
                    # Serialize and publish
                    msg_bytes = serialize_batch(batch_to_send)
                    msg_size_mb = len(msg_bytes) / (1024 * 1024)

                    self.pbar.write(f"📤 Publishing batch of {len(batch_to_send['rewards'])} steps, size: {msg_size_mb:.2f} MB")

                    ack = await self.js.publish(
                        subject="rover.experience",
                        payload=msg_bytes,
                        timeout=10.0
                    )
                    # self.pbar.write(f"✅ Batch published (seq={ack.seq})") # Reduce spam
                    
                    # Reset pbar for next batch
                    # self.pbar.reset() # Don't reset, we track server progress now

            except Exception as e:
                self.get_logger().error(f"Experience publish error: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()

            await asyncio.sleep(0.1)

    def destroy_node(self):
        self._stop_event.set()
        if self._nats_thread.is_alive():
            self._nats_thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SACEpisodeRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
