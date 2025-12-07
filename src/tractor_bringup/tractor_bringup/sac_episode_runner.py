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

from tractor_bringup.occupancy_processor import DepthToOccupancy, ScanToOccupancy, LocalMapper, MultiChannelOccupancy

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

        # State
        self._latest_rgb = None # Keep for debug/logging if needed, but not used for model
        self._latest_depth = None
        self._latest_scan = None
        self._latest_grid = None # (4, 128, 128) float32
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
            'grid': [], 'proprio': [], 
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
        # New multi-channel occupancy processor for enhanced SAC training
        # NOTE: Using higher thresholds to avoid ground plane false positives
        self.occupancy_processor = MultiChannelOccupancy(
            grid_size=128,  # Increased to 128 for better resolution
            range_m=4.0,
            width=424, height=240,
            camera_height=0.18,  # Updated: 180mm from ground
            camera_tilt_deg=0.0,
            obstacle_height_thresh=0.15,  # Increased: Only consider objects > 15cm as obstacles
            floor_thresh=0.12  # Increased: ±12cm tolerance for ground plane
        )
        # Keep old processors for backward compatibility (can be removed later)
        self.scan_processor = ScanToOccupancy(grid_size=64, grid_range=3.0)
        self.local_mapper = LocalMapper(map_size=256, decay_rate=0.995)
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
        # self.create_subscription(Image, '/camera/camera/color/image_raw', self._rgb_cb, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self._depth_cb, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, qos_profile_sensor_data)
        
        # Odometry: Prefer RF2O if available, otherwise Wheel Odom
        # We subscribe to both but update logic will prioritize
        self.create_subscription(Odometry, '/odom', self._wheel_odom_cb, 10)
        self.create_subscription(Odometry, '/odom_rf2o', self._rf2o_odom_cb, 10)

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
    def _rgb_cb(self, msg): pass # self._latest_rgb = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
    def _depth_cb(self, msg): 
        # Use passthrough to get raw 16-bit depth
        d = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        self._latest_depth = d
        
        # Process to grid immediately (or in control loop? Control loop is better for rate limiting)
        # But doing it here ensures we always have fresh grid
        # Let's do it in control loop to save CPU if inference is slower than camera
        pass
    def _scan_cb(self, msg):
        self._latest_scan = msg
        
    def _wheel_odom_cb(self, msg):
        # Fallback if no RF2O
        if not hasattr(self, '_last_rf2o_time') or (time.time() - self._last_rf2o_time > 0.5):
            self._latest_odom = (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.twist.twist.linear.x, msg.twist.twist.angular.z)
            
    def _rf2o_odom_cb(self, msg):
        self._last_rf2o_time = time.time()
        self._latest_odom = (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.twist.twist.linear.x, msg.twist.twist.angular.z)
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
        smoothed = np.convolve(ranges_padded, np.ones(window_size)/window_size, mode='valid')
        
        # Find best index in smoothed (matches length of ranges_gap potentially, or close)
        # 'valid' mode output length = N - K + 1. 
        # Actually simplest is mode='same' on original and handle wrapping manually or ignore edge effect.
        # Let's use mode='same' on unpadded for simplicity, assuming adequate standard scan.
        # But wait, gap might be at -PI/PI boundary (behind?). 
        # Typically we want forward gaps.
        # Let's focus on [-PI/2, PI/2] (Front 180).
        
        # Filter for front sector only (-1.57 to 1.57) to avoid driving backwards
        front_mask = (sorted_angles > -1.6) & (sorted_angles < 1.6)
        
        # If we have front data
        if np.any(front_mask):
            # Extract front arc
            front_ranges = ranges_gap[front_mask]
            front_angles = sorted_angles[front_mask]
            
            # Smooth front arc
            if len(front_ranges) >= window_size:
                smoothed_front = np.convolve(front_ranges, np.ones(window_size)/window_size, mode='same')
                best_idx_local = np.argmax(smoothed_front)
                best_angle = front_angles[best_idx_local]
            else:
                # Too few points, pick max individual
                best_idx_local = np.argmax(front_ranges)
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

        if self._latest_depth is None:
            # self.get_logger().warn('Waiting for depth data...', throttle_duration_sec=5.0)
            return

        # 1. Prepare Inputs
        # Process Depth + LiDAR -> Multi-Channel Occupancy Grid
        t0 = time.time()

        # Get robot pose for exploration history
        robot_pose = None
        if self._latest_odom:
            x, y, _, yaw = self._latest_odom
            robot_pose = (x, y, yaw)

        # NEW: Use MultiChannelOccupancy processor
        # Returns: (4, 128, 128) float32 array, already normalized to [0, 1]
        grid_multichannel = self.occupancy_processor.process(
            depth_img=self._latest_depth,
            laser_scan=self._latest_scan,
            robot_pose=robot_pose
        )

        # Grid Input for Model: (1, 4, 128, 128)
        grid_input = grid_multichannel[None, ...]  # Add batch dimension

        # For compatibility with visualization/logging, extract channel 0 (distance)
        # and convert back to uint8 format (0-255)
        # grid_for_viz = (grid_multichannel[0] * 255).astype(np.uint8)
        
        # CRITICAL FIX: The model and training server expect the FULL 4-channel grid!
        self._latest_grid = grid_multichannel 
        
        # Gap Following Analysis (for heuristic warmup and reward)
        # Use distance channel for gap finding
        distance_channel = grid_multichannel[0]  # (128, 128) normalized [0, 1]

        # Simple heuristic: find column with maximum average distance
        # Higher values = more free space
        col_scores = np.mean(distance_channel, axis=0)

        # Smooth scores
        col_scores = np.convolve(col_scores, np.ones(7)/7, mode='same')

        best_col = np.argmax(col_scores)

        # Map col 0..127 to heading -1..1
        # Col 0 = LEFT, Col 127 = RIGHT, Col 64 = CENTER
        # Angle: Left is +1.0, Right is -1.0
        self._target_heading = (64 - best_col) / 64.0

        # Calculate min_forward_dist from distance channel (for reward function)
        # Scan center strip (width ~30cm -> ~10 pixels at 3.125cm/pixel)
        # Center col is 64. 64 +/- 5 = 59..69
        # Only look at the bottom (closest to robot) to get safety distance!
        # Robot is at row 127. Look at last 10 rows.
        center_patch = distance_channel[118:128, 59:69]
        
        # Find minimum distance in front (inverse of distance = obstacle proximity)
        # Distance channel: 1.0 = far (free), 0.0 = close (occupied)
        min_normalized_dist = np.min(center_patch) if center_patch.size > 0 else 0.0
        # Convert back to meters (denormalize)
        self._min_forward_dist = min_normalized_dist * 4.0  # range_m = 4.0

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
            # Input: [grid, proprio]
            outputs = self._rknn_runtime.inference(inputs=[grid_input, proprio])

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
                self.get_logger().error(f"   Grid input range: [{grid_input.min():.3f}, {grid_input.max():.3f}]")
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
        if self._current_model_version == 0:
            if not self._warmup_active:
                self._warmup_active = True
                self.get_logger().info('🔥 Starting Heuristic Warmup (Gap Follower)...')
                self.pbar.set_description("🔥 Warmup: Gap Follower")

            # Heuristic Policy:
            # 1. Steer towards _target_heading (gap direction from LiDAR)
            # 2. Drive fast if aligned and clear, slow if turning or blocked
            
            # Angular action: directly map target heading (-1..1)
            # _target_heading is already normalized: +1 (Left) to -1 (Right)
            heuristic_angular = np.clip(self._target_heading, -1.0, 1.0)
            
            # Linear action based on LiDAR clearance (more reliable than depth)
            # lidar_min is the 360-degree safety bubble from actual LiDAR
            # Use LiDAR for safety distance check since it's more reliable than depth-grid computation
            clearance_dist = lidar_min if lidar_min > 0.05 else self._min_forward_dist
            
            # DEBUG: Log clearance values to diagnose movement issues
            if not hasattr(self, '_debug_log_count'):
                self._debug_log_count = 0
            self._debug_log_count += 1
            if self._debug_log_count % 30 == 0:  # Log every 1 second
                self.get_logger().info(f"🔍 Warmup Debug: LiDAR_min={lidar_min:.3f}m, Depth_min={self._min_forward_dist:.3f}m, clearance={clearance_dist:.3f}m, target={self._target_heading:.2f}")
            
            # Determine linear speed based on alignment AND clearance
            # NOTE: These are normalized actions [-1, 1] that get scaled by max_speed later
            # Make sure values are high enough to overcome motor deadzone
            # LOWERED thresholds since we're getting stuck at 0 velocity
            if abs(heuristic_angular) < 0.3 and clearance_dist > 0.35:
                # Aligned and clear - go fast
                heuristic_linear = 1.0
            elif abs(heuristic_angular) < 0.3 and clearance_dist > 0.2:
                # Aligned but getting close - moderate
                heuristic_linear = 0.8
            elif abs(heuristic_angular) < 0.6 and clearance_dist > 0.2:
                # Turning and clear - moderate speed
                heuristic_linear = 0.7
            elif clearance_dist > 0.15:
                # Close but some room - slower but still moving
                heuristic_linear = 0.5
            else:
                # Very close - slow crawl (not full stop, to collect data)
                heuristic_linear = 0.3
                
            # Override model action with heuristic
            action = np.array([heuristic_linear, heuristic_angular], dtype=np.float32)
            
            # Add small noise to heuristic so we don't just record identical straight lines
            # This helps the policy learn robustness
            noise = np.random.normal(0, 0.1, size=2) 
            action = np.clip(action + noise, -1.0, 1.0)
            
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
            if self._current_model_version == 0:
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
        with self._buffer_lock:
            self._data_buffer['grid'].append(self._latest_grid)
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
                np.savez_compressed(
                    save_path,
                    grid=self._latest_grid,
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
