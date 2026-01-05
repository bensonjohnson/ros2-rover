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

from tractor_bringup.occupancy_processor import DepthToOccupancy, ScanToOccupancy, LocalMapper, MultiChannelOccupancy, UnifiedBEVProcessor, RGBDProcessor

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


# ============================================================================
# Helper Classes for Improved Warmup/Centering
# ============================================================================

class PDController:
    """
    PD controller for smooth gap following with adaptive damping.

    Improvements over simple P-controller:
    - Derivative term prevents overshoot and oscillation
    - Deadband reduces jitter from sensor noise
    - Rate limiting for stability
    """
    def __init__(self, kp=0.5, kd=0.15, deadband=0.05, max_rate=0.3):
        self.kp = kp
        self.kd = kd
        self.deadband = deadband
        self.max_rate = max_rate  # Max change per step
        self.prev_error = 0.0
        self.prev_output = 0.0
        self.prev_time = None  # Initialize as None to detect first call

    def update(self, target_heading):
        """
        Args:
            target_heading: [-1, 1] normalized gap heading
        Returns:
            angular_cmd: [-1, 1] angular velocity command
        """
        current_time = time.time()

        # First call initialization
        if self.prev_time is None:
            self.prev_time = current_time
            self.prev_error = target_heading
            # Return simple proportional on first call
            output = self.kp * target_heading
            self.prev_output = output
            return np.clip(output, -1.0, 1.0)

        dt = current_time - self.prev_time
        # Clamp dt to reasonable range (avoid huge derivatives)
        dt = np.clip(dt, 0.01, 0.2)

        error = target_heading

        # Deadband to reduce jitter
        if abs(error) < self.deadband:
            error = 0.0

        # Derivative term (rate of change of error)
        derivative = (error - self.prev_error) / dt

        # PD control law
        output = self.kp * error + self.kd * derivative

        # Rate limiting: prevent large changes
        delta = output - self.prev_output
        if abs(delta) > self.max_rate:
            delta = np.sign(delta) * self.max_rate
        output = self.prev_output + delta

        # Update state
        self.prev_error = error
        self.prev_output = output
        self.prev_time = current_time

        return np.clip(output, -1.0, 1.0)


class StuckDetector:
    """
    Detect if robot is stuck and trigger recovery behavior.

    Monitors odometry to detect:
    - Low movement over time window
    - Repetitive oscillation patterns
    """
    def __init__(self, window_size=60, stuck_threshold=0.15):
        """
        Args:
            window_size: Number of samples to track (60 = 2s at 30Hz)
            stuck_threshold: Minimum distance traveled (meters) to not be stuck
        """
        self.window_size = window_size
        self.stuck_threshold = stuck_threshold
        self.position_history = deque(maxlen=window_size)
        self.stuck_counter = 0
        self.recovery_mode = False
        self.recovery_steps = 0

    def update(self, odom_msg):
        """
        Check if stuck based on odometry.

        Args:
            odom_msg: Odometry message

        Returns:
            is_stuck (bool): True if robot hasn't moved significantly
        """
        pos = (odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y)
        self.position_history.append(pos)

        if len(self.position_history) < self.position_history.maxlen:
            return False  # Not enough history

        # Compute total traveled distance over window
        positions = np.array(self.position_history)
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        total_distance = np.sum(distances)

        # Stuck if moved < threshold in time window
        is_stuck = total_distance < self.stuck_threshold

        # Count consecutive stuck detections
        if is_stuck:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        # Trigger recovery after 30 consecutive stuck detections (~1 second)
        if self.stuck_counter > 30 and not self.recovery_mode:
            self.recovery_mode = True
            self.recovery_steps = 45  # 1.5 seconds of recovery

        return is_stuck

    def get_recovery_action(self):
        """
        Generate recovery action (back up and turn).

        Returns:
            action: [linear, angular] recovery command
        """
        if self.recovery_steps > 0:
            self.recovery_steps -= 1
            # Back up and turn randomly
            return np.array([-0.5, np.random.uniform(-0.8, 0.8)])
        else:
            self.recovery_mode = False
            self.stuck_counter = 0
            return None


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
        self._latest_bev = None  # (2, 128, 128) float32 - Unified BEV grid
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

        # Buffers for batching (unified BEV mode)
        self._data_buffer = {
            'bev': [], 'proprio': [],
            'actions': [], 'rewards': [], 'dones': [],
            'is_eval': []
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
        
        # Evaluation State
        self._episodes_since_eval = 0
        self._is_eval_episode = False  # Current episode type
        
        # Warmup State
        self._warmup_start_time = 0.0
        self._warmup_active = False
        self._sensor_warmup_complete = False  # Flag for sensor stabilization
        self._sensor_warmup_countdown = 90  # ~3.0 seconds at 30Hz (increased from 50)
        self._prev_odom_update = None

        # Warmup Controllers (improved)
        self._pd_controller = PDController(kp=0.5, kd=0.15, deadband=0.05, max_rate=0.3)
        self._stuck_detector = StuckDetector(window_size=60, stuck_threshold=0.15)

        # Safety flags for new features
        self._enable_corridor_centering = False  # Disabled by default until tested
        self._enable_stuck_recovery = True
        
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

        # Rotation tracking for spin penalty (NEW)
        self._cumulative_rotation = 0.0  # Radians since last forward progress
        self._last_yaw_for_rotation = None  # Previous yaw for delta calculation
        self._forward_progress_threshold = 0.8  # meters to reset rotation counter
        self._last_position_for_rotation = None  # (x, y) for forward progress detection
        self._revolution_penalty_triggered = False
        self._latest_fused_yaw = 0.0  # EKF-fused yaw from odometry

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
        # Unified BEV processor for single-encoder SAC architecture
        # Processes LiDAR + depth into 2-channel 128×128 BEV grid
        self.occupancy_processor = UnifiedBEVProcessor(
            grid_size=128,  # BEV grid size
            max_range=4.0   # Maximum sensor range for normalization
        )
        # Depth-only processing (no RGB needed)
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
        # No RGB camera subscription - depth-only mode
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
    # RGB callback removed - depth-only mode
    def _depth_cb(self, msg):
        # Use passthrough to get raw 16-bit depth (uint16 mm)
        d = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        self._latest_depth_raw = d  # Store raw for processing in control loop

    def _process_depth(self, depth_raw):
        """Process raw depth to normalized (100, 848) float32 [0, 1].

        848x100 mode is center-cropped - captures less floor, more forward view.
        """
        if depth_raw is None or depth_raw.size == 0:
            return np.ones((100, 848), dtype=np.float32)  # All far/unknown

        # Convert uint16 mm → float32 meters
        if depth_raw.dtype == np.uint16:
            depth = depth_raw.astype(np.float32) * 0.001
        else:
            depth = depth_raw.astype(np.float32)

        # Apply median filter to reduce noise
        depth_uint16 = np.clip(depth * 1000, 0, 65535).astype(np.uint16)
        depth = cv2.medianBlur(depth_uint16, 3).astype(np.float32) * 0.001

        # Normalize to [0, 1] with max_range=4.0
        depth_clipped = np.clip(depth, 0.0, 4.0)
        depth_normalized = depth_clipped / 4.0
        depth_normalized[depth == 0.0] = 1.0  # Invalid -> far

        return depth_normalized  # (100, 848) float32
    def _scan_cb(self, msg):
        self._latest_scan = msg
        
    def _odom_cb(self, msg):
        # Extract position and velocity
        self._latest_odom = (
            msg.pose.pose.position.x, 
            msg.pose.pose.position.y, 
            msg.twist.twist.linear.x, 
            msg.twist.twist.angular.z
        )
        # Extract EKF-fused yaw from quaternion for rotation tracking
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self._latest_fused_yaw = math.atan2(siny_cosp, cosy_cosp)
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

    def _detect_corridor_and_center(self, scan_msg) -> Tuple[bool, float]:
        """
        Detect if robot is in a corridor and compute centering correction.

        Args:
            scan_msg: LaserScan message

        Returns:
            is_corridor (bool): True if symmetric environment detected
            centering_error (float): [-1, 1] where negative = too far left, positive = too far right
        """
        if not scan_msg:
            return False, 0.0

        ranges = np.array(scan_msg.ranges)
        angles = scan_msg.angle_min + np.arange(len(ranges)) * scan_msg.angle_increment
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        valid = (ranges > 0.15) & (ranges < scan_msg.range_max) & np.isfinite(ranges)

        # Left/Right 90° sectors (perpendicular to robot)
        # Left: 20° to 90° (avoiding too far forward/back)
        # Right: -90° to -20°
        left_mask = (angles > 0.35) & (angles < 1.57) & valid  # ~20° to 90°
        right_mask = (angles > -1.57) & (angles < -0.35) & valid  # -90° to -20°

        if not (np.any(left_mask) and np.any(right_mask)):
            return False, 0.0

        # Use median for robustness against outliers
        left_dist = np.median(ranges[left_mask])
        right_dist = np.median(ranges[right_mask])

        # Corridor detection: Both sides at similar distance (within 20% for stricter detection)
        avg_side = (left_dist + right_dist) / 2.0
        diff = abs(left_dist - right_dist)

        # Stricter corridor criteria: very similar side distances AND close walls
        # Reduced threshold from 30% to 20% to avoid false positives
        # Increased min distance to 0.5m to only activate in actual corridors
        is_corridor = (diff / (avg_side + 0.01) < 0.20) and (avg_side < 1.5) and (avg_side > 0.5)

        if is_corridor:
            # Centering error: positive = too far left (need to move right)
            # Negative = too far right (need to move left)
            # right_dist > left_dist means right wall is farther, robot is too far left
            centering_error = (right_dist - left_dist) / avg_side
            return True, np.clip(centering_error, -1.0, 1.0)

        return False, 0.0

    def _find_best_gap_multiscale(self, ranges, angles, valid, scan_msg) -> Tuple[float, float]:
        """
        Full 360° multi-scale gap detection with forward bias.

        Searches entire LiDAR field of view for gaps, allowing the rover to find
        escape routes in any direction when blocked. Forward gaps are preferred
        via scoring bias, but side/rear gaps can be found when needed.

        Args:
            ranges: Range array
            angles: Angle array (wrapped to [-π, π])
            valid: Valid data mask
            scan_msg: LaserScan message for increment

        Returns:
            best_angle (float): Angle of best gap in radians [-π, π]
            best_depth (float): Average depth of best gap
        """
        if not np.any(valid):
            return 0.0, 0.0

        # Use ALL valid ranges (full 360°)
        all_ranges = ranges.copy()
        all_ranges[~valid] = 0.0  # Treat invalid as obstacles

        # Sort by angle for proper convolution
        sort_idx = np.argsort(angles)
        sorted_angles = angles[sort_idx]
        sorted_ranges = all_ranges[sort_idx]

        if len(sorted_ranges) < 5:
            return 0.0, 0.0

        best_gap = {'angle': 0.0, 'depth': 0.0, 'score': -np.inf}

        # Multi-scale: try 15°, 25°, 35° windows
        for window_deg in [15, 25, 35]:
            window_rad = np.radians(window_deg)
            window_size = int(window_rad / scan_msg.angle_increment)
            window_size = max(3, min(window_size, len(sorted_ranges) // 3))

            # Convolve to smooth
            if len(sorted_ranges) >= window_size:
                smoothed = np.convolve(sorted_ranges, np.ones(window_size)/window_size, mode='same')

                # Find local maxima
                for i in range(window_size, len(smoothed) - window_size):
                    if smoothed[i] < 0.5:  # Too close
                        continue

                    # Check if local maximum
                    if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
                        angle = sorted_angles[i]
                        
                        # Forward bias: prefer gaps ahead, but allow all directions
                        # 1.0 at center (0°), 0.3 at sides (±90°), 0.1 at rear (±180°)
                        abs_angle = abs(angle)
                        if abs_angle < math.pi / 2:  # Front hemisphere
                            forward_bias = 1.0 - (abs_angle / (math.pi / 2)) * 0.7  # 1.0 → 0.3
                        else:  # Rear hemisphere
                            forward_bias = 0.3 - ((abs_angle - math.pi / 2) / (math.pi / 2)) * 0.2  # 0.3 → 0.1

                        width_bonus = 1.0 + (window_deg / 35.0) * 0.3  # Prefer wider gaps

                        score = smoothed[i] * forward_bias * width_bonus

                        if score > best_gap['score']:
                            best_gap = {
                                'angle': angle,
                                'depth': smoothed[i],
                                'score': score
                            }

        return best_gap['angle'], best_gap['depth']

    def _compute_safe_speed(self, angular_cmd, clearance_front, clearance_sides) -> float:
        """
        Compute linear speed using continuous functions instead of hardcoded thresholds.

        Args:
            angular_cmd: [-1, 1] angular velocity command
            clearance_front: meters ahead
            clearance_sides: average lateral clearance

        Returns:
            linear_cmd: [0, 1] normalized linear speed
        """
        # 1. Alignment factor: slower when turning (sigmoid function)
        # 1.0 when straight, ~0.3 when max turn
        alignment_factor = 1.0 / (1.0 + 3.0 * abs(angular_cmd))

        # 2. Clearance factor: slower when close to obstacles
        # Sigmoid: 1.0 when clear (>1m), 0.3 when close (<0.2m)
        clearance_factor = 1.0 / (1.0 + np.exp(-5.0 * (clearance_front - 0.5)))
        clearance_factor = np.clip(clearance_factor, 0.2, 1.0)

        # 3. Lateral clearance: slower in narrow spaces
        # 1.0 when wide (>1m), 0.7 when narrow (<0.5m)
        lateral_factor = 0.7 + 0.3 * np.clip(clearance_sides / 1.0, 0.0, 1.0)

        # Combine factors
        speed = alignment_factor * clearance_factor * lateral_factor

        # Enforce minimum speed to overcome motor deadzone
        return np.clip(speed, 0.25, 1.0)

    def _process_lidar_metrics(self, scan_msg) -> Tuple[float, float, float]:
        """
        Extract key metrics from LiDAR scan using improved multi-scale gap detection.

        Returns:
            min_dist_all (float): Closest obstacle in 360 degrees (Safety Bubble)
            mean_side_dist (float): Average distance on left/right (for lateral clearance)
            gap_heading (float): Heading towards largest open space (-1..1)
        """
        if not scan_msg:
            return 0.0, 0.0, 0.0

        ranges = np.array(scan_msg.ranges)

        # 1. Strict Filtering (NaN, Inf, Range Limits)
        valid = (ranges > 0.15) & (ranges < scan_msg.range_max) & np.isfinite(ranges)

        if not np.any(valid):
            return 3.0, 3.0, 0.0

        valid_ranges = ranges[valid]

        # 1. Safety Bubble (minimum distance in 360°)
        min_dist_all = np.min(valid_ranges)

        # 2. Angle Handling (wrap to [-π, π])
        angles = scan_msg.angle_min + np.arange(len(ranges)) * scan_msg.angle_increment
        angles = (angles + np.pi) % (2 * np.pi) - np.pi

        # 3. Side Clearance (for speed control)
        # Left: +45° to +135° (+0.78 to +2.35 rad)
        # Right: -135° to -45° (-2.35 to -0.78 rad)
        left_mask = (angles > 0.78) & (angles < 2.35) & valid
        right_mask = (angles > -2.35) & (angles < -0.78) & valid

        l_dist = np.mean(ranges[left_mask]) if np.any(left_mask) else 3.0
        r_dist = np.mean(ranges[right_mask]) if np.any(right_mask) else 3.0
        mean_side_dist = (l_dist + r_dist) / 2.0

        # 4. Improved Multi-Scale Gap Finding
        best_angle, best_depth = self._find_best_gap_multiscale(ranges, angles, valid, scan_msg)

        # 5. Corridor Detection and Centering (if enabled)
        # Normalize best_angle from [-π, π] to [-1, 1] for full 360° coverage
        target = np.clip(best_angle / math.pi, -1.0, 1.0)

        if self._enable_corridor_centering:
            is_corridor, centering_error = self._detect_corridor_and_center(scan_msg)

            if is_corridor:
                # In corridor: blend gap following with lateral centering
                # Use 85% gap heading + 15% centering correction
                gap_target = target
                target = 0.85 * gap_target + 0.15 * centering_error

                # DEBUG: Log corridor centering
                if not hasattr(self, '_corridor_log_count'):
                    self._corridor_log_count = 0
                self._corridor_log_count += 1
                if self._corridor_log_count % 30 == 0:  # Every 1 second
                    self.get_logger().info(f"🛤️  Corridor Mode: Gap={gap_target:.2f}, Center={centering_error:.2f}, Blended={target:.2f}")

        # DEBUG: Log if target is stuck at extremes
        if abs(target) > 0.9:
            self.get_logger().info(f"🔍 Gap Debug: Best Angle={best_angle:.2f} rad, Target={target:.2f}, Depth={best_depth:.2f}m")
            self.get_logger().info(f"   Ranges: Min={min_dist_all:.2f}, Max={np.max(valid_ranges):.2f}, Mean={np.mean(valid_ranges):.2f}")

        return min_dist_all, mean_side_dist, target

    def _compute_reward(self, action, linear_vel, angular_vel, min_lidar_dist, side_clearance, collision):
        """
        Direct Track Control reward function:
        - action[0] = left track speed (-1 to 1)
        - action[1] = right track speed (-1 to 1)
        
        Rewards:
        1. Forward progress (both tracks positive and similar)
        2. Opposite track penalty (spinning in place)
        3. Asymmetric track penalty (one track only) 
        4. Side clearance bonus
        5. Smooth control penalty
        6. Full revolution penalty
        
        Range: [-1.0, 1.0]
        """
        reward = 0.0
        target_speed = self._curriculum_max_speed
        
        left_track = action[0]
        right_track = action[1]
        
        # 1. Base idle penalty - always applied, must be overcome by forward motion
        reward -= 0.2
        
        # 2. Forward Motion Reward - based on actual velocity
        if linear_vel > 0.01:
            speed_ratio = np.clip(linear_vel / target_speed, 0.0, 1.0)
            reward += speed_ratio * 1.2  # Up to +1.0 net (cancels idle penalty)
        
        # 3. Backward penalty - both tracks negative
        if linear_vel < -0.01:
            backward_penalty = 0.5 + abs(linear_vel) * 2.0
            reward -= backward_penalty
        
        # 4. TRACK COORDINATION PENALTIES (new for direct track control)
        
        # 4a. Opposite Direction Penalty - tracks spinning in opposite directions = pure rotation
        # This is the key penalty that prevents "spin in place" behavior
        if left_track * right_track < 0:  # Different signs = opposite directions
            # Penalty proportional to how opposite they are
            opposition_magnitude = min(abs(left_track), abs(right_track))
            reward -= opposition_magnitude * 0.8  # Strong penalty for spinning
        
        # 4b. Asymmetric Track Penalty - one track moving, other barely moving
        # Prevents "spin one track backward" behavior
        track_diff = abs(left_track - right_track)
        track_avg = (abs(left_track) + abs(right_track)) / 2.0
        
        if track_avg > 0.1:  # Only apply if tracks are actually active
            asymmetry_ratio = track_diff / (track_avg + 0.1)  # 0 = symmetric, 2 = max asymmetric
            if asymmetry_ratio > 0.8:  # One track significantly different from other
                reward -= 0.3 * asymmetry_ratio
        
        # 4c. Both Tracks Forward Bonus - reward coordinated forward motion
        if left_track > 0.2 and right_track > 0.2:
            # Both tracks are moving forward
            coordination_bonus = min(left_track, right_track) * 0.3
            reward += coordination_bonus
        
        # 5. Side Clearance Reward
        if min_lidar_dist > 0.5:
            reward += 0.1
        elif min_lidar_dist < 0.2:
            reward -= 0.2
        
        # 6. Smooth Control Penalty
        if hasattr(self, '_prev_action'):
            action_diff = np.abs(action - self._prev_action)
            reward -= np.mean(action_diff) * 0.1
        
        # 7. Full Revolution Penalty
        if self._revolution_penalty_triggered:
            reward -= 0.8
            self._revolution_penalty_triggered = False
        
        # 8. Angular velocity penalty (less harsh now that track penalties exist)
        if abs(angular_vel) > 0.3 and abs(linear_vel) < 0.05:
            # Spinning in place without forward progress
            reward -= abs(angular_vel) * 0.3
        
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
        # Process sensors with UnifiedBEVProcessor
        # Returns: bev_grid (2, 128, 128) - Channel 0: LiDAR, Channel 1: Depth
        bev_grid = self.occupancy_processor.process(
            depth_img=self._latest_depth_raw,
            laser_scan=self._latest_scan
        )

        # Store processed sensor data
        self._latest_bev = bev_grid  # (2, 128, 128) float32 [0, 1]

        # Gap Following Analysis using LiDAR occupancy (channel 0)
        # bev_grid[0]: 0.0 = free, 1.0 = occupied
        # For gap finding, invert: free space = high score
        laser_channel = bev_grid[0]  # (128, 128) LiDAR occupancy
        
        # FOCUS ON FRONT HALF ONLY (rows 64-128 = closer to robot)
        # This prevents far-away features from affecting navigation
        front_half = laser_channel[64:128, :]  # (64, 128) - front portion only
        free_space = 1.0 - front_half  # (64, 128), free space = 1.0

        # Find column with maximum average free space
        col_scores = np.mean(free_space, axis=0)  # Average down columns
        
        # Add FORWARD BIAS: prefer driving straight unless gap is significantly better
        # Center columns (54-74) get a bonus (20 cols centered on 64)
        forward_bias = np.zeros(128)
        forward_bias[54:74] = 0.1  # Slight preference for center 20 columns
        col_scores = col_scores + forward_bias

        # WIDER smoothing window to reduce oscillation (scaled for 128: was 25 for 256)
        col_scores = np.convolve(col_scores, np.ones(13)/13, mode='same')

        best_col = np.argmax(col_scores)

        # Map col 0..127 to heading -1..1
        # Col 0 = LEFT, Col 127 = RIGHT, Col 64 = CENTER
        # Angle: Left is +1.0, Right is -1.0
        raw_heading = (64 - best_col) / 64.0
        
        # TEMPORAL SMOOTHING: low-pass filter to prevent rapid oscillation
        # new_heading = alpha * raw + (1 - alpha) * old
        alpha = 0.3  # Low alpha = more smoothing
        if hasattr(self, '_target_heading') and self._target_heading is not None:
            self._target_heading = alpha * raw_heading + (1 - alpha) * self._target_heading
        else:
            self._target_heading = raw_heading

        # Calculate min_forward_dist from laser for reward function
        # LiDAR: 0.0 = free, 1.0 = occupied
        # Center strip: cols 59..69 (±5 from center col 64)
        # Bottom 10 rows: 118..128 (closest to robot at row 127)
        center_patch = laser_channel[118:128, 59:69]

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

        # Get current velocities from EKF-fused odometry for reward/proprioception
        # EKF fuses IMU + wheel encoders for more accurate velocity estimates
        current_linear = self._latest_odom[2] if self._latest_odom else 0.0
        current_angular = self._latest_odom[3] if self._latest_odom else 0.0

        # ========== ROTATION TRACKING FOR REVOLUTION PENALTY (NEW) ==========
        # Track cumulative rotation using EKF-fused yaw for spin-in-place penalty
        if self._last_yaw_for_rotation is not None:
            # Calculate yaw delta (handle wrap-around at ±π)
            yaw_delta = self._latest_fused_yaw - self._last_yaw_for_rotation
            if yaw_delta > math.pi:
                yaw_delta -= 2 * math.pi
            elif yaw_delta < -math.pi:
                yaw_delta += 2 * math.pi
            
            self._cumulative_rotation += abs(yaw_delta)
            
            # Check for full revolution without forward progress
            if self._cumulative_rotation >= 2 * math.pi:
                self._revolution_penalty_triggered = True
                self._cumulative_rotation = 0.0  # Reset after penalty
                self.get_logger().warn('⚠️ Full revolution detected without forward progress!')

        self._last_yaw_for_rotation = self._latest_fused_yaw

        # Check for forward progress to reset rotation counter
        if self._latest_odom and self._last_position_for_rotation is not None:
            x, y = self._latest_odom[0], self._latest_odom[1]
            last_x, last_y = self._last_position_for_rotation
            distance = math.sqrt((x - last_x)**2 + (y - last_y)**2)
            
            if distance > self._forward_progress_threshold:
                self._cumulative_rotation = 0.0  # Reset rotation counter
                self._last_position_for_rotation = (x, y)
        elif self._latest_odom:
            self._last_position_for_rotation = (self._latest_odom[0], self._latest_odom[1])
        # ========== END ROTATION TRACKING ==========

        # Construct 6D proprio: [lidar_min, prev_lin, prev_ang, current_lin, current_ang, gap_heading]
        # Added gap_heading to give model explicit signal of where open space is
        proprio = np.array([[
            lidar_min,                  # Min LiDAR distance (360 Safety)
            self._prev_action[0],       # Previous linear action
            self._prev_action[1],       # Previous angular action
            current_linear,             # Current fused linear velocity (from EKF)
            current_angular,            # Current fused angular velocity (from EKF)
            self._target_heading        # Gap heading: direction to open space [-1, 1]
        ]], dtype=np.float32)

        # 2. Inference (RKNN)
        # Returns: [action_mean] (value head not exported in actor ONNX)
        if self._rknn_runtime:
            # Stateless inference (LSTM removed for export compatibility)
            # Input: [bev, proprio] - Unified BEV grid
            # Add Batch dimension
            bev_input = bev_grid[None, ...]  # (1, 2, 128, 128)

            outputs = self._rknn_runtime.inference(inputs=[bev_input, proprio])

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
                self.get_logger().error(f"   BEV input range: [{bev_input.min():.3f}, {bev_input.max():.3f}]")
                self.get_logger().error(f"   Proprio input: {proprio}")
                # Use zeros and continue
                action_mean = np.zeros(2)

            # We don't get value from actor model, so estimate or ignore
            value = 0.0
        else:
            action_mean = np.zeros(2)
            action = np.zeros(2) # Initialize action to avoid UnboundLocalError
            value = 0.0

        # WARMUP / SEEDING (RANDOM EXPLORATION)
        # Use random actions for model v0 to seed replay buffer with diverse data
        # SAC learns well from random data thanks to entropy maximization
        # Bias towards forward motion to avoid excessive collisions
        # WARMUP / SEEDING (RANDOM EXPLORATION)
        # Use random actions for model v0 to seed replay buffer with diverse data
        # SAC learns well from random data thanks to entropy maximization
        # Bias towards forward motion to avoid excessive collisions
        if self._current_model_version <= 0:
            if not self._warmup_active:
                self._warmup_active = True
                self.get_logger().info('🔥 Starting Random Exploration Warmup...')
                self.pbar.set_description("🔥 Warmup: Random Exploration")

            # DIRECT TRACK CONTROL: Random exploration with forward bias
            # action[0] = left track, action[1] = right track
            # Bias towards both tracks forward for forward motion

            # Base random track speeds - bias towards forward
            base_speed = np.random.uniform(0.3, 1.0)  # Forward bias
            track_variance = np.random.uniform(-0.3, 0.3)  # Slight turn variance
            
            random_left = base_speed + track_variance
            random_right = base_speed - track_variance

            # Safety-based speed scaling
            clearance_dist = lidar_min if lidar_min > 0.05 else self._min_forward_dist

            # Slow down if close to obstacles
            if clearance_dist < 0.25:
                speed_scale = 0.3
            elif clearance_dist < 0.4:
                speed_scale = 0.6
            else:
                speed_scale = 1.0

            random_left = np.clip(random_left * speed_scale, -1.0, 1.0)
            random_right = np.clip(random_right * speed_scale, -1.0, 1.0)

            # DEBUG: Log warmup state occasionally
            if not hasattr(self, '_debug_log_count'):
                self._debug_log_count = 0
            self._debug_log_count += 1
            if self._debug_log_count % 30 == 0:  # Log every 1 second
                self.get_logger().info(
                    f"🔍 Warmup Random: LiDAR={lidar_min:.2f}m, Clearance={clearance_dist:.2f}m, "
                    f"LeftTrack={random_left:.2f}, RightTrack={random_right:.2f}"
                )

            # Override model action with random exploration (direct track format)
            action = np.array([random_left, random_right], dtype=np.float32)

        elif self._warmup_active:
            self.get_logger().info('✅ Warmup Complete (Model v1+ loaded). Switching to learned policy.')
            self._warmup_active = False
            self.pbar.set_description(f"⏳ Training v{self._current_model_version}")

        # 3. Add Exploration Noise (Gaussian)
        # Only apply standard noise if NOT in warmup (Model > 0) AND NOT eval episode
        if self._current_model_version > 0:
            if self._is_eval_episode:
                # Deterministic Evaluation Mode
                # Use action_mean directly (tanh applied by model)
                action = action_mean
                
                # Visual indicator for logs
                if not hasattr(self, '_eval_log_count'):
                    self._eval_log_count = 0
                self._eval_log_count += 1
                if self._eval_log_count % 30 == 0:
                    self.get_logger().info(f"🧪 Evaluation Mode: deterministic action {action}")
            else:
                # Training Mode: Add exploration noise (REDUCED from 0.5 to 0.15)
                noise = np.random.normal(0, 0.15, size=2)  # Lower noise for stable learning
                action = np.clip(action_mean + noise, -1.0, 1.0)

            # Safety check: if action_mean has NaN, use zero and warn
            if np.isnan(action_mean).any():
                self.get_logger().error("⚠️  NaN detected in action_mean from model! Using zero action.")
                action_mean = np.zeros(2)
                action = np.clip(np.random.normal(0, 0.5, size=2), -1.0, 1.0) if not self._is_eval_episode else np.zeros(2)

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
            actual_action = np.array([-0.5, -0.5])  # Both tracks back = reverse
            
            # Only mark as collision if sensors are warmed up
            # During warmup, still stop but don't trigger episode resets
            collision = self._sensor_warmup_complete
        else:
            # Normal execution with DIRECT TRACK CONTROL
            # action[0] = left track speed (-1 to 1)
            # action[1] = right track speed (-1 to 1)
            left_track = action[0]
            right_track = action[1]
            
            # SOFT DEADZONE Compensation for each track
            # Maps [-1, 1] model output to [-1, -MIN] and [MIN, 1] physical range
            def apply_soft_deadzone(val, min_val):
                if abs(val) < 0.001: return 0.0
                return math.copysign(min_val + (1.0 - min_val) * abs(val), val)

            MIN_TRACK = 0.25  # Minimum to overcome motor friction
            
            left_track = apply_soft_deadzone(action[0], MIN_TRACK)
            right_track = apply_soft_deadzone(action[1], MIN_TRACK)
            
            # Convert track speeds to linear/angular for Twist message
            # linear = (left + right) / 2
            # angular = (right - left) / wheel_separation
            wheel_sep = 0.3  # meters (matches motor driver)
            linear_vel = (left_track + right_track) / 2.0
            angular_vel = (right_track - left_track) / wheel_sep
            
            # Apply velocity scaling
            max_speed = self.max_linear if self._current_model_version <= 0 else self._curriculum_max_speed
            cmd.linear.x = float(linear_vel * max_speed)
            cmd.angular.z = float(angular_vel * self.max_angular)
            
            # Record actual track commands (not converted values)
            actual_action = np.array([left_track, right_track])
            collision = False
            
        self.cmd_pub.publish(cmd)

        # 5. Compute Reward
        # Note: current_linear and current_angular already extracted earlier for proprioception

        reward = self._compute_reward(
            actual_action, current_linear, current_angular,
            lidar_min, lidar_sides, collision
        )

        # Safety check: NaN in reward
        if np.isnan(reward) or np.isinf(reward):
            self.get_logger().warn("⚠️  NaN/Inf in reward, skipping data collection")
            return

        # 6. Store Transition
        # Check BEV is ready
        if self._latest_bev is None:
            # No valid sensor data yet
            self.get_logger().warn("⚠️  No BEV data available, skipping data collection")
            return

        with self._buffer_lock:
            self._data_buffer['bev'].append(self._latest_bev)  # (2, 128, 128)
            self._data_buffer['proprio'].append(proprio[0])
            self._data_buffer['actions'].append(actual_action)
            self._data_buffer['rewards'].append(reward)
            self._data_buffer['dones'].append(collision)
            self._data_buffer['is_eval'].append(self._is_eval_episode)
            
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

                # Save depth-only for calibration
                depth_calib = self._process_depth(self._latest_depth_raw)  # (100, 848)

                np.savez_compressed(
                    save_path,
                    bev=self._latest_bev,  # (2, 128, 128)
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

        # Update Evaluation State for NEXT episode
        if self._is_eval_episode:
            self.get_logger().info(f"📊 Eval Episode Complete. Result: {'COLLISION' if self._latest_bev is not None and np.max(self._latest_bev[0, 236:, 118:138]) > 0.5 else 'TIMEOUT'}")
            self._is_eval_episode = False
            self._episodes_since_eval = 0
        else:
            self._episodes_since_eval += 1
            if self._episodes_since_eval >= 5:
                self._is_eval_episode = True
                self.get_logger().info("🧪 Next episode will be EVALUATION (No Noise)")
            else:
                self._is_eval_episode = False

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
