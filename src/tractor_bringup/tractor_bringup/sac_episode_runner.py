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
from sensor_msgs.msg import Image, Imu, JointState, LaserScan
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
        self.invert_linear_vel = self.declare_parameter('invert_linear_vel', False).value


        # TQDM Dashboard (Initialize FIRST to avoid race conditions with threads)
        print("\033[H\033[J", end="") # Clear screen
        print("==================================================")
        print("         SAC ROVER RUNNER (V620)                  ")
        print("==================================================")
        self.pbar = tqdm(total=2000, desc="⏳ Server Training", unit="step", dynamic_ncols=True)
        self.total_steps = 0
        self.episode_reward = 0.0

        # State (raw sensors)
        self._latest_depth_raw = None  # Raw depth from camera (424, 240) uint16
        self._latest_scan = None
        # Processed sensor data for model
        self._latest_bev = None  # (2, 128, 128) float32 - Unified BEV grid
        self._latest_odom = None  # EKF fused odometry (position from LiDAR, velocity from wheels)
        self._latest_rf2o_odom = None  # LiDAR-only odometry from rf2o (for velocity)
        self._latest_imu = None
        self._latest_wheel_vels = None
        self._min_forward_dist = 10.0
        self._safety_override = False
        self._velocity_confidence = 1.0  # Velocity estimate confidence (0.0-1.0)

        # Curriculum State (updated by server)
        self._curriculum_collision_dist = 0.5
        self._curriculum_max_speed = self.max_linear  # Use configured max speed (0.18 m/s)

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
        self._warmup_active = False
        self._sensor_warmup_complete = False  # Flag for sensor stabilization
        self._sensor_warmup_countdown = 90  # ~3.0 seconds at 30Hz (increased from 50)

        # Previous action for smoothness reward
        self._prev_action = np.array([0.0, 0.0])
        self._prev_linear_cmds = deque(maxlen=20) # For oscillation detection

        # Gap Following State
        self._target_heading = 0.0 # -1.0 (Left) to 1.0 (Right) - 360 gap heading for reward
        self._bev_heading = 0.0  # Front-biased BEV heading for proprioception
        self._prev_target_heading = 0.0 # For rate limiting
        self._max_depth_val = 0.0

        # Curriculum and Exploration State (NEW)
        self._training_phase = 'exploration'  # exploration, learning, refinement
        self._episode_reward_history = deque(maxlen=50)  # Track episode rewards
        self._current_episode_reward = 0.0  # For episode-based phase tracking
        self._last_phase_transition_step = 0
        self._eval_reward_window = deque(maxlen=20)  # Last 20 eval episode rewards
        self._state_visits = {}  # State visit counts for exploration bonus
        self._prev_actions_buffer = deque(maxlen=30)  # Action history for diversity bonus

        # IMU State for Stitching
        self._latest_imu_yaw = 0.0
        self._prev_imu_yaw = None

        # Rotation tracking for spin penalty
        self._cumulative_rotation = 0.0  # Radians since last forward progress
        self._last_yaw_for_rotation = None  # Previous yaw for delta calculation
        self._forward_progress_threshold = 0.3  # meters to reset rotation counter (reduced from 0.8)
        self._last_position_for_rotation = None  # (x, y) for forward progress detection
        self._revolution_penalty_triggered = False

        # Unstuck reward tracking
        self._prev_min_clearance = 10.0  # Previous step's minimum clearance distance
        self._steps_in_tight_space = 0  # Counter for how long in tight space
        self._latest_fused_yaw = 0.0  # EKF-fused yaw from odometry
        
        # Stuck Detector
        self.stuck_detector = StuckDetector(stuck_threshold=0.15)
        self._is_stuck = False

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
        # Odometry: Use Fused EKF Output (for position/yaw)
        self.create_subscription(Odometry, '/odometry/filtered', self._odom_cb, 10)
        # LiDAR odometry from rf2o (for velocity - more reliable for reward calculation)
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
    # RGB callback removed - depth-only mode
    def _depth_cb(self, msg):
        # Use passthrough to get raw 16-bit depth (uint16 mm)
        d = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        self._latest_depth_raw = d  # Store raw for processing in control loop

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
        
        # Update stuck detector
        self._is_stuck = self.stuck_detector.update(msg)

    def _rf2o_odom_cb(self, msg):
        """Callback for rf2o LiDAR odometry - stores velocity from LiDAR scan matching."""
        # Extract linear and angular velocity from rf2o (LiDAR-derived, more reliable)
        self._latest_rf2o_odom = (
            msg.twist.twist.linear.x,   # LiDAR-derived forward velocity
            msg.twist.twist.angular.z   # LiDAR-derived angular velocity
        )
        
    def _imu_cb(self, msg):
        self._latest_imu = (
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z
        )
    def _joint_cb(self, msg):
        if len(msg.velocity) >= 4: self._latest_wheel_vels = (msg.velocity[2], msg.velocity[3])
    def _safety_cb(self, msg): self._safety_override = msg.data
    def _vel_conf_cb(self, msg): self._velocity_confidence = msg.data

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

        # Improved Multi-Scale Gap Finding
        best_angle, best_depth = self._find_best_gap_multiscale(ranges, angles, valid, scan_msg)

        # Normalize best_angle from [-π, π] to [-1, 1]
        target = np.clip(best_angle / math.pi, -1.0, 1.0)

        # DEBUG: Log if target is stuck at extremes
        if abs(target) > 0.9:
            self.get_logger().info(f"🔍 Gap Debug: Best Angle={best_angle:.2f} rad, Target={target:.2f}, Depth={best_depth:.2f}m")
            self.get_logger().info(f"   Ranges: Min={min_dist_all:.2f}, Max={np.max(valid_ranges):.2f}, Mean={np.mean(valid_ranges):.2f}")

        return min_dist_all, mean_side_dist, target

    def _get_current_phase(self):
        """Get current training phase based on performance metrics."""
        if self._current_model_version <= 5:
            return 'exploration'
        elif self._current_model_version <= 15:
            return 'learning'
        else:
            return 'refinement'
    
    def _compute_exploration_bonus(self, bev_grid):
        """
        Compute exploration bonus based on state visitation.
        
        Returns:
            bonus (float): Additional reward for visiting novel states
        """
        # Quantize BEV to 16x16 for hashing (reduces state space)
        bev_small = cv2.resize(bev_grid[0], (16, 16))
        state_hash = tuple(bev_small.flatten().astype(np.uint8))
        
        # Count visits
        if state_hash not in self._state_visits:
            self._state_visits[state_hash] = 0
        
        visit_count = self._state_visits[state_hash]
        
        # Bonus: +0.1 for first visit, decays with visits
        # Using inverse count heuristic: 1/sqrt(count)
        bonus = 0.1 / (1.0 + np.sqrt(visit_count))
        
        # Increment visit count
        self._state_visits[state_hash] += 1
        
        return bonus
    
    def _compute_action_diversity_bonus(self):
        """
        Compute bonus for action diversity.
        
        Returns:
            bonus (float): Additional reward for varied actions
        """
        if len(self._prev_actions_buffer) < 10:
            return 0.0
        
        # Compute std of recent actions
        actions = np.array(list(self._prev_actions_buffer))
        
        # Linear action std (first column)
        lin_std = np.std(actions[:, 0])
        
        # Angular action std (second column)
        ang_std = np.std(actions[:, 1])
        
        # Reward for diversity in both dimensions
        # Normalize to ~[0, 0.1] range
        bonus = 0.05 * (lin_std + ang_std)
        
        return np.clip(bonus, 0.0, 0.15)
    
    def _compute_reward(self, action, linear_vel, angular_vel, min_lidar_dist, side_clearance, collision, is_stuck):
        """
        Direct Track Control reward function:
        - action[0] = left track speed (-1 to 1)
        - action[1] = right track speed (-1 to 1)

        Rewards:
        1. Forward progress (both tracks positive and similar)
        2. Opposite track penalty (spinning in place) - RELAXED
        3. Asymmetric track penalty (one track only) - RELAXED
        4. Side clearance bonus
        5. Smooth control penalty
        6. Full revolution penalty
        7. Exploration bonus (state visitation)
        8. Action diversity bonus

        Range: [-1.0, 1.0]
        """
        # Apply velocity deadband to filter odometry noise
        VELOCITY_DEADBAND = 0.03  # 3 cm/s - filter encoder noise and drift
        if abs(linear_vel) < VELOCITY_DEADBAND:
            linear_vel = 0.0

        reward = 0.0
        target_speed = self._curriculum_max_speed

        left_track = action[0]
        right_track = action[1]

        # Track current phase for reward scaling
        phase = self._get_current_phase()
        
        # ========== PHASE-DEPENDENT REWARD SCALING ==========
        if phase == 'exploration':
            # Phase 1: Exploration - relax all penalties, encourage ANY motion
            idle_penalty = 0.05  # Very mild penalty
            forward_bonus_mult = 1.5  # Normalized from 3.0
            spin_penalty_scale = 0.3  # Very mild spin penalty
        elif phase == 'learning':
            # Phase 2: Learning - moderate penalties, encourage forward motion
            idle_penalty = 0.15
            forward_bonus_mult = 1.0  # Normalized from 2.0
            spin_penalty_scale = 0.6
        else:  # refinement
            # Phase 3: Refinement - full reward function
            idle_penalty = 0.1
            forward_bonus_mult = 0.8  # Normalized from 1.5
            spin_penalty_scale = 1.0


        # 1. Base idle penalty - only apply when NOT moving forward
        # If moving forward, skip idle penalty (already rewarded via forward bonus)
        if linear_vel < 0.05:  # Only penalize when nearly stationary
            reward -= idle_penalty

        # 2. Coupled Forward Reward (Action + Velocity)
        # Verify that INTENT (Action) matches OUTCOME (Velocity)
        cmd_fwd = (left_track + right_track) / 2.0
        
        # Normalized measured velocity (0 to 1+)
        meas_fwd = linear_vel / target_speed if target_speed > 0 else 0.0

        # Only reward if BOTH are positive
        # If cmd is high but meas is low -> Stuck/Slipping -> Low Reward
        # If meas is high but cmd is low -> Drifting -> Low Reward
        # If cmd is zero (spinning) -> Zero Forward Reward (Neutral)
        base_fwd_reward = 0.0
        if cmd_fwd > 0 and meas_fwd > 0:
            base_fwd_reward = min(cmd_fwd, meas_fwd) * forward_bonus_mult

        reward += base_fwd_reward

        # 3. Backward penalty (unchanged)
        if linear_vel < -0.03:
            backward_penalty = 0.15 + abs(linear_vel) * 0.8
            reward -= backward_penalty
        
        # 4. Tank Steering Efficiency Rules
        
        # 4a. Zero Turn Logic (Spinning in place)
        # If commanded forward speed is low (spinning), enforce symmetry
        if abs(cmd_fwd) < 0.1: 
            # Check for Dragging/Drifting (one track moving, one stopped)
            # Valid spin: L=-0.5, R=0.5 -> diff=1.0, sum_abs=1.0 -> ratio=1.0
            # Invalid spin: L=0.0, R=0.5 -> diff=0.5, sum_abs=0.5 -> ratio=1.0 (Wait, ratio doesn't catch this)
            
            # Better check: Symmetry Error
            # abs(abs(L) - abs(R)) should be low
            symmetry_error = abs(abs(left_track) - abs(right_track))
            
            if symmetry_error > 0.2:
                # Penalize asymmetric zero turns (drifting/pivoting on dead track)
                reward -= 0.2 * symmetry_error * spin_penalty_scale
        
        # 4b. Forward Turn Logic (Arcing)
        # If commanded forward is high, allow asymmetry
        elif cmd_fwd > 0.3:
            # Check if one track is dragging (too slow for the speed)
            min_track = min(left_track, right_track)
            max_track = max(left_track, right_track)
            
            # If fast track is fast (>0.6) but slow track is dead (<0.1) -> Pivot Turn (Inefficient)
            if max_track > 0.6 and min_track < 0.1:
                 # Penalize "Curve Dragging"
                 reward -= 0.1

        # 5. Side Clearance Reward (unchanged)
        if min_lidar_dist > 0.5:
            reward += 0.1
        elif min_lidar_dist < 0.2:
            reward -= 0.2
        
        # 6. Smooth Control Penalty (relaxed)
        if hasattr(self, '_prev_action'):
            action_diff = np.abs(action - self._prev_action)
            reward -= np.mean(action_diff) * 0.05

        
        # ========== EXPLORATION BONUS ==========
        
        # 9. State Visit Bonus (only in exploration/learning phases)
        if phase != 'refinement' and hasattr(self, '_latest_bev'):
            state_bonus = self._compute_exploration_bonus(self._latest_bev)
            reward += state_bonus
        
        # 10. Action Diversity Bonus (always active)
        if hasattr(self, '_prev_actions_buffer'):
            action_bonus = self._compute_action_diversity_bonus()
            reward += action_bonus
        
        # 11. Gap-Heading Reward Shaping
        if hasattr(self, '_target_heading') and abs(self._target_heading) > 0.1:
            intended_turn = right_track - left_track
            gap_direction = self._target_heading

            if linear_vel > 0.03:
                alignment_with_gap = intended_turn * gap_direction
                if alignment_with_gap > 0:
                    reward += 0.15 * min(alignment_with_gap, 0.4)
                elif abs(gap_direction) > 0.5:
                    reward -= 0.05

        # 12. Unstuck Reward Shaping - Encourage escaping tight spaces
        if hasattr(self, '_prev_min_clearance'):
            TIGHT_SPACE_THRESHOLD = 0.35  # Meters - consider "tight" if closer than this

            # Track if we're in a tight space
            if min_lidar_dist < TIGHT_SPACE_THRESHOLD:
                self._steps_in_tight_space += 1
            else:
                self._steps_in_tight_space = 0

            # Clearance improvement bonus - reward escaping from tight spaces
            clearance_delta = min_lidar_dist - self._prev_min_clearance

            if min_lidar_dist < TIGHT_SPACE_THRESHOLD and clearance_delta > 0.02:
                # Escaping! Bonus scales with improvement and how tight it was
                tightness_factor = (TIGHT_SPACE_THRESHOLD - min_lidar_dist) / TIGHT_SPACE_THRESHOLD
                escape_bonus = clearance_delta * 2.0 * (1.0 + tightness_factor)
                reward += np.clip(escape_bonus, 0.0, 0.3)

            # Angular action bonus when stuck - encourage trying to rotate out
            if min_lidar_dist < TIGHT_SPACE_THRESHOLD and self._steps_in_tight_space > 15:
                # Stuck for 0.5s+, encourage rotation attempts
                angular_action_magnitude = abs(right_track - left_track)
                if angular_action_magnitude > 0.3:  # Significant rotation attempt
                    reward += 0.1 * angular_action_magnitude

        # 13. STUCK RECOVERY OVERRIDE (High Priority)
        # If physically stuck (detected by odometry), FORCE the policy to learn zero-turns
        if is_stuck:
            # Penalize ANY forward/backward command attempts when stuck
            # (Spinning wheels in place is bad)
            fwd_effort = abs(left_track + right_track)
            reward -= 1.0 * fwd_effort
            
            # Reward PURE rotation (Zero Turn)
            # Max difference is 2.0 (-1 vs 1)
            rot_effort = abs(left_track - right_track)
            reward += 1.0 * rot_effort
            
            # Additional penalty if tracks are not opposing
            if left_track * right_track > 0: # Both positive or both negative
                reward -= 0.5
                
        # 14. ARC TURN BONUS (Agile Obstacle Avoidance)
        # If moving fast AND turning, while near obstacles but safe -> Good!
        # Encourages "swooping" around mapped obstacles rather than stop-turn-go
        if linear_vel > 0.2 and abs(angular_vel) > 0.5 and min_lidar_dist > 0.3:
            # Reward maintaining speed while turning
            # Max bonus ~ 0.3 * 0.5 * 1.0 = 0.15
            reward += 0.3 * abs(linear_vel) * abs(angular_vel)

        # 15. WALL PROXIMITY PENALTY (Don't drive into walls)
        # If very close to wall and driving TOWARDS it -> Penalty
        if min_lidar_dist < 0.5 and linear_vel > 0.1:
            # Check if driving towards the closest point
            # We don't have exact vector here, but high linear velocity when close is dangerous
            reward -= 0.5 * (0.5 - min_lidar_dist) * linear_vel

            # Update previous clearance for next step
            self._prev_min_clearance = min_lidar_dist

        return np.clip(reward, -1.0, 1.0)

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
        if hasattr(self, '_bev_heading'):
            self._bev_heading = alpha * raw_heading + (1 - alpha) * self._bev_heading
        else:
            self._bev_heading = raw_heading

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

        # LiDAR Metrics for Reward (360° gap heading)
        lidar_min, lidar_sides, gap_heading = self._process_lidar_metrics(self._latest_scan)

        # Update target heading for reward shaping (360° awareness)
        # Note: _bev_heading (front-biased) is used for proprioception
        self._target_heading = gap_heading

        # Get current velocities from rf2o LiDAR odometry for reward calculation
        # rf2o provides more reliable velocity from scan matching (ignores wheel slip)
        # Falls back to EKF odometry if rf2o data unavailable
        current_linear = 0.0
        current_angular = 0.0
        
        if self._latest_rf2o_odom:
            # Use rf2o LiDAR odometry (primary source - more reliable)
            current_linear = self._latest_rf2o_odom[0]  # LiDAR-derived forward velocity
            current_angular = self._latest_rf2o_odom[1]  # LiDAR-derived angular velocity
        elif self._latest_odom:
            # Fallback to EKF odometry if rf2o unavailable
            current_linear = self._latest_odom[2]
            current_angular = self._latest_odom[3]
        
        # Log once when switching sources
        if not hasattr(self, '_odom_source_log'):
            self._odom_source_log = 0
        if self._odom_source_log < 5:
            source = "rf2o (LiDAR)" if self._latest_rf2o_odom else "EKF fallback"
            self.get_logger().info(f"odom velocity source: {source}")
            self.get_logger().info(f"odom velocity source: {source}")
            self._odom_source_log += 1

        # Apply velocity inversion if configured (fixes backwards mounting/penalty issues)
        if self.invert_linear_vel:
            current_linear = -current_linear


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

        # Construct 6D proprio: [lidar_min, prev_lin, prev_ang, current_lin, current_ang, bev_heading]
        # Using front-biased BEV heading (not 360° gap) to encourage forward driving
        proprio = np.array([[
            lidar_min,                  # Min LiDAR distance (360 Safety)
            self._prev_action[0],       # Previous linear action (left track)
            self._prev_action[1],       # Previous angular action (right track)
            current_linear,             # Current fused linear velocity (from EKF)
            current_angular,            # Current fused angular velocity (from EKF)
            self._bev_heading           # Front-biased BEV heading [-1, 1] (prefers forward)
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

            # DIRECT TRACK CONTROL: Tank-Style Warmup (Diverse Dynamics)
            # action[0] = left track, action[1] = right track
            
            # Modes:
            # 1. Zero Turn (Low speed, opposite tracks) - 20%
            # 2. Arc Turn (High speed, asymmetric) - 40%
            # 3. Straight (High speed, equal) - 20%
            # 4. Recovery/Chaos (Full random) - 20%
            
            rand_mode = np.random.rand()
            
            if rand_mode < 0.2: # Zero Turn
                # Spin in place at controlled speed (0.4 to 0.6)
                spin_speed = np.random.uniform(0.4, 0.6)
                if np.random.rand() < 0.5:
                    random_left = spin_speed
                    random_right = -spin_speed
                else:
                    random_left = -spin_speed
                    random_right = spin_speed
                    
            elif rand_mode < 0.6: # Arc Turn (Forward biased)
                # Base forward speed (0.5 to 1.0)
                base_speed = np.random.uniform(0.5, 1.0)
                # Apply turn offset to one track
                turn_offset = np.random.uniform(0.2, 0.5)
                if np.random.rand() < 0.5:
                    random_left = base_speed
                    random_right = base_speed - turn_offset
                else:
                    random_left = base_speed - turn_offset
                    random_right = base_speed
                    
            elif rand_mode < 0.8: # Straight Forward
                speed = np.random.uniform(0.5, 1.0)
                random_left = speed
                random_right = speed
                
            else: # Recovery / Chaos (Full range)
                random_left = np.random.uniform(-1.0, 1.0)
                random_right = np.random.uniform(-1.0, 1.0)

            # Safety-based speed scaling (prevent high-speed crashes)
            clearance_dist = lidar_min if lidar_min > 0.05 else self._min_forward_dist

            # Slow down if close to obstacles
            if clearance_dist < 0.4:
                speed_scale = 0.5
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
                    f"🎲 Warmup Chaos: LiDAR={lidar_min:.2f}m, "
                    f"L={random_left:.2f}, R={random_right:.2f}"
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
        # Allow rotation if stuck, even if close!
        is_stuck = self._is_stuck
        
        # If stuck, RELAX the safety distance to allow zero-turn recovery
        # But force stop of linear motion if very close
        if is_stuck and effective_min_dist < 0.12:
            safety_triggered = False # Allow control, but reward will penalize forward
            # Force zero linear command in safety check? No, let RL learn it via reward
            # But limits might be needed to prevent damage
            if effective_min_dist < 0.05: # Extremely close
                safety_triggered = True
        else:
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

        # Track action for diversity bonus (always, including eval episodes)
        if self._current_model_version > 0:
            # Store actual executed action for diversity tracking
            self._prev_actions_buffer.append(actual_action.copy())

        self.cmd_pub.publish(cmd)

        # 5. Compute Reward
        # Note: current_linear and current_angular already extracted earlier for proprioception

        reward = self._compute_reward(
            actual_action, current_linear, current_angular,
            lidar_min, lidar_sides, collision, is_stuck
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

                save_path = self._calibration_dir / f"calib_{timestamp}.npz"

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
        
        # Log reward breakdown occasionally for debugging (using lidar_sides instead of side_clearance)
        if np.random.rand() < 0.05:  # ~5% of steps
            self._log_reward_breakdown(
                actual_action, current_linear, current_angular,
                lidar_min, lidar_sides, is_stuck
            )

    def _trigger_episode_reset(self):
        """Call motor driver to reset encoder baselines."""
        if not self.reset_episode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Episode reset service unavailable')
            return

        # Record episode reward for phase tracking (before resetting)
        episode_reward = self.episode_reward
        
        # Update Evaluation State for NEXT episode
        if self._is_eval_episode:
            # Store eval reward for performance tracking
            self._eval_reward_window.append(episode_reward)
            
            # Log eval episode summary
            avg_eval = np.mean(list(self._eval_reward_window)[-10:]) if self._eval_reward_window else 0.0
            self.get_logger().info(
                f"📊 Eval Episode Complete: Reward={episode_reward:.2f}, "
                f"Avg10={avg_eval:.2f}"
            )
            
            self._is_eval_episode = False
            self._episodes_since_eval = 0
            
            # Check for phase transition based on eval performance
            self._check_phase_transition()
        else:
            # Store training episode reward for history
            if hasattr(self, '_episode_reward_history'):
                self._episode_reward_history.append(episode_reward)
            
            # Log training episode summary
            avg_train = np.mean(list(self._episode_reward_history)[-10:]) if self._episode_reward_history else 0.0
            self.get_logger().info(
                f"📊 Training Episode Complete: Reward={episode_reward:.2f}, "
                f"Avg10={avg_train:.2f}"
            )
            
            self._episodes_since_eval += 1
            if self._episodes_since_eval >= 5:
                self._is_eval_episode = True
                self.get_logger().info("🧪 Next episode will be EVALUATION (No Noise)")
            else:
                self._is_eval_episode = False

        # Reset episode reward counter
        self.episode_reward = 0.0

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
    
    def _log_reward_breakdown(self, action, linear_vel, angular_vel, min_lidar_dist, side_clearance, is_stuck=False):
        """Log detailed reward components for debugging."""
        # Get phase
        phase = self._get_current_phase()
        left_track, right_track = action[0], action[1]
        
        # Calculate components
        components = {'forward': 0.0, 'idle_penalty': 0.0, 'backward': 0.0,
                      'spin': 0.0, 'smoothness': 0.0, 'track_coord': 0.0,
                      'exploration': 0.0, 'diversity': 0.0, 'unstuck': 0.0}

        target_speed = self._curriculum_max_speed

        if phase == 'exploration':
            idle_penalty = 0.05
            forward_mult = 1.5
            spin_penalty_scale = 0.3
        elif phase == 'learning':
            idle_penalty = 0.15
            forward_mult = 1.0
            spin_penalty_scale = 0.6
        else:
            idle_penalty = 0.1
            forward_mult = 0.8
            spin_penalty_scale = 1.0


        # Idle penalty only when nearly stationary
        if linear_vel < 0.05:
            components['idle_penalty'] = -idle_penalty
        else:
            components['idle_penalty'] = 0.0

        # Coupled Forward Reward
        cmd_fwd = (left_track + right_track) / 2.0
        meas_fwd = linear_vel / target_speed if target_speed > 0 else 0.0
        
        if cmd_fwd > 0 and meas_fwd > 0:
            components['forward'] = min(cmd_fwd, meas_fwd) * forward_mult
        
        # Backward Penalty (unchanged)
        if linear_vel < -0.03:
            components['backward'] = -(0.15 + abs(linear_vel) * 0.8)

        # Tank Steering Rules
        if abs(cmd_fwd) < 0.1:
            symmetry_error = abs(abs(left_track) - abs(right_track))
            if symmetry_error > 0.2:
                components['track_coord'] = -0.2 * symmetry_error * spin_penalty_scale
        elif cmd_fwd > 0.3:
            min_track = min(left_track, right_track)
            max_track = max(left_track, right_track)
            if max_track > 0.6 and min_track < 0.1:
                components['track_coord'] = -0.1 * 0.3 * spin_penalty_scale



        if hasattr(self, '_latest_bev') and phase != 'refinement':
            components['exploration'] = self._compute_exploration_bonus(self._latest_bev)

        if hasattr(self, '_prev_actions_buffer'):
            components['diversity'] = self._compute_action_diversity_bonus()

        # Unstuck bonus computation (matches reward function)
        if hasattr(self, '_prev_min_clearance'):
            TIGHT_SPACE_THRESHOLD = 0.35
            clearance_delta = min_lidar_dist - self._prev_min_clearance

            if min_lidar_dist < TIGHT_SPACE_THRESHOLD and clearance_delta > 0.02:
                tightness_factor = (TIGHT_SPACE_THRESHOLD - min_lidar_dist) / TIGHT_SPACE_THRESHOLD
                escape_bonus = clearance_delta * 2.0 * (1.0 + tightness_factor)
                components['unstuck'] += np.clip(escape_bonus, 0.0, 0.3)

            if min_lidar_dist < TIGHT_SPACE_THRESHOLD and self._steps_in_tight_space > 15:
                left_track, right_track = action[0], action[1]
                angular_action_magnitude = abs(right_track - left_track)
                if angular_action_magnitude > 0.3:
                    components['unstuck'] += 0.1 * angular_action_magnitude

        if is_stuck:
             fwd_effort = abs(left_track + right_track)
             rot_effort = abs(left_track - right_track)
             components['stuck_state'] = -1.0 * fwd_effort + 1.0 * rot_effort
             if left_track * right_track > 0:
                 components['stuck_state'] -= 0.5
        else:
             components['stuck_state'] = 0.0
             
        # Arc Turn Bonus
        if linear_vel > 0.2 and abs(angular_vel) > 0.5 and min_lidar_dist > 0.3:
            components['arc_turn'] = 0.3 * abs(linear_vel) * abs(angular_vel)
        else:
            components['arc_turn'] = 0.0
            
        # Wall Avoidance
        if min_lidar_dist < 0.5 and linear_vel > 0.1:
            components['wall_avoid'] = -0.5 * (0.5 - min_lidar_dist) * linear_vel
        else:
            components['wall_avoid'] = 0.0

        # Log
        total = sum(components.values())
        self.get_logger().info(
            f"📊 Reward Breakdown (phase={phase}): "
            f"Total={total:.3f} | Fwd={components['forward']:.2f} "
            f"(idle:{components['idle_penalty']:+.2f}) | "
            f"Bwd={components['backward']:.2f} Spin={components['spin']:.2f} | "
            f"TrkCoord={components['track_coord']:.2f} Smooth={components['smoothness']:.2f} | "
            f"Exp={components['exploration']:+.2f} Div={components['diversity']:+.2f} Unstuck={components['unstuck']:+.2f}\n"
            f"   StuckState={components['stuck_state']:+.2f} Arc={components['arc_turn']:+.2f} Wall={components['wall_avoid']:+.2f}"
        )
    
    def _check_phase_transition(self):
        """Check if eval performance warrants phase transition."""
        if len(self._eval_reward_window) < 5:
            return
        
        # Only check for transitions after model v6 (exploration phase done)
        if self._current_model_version < 6:
            return
        
        # Calculate recent average
        avg_reward = np.mean(list(self._eval_reward_window)[-5:])
        
        # Get current and target phase
        current_phase = self._get_current_phase()
        
        if current_phase == 'exploration' and avg_reward > -0.3:
            # Phase 1 -> 2: Exploration -> Learning
            self._current_model_version = max(self._current_model_version, 6)
            self.get_logger().info(
                f"⭐ PHASE TRANSITION: exploration → learning "
                f"(avg eval reward = {avg_reward:.3f} > -0.3)"
            )
        elif current_phase == 'learning' and avg_reward > -0.1:
            # Phase 2 -> 3: Learning → Refinement
            self._current_model_version = max(self._current_model_version, 16)
            self.get_logger().info(
                f"⭐ PHASE TRANSITION: learning → refinement "
                f"(avg eval reward = {avg_reward:.3f} > -0.1)"
            )

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
