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

from tractor_bringup.occupancy_processor import DepthToOccupancy

# ROS2 Messages
from sensor_msgs.msg import Image, Imu, JointState, MagneticField
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
        self.pbar = tqdm(total=200, desc="⏳ Server Training", unit="step", dynamic_ncols=True)
        self.total_steps = 0
        self.episode_reward = 0.0

        # State
        self._latest_rgb = None # Keep for debug/logging if needed, but not used for model
        self._latest_depth = None
        self._latest_grid = None # (64, 64) uint8
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
        
        # Previous action for smoothness reward
        self._prev_action = np.array([0.0, 0.0])
        self._prev_linear_cmds = deque(maxlen=20) # For oscillation detection
        self._prev_angular_actions = deque(maxlen=10) # For action smoothness tracking

        # Gap Following State
        self._target_heading = 0.0 # -1.0 (Left) to 1.0 (Right)
        self._max_depth_val = 0.0

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
        self.occupancy_processor = DepthToOccupancy(
            width=424, height=240,
            camera_height=0.123, # Calculated from URDF: 0.029 + 0.08025 + 0.01375
            camera_tilt_deg=0.0,
            obstacle_height_thresh=0.1,
            floor_thresh=0.08
        )
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
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.create_subscription(Imu, '/imu/data', self._imu_cb, qos_profile_sensor_data)
        self.create_subscription(MagneticField, '/imu/mag', self._mag_cb, qos_profile_sensor_data)
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
    def _odom_cb(self, msg): 
        self._latest_odom = (msg.pose.pose.position.x, msg.pose.pose.position.y, msg.twist.twist.linear.x, msg.twist.twist.angular.z)
    def _imu_cb(self, msg): 
        self._latest_imu = (
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z
        )
    def _mag_cb(self, msg): self._latest_mag = (msg.magnetic_field.x, msg.magnetic_field.y, msg.magnetic_field.z)
    def _joint_cb(self, msg):
        if len(msg.velocity) >= 4: self._latest_wheel_vels = (msg.velocity[2], msg.velocity[3])
    def _safety_cb(self, msg): self._safety_override = msg.data
    def _vel_conf_cb(self, msg): self._velocity_confidence = msg.data

    def _compute_reward(self, action, linear_vel, angular_vel, clearance, collision):
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
        # Process Depth -> Occupancy Grid
        t0 = time.time()
        grid = self.occupancy_processor.process(self._latest_depth)
        self._latest_grid = grid
        
        # Grid Input for Model: (1, 1, 64, 64)
        # Normalize to [0, 1] for model
        grid_normalized = grid.astype(np.float32) / 255.0
        grid_input = grid_normalized[None, None, ...] 
        
        # Gap Following Analysis (using Grid now!)
        # Find best heading from grid
        # Simple approach: Find column with most "free" space (128)
        # Scan rows from bottom up
        
        # ... (Keep existing gap logic or adapt to grid? Existing used raw depth strips)
        # Let's adapt to grid for consistency
        
        # Sum free space in columns
        # Grid: 0=Unknown, 128=Free, 255=Occupied
        # We want columns with 128 and NO 255 close to robot
        
        # Simple heuristic: Sum of (is_free) - Sum of (is_occupied * penalty)
        free_mask = (grid == 128).astype(np.float32)
        occ_mask = (grid == 255).astype(np.float32)
        
        col_scores = np.sum(free_mask, axis=0) - np.sum(occ_mask * 5, axis=0)
        
        # Smooth scores
        col_scores = np.convolve(col_scores, np.ones(5)/5, mode='same')
        
        best_col = np.argmax(col_scores)
        
        # Map col 0..63 to -1..1
        # Col 0 is Left (max Y), Col 63 is Right (min Y)
        # So 0 -> 1.0, 63 -> -1.0
        # Wait, in processor:
        # grid_cols = Center + Y * scale
        # Y positive is Left.
        # So Col > Center is Left.
        # Col < Center is Right.
        # 32 is Center.
        # 63 is Left. 0 is Right.
        
        self._target_heading = (best_col - 32) / 32.0
        
        # Calculate min_forward_dist from grid (for reward function)
        # Scan center strip (width ~30cm -> 6 pixels)
        # Grid resolution: 3.0m / 64px = 0.047m/px
        # Center col is 32. 32 +/- 3 = 29..35
        center_strip = grid[:, 29:35]
        # Find obstacles (255)
        obs_rows, _ = np.where(center_strip == 255)
        
        if len(obs_rows) > 0:
            # Closest obstacle is the one with largest row index (closest to bottom/robot)
            closest_obs_row = np.max(obs_rows)
            # Distance in pixels
            dist_px = 63 - closest_obs_row
            # Distance in meters
            self._min_forward_dist = dist_px * (3.0 / 64.0)
        else:
            self._min_forward_dist = 3.0 # Max range
            
        # Proprioception (10 values: 9-axis IMU + min_dist)
        # Get IMU data
        if self._latest_imu:
            ax, ay, az, gx, gy, gz = self._latest_imu
        else:
            ax, ay, az, gx, gy, gz = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Get Magnetometer data
        if self._latest_mag:
            mx, my, mz = self._latest_mag
        else:
            mx, my, mz = 0.0, 0.0, 0.0

        # Construct 10D proprio: [ax, ay, az, gx, gy, gz, mx, my, mz, min_dist]
        proprio = np.array([[
            ax, ay, az, gx, gy, gz, mx, my, mz, self._min_forward_dist
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

        # WARMUP DISABLED - New reward function provides strong forward motion incentive
        # Old warmup forced straight driving during model version 0
        # With 3x stronger forward rewards (24 max vs 8 old), this is no longer needed
        # if self._current_model_version == 0:
        #     if not self._warmup_active:
        #         self._warmup_active = True
        #         self.get_logger().info('🔥 Starting Continuous Warmup: Driving Straight until Model 1...')
        #     action = np.array([0.8, 0.0], dtype=np.float32)
        # elif self._warmup_active:
        #     self.get_logger().info('✅ Warmup Complete (Model loaded). Switching to policy.')
        #     self._warmup_active = False

        # 3. Add Exploration Noise (Gaussian)
        # Noise is always applied (warmup disabled, no exception needed)
        # We use a fixed std dev for exploration on rover, or could receive it from server
        noise = np.random.normal(0, 0.5, size=2) # 0.5 std dev
        action = np.clip(action_mean + noise, -1.0, 1.0)

        # Safety check: if action_mean has NaN, use zero and warn
        if np.isnan(action_mean).any():
            self.get_logger().error("⚠️  NaN detected in action_mean from model! Using zero action.")
            action_mean = np.zeros(2)
            action = np.clip(noise, -1.0, 1.0)  # Pure random

        # 4. Execute Action
        cmd = Twist()
        
        # Safety Override Logic
        if self._safety_override or self._min_forward_dist < 0.12:
            # Override: Stop and reverse slightly
            cmd.linear.x = -0.05
            cmd.angular.z = 0.0
            actual_action = np.array([-0.5, 0.0]) # Record that we stopped
            collision = True
        else:
            # Normal execution
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
            self._min_forward_dist, collision
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
            
            # Update progress bar to show steps towards next model (modulo 200)
            progress = total_steps % 200
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
