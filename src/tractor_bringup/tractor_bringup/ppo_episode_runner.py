#!/usr/bin/env python3
"""PPO Autonomous Episode Runner for Rover.

Runs continuous PPO inference on NPU, collects experience tuples,
calculates dense rewards, and asynchronously syncs with V620 server.
"""

import os
import math
import time
import threading
import queue
import tempfile
import subprocess
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import zmq
import cv2
from cv_bridge import CvBridge

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

class PPOEpisodeRunner(Node):
    """Continuous PPO runner with async data collection."""

    def __init__(self) -> None:
        super().__init__('ppo_episode_runner')

        # Parameters
        self.declare_parameter('server_addr', 'tcp://10.0.0.200:5556')
        self.declare_parameter('max_linear_speed', 0.18)
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('inference_rate_hz', 30.0)
        self.declare_parameter('batch_size', 256)  # Send data every N steps
        self.declare_parameter('collection_duration', 180.0) # Seconds to collect before triggering training

        self.server_addr = str(self.get_parameter('server_addr').value)
        self.max_linear = float(self.get_parameter('max_linear_speed').value)
        self.max_angular = float(self.get_parameter('max_angular_speed').value)
        self.inference_rate = float(self.get_parameter('inference_rate_hz').value)
        self.batch_size = int(self.get_parameter('batch_size').value)
        self.collection_duration = float(self.get_parameter('collection_duration').value)

        # State
        self._latest_rgb = None
        self._latest_depth = None
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
            'rgb': [], 'depth': [], 'proprio': [], 
            'actions': [], 'rewards': [], 'dones': [], 
            'log_probs': [], 'values': []
        }
        self._buffer_lock = threading.Lock()

        # Model State
        self._rknn_runtime = None
        self._model_ready = True  # Allow random exploration initially
        self._temp_dir = Path(tempfile.mkdtemp(prefix='ppo_rover_'))
        self._calibration_dir = Path("./calibration_data")
        self._calibration_dir.mkdir(exist_ok=True)
        self._current_model_version = -1
        self._model_update_needed = False
        
        # Warmup State
        self._warmup_start_time = 0.0
        self._warmup_active = False
        
        # LSTM State (if used in future, currently stateless PPO)
        self._lstm_h = None
        self._lstm_c = None

        # Previous action for smoothness reward
        self._prev_action = np.array([0.0, 0.0])
        self._prev_linear_cmds = deque(maxlen=20) # For oscillation detection
        self._prev_angular_actions = deque(maxlen=10) # For action smoothness tracking

        # Clearance tracking for centering rewards
        self._left_clearance = 5.0
        self._right_clearance = 5.0

        # ZMQ Setup
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.REQ)
        self.zmq_socket.connect(self.server_addr)
        
        # Background Threads
        self._stop_event = threading.Event()
        self._initial_sync_done = threading.Event() # Wait for server handshake
        self._last_model_update = 0.0
        self._sync_thread = threading.Thread(target=self._sync_loop)
        self._sync_thread.start()

        # ROS2 Setup
        self.bridge = CvBridge()
        self._setup_subscribers()
        self._setup_publishers()
        
        # Inference Timer
        self.create_timer(1.0 / self.inference_rate, self._control_loop)

        # Episode reset client for encoder baseline reset
        self.reset_episode_client = self.create_client(Trigger, '/reset_episode')

        self.get_logger().info('🚀 PPO Runner Initialized')

    def _setup_subscribers(self):
        self.create_subscription(Image, '/camera/camera/color/image_raw', self._rgb_cb, qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self._depth_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.create_subscription(Imu, '/imu/data', self._imu_cb, qos_profile_sensor_data)
        self.create_subscription(MagneticField, '/imu/mag', self._mag_cb, qos_profile_sensor_data)
        self.create_subscription(JointState, '/joint_states', self._joint_cb, 10)
        self.create_subscription(Float32, '/min_forward_distance', self._dist_cb, 10)
        self.create_subscription(Bool, '/safety_monitor_status', self._safety_cb, 10)
        self.create_subscription(Float32, '/velocity_confidence', self._vel_conf_cb, 10)

    def _setup_publishers(self):
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel_ai', 10)

    # Callbacks
    def _rgb_cb(self, msg): self._latest_rgb = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
    def _depth_cb(self, msg): 
        d = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        if d.dtype == np.uint16: d = d.astype(np.float32) * 0.001
        self._latest_depth = d
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
    def _dist_cb(self, msg): self._min_forward_dist = msg.data
    def _safety_cb(self, msg): self._safety_override = msg.data
    def _vel_conf_cb(self, msg): self._velocity_confidence = msg.data

    def _compute_reward(self, action, linear_vel, angular_vel, clearance, collision):
        """Tank-steering optimized reward with conditional spinning penalty.

        Key improvements for tank steering:
        - Conditional spinning penalty based on environment
        - Allows point-turns in tight spaces
        - Rewards turning toward open space
        - Higher weight on centering for corridor navigation
        """
        reward = 0.0
        target_speed = self._curriculum_max_speed

        # 1. Forward Progress Bonus (quadratic for max speed)
        forward_vel = max(0.0, linear_vel)
        if forward_vel > 0.01:
            speed_reward = (forward_vel / target_speed) ** 2 * 5.0
            reward += speed_reward
            if forward_vel > 0.08:
                reward += 3.0

        # 2. Backward Motion Penalty
        if linear_vel < -0.01:
            reward -= abs(linear_vel) * 20.0

        # 3. CONDITIONAL Spinning Penalty (Context-Aware for Tank Steering)
        if abs(linear_vel) < 0.05 and abs(angular_vel) > 0.3:
            min_side_clearance = min(self._left_clearance, self._right_clearance)

            if clearance > 1.0 and min_side_clearance > 0.8:
                # Wide open space - penalize stationary spinning
                reward -= 12.0 + (abs(angular_vel) * 3.0)
            elif clearance < 0.5 or min_side_clearance < 0.4:
                # Tight space - ALLOW point-turns (energy cost only)
                reward -= abs(angular_vel) * 0.5
                # Bonus for turning toward open space
                if (self._left_clearance < self._right_clearance and angular_vel > 0) or \
                   (self._right_clearance < self._left_clearance and angular_vel < 0):
                    reward += 2.0
            else:
                # Medium clearance - moderate penalty
                reward -= abs(angular_vel) * 3.0

        # 4. Clearance Adaptation
        if clearance > 1.5:
            reward += forward_vel * 2.0  # Safe - encourage speed
        elif clearance < 0.5:
            if forward_vel > 0.05:
                reward -= forward_vel * 3.0  # Risky - slow down

        # 5. Centering Reward
        if self._left_clearance < 2.0 or self._right_clearance < 2.0:
            balance_diff = abs(self._left_clearance - self._right_clearance)
            if balance_diff > 0.3:
                reward -= balance_diff * 4.0

            min_side = min(self._left_clearance, self._right_clearance)
            if min_side > 0.4:
                reward += (min_side - 0.4) * 3.0

        # 6. Collision Penalty
        if collision or self._safety_override:
            reward -= 50.0

        # 7. Action Smoothness
        if len(self._prev_angular_actions) > 0:
            angular_jerk = abs(action[1] - self._prev_angular_actions[-1])
            if angular_jerk > 0.4:
                reward -= angular_jerk * 6.0
        self._prev_angular_actions.append(action[1])

        # 8. Oscillation Penalty
        if len(self._prev_linear_cmds) > 2:
            if self._prev_linear_cmds[-1] * self._prev_linear_cmds[-2] < -0.01:
                reward -= 10.0

        # 9. Smooth Obstacle Navigation Bonus
        if forward_vel > 0.08 and abs(angular_vel) > 0.2 and clearance < 0.8:
            action_diff = np.abs(action - self._prev_action)
            if np.sum(action_diff) < 0.15:
                reward += 6.0

        # 10. General angular penalty (prefer straight) - REDUCED from 0.5 to 0.3
        reward -= abs(angular_vel) * 0.3

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

        if self._latest_rgb is None:
            # self.get_logger().warn('Waiting for RGB data...', throttle_duration_sec=5.0)
            return

        if self._latest_depth is None:
            # self.get_logger().warn('Waiting for depth data...', throttle_duration_sec=5.0)
            return

        # 1. Prepare Inputs
        rgb = cv2.resize(self._latest_rgb, (424, 240)) # Resize to model input
        # CRITICAL: Pass uint8 [0,255] to RKNN (it handles normalization via config)
        # rgb = rgb.astype(np.float32) / 255.0  <-- REMOVED
        rgb_input = rgb[None, ...] # (1, 240, 424, 3) - RKNN expects NHWC for uint8 input usually, but let's check config
        # Actually, RKNN API usually expects NHWC for images.
        # But our model was exported with NCHW input layout in PyTorch.
        # RKNN config 'mean_values' applies to channel dimension.
        # If we pass NHWC uint8, RKNN converts to NCHW float internally if model expects it.
        # Let's keep it simple: Pass NHWC uint8.
        rgb_input = rgb[None, ...] # (1, 240, 424, 3)

        depth = cv2.resize(self._latest_depth, (424, 240))

        # Calculate left/right clearance for centering rewards (like MAP-Elites)
        h, w = depth.shape
        roi_y_start = h // 2  # Bottom half only

        # Left 30% and Right 30%
        left_roi = depth[roi_y_start:, :int(w*0.3)]
        right_roi = depth[roi_y_start:, int(w*0.7):]

        # Filter valid depths (>0.1m) and cap at 5.0m
        valid_left = left_roi[(left_roi > 0.1) & (left_roi < 5.0)]
        valid_right = right_roi[(right_roi > 0.1) & (right_roi < 5.0)]

        self._left_clearance = float(np.min(valid_left)) if len(valid_left) > 0 else 5.0
        self._right_clearance = float(np.min(valid_right)) if len(valid_right) > 0 else 5.0
        # CRITICAL: Normalize depth from [0,6m] to [0,1] to match RKNN calibration
        depth_normalized = depth / 6.0
        depth_input = depth_normalized[None, None, ...] # (1, 1, 240, 424)
        
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
            outputs = self._rknn_runtime.inference(inputs=[rgb_input, depth_input, proprio])

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
                self.get_logger().error(f"   RGB input range: [{rgb_input.min():.3f}, {rgb_input.max():.3f}]")
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

        # INTELLIGENT WARMUP SEQUENCE (Model 0)
        # This overrides any model output if we are in warmup phase
        if self._current_model_version == 0:
            if not self._warmup_active:
                self._warmup_active = True
                self.get_logger().info('🔥 Starting Continuous Warmup: Driving Straight until Model 1...')

            # Always drive forward
            action = np.array([0.8, 0.0], dtype=np.float32)
            
        elif self._warmup_active:
             # Just switched from 0 to >0
             self.get_logger().info('✅ Warmup Complete (Model loaded). Switching to policy.')
             self._warmup_active = False

        # 3. Add Exploration Noise (Gaussian)
        # Skip noise during warmup (Model 0)
        if self._current_model_version == 0:
             # During warmup, action is deterministic and hardcoded
             # We set action_mean to action so log_prob calculation works (it will be 0 distance)
             action_mean = action
             # No noise added
        else:
            # We use a fixed std dev for exploration on rover, or could receive it from server
            noise = np.random.normal(0, 0.5, size=2) # 0.5 std dev
            action = np.clip(action_mean + noise, -1.0, 1.0)

            # Safety check: if action_mean has NaN, use zero and warn
            if np.isnan(action_mean).any():
                self.get_logger().error("⚠️  NaN detected in action_mean from model! Using zero action.")
                action_mean = np.zeros(2)
                action = np.clip(noise, -1.0, 1.0)  # Pure random

        # Calculate log_prob of this action (needed for PPO)
        # Simplified: log_prob of Gaussian
        log_prob = -0.5 * np.sum(np.square((action - action_mean) / 0.5)) - np.log(0.5 * np.sqrt(2*np.pi))

        # Safety check: if log_prob is NaN, skip this step
        if np.isnan(log_prob) or np.isinf(log_prob):
            self.get_logger().warn("⚠️  NaN/Inf in log_prob, skipping data collection")
            return

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

        # Clip reward to prevent extreme values
        reward = np.clip(reward, -100.0, 100.0)

        # Safety check: NaN in reward
        if np.isnan(reward) or np.isinf(reward):
            self.get_logger().warn("⚠️  NaN/Inf in reward, skipping data collection")
            return

        # 6. Store Transition
        with self._buffer_lock:
            self._data_buffer['rgb'].append(rgb)
            self._data_buffer['depth'].append(depth)
            self._data_buffer['proprio'].append(proprio[0])
            self._data_buffer['actions'].append(actual_action)
            self._data_buffer['rewards'].append(reward)
            self._data_buffer['dones'].append(collision) # Treat collision as terminal for value estimation
            self._data_buffer['log_probs'].append(log_prob)
            self._data_buffer['values'].append(value)
            
        # Save calibration data (keep ~100 samples)
        # We save occasionally to avoid disk I/O spam
        if np.random.rand() < 0.1: # 10% chance to save sample
            calib_files = list(self._calibration_dir.glob('*.npz'))
            if len(calib_files) < 100:
                timestamp = int(time.time() * 1000)
                save_path = self._calibration_dir / f"calib_{timestamp}.npz"
                np.savez_compressed(
                    save_path,
                    rgb=rgb,
                    depth=depth,
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
                    self.get_logger().info(f'Episode reset: {response.message}')
                else:
                    self.get_logger().warn(f'Episode reset failed: {response.message}')
            except Exception as e:
                self.get_logger().error(f'Episode reset error: {e}')

        future.add_done_callback(_log_response)

    def _sync_loop(self):
        """Background thread to sync with server."""
        
        # Initial Handshake: Query server state before doing anything
        self.get_logger().info("🤝 Connecting to server to sync state...")
        while not self._stop_event.is_set():
            try:
                self.zmq_socket.send_pyobj({'type': 'check_status'})
                # Wait for response with timeout
                if self.zmq_socket.poll(timeout=2000):
                    response = self.zmq_socket.recv_pyobj()
                    if 'model_version' in response:
                        server_version = response['model_version']
                        self.get_logger().info(f"✅ Connected! Server is at v{server_version}")
                        self._current_model_version = server_version
                        
                        # If server has a trained model, we need to fetch it
                        if server_version > 0:
                            self._model_update_needed = True
                            
                        self._initial_sync_done.set()
                        break
                else:
                    self.get_logger().warn("⏳ Waiting for server response...")
            except Exception as e:
                self.get_logger().warn(f"Handshake failed: {e}. Retrying...")
                time.sleep(1.0)
        
        while not self._stop_event.is_set():
            # 1. Check if we have enough data to send
            batch_to_send = None
            with self._buffer_lock:
                if len(self._data_buffer['rewards']) >= self.batch_size:
                    # Extract batch
                    batch_to_send = {k: np.array(v) for k, v in self._data_buffer.items()}
                    # Clear buffer
                    for k in self._data_buffer:
                        self._data_buffer[k] = []
            
            # Check if collection duration exceeded
            time_based_trigger = False
            if time.time() - self._collection_start_time > self.collection_duration:
                time_based_trigger = True
                self.get_logger().info(f"⏰ Collection duration ({self.collection_duration}s) exceeded. Triggering training...")
                
                # Force flush remaining data even if < batch_size
                with self._buffer_lock:
                    if len(self._data_buffer['rewards']) > 0:
                        if batch_to_send is None:
                            batch_to_send = {k: np.array(v) for k, v in self._data_buffer.items()}
                        else:
                            # Append to existing batch
                            for k, v in self._data_buffer.items():
                                batch_to_send[k] = np.concatenate([batch_to_send[k], np.array(v)])
                        
                        # Clear buffer
                        for k in self._data_buffer:
                            self._data_buffer[k] = []

            if batch_to_send:
                try:
                    self.get_logger().info(f"📤 Sending batch of {len(batch_to_send['rewards'])} steps")
                    self.zmq_socket.send_pyobj({
                        'type': 'data_batch',
                        'data': batch_to_send
                    })
                    
                    # Receive response (curriculum updates)
                    response = self.zmq_socket.recv_pyobj()
                    if 'curriculum' in response:
                        curr = response['curriculum']
                        self._curriculum_collision_dist = curr['collision_dist']
                        self._curriculum_max_speed = curr['max_speed']
                        self.get_logger().info(f"🎓 Curriculum: Dist={self._curriculum_collision_dist:.2f}, Speed={self._curriculum_max_speed:.2f}")
                    
                    # Check for wait signal (Server is training)
                    if response.get('wait_for_training', False):
                        self.get_logger().info("🛑 Server is training. Pausing rover...")
                        self._model_ready = False # Stop inference loop
                        
                        # Stop robot immediately
                        stop_cmd = Twist()
                        self.cmd_pub.publish(stop_cmd)
                        
                        # Poll until ready
                        while not self._stop_event.is_set():
                            time.sleep(1.0)
                            try:
                                self.zmq_socket.send_pyobj({'type': 'check_status'})
                                status_resp = self.zmq_socket.recv_pyobj()
                                
                                if status_resp.get('status') == 'ready':
                                    self.get_logger().info("✅ Server training complete. Resuming...")
                                    # Check if we need to update model
                                    if status_resp.get('model_version', -1) > self._current_model_version:
                                        self._model_update_needed = True
                                    else:
                                        self._model_ready = True # Resume if no update needed
                                    break
                                else:
                                    self.get_logger().info("⏳ Waiting for training to finish...")
                            except Exception as e:
                                self.get_logger().error(f"Polling failed: {e}")
                                time.sleep(1.0)

                    # Check for explicit error from server
                    if response.get('type') == 'error':
                        self.get_logger().error(f"❌ Server reported error: {response.get('msg')}")

                    # Check for model update notification
                    if 'model_version' in response:
                        server_version = response['model_version']
                        if server_version > self._current_model_version:
                            self.get_logger().info(f"🔔 New model available: v{server_version} (Current: v{self._current_model_version})")
                            self.get_logger().info(f"🔔 New model available: v{server_version} (Current: v{self._current_model_version})")
                            self._model_update_needed = True
                        
                except Exception as e:
                    self.get_logger().error(f"Sync failed: {e}")
            
            # 1.5 Send Manual Trigger if time-based
            if time_based_trigger:
                try:
                    self.get_logger().info("🛑 Stopping robot and requesting training start...")
                    self._model_ready = False # Stop inference
                    
                    # Stop robot
                    stop_cmd = Twist()
                    self.cmd_pub.publish(stop_cmd)
                    
                    # Send trigger
                    self.zmq_socket.send_pyobj({'type': 'start_training'})
                    response = self.zmq_socket.recv_pyobj()
                    
                    if response.get('status') == 'training_queued':
                        self.get_logger().info("✅ Training queued successfully. Waiting for completion...")
                        
                        # Wait loop
                        while not self._stop_event.is_set():
                            time.sleep(1.0)
                            try:
                                self.zmq_socket.send_pyobj({'type': 'check_status'})
                                status_resp = self.zmq_socket.recv_pyobj()
                                
                                if status_resp.get('status') == 'ready':
                                    self.get_logger().info("✅ Training complete!")
                                    if status_resp.get('model_version', -1) > self._current_model_version:
                                        self._model_update_needed = True
                                    
                                    # Reset timer and resume
                                    self._collection_start_time = time.time()
                                    self._model_ready = True
                                    break
                                else:
                                    self.get_logger().info("⏳ Training in progress...")
                            except Exception as e:
                                self.get_logger().error(f"Polling error: {e}")
                                time.sleep(1.0)
                    else:
                        self.get_logger().warn(f"Trigger failed: {response}")
                        # Reset timer anyway to avoid loop
                        self._collection_start_time = time.time()
                        self._model_ready = True
                        
                except Exception as e:
                    self.get_logger().error(f"Failed to trigger training: {e}")
                    self._collection_start_time = time.time() # Reset to avoid stuck loop
            
            # 2. Request new model if notified
            if self._model_update_needed:
                try:
                    # SKIP download for Model 0 (Warmup Model)
                    # We don't need a neural network for the hardcoded warmup sequence
                    if self._current_model_version == -1 and response.get('model_version', -1) == 0:
                         # We are initializing to Model 0
                         self.get_logger().info("🔥 Initializing Warmup Sequence (Model 0) - Skipping download")
                         self._current_model_version = 0
                         self._model_ready = True
                         self._model_update_needed = False
                         # Ensure we don't try to download
                         continue

                    self.get_logger().info("📥 Requesting model update...")
                    self.zmq_socket.send_pyobj({'type': 'get_model'})
                    
                    response = self.zmq_socket.recv_pyobj()
                    if 'model_bytes' in response:
                        # Save ONNX to temp file
                        onnx_path = self._temp_dir / "latest_model.onnx"
                        with open(onnx_path, 'wb') as f:
                            f.write(response['model_bytes'])
                            f.flush()
                            os.fsync(f.fileno())
                            
                        self.get_logger().info(f"💾 Received ONNX model ({len(response['model_bytes'])} bytes)")
                        
                        # Update version tracking
                        if 'model_version' in response:
                            self._current_model_version = response['model_version']
                        
                        # Convert to RKNN
                        if HAS_RKNN:
                            self.get_logger().info("🔄 Converting to RKNN (this may take a minute)...")
                            rknn_path = str(onnx_path).replace('.onnx', '.rknn')
                            
                            # Call conversion script
                            cmd = ["./convert_onnx_to_rknn.sh", str(onnx_path), str(self._calibration_dir)]
                            
                            if not os.path.exists("convert_onnx_to_rknn.sh"):
                                self.get_logger().warn("⚠ convert_onnx_to_rknn.sh not found, skipping conversion")
                            else:
                                result = subprocess.run(cmd, capture_output=True, text=True)

                                # Log conversion output for debugging
                                if result.stdout:
                                    for line in result.stdout.strip().split('\n'):
                                        if 'Test output:' in line or 'RKNN model produces' in line or 'Range:' in line:
                                            self.get_logger().info(f"[RKNN] {line.strip()}")

                                if result.returncode == 0 and os.path.exists(rknn_path):
                                    self.get_logger().info("✅ RKNN Conversion successful")
                                    
                                    # Load new model
                                    self.get_logger().info("🔄 Loading new RKNN model...")
                                    new_runtime = RKNNLite()
                                    ret = new_runtime.load_rknn(rknn_path)
                                    if ret != 0:
                                        self.get_logger().error("Load RKNN failed")
                                    else:
                                        ret = new_runtime.init_runtime()
                                        if ret != 0:
                                            self.get_logger().error("Init RKNN runtime failed")
                                        else:
                                            # Swap runtime
                                            self._rknn_runtime = new_runtime
                                            self._model_ready = True
                                            self._model_update_needed = False # Reset flag only on success
                                            self.get_logger().info(f"🚀 New model v{self._current_model_version} loaded and active!")
                                else:
                                    self.get_logger().error(f"RKNN Conversion failed: {result.stderr}")
                        else:
                            # If no RKNN (e.g. testing on PC), just mark as updated
                            self._model_update_needed = False
                        
                except Exception as e:
                    self.get_logger().error(f"Model update failed: {e}")

            time.sleep(0.1)

    def destroy_node(self):
        self._stop_event.set()
        self._sync_thread.join()
        super().destroy_node()

from collections import deque

def main(args=None):
    rclpy.init(args=args)
    node = PPOEpisodeRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
