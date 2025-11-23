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

        self.server_addr = str(self.get_parameter('server_addr').value)
        self.max_linear = float(self.get_parameter('max_linear_speed').value)
        self.max_angular = float(self.get_parameter('max_angular_speed').value)
        self.inference_rate = float(self.get_parameter('inference_rate_hz').value)
        self.batch_size = int(self.get_parameter('batch_size').value)

        # State
        self._latest_rgb = None
        self._latest_depth = None
        self._latest_odom = None
        self._latest_imu = None
        self._latest_mag = None
        self._latest_wheel_vels = None
        self._min_forward_dist = 10.0
        self._safety_override = False
        
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
        self._current_model_version = -1
        self._model_update_needed = False
        
        # LSTM State (if used in future, currently stateless PPO)
        self._lstm_h = None
        self._lstm_c = None

        # Previous action for smoothness reward
        self._prev_action = np.array([0.0, 0.0])
        self._prev_linear_cmds = deque(maxlen=20) # For oscillation detection

        # ZMQ Setup
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.REQ)
        self.zmq_socket.connect(self.server_addr)
        
        # Background Threads
        self._stop_event = threading.Event()
        self._last_model_update = 0.0
        self._sync_thread = threading.Thread(target=self._sync_loop)
        self._sync_thread.start()

        # ROS2 Setup
        self.bridge = CvBridge()
        self._setup_subscribers()
        self._setup_publishers()
        
        # Inference Timer
        self.create_timer(1.0 / self.inference_rate, self._control_loop)
        
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
    def _imu_cb(self, msg): self._latest_imu = (msg.linear_acceleration.x, msg.linear_acceleration.y, msg.angular_velocity.z)
    def _mag_cb(self, msg): self._latest_mag = (msg.magnetic_field.x, msg.magnetic_field.y, msg.magnetic_field.z)
    def _joint_cb(self, msg): 
        if len(msg.velocity) >= 4: self._latest_wheel_vels = (msg.velocity[2], msg.velocity[3])
    def _dist_cb(self, msg): self._min_forward_dist = msg.data
    def _safety_cb(self, msg): self._safety_override = msg.data

    def _compute_reward(self, action, linear_vel, angular_vel, clearance, collision):
        """Compute dense reward for current step."""
        reward = 0.0
        
        # 1. Forward progress (scaled by curriculum speed)
        target_speed = self._curriculum_max_speed
        speed_reward = (linear_vel / target_speed) * 2.0
        reward += speed_reward
        
        # 2. Clearance adaptation
        if clearance > 1.5: # Safe
            reward += (linear_vel / target_speed) * 3.0 # Encourage max speed
        elif clearance < 0.5: # Risky
            reward -= abs(linear_vel) * 2.0 # Encourage slowing down
            
        # 3. Collision penalty (huge)
        if collision or self._safety_override:
            reward -= 50.0
            
        # 4. Smoothness (penalize jerky actions)
        action_diff = np.abs(action - self._prev_action)
        reward -= np.sum(action_diff) * 1.0
        
        # 5. Oscillation (penalize sign flips)
        if len(self._prev_linear_cmds) > 2:
            # Check if sign flipped recently
            if self._prev_linear_cmds[-1] * self._prev_linear_cmds[-2] < -0.01:
                reward -= 5.0
                
        # 6. Spinning without moving
        if abs(linear_vel) < 0.05 and abs(angular_vel) > 0.5:
            reward -= 2.0
            
        return reward

    def _control_loop(self):
        """Main control loop running at 30Hz."""
        if not self._model_ready:
            return

        if self._latest_rgb is None:
            # self.get_logger().warn('Waiting for RGB data...', throttle_duration_sec=5.0)
            return

        # 1. Prepare Inputs
        rgb = cv2.resize(self._latest_rgb, (424, 240)) # Resize to model input
        rgb_input = np.transpose(rgb, (2, 0, 1))[None, ...] # (1, 3, 240, 424)
        
        depth = cv2.resize(self._latest_depth, (424, 240))
        depth_input = depth[None, None, ...] # (1, 1, 240, 424)
        
        # Proprioception
        w_l, w_r = self._latest_wheel_vels if self._latest_wheel_vels else (0,0)
        ax, ay, gz = self._latest_imu if self._latest_imu else (0,0,0)
        mx, my, mz = self._latest_mag if self._latest_mag else (0,0,0)
        proprio = np.array([[w_l, w_r, ax, ay, gz, mx, my, mz, self._min_forward_dist]], dtype=np.float32)

        # 2. Inference (RKNN)
        # Returns: [action_mean, lstm_h, lstm_c] (value head not exported in actor ONNX)
        if self._rknn_runtime:
            # Prepare dummy LSTM states (since we don't use them yet but model expects them)
            lstm_h = np.zeros((1, 1, 128), dtype=np.float32)
            lstm_c = np.zeros((1, 1, 128), dtype=np.float32)
            
            outputs = self._rknn_runtime.inference(inputs=[rgb_input, depth_input, proprio, lstm_h, lstm_c])
            
            # Output 0 is action (1, 2)
            action_mean = outputs[0][0] 
            
            # We don't get value from actor model, so estimate or ignore
            value = 0.0 
        else:
            action_mean = np.zeros(2)
            value = 0.0

        # 3. Add Exploration Noise (Gaussian)
        # We use a fixed std dev for exploration on rover, or could receive it from server
        noise = np.random.normal(0, 0.5, size=2) # 0.5 std dev
        action = np.clip(action_mean + noise, -1.0, 1.0)
        
        # Calculate log_prob of this action (needed for PPO)
        # Simplified: log_prob of Gaussian
        log_prob = -0.5 * np.sum(np.square((action - action_mean) / 0.5)) - np.log(0.5 * np.sqrt(2*np.pi))

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
        current_angular = self._latest_odom[3] if self._latest_odom else 0.0
        
        reward = self._compute_reward(
            actual_action, current_linear, current_angular, 
            self._min_forward_dist, collision
        )

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
            
        # Update state
        self._prev_action = actual_action
        self._prev_linear_cmds.append(actual_action[0])

    def _sync_loop(self):
        """Background thread to sync with server."""
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
            
            if batch_to_send:
                try:
                    # Send data
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
                    
                    # Check for model update notification
                    if 'model_version' in response:
                        server_version = response['model_version']
                        if server_version > self._current_model_version:
                            self.get_logger().info(f"🔔 New model available: v{server_version} (Current: v{self._current_model_version})")
                            self._model_update_needed = True
                        
                except Exception as e:
                    self.get_logger().error(f"Sync failed: {e}")
            
            # 2. Request new model if notified
            if self._model_update_needed:
                try:
                    self.get_logger().info("📥 Requesting model update...")
                    self.zmq_socket.send_pyobj({'type': 'get_model'})
                    
                    response = self.zmq_socket.recv_pyobj()
                    if 'model_bytes' in response:
                        # Save ONNX to temp file
                        onnx_path = self._temp_dir / "latest_model.onnx"
                        with open(onnx_path, 'wb') as f:
                            f.write(response['model_bytes'])
                            
                        self.get_logger().info(f"💾 Received ONNX model ({len(response['model_bytes'])} bytes)")
                        
                        # Update version tracking
                        if 'model_version' in response:
                            self._current_model_version = response['model_version']
                        
                        # Convert to RKNN
                        if HAS_RKNN:
                            self.get_logger().info("🔄 Converting to RKNN (this may take a minute)...")
                            rknn_path = str(onnx_path).replace('.onnx', '.rknn')
                            
                            # Call conversion script
                            cmd = ["./convert_onnx_to_rknn.sh", str(onnx_path)]
                            
                            if not os.path.exists("convert_onnx_to_rknn.sh"):
                                self.get_logger().warn("⚠ convert_onnx_to_rknn.sh not found, skipping conversion")
                            else:
                                result = subprocess.run(cmd, capture_output=True, text=True)
                                
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
