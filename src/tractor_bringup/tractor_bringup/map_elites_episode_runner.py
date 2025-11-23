#!/usr/bin/env python3
"""MAP-Elites autonomous episode runner for rover.

Runs autonomous episodes and reports results back to V620 server.
No teleoperation required - rover explores autonomously!
"""

import os
import math
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import zmq

# Import torch for deserializing PyTorch models
try:
    import torch
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    print("⚠ PyTorch not available - cannot receive models from V620")

from sensor_msgs.msg import Image, Imu, JointState, MagneticField
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge

try:
    from rknnlite.api import RKNNLite
    HAS_RKNN = True
except ImportError:
    HAS_RKNN = False
    print("⚠ RKNNLite not available - cannot run on NPU")


class MAPElitesEpisodeRunner(Node):
    """Autonomous episode runner for MAP-Elites."""

    def __init__(self) -> None:
        super().__init__('map_elites_episode_runner')

        # Parameters
        self.declare_parameter('server_addr', 'tcp://10.0.0.200:5556')
        self.declare_parameter('episode_duration', 60.0)  # seconds
        self.declare_parameter('max_linear_speed', 0.18)
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('collision_distance', 0.12)  # m
        self.declare_parameter('inference_rate_hz', 30.0)  # Match camera frame rate
        self.declare_parameter('use_npu', True)

        self.server_addr = str(self.get_parameter('server_addr').value)
        self.episode_duration = float(self.get_parameter('episode_duration').value)
        self.max_linear = float(self.get_parameter('max_linear_speed').value)
        self.max_angular = float(self.get_parameter('max_angular_speed').value)
        self.collision_dist = float(self.get_parameter('collision_distance').value)
        self.inference_rate = float(self.get_parameter('inference_rate_hz').value)
        self.use_npu = bool(self.get_parameter('use_npu').value)

        # State
        self._latest_rgb = None
        self._latest_depth = None
        self._latest_odom = None
        self._latest_imu = None
        self._latest_mag = None
        self._latest_wheel_vels = None
        self._min_forward_dist = 10.0
        self._episode_running = False
        self._rknn_runtime = None
        self._current_model_state = None
        
        # LSTM hidden state for temporal memory
        self._lstm_hidden_h = None  # (1, 1, 128) when initialized
        self._lstm_hidden_c = None  # (1, 1, 128) when initialized

        # Episode metrics
        self._episode_start_time = 0.0
        self._episode_start_pos = None
        self._total_distance = 0.0
        self._collision_count = 0
        self._speed_samples = []
        self._clearance_samples = []
        self._left_clearance_samples = []  # NEW: Left side clearance
        self._right_clearance_samples = []  # NEW: Right side clearance
        
        # NEW: Tank-specific metrics
        self._heading_samples = []  # For turn efficiency calculation
        self._stationary_rotation_time = 0.0  # Time spent pivoting in place
        self._track_slip_detected = False  # Flag for track slippage
        self._last_commanded_speed = 0.0  # For slip detection
        self._last_actual_speed = 0.0  # For slip detection
        
        # NEW: Coverage and Oscillation
        self._visited_grid = set()
        self._grid_resolution = 0.5  # 0.5m grid cells
        self._oscillation_count = 0
        self._recent_linear_actions = []
        self._last_linear_cmd = 0.0
        self._last_angular_cmd = 0.0

        # Trajectory collection (for gradient refinement)
        self._collect_trajectory = True  # Always collect by default (uses RAM but saves episode time)
        self._trajectory_rgb = []
        self._trajectory_depth = []
        self._trajectory_proprio = []
        self._trajectory_actions = []
        self._trajectory_headings = []  # NEW: Store headings for turn efficiency

        # Trajectory cache per model (32GB RAM on rover)
        self._trajectory_cache = {}  # model_id -> trajectory_data
        self._max_cache_size = 10  # Keep last 10 model trajectories (~1.5GB each = 15GB total)

        self.bridge = CvBridge()

        # ZeroMQ REQ socket for bidirectional communication with V620
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.REQ)
        self.zmq_socket.connect(self.server_addr)
        self.zmq_socket.setsockopt(zmq.LINGER, 0)

        # Current model being evaluated
        self._current_model_id = None

        # Generate unique client ID for debugging multiple instances
        import uuid
        self.client_id = str(uuid.uuid4())[:8]
        self.get_logger().info(f'Client ID: {self.client_id}')

        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw',
            self.rgb_callback, qos_profile_sensor_data
        )
        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback, qos_profile_sensor_data
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom',
            self.odom_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data',
            self.imu_callback, qos_profile_sensor_data
        )
        self.mag_sub = self.create_subscription(
            MagneticField, '/imu/mag',
            self.mag_callback, qos_profile_sensor_data
        )
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states',
            self.joint_state_callback, 10
        )
        self.min_dist_sub = self.create_subscription(
            Float32, '/min_forward_distance',
            self.min_dist_callback, 10
        )

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel_ai', 10)

        # Inference timer
        self.inference_timer = self.create_timer(
            1.0 / self.inference_rate,
            self.inference_callback
        )

        # Create temp directory for model conversions
        self.temp_dir = Path(tempfile.mkdtemp(prefix='map_elites_'))
        self.get_logger().info(f'Temp directory: {self.temp_dir}')

        self.get_logger().info('MAP-Elites episode runner initialized')
        self.get_logger().info(f'Server: {self.server_addr}')
        self.get_logger().info(f'Episode duration: {self.episode_duration}s')

        # Request first model from server
        self.request_new_model()

    def convert_pytorch_from_file(self, pt_path: str, model_id: int) -> Optional[str]:
        """Convert PyTorch .pt file to RKNN model.

        Args:
            pt_path: Path to PyTorch .pt file
            model_id: Model ID for naming

        Returns:
            Path to RKNN model file, or None if conversion failed
        """
        try:
            self.get_logger().info(f'  PyTorch model: {pt_path}')

            # Export to ONNX
            onnx_path = self.temp_dir / f'model_{model_id}.onnx'

            self.get_logger().info('  Exporting to ONNX...')
            result = subprocess.run([
                'python3',
                '/home/ubuntu/ros2-rover/src/tractor_bringup/tractor_bringup/export_actor_to_onnx.py',
                str(pt_path),
                str(onnx_path)
            ], capture_output=True, text=True, timeout=60)

            if result.returncode != 0:
                self.get_logger().error(f'ONNX export failed: {result.stderr}')
                return None

            self.get_logger().info(f'  ✓ ONNX exported: {onnx_path}')

            # Convert ONNX to RKNN
            rknn_path = self.temp_dir / f'model_{model_id}.rknn'

            self.get_logger().info('  Converting to RKNN...')
            result = subprocess.run([
                'python3',
                '/home/ubuntu/ros2-rover/src/tractor_bringup/tractor_bringup/convert_onnx_to_rknn.py',
                str(onnx_path),
                '--output', str(rknn_path)
            ], capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                self.get_logger().error(f'RKNN conversion failed: {result.stderr}')
                return None

            if not rknn_path.exists():
                self.get_logger().error('RKNN file not created')
                return None

            self.get_logger().info(f'  ✓ RKNN converted: {rknn_path} ({rknn_path.stat().st_size / 1024:.1f} KB)')

            return str(rknn_path)

        except Exception as e:
            self.get_logger().error(f'Model conversion failed: {e}')
            import traceback
            traceback.print_exc()
            return None

    def load_rknn_model(self, rknn_path: str) -> bool:
        """Load RKNN model into runtime.

        Args:
            rknn_path: Path to RKNN model file

        Returns:
            True if successful
        """
        if not HAS_RKNN or not self.use_npu:
            self.get_logger().warn('RKNN not available, will use random actions')
            return False

        try:
            # Release old runtime if exists
            if self._rknn_runtime is not None:
                self._rknn_runtime.release()

            # Create new runtime
            self._rknn_runtime = RKNNLite()

            # Load model
            ret = self._rknn_runtime.load_rknn(rknn_path)
            if ret != 0:
                self.get_logger().error(f'Failed to load RKNN model: {ret}')
                self._rknn_runtime = None
                return False

            # Init runtime
            ret = self._rknn_runtime.init_runtime()
            if ret != 0:
                self.get_logger().error(f'Failed to init RKNN runtime: {ret}')
                self._rknn_runtime = None
                return False

            self.get_logger().info('✓ RKNN model loaded and ready')
            return True

        except Exception as e:
            self.get_logger().error(f'Failed to load RKNN model: {e}')
            self._rknn_runtime = None
            return False

    def rgb_callback(self, msg: Image) -> None:
        """Store latest RGB image."""
        try:
            self._latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().warn(f'RGB conversion failed: {e}')

    def depth_callback(self, msg: Image) -> None:
        """Store latest depth image."""
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) * 0.001  # mm to m
            self._latest_depth = depth.astype(np.float32)

            # NEW: Calculate left/right clearance for centering reward
            if self._episode_running:
                h, w = self._latest_depth.shape
                # Bottom half only (like safety monitor)
                roi_y_start = h // 2
                
                # Left 30%
                left_roi = self._latest_depth[roi_y_start:, :int(w*0.3)]
                # Right 30%
                right_roi = self._latest_depth[roi_y_start:, int(w*0.7):]
                
                # Filter valid depths (>0.1m) and cap at 5.0m
                valid_left = left_roi[(left_roi > 0.1) & (left_roi < 5.0)]
                valid_right = right_roi[(right_roi > 0.1) & (right_roi < 5.0)]
                
                left_clearance = float(np.min(valid_left)) if len(valid_left) > 0 else 5.0
                right_clearance = float(np.min(valid_right)) if len(valid_right) > 0 else 5.0
                
                self._left_clearance_samples.append(left_clearance)
                self._right_clearance_samples.append(right_clearance)

        except Exception as e:
            self.get_logger().warn(f'Depth conversion failed: {e}')

    def odom_callback(self, msg: Odometry) -> None:
        """Store latest odometry."""
        # Extract quaternion and convert to yaw (heading)
        q = msg.pose.pose.orientation
        # Yaw from quaternion: atan2(2*(w*z + x*y), 1 - 2*(y² + z²))
        yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        
        self._latest_odom = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.twist.twist.linear.x,
            msg.twist.twist.angular.z,
            yaw  # NEW: Add heading
        )
        
        # NEW: Update coverage grid
        if self._episode_running:
            grid_x = math.floor(msg.pose.pose.position.x / self._grid_resolution)
            grid_y = math.floor(msg.pose.pose.position.y / self._grid_resolution)
            self._visited_grid.add((grid_x, grid_y))

    def imu_callback(self, msg: Imu) -> None:
        """Store latest IMU data."""
        self._latest_imu = (
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.angular_velocity.z
        )

    def mag_callback(self, msg: MagneticField) -> None:
        """Store latest Magnetometer data."""
        self._latest_mag = (
            msg.magnetic_field.x,
            msg.magnetic_field.y,
            msg.magnetic_field.z
        )

    def joint_state_callback(self, msg: JointState) -> None:
        """Store latest wheel velocities."""
        # Assuming velocity[2] is left and velocity[3] is right based on Hiwonder driver
        # joint_msg.velocity = [0.0, 0.0, self.left_velocity, self.right_velocity]
        if len(msg.velocity) >= 4:
            self._latest_wheel_vels = (msg.velocity[2], msg.velocity[3])

    def min_dist_callback(self, msg: Float32) -> None:
        """Store minimum forward distance."""
        self._min_forward_dist = msg.data

        # Check for collision during episode
        if self._episode_running and self._min_forward_dist < self.collision_dist:
            self._collision_count += 1
            self.get_logger().warn(f'Collision detected! ({self._min_forward_dist:.2f}m)')

    def request_new_model(self) -> None:
        """Request new model from V620 server."""
        try:
            # Send model request
            request = {
                'type': 'request_model',
                'client_id': self.client_id
            }
            self.zmq_socket.send_pyobj(request)

            self.get_logger().info(f'Requesting new model from V620 (Client {self.client_id})...')

            # Wait for response (infinite timeout - may be delayed by refinement/tournament)
            if self.zmq_socket.poll(timeout=-1):  # Infinite timeout
                response = self.zmq_socket.recv_pyobj()

                if response['type'] == 'model':
                    self._current_model_id = response['model_id']
                    model_bytes = response['model_bytes']

                    self.get_logger().info(
                        f'✓ Received model #{self._current_model_id} '
                        f'({response["generation_type"]}) - {len(model_bytes)/1024:.1f} KB'
                    )

                    # Save model bytes directly to file (no torch needed)
                    pt_path = self.temp_dir / f'model_{self._current_model_id}.pt'
                    with open(pt_path, 'wb') as f:
                        f.write(model_bytes)

                    self._current_model_state = str(pt_path)  # Store path instead of dict

                    # Convert PyTorch → ONNX → RKNN
                    self.get_logger().info('Converting model to RKNN...')
                    rknn_path = self.convert_pytorch_from_file(
                        str(pt_path),
                        self._current_model_id
                    )

                    if rknn_path:
                        # Load into RKNN runtime
                        success = self.load_rknn_model(rknn_path)
                        if success:
                            self.get_logger().info('🚀 Model ready for inference!')
                        else:
                            self.get_logger().warn('⚠ Using random actions (RKNN load failed)')
                    else:
                        self.get_logger().warn('⚠ Using random actions (conversion failed)')

                    # Start new episode
                    self.start_episode()

                else:
                    self.get_logger().error(f'Unexpected response type: {response.get("type")}')
                    # Try again after delay
                    time.sleep(5.0)
                    self.request_new_model()

            else:
                self.get_logger().error('Timeout waiting for model from V620')
                # Try again
                time.sleep(5.0)
                self.request_new_model()

        except Exception as e:
            self.get_logger().error(f'Failed to request model: {e}')
            time.sleep(5.0)
            self.request_new_model()

    def start_episode(self) -> None:
        """Start new autonomous episode."""
        self._episode_running = True
        self._episode_start_time = time.time()

        # Wait for odometry
        while self._latest_odom is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        # Reset metrics
        self._episode_start_time = time.time()
        self._episode_start_pos = (self._latest_odom[0], self._latest_odom[1])
        self._total_distance = 0.0
        self._collision_count = 0
        self._speed_samples = []
        self._clearance_samples = []
        self._left_clearance_samples = []
        self._right_clearance_samples = []
        self._heading_samples = []
        self._stationary_rotation_time = 0.0
        self._track_slip_detected = False
        
        # NEW: Reset coverage and oscillation
        self._visited_grid.clear()
        self._oscillation_count = 0
        self._recent_linear_actions = []
        self._last_linear_cmd = 0.0
        self._last_angular_cmd = 0.0

        # Reset trajectory buffers
        self._trajectory_rgb = []
        self._trajectory_depth = []
        self._trajectory_proprio = []
        self._trajectory_actions = []
        self._trajectory_headings = []  # NEW: Clear heading trajectory

        # NEW: Reset LSTM hidden state for new episode
        self._lstm_hidden_h = None
        self._lstm_hidden_c = None

        self.get_logger().info('🚀 Episode started')

    def end_episode(self) -> None:
        """End episode and send results to server."""
        self._episode_running = False

        # Stop rover
        stop_cmd = Twist()
        self.cmd_pub.publish(stop_cmd)

        # Compute episode metrics
        duration = time.time() - self._episode_start_time

        # Compute total distance traveled
        if self._latest_odom and self._episode_start_pos:
            dx = self._latest_odom[0] - self._episode_start_pos[0]
            dy = self._latest_odom[1] - self._episode_start_pos[1]
            self._total_distance = np.sqrt(dx**2 + dy**2)

        # Compute averages
        avg_speed = np.mean(self._speed_samples) if self._speed_samples else 0.0
        avg_clearance = np.mean(self._clearance_samples) if self._clearance_samples else 0.0
        avg_left_clearance = np.mean(self._left_clearance_samples) if self._left_clearance_samples else 0.0
        avg_right_clearance = np.mean(self._right_clearance_samples) if self._right_clearance_samples else 0.0
        
        # NEW: Compute turn efficiency (distance per heading change)
        turn_efficiency = 0.0
        if len(self._heading_samples) > 1 and self._total_distance > 0.1:
            total_heading_change = np.sum(np.abs(np.diff(self._heading_samples)))
            turn_efficiency = self._total_distance / (total_heading_change + 1e-6)
        
        # NEW: Detect track slippage (if we have commanded vs actual speed data)
        # This is a simple detection - can be enhanced with more sophisticated slip detection
        track_slip_detected = self._track_slip_detected

        # Compute action smoothness (lower is better = smoother turning)
        action_smoothness = 0.0
        avg_linear_action = 0.0
        avg_angular_action = 0.0
        if len(self._trajectory_actions) > 1:
            actions = np.array(self._trajectory_actions)  # (N, 2) - [linear, angular]
            # Measure angular velocity changes (jerkiness in turning)
            angular_actions = actions[:, 1]  # Extract angular component
            angular_diffs = np.diff(angular_actions)  # Changes between timesteps
            action_smoothness = float(np.std(angular_diffs))  # Standard deviation of changes

            # Measure forward vs rotation bias (for tank steering)
            linear_actions = actions[:, 0]  # Extract linear component
            avg_linear_action = float(np.mean(np.abs(linear_actions)))  # Average forward intent
            avg_angular_action = float(np.mean(np.abs(angular_actions)))  # Average rotation intent

        # Cache trajectory data for this model (if we collected any)
        if len(self._trajectory_rgb) > 0:
            self.cache_trajectory_data()
            self.get_logger().info(
                f'  Cached trajectory: {len(self._trajectory_rgb)} samples, '
                f'{len(self._trajectory_cache)} models in cache'
            )

        # Package episode results
        episode_result = {
            'type': 'episode_result',
            'client_id': self.client_id,
            'model_id': self._current_model_id,
            'total_distance': float(self._total_distance),
            'collision_count': int(self._collision_count),
            'avg_speed': float(avg_speed),
            'avg_clearance': float(avg_clearance),
            'avg_left_clearance': float(avg_left_clearance),
            'avg_right_clearance': float(avg_right_clearance),
            'duration': float(duration),
            'action_smoothness': float(action_smoothness),
            'avg_linear_action': float(avg_linear_action),
            'avg_angular_action': float(avg_angular_action),
            # NEW: Tank-specific metrics
            'turn_efficiency': float(turn_efficiency),
            'stationary_rotation_time': float(self._stationary_rotation_time),
            'track_slip_detected': bool(track_slip_detected),
            'coverage_count': int(len(self._visited_grid)),
            'oscillation_count': int(self._oscillation_count),
        }

        # Send results to server and wait for acknowledgment
        try:
            self.zmq_socket.send_pyobj(episode_result)

            self.get_logger().info(
                f'✓ Episode #{self._current_model_id} complete: '
                f'dist={self._total_distance:.2f}m, '
                f'collisions={self._collision_count}, '
                f'avg_speed={avg_speed:.3f}m/s, '
                f'avg_clearance={avg_clearance:.2f}m, '
                f'L/R={avg_left_clearance:.1f}/{avg_right_clearance:.1f}m'
            )

            # Wait for acknowledgment (infinite timeout - server might be checkpointing)
            if self.zmq_socket.poll(timeout=-1):
                ack = self.zmq_socket.recv_pyobj()
                if ack.get('type') == 'ack':
                    self.get_logger().info('✓ Results acknowledged by V620')

                    # Check if server wants trajectory data for gradient refinement
                    if ack.get('collect_trajectory', False):
                        model_id = ack.get('model_id', self._current_model_id)
                        self.get_logger().info(f'→ Server requested trajectory for model #{model_id}')

                        # Check if we have cached trajectory
                        if model_id in self._trajectory_cache:
                            self.get_logger().info('  ✓ Sending cached trajectory (instant replay!)')
                            self.send_cached_trajectory_data(model_id)
                        else:
                            self.get_logger().warn(f'  ⚠ No cached trajectory for model #{model_id}')
                            # Fall back to sending empty or skip
                            self.zmq_socket.send_pyobj({'type': 'trajectory_data_unavailable', 'model_id': model_id})
                            if self.zmq_socket.poll(timeout=10000):
                                self.zmq_socket.recv_pyobj()  # Consume ack

                        # Request next model after trajectory refinement completes
                        self.get_logger().info('Waiting 3s before next episode...')
                        time.sleep(3.0)
                        self.request_new_model()
                        return

                else:
                    self.get_logger().warn(f'Unexpected ack: {ack}')
            else:
                self.get_logger().error('Timeout waiting for acknowledgment')

        except Exception as e:
            self.get_logger().error(f'Failed to send episode results: {e}')

        # Request next model
        self.get_logger().info('Waiting 3s before next episode...')
        time.sleep(3.0)
        self.request_new_model()

    def cache_trajectory_data(self) -> None:
        """Cache trajectory data for current model."""
        if len(self._trajectory_rgb) == 0:
            return  # Nothing to cache

        # Convert to numpy arrays
        trajectory_data = {
            'rgb': np.array(self._trajectory_rgb, dtype=np.uint8),
            'depth': np.array(self._trajectory_depth, dtype=np.float32),
            'proprio': np.array(self._trajectory_proprio, dtype=np.float32),
            'actions': np.array(self._trajectory_actions, dtype=np.float32),
        }

        # Cache for this model
        self._trajectory_cache[self._current_model_id] = trajectory_data

        # Manage cache size (FIFO)
        if len(self._trajectory_cache) > self._max_cache_size:
            # Remove oldest entry
            oldest_model_id = min(self._trajectory_cache.keys())
            del self._trajectory_cache[oldest_model_id]
            self.get_logger().debug(f'  Evicted trajectory for model #{oldest_model_id} from cache')

    def send_cached_trajectory_data(self, model_id: int) -> None:
        """Send cached trajectory data to V620 server.

        Args:
            model_id: Model ID to send trajectory for
        """
        if model_id not in self._trajectory_cache:
            self.get_logger().error(f'Model #{model_id} not in cache!')
            return

        try:
            import time
            import zstandard as zstd
            start_time = time.time()

            trajectory_data = self._trajectory_cache[model_id]

            # Compress with Zstandard (level 10 - good compression/speed balance)
            cctx = zstd.ZstdCompressor(level=10)
            rgb_compressed = cctx.compress(trajectory_data['rgb'].tobytes())
            depth_compressed = cctx.compress(trajectory_data['depth'].tobytes())

            original_mb = (trajectory_data['rgb'].nbytes + trajectory_data['depth'].nbytes) / 1024 / 1024
            compressed_mb = (len(rgb_compressed) + len(depth_compressed)) / 1024 / 1024

            self.get_logger().info(
                f'  Compressed (Zstd-10): {original_mb:.1f} MB → {compressed_mb:.1f} MB '
                f'({original_mb/compressed_mb:.1f}x)'
            )

            # Prepare message
            trajectory_message = {
                'type': 'trajectory_data',
                'model_id': model_id,
                'compressed': True,
                'compression': 'zstd',  # Specify compression type
                'trajectory': {
                    'rgb': rgb_compressed,
                    'rgb_shape': trajectory_data['rgb'].shape,
                    'depth': depth_compressed,
                    'depth_shape': trajectory_data['depth'].shape,
                    'proprio': trajectory_data['proprio'],
                    'actions': trajectory_data['actions'],
                }
            }

            # Send
            self.zmq_socket.send_pyobj(trajectory_message)
            send_time = time.time() - start_time
            self.get_logger().info(f'  Sent in {send_time:.1f}s, waiting for ack...')

            # Wait for acknowledgment (infinite timeout - wait for refinement to complete)
            if self.zmq_socket.poll(timeout=-1):  # Infinite timeout
                ack = self.zmq_socket.recv_pyobj()
                total_time = time.time() - start_time
                if ack.get('type') == 'ack':
                    if ack.get('refined'):
                        self.get_logger().info(f'✓ Model refined by V620 ({total_time:.1f}s total)')
                    else:
                        self.get_logger().info(f'✓ Trajectory acknowledged ({total_time:.1f}s total)')
                else:
                    self.get_logger().warn(f'Unexpected ack: {ack}')
            else:
                self.get_logger().error('Timeout waiting for trajectory ack (infinite wait)')

        except Exception as e:
            self.get_logger().error(f'Failed to send cached trajectory: {e}')
            import traceback
            traceback.print_exc()

    def send_trajectory_data(self) -> None:
        """Send collected trajectory data to V620 server."""
        try:
            import time
            import zstandard as zstd
            start_time = time.time()

            # Convert lists to numpy arrays
            rgb_array = np.array(self._trajectory_rgb, dtype=np.uint8)  # (N, H, W, 3)
            depth_array = np.array(self._trajectory_depth, dtype=np.float32)  # (N, H, W)
            proprio_array = np.array(self._trajectory_proprio, dtype=np.float32)  # (N, 6)
            actions_array = np.array(self._trajectory_actions, dtype=np.float32)  # (N, 2)

            self.get_logger().info(
                f'→ Preparing trajectory data: {len(actions_array)} samples'
            )

            # Compress large arrays with Zstandard (level 10 - good compression/speed balance)
            cctx = zstd.ZstdCompressor(level=10)
            rgb_compressed = cctx.compress(rgb_array.tobytes())
            depth_compressed = cctx.compress(depth_array.tobytes())

            # Calculate sizes
            original_mb = (rgb_array.nbytes + depth_array.nbytes) / 1024 / 1024
            compressed_mb = (len(rgb_compressed) + len(depth_compressed)) / 1024 / 1024
            ratio = original_mb / compressed_mb if compressed_mb > 0 else 1.0

            self.get_logger().info(
                f'  Compressed (Zstd-10): {original_mb:.1f} MB → {compressed_mb:.1f} MB '
                f'({ratio:.1f}x ratio)'
            )

            # Prepare message with compressed data and metadata
            trajectory_message = {
                'type': 'trajectory_data',
                'model_id': self._current_model_id,
                'compressed': True,
                'compression': 'zstd',  # Specify compression type
                'trajectory': {
                    'rgb': rgb_compressed,
                    'rgb_shape': rgb_array.shape,
                    'depth': depth_compressed,
                    'depth_shape': depth_array.shape,
                    'proprio': proprio_array,  # Small, no need to compress
                    'actions': actions_array,  # Small, no need to compress
                }
            }

            # Send to server
            self.zmq_socket.send_pyobj(trajectory_message)
            send_time = time.time() - start_time
            self.get_logger().info(f'  Sent in {send_time:.1f}s, waiting for ack...')

            # Wait for acknowledgment (server is doing gradient descent, this takes time)
            if self.zmq_socket.poll(timeout=300000):  # 5 minute timeout for gradient refinement
                ack = self.zmq_socket.recv_pyobj()
                total_time = time.time() - start_time
                if ack.get('type') == 'ack':
                    if ack.get('refined'):
                        self.get_logger().info(f'✓ Model refined by V620 ({total_time:.1f}s total)')
                    else:
                        self.get_logger().info(f'✓ Trajectory data acknowledged ({total_time:.1f}s total)')
                else:
                    self.get_logger().warn(f'Unexpected ack: {ack}')
            else:
                self.get_logger().error('Timeout waiting for trajectory ack (5 min)')

        except Exception as e:
            self.get_logger().error(f'Failed to send trajectory data: {e}')
            import traceback
            traceback.print_exc()

    def inference_callback(self) -> None:
        """Run inference and publish commands."""
        # Check if episode should end
        if self._episode_running:
            elapsed = time.time() - self._episode_start_time
            if elapsed >= self.episode_duration:
                self.end_episode()
                return

        # Check for required data
        if not self._episode_running:
            return

        if self._latest_rgb is None or self._latest_depth is None or self._latest_odom is None:
            return

        try:
            # Use RKNN model if available, otherwise random actions
            if self._rknn_runtime is not None:
                # Prepare inputs for RKNN
                rgb = self._latest_rgb.astype(np.uint8)
                rgb = np.transpose(rgb, (2, 0, 1))  # (3, H, W)
                rgb = np.expand_dims(rgb, axis=0)  # (1, 3, H, W)

                depth = self._latest_depth.astype(np.float32) / 5.0  # Normalize
                depth = np.expand_dims(depth, axis=0)  # (1, H, W)
                depth = np.expand_dims(depth, axis=0)  # (1, 1, H, W)

                # Construct new proprioception vector:
                # [wheel_vel_left, wheel_vel_right, imu_accel_x, imu_accel_y, imu_ang_vel_z, mag_x, mag_y, mag_z, min_forward_dist]
                
                # Defaults if sensors not ready
                w_left, w_right = 0.0, 0.0
                if self._latest_wheel_vels:
                    w_left, w_right = self._latest_wheel_vels
                
                accel_x, accel_y, gyro_z = 0.0, 0.0, 0.0
                if self._latest_imu:
                    accel_x, accel_y, gyro_z = self._latest_imu

                mag_x, mag_y, mag_z = 0.0, 0.0, 0.0
                if self._latest_mag:
                    mag_x, mag_y, mag_z = self._latest_mag
                
                proprio = np.array([[
                    w_left, w_right, accel_x, accel_y, gyro_z, mag_x, mag_y, mag_z, self._min_forward_dist
                ]], dtype=np.float32)  # (1, 9)

                # Prepare LSTM hidden state inputs
                # If None, initialize to zeros (will be done by RKNN internally or pass zeros)
                if self._lstm_hidden_h is None:
                    # Initialize hidden states to zeros
                    lstm_h = np.zeros((1, 1, 128), dtype=np.float32)
                    lstm_c = np.zeros((1, 1, 128), dtype=np.float32)
                else:
                    lstm_h = self._lstm_hidden_h
                    lstm_c = self._lstm_hidden_c

                # Run RKNN inference with LSTM hidden states
                # RKNN model expects: [rgb, depth, proprio, lstm_h, lstm_c]
                # Returns: [action, new_lstm_h, new_lstm_c]
                outputs = self._rknn_runtime.inference(inputs=[rgb, depth, proprio, lstm_h, lstm_c])

                # Parse outputs
                action = outputs[0][0]  # (2,) action
                new_lstm_h = outputs[1]  # (1, 1, 128) new hidden state
                new_lstm_c = outputs[2]  # (1, 1, 128) new cell state
                
                # Store updated hidden states for next timestep
                self._lstm_hidden_h = new_lstm_h
                self._lstm_hidden_c = new_lstm_c
                
                linear_cmd = float(np.clip(action[0], -1.0, 1.0) * self.max_linear)
                angular_cmd = float(np.clip(action[1], -1.0, 1.0) * self.max_angular)

            else:
                # Fallback to random policy
                linear_cmd = np.random.uniform(-0.5, 1.0) * self.max_linear
                angular_cmd = np.random.uniform(-1.0, 1.0) * self.max_angular

            # Safety: stop if too close to obstacle
            if self._min_forward_dist < self.collision_dist * 2:
                linear_cmd = 0.0

            # Publish command
            cmd = Twist()
            cmd.linear.x = float(linear_cmd)
            cmd.angular.z = float(angular_cmd)
            self.cmd_pub.publish(cmd)

            # Record metrics
            current_speed = abs(self._latest_odom[2])  # Linear velocity
            self._speed_samples.append(current_speed)
            self._clearance_samples.append(self._min_forward_dist)
            
            # NEW: Record heading for turn efficiency
            self._heading_samples.append(self._latest_odom[4])  # yaw/heading
            
            # NEW: Detect stationary rotation (pivot turns)
            if abs(self._latest_odom[2]) < 0.02 and abs(angular_cmd) > 0.1:  # Low linear, high angular command
                self._stationary_rotation_time += 1.0 / self.inference_rate  # Add timestep (now 1/30s)
            
            # NEW: Simple track slippage detection
            commanded_speed = abs(linear_cmd)
            actual_speed = abs(self._latest_odom[2])
            if commanded_speed > 0.05 and actual_speed < commanded_speed * 0.3:  # Commanded but not moving much
                self._track_slip_detected = True
            
            # NEW: Oscillation detection
            self._recent_linear_actions.append(linear_cmd)
            if len(self._recent_linear_actions) > 20:
                self._recent_linear_actions.pop(0)
                # Count sign flips
                sign_flips = 0
                for i in range(1, len(self._recent_linear_actions)):
                    if self._recent_linear_actions[i] * self._recent_linear_actions[i-1] < -0.01: # Significant sign flip
                        sign_flips += 1
                if sign_flips > 5:
                    self._oscillation_count += 1

            # Store last commands for next proprioception
            self._last_linear_cmd = linear_cmd
            self._last_angular_cmd = angular_cmd

            # Collect trajectory data if requested (subsample to avoid too much data)
            # Max 1800 samples (~60s at 30Hz)
            if self._collect_trajectory and len(self._trajectory_rgb) < 1800:
                # Store observation data
                self._trajectory_rgb.append(self._latest_rgb.copy())
                self._trajectory_depth.append(self._latest_depth.copy())

                # Store proprioception
                # [wheel_vel_left, wheel_vel_right, imu_accel_x, imu_accel_y, imu_ang_vel_z, mag_x, mag_y, mag_z, min_forward_dist]
                w_left, w_right = 0.0, 0.0
                if self._latest_wheel_vels:
                    w_left, w_right = self._latest_wheel_vels
                
                accel_x, accel_y, gyro_z = 0.0, 0.0, 0.0
                if self._latest_imu:
                    accel_x, accel_y, gyro_z = self._latest_imu

                mag_x, mag_y, mag_z = 0.0, 0.0, 0.0
                if self._latest_mag:
                    mag_x, mag_y, mag_z = self._latest_mag

                proprio_vec = [w_left, w_right, accel_x, accel_y, gyro_z, mag_x, mag_y, mag_z, self._min_forward_dist]
                self._trajectory_proprio.append(proprio_vec)

                # Store action taken
                self._trajectory_actions.append([linear_cmd / self.max_linear, angular_cmd / self.max_angular])
                
                # NEW: Store heading
                self._trajectory_headings.append(self._latest_odom[4])

        except Exception as e:
            self.get_logger().error(f'Inference failed: {e}')
            import traceback
            traceback.print_exc()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MAPElitesEpisodeRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
