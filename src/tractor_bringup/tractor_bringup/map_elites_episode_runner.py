#!/usr/bin/env python3
"""MAP-Elites autonomous episode runner for rover.

Runs autonomous episodes and reports results back to V620 server.
No teleoperation required - rover explores autonomously!
"""

import os
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

from sensor_msgs.msg import Image
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
        self.declare_parameter('inference_rate_hz', 10.0)
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
        self._min_forward_dist = 10.0
        self._episode_running = False
        self._rknn_runtime = None
        self._current_model_state = None

        # Episode metrics
        self._episode_start_time = 0.0
        self._episode_start_pos = None
        self._total_distance = 0.0
        self._collision_count = 0
        self._speed_samples = []
        self._clearance_samples = []

        self.bridge = CvBridge()

        # ZeroMQ REQ socket for bidirectional communication with V620
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.REQ)
        self.zmq_socket.connect(self.server_addr)
        self.zmq_socket.setsockopt(zmq.LINGER, 0)

        # Current model being evaluated
        self._current_model_id = None

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

    def convert_pytorch_to_rknn(self, model_state: dict, model_id: int) -> Optional[str]:
        """Convert PyTorch state_dict to RKNN model.

        Args:
            model_state: PyTorch state_dict
            model_id: Model ID for naming

        Returns:
            Path to RKNN model file, or None if conversion failed
        """
        try:
            import torch

            # Save PyTorch model
            pt_path = self.temp_dir / f'model_{model_id}.pt'
            torch.save(model_state, pt_path)
            self.get_logger().info(f'  Saved PyTorch model: {pt_path}')

            # Export to ONNX
            onnx_path = self.temp_dir / f'model_{model_id}.onnx'

            self.get_logger().info('  Exporting to ONNX...')
            result = subprocess.run([
                'python3', '-c',
                f"""
import torch
import torch.nn as nn
import sys
sys.path.append('/home/ubuntu/ros2-rover/remote_training_server')
from v620_ppo_trainer import RGBDEncoder, PolicyHead

class ActorNetwork(nn.Module):
    def __init__(self, proprio_dim=6):
        super().__init__()
        self.encoder = RGBDEncoder()
        self.policy_head = PolicyHead(self.encoder.output_dim, proprio_dim)

    def forward(self, rgb, depth, proprio):
        features = self.encoder(rgb, depth)
        action = self.policy_head(features, proprio)
        return action

# Load model
model = ActorNetwork()
model.load_state_dict(torch.load('{pt_path}'))
model.eval()

# Dummy inputs
rgb = torch.randn(1, 3, 240, 424)
depth = torch.randn(1, 1, 240, 424)
proprio = torch.randn(1, 6)

# Export
torch.onnx.export(
    model,
    (rgb, depth, proprio),
    '{onnx_path}',
    input_names=['rgb', 'depth', 'proprio'],
    output_names=['action'],
    opset_version=11,
    do_constant_folding=True
)
print('ONNX export complete')
"""
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
        except Exception as e:
            self.get_logger().warn(f'Depth conversion failed: {e}')

    def odom_callback(self, msg: Odometry) -> None:
        """Store latest odometry."""
        self._latest_odom = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.twist.twist.linear.x,
            msg.twist.twist.angular.z
        )

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
            request = {'type': 'request_model'}
            self.zmq_socket.send_pyobj(request)

            self.get_logger().info('Requesting new model from V620...')

            # Wait for response (with timeout)
            if self.zmq_socket.poll(timeout=30000):  # 30 second timeout
                response = self.zmq_socket.recv_pyobj()

                if response['type'] == 'model':
                    self._current_model_id = response['model_id']
                    self._current_model_state = response['model_state']

                    self.get_logger().info(
                        f'✓ Received model #{self._current_model_id} '
                        f'({response["generation_type"]})'
                    )

                    # Convert PyTorch → ONNX → RKNN
                    self.get_logger().info('Converting model to RKNN...')
                    rknn_path = self.convert_pytorch_to_rknn(
                        self._current_model_state,
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

        self._episode_start_pos = (self._latest_odom[0], self._latest_odom[1])
        self._total_distance = 0.0
        self._collision_count = 0
        self._speed_samples = []
        self._clearance_samples = []

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

        # Package episode results
        episode_result = {
            'type': 'episode_result',
            'model_id': self._current_model_id,
            'total_distance': float(self._total_distance),
            'collision_count': int(self._collision_count),
            'avg_speed': float(avg_speed),
            'avg_clearance': float(avg_clearance),
            'duration': float(duration),
        }

        # Send results to server and wait for acknowledgment
        try:
            self.zmq_socket.send_pyobj(episode_result)

            self.get_logger().info(
                f'✓ Episode #{self._current_model_id} complete: '
                f'dist={self._total_distance:.2f}m, '
                f'collisions={self._collision_count}, '
                f'avg_speed={avg_speed:.3f}m/s, '
                f'avg_clearance={avg_clearance:.2f}m'
            )

            # Wait for acknowledgment
            if self.zmq_socket.poll(timeout=10000):  # 10 second timeout
                ack = self.zmq_socket.recv_pyobj()
                if ack.get('type') == 'ack':
                    self.get_logger().info('✓ Results acknowledged by V620')
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

                lin_vel, ang_vel = self._latest_odom[2], self._latest_odom[3]
                proprio = np.array([[
                    lin_vel, ang_vel, 0.0, 0.0, 0.0, self._min_forward_dist
                ]], dtype=np.float32)  # (1, 6)

                # Run RKNN inference
                outputs = self._rknn_runtime.inference(inputs=[rgb, depth, proprio])

                # Parse output
                action = outputs[0][0]  # (2,)
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
