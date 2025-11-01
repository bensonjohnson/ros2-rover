#!/usr/bin/env python3
"""MAP-Elites autonomous episode runner for rover.

Runs autonomous episodes and reports results back to V620 server.
No teleoperation required - rover explores autonomously!
"""

import time
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

        # ZeroMQ - send episode results to V620
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_socket.connect(self.server_addr)
        self.zmq_socket.setsockopt(zmq.SNDHWM, 100)
        self.zmq_socket.setsockopt(zmq.LINGER, 0)

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

        self.get_logger().info('MAP-Elites episode runner initialized')
        self.get_logger().info(f'Server: {self.server_addr}')
        self.get_logger().info(f'Episode duration: {self.episode_duration}s')

        # Request first model from server
        self.request_new_model()

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
        """Request new model from V620 server (placeholder).

        In full implementation, server would send model weights.
        For now, we'll use a random model.
        """
        # TODO: Implement model request protocol
        # For now, initialize with random weights
        if self._rknn_runtime is None and HAS_RKNN and self.use_npu:
            # Load a model (would come from server)
            pass

        # Start new episode
        self.start_episode()

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

        # Package episode data
        episode_data = {
            'model_state': self._current_model_state,  # Would be actual model state
            'total_distance': float(self._total_distance),
            'collision_count': int(self._collision_count),
            'avg_speed': float(avg_speed),
            'avg_clearance': float(avg_clearance),
            'duration': float(duration),
        }

        # Send to server
        try:
            self.zmq_socket.send_pyobj(episode_data, flags=zmq.NOBLOCK)
            self.get_logger().info(
                f'✓ Episode complete: dist={self._total_distance:.2f}m, '
                f'collisions={self._collision_count}, '
                f'avg_speed={avg_speed:.3f}m/s, '
                f'avg_clearance={avg_clearance:.2f}m'
            )
        except zmq.error.Again:
            self.get_logger().warn('Failed to send episode data (buffer full)')

        # Request next model
        self.get_logger().info('Waiting 5s before next episode...')
        time.sleep(5.0)
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
            # For now, use simple random policy
            # In full implementation, this would use RKNN model
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
