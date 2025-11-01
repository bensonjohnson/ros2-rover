#!/usr/bin/env python3
"""Remote Training Data Collector for V620 ROCm Training.

This node collects RGB-D images, proprioceptive data (IMU + encoders), actions,
and rewards from the rover and streams them to a remote training server via ZeroMQ.

Data format sent to V620:
- RGB image: (240, 424, 3) uint8
- Depth image: (240, 424) float32 (meters)
- Proprioception: [linear_vel, angular_vel, roll, pitch, accel_mag, min_forward_dist] (6 floats)
- Action: [linear_cmd, angular_cmd] (2 floats)
- Reward: float32
- Done: bool
"""

import time
import os
from typing import Optional, Tuple
import json

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32

from cv_bridge import CvBridge
import cv2

try:
    import zmq
    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False


class RemoteTrainingCollector(Node):
    """Collects rover sensor data and streams to remote training server."""

    def __init__(self) -> None:
        super().__init__('remote_training_collector')

        # Parameters
        self.declare_parameter('server_address', 'tcp://192.168.1.100:5555')  # V620 IP
        self.declare_parameter('rgb_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('imu_topic', '/lsm9ds1_imu_publisher/imu/data')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('cmd_vel_topic', 'cmd_vel_ai')
        self.declare_parameter('min_distance_topic', '/min_forward_distance')
        self.declare_parameter('collection_rate_hz', 10.0)
        self.declare_parameter('depth_scale', 0.001)  # For uint16 → meters
        self.declare_parameter('depth_clip_m', 6.0)
        self.declare_parameter('enable_compression', True)
        self.declare_parameter('jpeg_quality', 85)
        self.declare_parameter('save_calibration_samples', True)
        self.declare_parameter('calibration_sample_interval', 10.0)  # seconds
        self.declare_parameter('calibration_sample_count', 100)
        self.declare_parameter('calibration_data_dir', 'calibration_data')

        self.server_addr = str(self.get_parameter('server_address').value)
        self.rgb_topic = str(self.get_parameter('rgb_topic').value)
        self.depth_topic = str(self.get_parameter('depth_topic').value)
        self.imu_topic = str(self.get_parameter('imu_topic').value)
        self.odom_topic = str(self.get_parameter('odom_topic').value)
        self.cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        self.min_distance_topic = str(self.get_parameter('min_distance_topic').value)
        self.collection_rate = float(self.get_parameter('collection_rate_hz').value)
        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.depth_clip = float(self.get_parameter('depth_clip_m').value)
        self.enable_compression = bool(self.get_parameter('enable_compression').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        self.save_calibration = bool(self.get_parameter('save_calibration_samples').value)
        self.calibration_interval = float(self.get_parameter('calibration_sample_interval').value)
        self.calibration_count = int(self.get_parameter('calibration_sample_count').value)
        self.calibration_dir = str(self.get_parameter('calibration_data_dir').value)

        # Create calibration directory
        if self.save_calibration:
            os.makedirs(self.calibration_dir, exist_ok=True)
            self.get_logger().info(f'Saving calibration samples to: {self.calibration_dir}')

        if not HAS_ZMQ:
            self.get_logger().error('ZeroMQ not installed! Install: pip install pyzmq')
            raise RuntimeError('ZeroMQ required for remote training')

        # Initialize ZeroMQ
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUSH)
        self.zmq_socket.setsockopt(zmq.SNDHWM, 10)  # High water mark - drop old data if buffer full
        self.zmq_socket.connect(self.server_addr)
        self.get_logger().info(f'Connected to training server: {self.server_addr}')

        self.bridge = CvBridge()

        # State caches
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_imu: Optional[Tuple[float, float, float]] = None  # roll, pitch, accel_mag
        self._latest_vel: Optional[Tuple[float, float]] = None  # linear, angular
        self._latest_action: Optional[Tuple[float, float]] = None  # cmd linear, angular
        self._min_forward_dist: float = 10.0
        self._prev_position: Optional[Tuple[float, float]] = None
        self._episode_step = 0
        self._total_samples_sent = 0
        self._last_calibration_save = time.time()
        self._calibration_samples_saved = 0

        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image, self.rgb_topic, self.rgb_callback, qos_profile_sensor_data
        )
        self.depth_sub = self.create_subscription(
            Image, self.depth_topic, self.depth_callback, qos_profile_sensor_data
        )
        self.imu_sub = self.create_subscription(
            Imu, self.imu_topic, self.imu_callback, qos_profile_sensor_data
        )
        self.odom_sub = self.create_subscription(
            Odometry, self.odom_topic, self.odom_callback, 20
        )
        self.cmd_vel_sub = self.create_subscription(
            Twist, self.cmd_vel_topic, self.cmd_vel_callback, 10
        )
        self.min_dist_sub = self.create_subscription(
            Float32, self.min_distance_topic, self.min_dist_callback, 10
        )

        # Collection timer
        timer_period = 1.0 / max(self.collection_rate, 0.1)
        self.timer = self.create_timer(timer_period, self.collect_and_send)

        self.get_logger().info('Remote training collector initialized')

    def rgb_callback(self, msg: Image) -> None:
        try:
            # Convert to numpy (BGR format from cv_bridge)
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Convert BGR to RGB
            self._latest_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception as exc:
            self.get_logger().warn(f'RGB conversion failed: {exc}')

    def depth_callback(self, msg: Image) -> None:
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) * self.depth_scale
            elif depth.dtype != np.float32:
                depth = depth.astype(np.float32)
            depth = np.nan_to_num(depth, nan=self.depth_clip, posinf=self.depth_clip, neginf=0.0)
            depth = np.clip(depth, 0.0, self.depth_clip)
            self._latest_depth = depth
        except Exception as exc:
            self.get_logger().warn(f'Depth conversion failed: {exc}')

    def imu_callback(self, msg: Imu) -> None:
        try:
            roll, pitch = self._roll_pitch_from_quaternion(
                msg.orientation.x, msg.orientation.y,
                msg.orientation.z, msg.orientation.w
            )
            ax, ay, az = msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
            accel_mag = float(np.sqrt(ax * ax + ay * ay + az * az))
            self._latest_imu = (roll, pitch, accel_mag)
        except Exception as exc:
            self.get_logger().debug(f'IMU parsing failed: {exc}')

    def odom_callback(self, msg: Odometry) -> None:
        try:
            lin = float(msg.twist.twist.linear.x)
            ang = float(msg.twist.twist.angular.z)
            self._latest_vel = (lin, ang)
            # Track position for reward computation
            pos_x = msg.pose.pose.position.x
            pos_y = msg.pose.pose.position.y
            if self._prev_position is None:
                self._prev_position = (pos_x, pos_y)
        except Exception as exc:
            self.get_logger().debug(f'Odom parsing failed: {exc}')

    def cmd_vel_callback(self, msg: Twist) -> None:
        lin = float(msg.linear.x)
        ang = float(msg.angular.z)
        self._latest_action = (lin, ang)

    def min_dist_callback(self, msg: Float32) -> None:
        self._min_forward_dist = float(msg.data)

    def collect_and_send(self) -> None:
        """Collect current state and send to training server."""
        # Check if we have all required data
        if (self._latest_rgb is None or self._latest_depth is None or
            self._latest_imu is None or self._latest_vel is None or
            self._latest_action is None):
            return

        # Compute reward based on forward progress
        reward = self._compute_reward()

        # Check if episode should terminate (collision or timeout)
        done = self._check_done()

        # Build proprioception vector
        lin_vel, ang_vel = self._latest_vel
        roll, pitch, accel_mag = self._latest_imu
        proprio = np.array([
            lin_vel, ang_vel, roll, pitch, accel_mag, self._min_forward_dist
        ], dtype=np.float32)

        # Build action vector
        action = np.array(list(self._latest_action), dtype=np.float32)

        # Prepare data packet
        try:
            packet = self._build_packet(
                self._latest_rgb,
                self._latest_depth,
                proprio,
                action,
                reward,
                done
            )

            # Send via ZeroMQ (non-blocking)
            self.zmq_socket.send(packet, zmq.NOBLOCK)
            self._total_samples_sent += 1
            self._episode_step += 1

            if self._total_samples_sent % 100 == 0:
                self.get_logger().info(
                    f'Sent {self._total_samples_sent} samples | Episode step: {self._episode_step}'
                )

            if done:
                self.get_logger().info(f'Episode finished at step {self._episode_step}')
                self._episode_step = 0

            # Save calibration samples periodically
            if self.save_calibration and self._calibration_samples_saved < self.calibration_count:
                now = time.time()
                if (now - self._last_calibration_save) >= self.calibration_interval:
                    self._save_calibration_sample(self._latest_rgb, self._latest_depth, proprio)
                    self._last_calibration_save = now

        except zmq.error.Again:
            self.get_logger().warn_throttle(5.0, 'Training server buffer full, dropping sample')
        except Exception as exc:
            self.get_logger().error(f'Failed to send packet: {exc}')

    def _build_packet(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        proprio: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool
    ) -> bytes:
        """Build binary packet for network transmission."""
        # Metadata
        metadata = {
            'timestamp': time.time(),
            'episode_step': self._episode_step,
            'reward': float(reward),
            'done': bool(done),
            'rgb_shape': list(rgb.shape),
            'depth_shape': list(depth.shape),
            'proprio_shape': list(proprio.shape),
            'action_shape': list(action.shape),
        }

        if self.enable_compression:
            # Compress RGB with JPEG
            _, rgb_encoded = cv2.imencode(
                '.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            )
            rgb_bytes = rgb_encoded.tobytes()
            metadata['rgb_compressed'] = True
        else:
            rgb_bytes = rgb.tobytes()
            metadata['rgb_compressed'] = False

        # Depth as float32 (no compression for now - could use PNG16 for lossless)
        depth_bytes = depth.astype(np.float32).tobytes()

        # Proprio and action as float32
        proprio_bytes = proprio.tobytes()
        action_bytes = action.tobytes()

        # Serialize metadata as JSON
        metadata_json = json.dumps(metadata).encode('utf-8')
        metadata_len = len(metadata_json)

        # Build packet: [metadata_len(4B)][metadata][rgb][depth][proprio][action]
        packet = (
            metadata_len.to_bytes(4, byteorder='little') +
            metadata_json +
            rgb_bytes +
            depth_bytes +
            proprio_bytes +
            action_bytes
        )

        return packet

    def _compute_reward(self) -> float:
        """Compute reward based on forward movement and safety."""
        reward = 0.0

        # Reward forward linear velocity
        if self._latest_vel is not None:
            lin_vel, ang_vel = self._latest_vel
            reward += lin_vel * 2.0  # Encourage forward movement
            reward -= abs(ang_vel) * 0.5  # Penalize spinning

        # Penalize being close to obstacles
        if self._min_forward_dist < 0.3:
            reward -= 5.0  # Heavy penalty for being too close
        elif self._min_forward_dist < 0.5:
            reward -= 1.0  # Light penalty

        # Small time penalty to encourage efficiency
        reward -= 0.01

        return float(reward)

    def _check_done(self) -> bool:
        """Check if episode should terminate."""
        # Collision detection
        if self._min_forward_dist < 0.15:
            self.get_logger().warn('Episode terminated: Collision detected')
            return True

        # Episode timeout (300 steps at 10Hz = 30 seconds)
        if self._episode_step >= 300:
            self.get_logger().info('Episode terminated: Timeout')
            return True

        return False

    def _save_calibration_sample(self, rgb: np.ndarray, depth: np.ndarray, proprio: np.ndarray) -> None:
        """Save calibration sample for RKNN quantization."""
        try:
            sample_path = os.path.join(
                self.calibration_dir,
                f'calibration_{self._calibration_samples_saved:04d}.npz'
            )

            # Save RGB, depth, and proprioception
            np.savez_compressed(
                sample_path,
                rgb=rgb,
                depth=depth,
                proprio=proprio
            )

            self._calibration_samples_saved += 1

            if self._calibration_samples_saved % 10 == 0:
                self.get_logger().info(
                    f'Saved {self._calibration_samples_saved}/{self.calibration_count} calibration samples'
                )

            if self._calibration_samples_saved >= self.calibration_count:
                self.get_logger().info(
                    f'✓ Calibration dataset complete: {self._calibration_samples_saved} samples saved'
                )

        except Exception as exc:
            self.get_logger().error(f'Failed to save calibration sample: {exc}')

    @staticmethod
    def _roll_pitch_from_quaternion(x: float, y: float, z: float, w: float) -> Tuple[float, float]:
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch = np.arcsin(t2)
        return float(roll), float(pitch)

    def __del__(self):
        """Cleanup ZeroMQ resources."""
        if hasattr(self, 'zmq_socket'):
            self.zmq_socket.close()
        if hasattr(self, 'zmq_context'):
            self.zmq_context.term()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RemoteTrainingCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
