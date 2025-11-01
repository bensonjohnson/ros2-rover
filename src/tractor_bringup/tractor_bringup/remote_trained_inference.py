#!/usr/bin/env python3
"""NPU Inference node for remotely-trained RGB-D models.

This node runs inference using models trained on the V620 server,
converted to RKNN format, and deployed to the rover's RK3588 NPU.

Input: RGB image + Depth image + Proprioception
Output: Velocity commands (linear, angular)
"""

import os
import time
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from std_srvs.srv import Trigger

from cv_bridge import CvBridge
import cv2

try:
    from rknnlite.api import RKNNLite
    HAS_RKNN = True
except ImportError:
    HAS_RKNN = False
    print("WARNING: rknnlite not available, will use CPU fallback (very slow)")


class RemoteTrainedInference(Node):
    """Runs inference with remotely-trained models on RK3588 NPU."""

    def __init__(self) -> None:
        super().__init__('remote_trained_inference')

        # Parameters
        self.declare_parameter('model_path', '/home/benson/Documents/ros2-rover/models/remote_trained.rknn')
        self.declare_parameter('rgb_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('imu_topic', '/lsm9ds1_imu_publisher/imu/data')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('min_distance_topic', '/min_forward_distance')
        self.declare_parameter('cmd_vel_topic', 'cmd_vel_ai')
        self.declare_parameter('max_linear_speed', 0.18)
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('inference_rate_hz', 10.0)
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('depth_clip_m', 6.0)
        self.declare_parameter('use_npu', True)
        self.declare_parameter('npu_core_mask', 0)  # 0=auto, 1=core0, 2=core1, 3=core2

        self.model_path = str(self.get_parameter('model_path').value)
        self.rgb_topic = str(self.get_parameter('rgb_topic').value)
        self.depth_topic = str(self.get_parameter('depth_topic').value)
        self.imu_topic = str(self.get_parameter('imu_topic').value)
        self.odom_topic = str(self.get_parameter('odom_topic').value)
        self.min_distance_topic = str(self.get_parameter('min_distance_topic').value)
        self.cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        self.max_linear = float(self.get_parameter('max_linear_speed').value)
        self.max_angular = float(self.get_parameter('max_angular_speed').value)
        self.inference_rate = float(self.get_parameter('inference_rate_hz').value)
        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.depth_clip = float(self.get_parameter('depth_clip_m').value)
        self.use_npu = bool(self.get_parameter('use_npu').value)
        self.npu_core_mask = int(self.get_parameter('npu_core_mask').value)

        self.bridge = CvBridge()

        # State
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_imu: Optional[Tuple[float, float, float]] = None
        self._latest_vel: Optional[Tuple[float, float]] = None
        self._min_forward_dist: float = 10.0
        self._rknn_runtime: Optional[RKNNLite] = None
        self._inference_count = 0
        self._last_inference_time = time.time()

        # Load model
        self._load_model()

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

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
        self.min_dist_sub = self.create_subscription(
            Float32, self.min_distance_topic, self.min_dist_callback, 10
        )

        # Service for reloading model
        self.reload_srv = self.create_service(
            Trigger, 'reload_remote_model', self.reload_model_callback
        )

        # Inference timer
        timer_period = 1.0 / max(self.inference_rate, 0.1)
        self.timer = self.create_timer(timer_period, self.run_inference)

        self.get_logger().info(f'Remote trained inference node ready (NPU: {self.use_npu})')

    def _load_model(self) -> bool:
        """Load RKNN model onto NPU."""
        if not os.path.exists(self.model_path):
            self.get_logger().error(f'Model not found: {self.model_path}')
            return False

        if not HAS_RKNN or not self.use_npu:
            self.get_logger().warn('RKNN not available or disabled, using CPU fallback')
            # TODO: Implement ONNX CPU fallback
            return False

        try:
            self._rknn_runtime = RKNNLite(verbose=False)

            # Load RKNN model
            ret = self._rknn_runtime.load_rknn(self.model_path)
            if ret != 0:
                self.get_logger().error(f'Failed to load RKNN model: {ret}')
                return False

            # Initialize runtime
            if self.npu_core_mask == 0:
                core_mask = RKNNLite.NPU_CORE_AUTO
            elif self.npu_core_mask == 1:
                core_mask = RKNNLite.NPU_CORE_0
            elif self.npu_core_mask == 2:
                core_mask = RKNNLite.NPU_CORE_1
            else:
                core_mask = RKNNLite.NPU_CORE_2

            ret = self._rknn_runtime.init_runtime(core_mask=core_mask)
            if ret != 0:
                self.get_logger().error(f'Failed to init RKNN runtime: {ret}')
                return False

            self.get_logger().info(f'✓ RKNN model loaded: {self.model_path}')
            return True

        except Exception as exc:
            self.get_logger().error(f'Exception loading RKNN: {exc}')
            return False

    def reload_model_callback(self, request, response):
        """Service callback to reload model."""
        self.get_logger().info('Reloading model...')

        # Release old model
        if self._rknn_runtime is not None:
            self._rknn_runtime.release()
            self._rknn_runtime = None

        # Load new model
        success = self._load_model()

        response.success = success
        response.message = 'Model reloaded successfully' if success else 'Failed to reload model'
        return response

    def rgb_callback(self, msg: Image) -> None:
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
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
            self._latest_depth = np.clip(depth, 0.0, self.depth_clip)
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
        except Exception as exc:
            self.get_logger().debug(f'Odom parsing failed: {exc}')

    def min_dist_callback(self, msg: Float32) -> None:
        self._min_forward_dist = float(msg.data)

    def run_inference(self) -> None:
        """Run NPU inference and publish velocity commands."""
        if self._rknn_runtime is None:
            return

        # Check for required inputs (IMU is optional)
        if (self._latest_rgb is None or self._latest_depth is None or
            self._latest_vel is None):
            return

        try:
            # Prepare inputs
            # RGB: (H, W, 3) -> (1, 3, H, W) uint8
            rgb = self._latest_rgb.astype(np.uint8)
            rgb = np.transpose(rgb, (2, 0, 1))  # (3, H, W)
            rgb = np.expand_dims(rgb, axis=0)  # (1, 3, H, W)

            # Depth: (H, W) -> (1, 1, H, W) float32 normalized to [0, 1]
            depth = (self._latest_depth / self.depth_clip).astype(np.float32)
            depth = np.expand_dims(depth, axis=0)  # (1, H, W)
            depth = np.expand_dims(depth, axis=0)  # (1, 1, H, W)

            # Proprioception: [lin_vel, ang_vel, roll, pitch, accel_mag, min_dist] -> (1, 6)
            lin_vel, ang_vel = self._latest_vel

            # IMU is optional - use zeros if not available
            if self._latest_imu is not None:
                roll, pitch, accel_mag = self._latest_imu
            else:
                roll, pitch, accel_mag = 0.0, 0.0, 0.0

            proprio = np.array([[
                lin_vel, ang_vel, roll, pitch, accel_mag, self._min_forward_dist
            ]], dtype=np.float32)  # (1, 6)

            # Run inference
            start_time = time.time()
            outputs = self._rknn_runtime.inference(inputs=[rgb, depth, proprio])
            inference_time = (time.time() - start_time) * 1000  # ms

            # Parse output: [linear_vel, angular_vel]
            action = outputs[0][0]  # Assuming output shape (1, 2)
            linear_cmd = float(np.clip(action[0], -1.0, 1.0) * self.max_linear)
            angular_cmd = float(np.clip(action[1], -1.0, 1.0) * self.max_angular)

            # Publish command
            cmd = Twist()
            cmd.linear.x = linear_cmd
            cmd.angular.z = angular_cmd
            self.cmd_pub.publish(cmd)

            self._inference_count += 1

            # Log performance
            if self._inference_count % 50 == 0:
                self.get_logger().info(
                    f'Inference #{self._inference_count} | '
                    f'Time: {inference_time:.1f}ms | '
                    f'Cmd: [{linear_cmd:.3f}, {angular_cmd:.3f}]'
                )

        except Exception as exc:
            self.get_logger().error(f'Inference failed: {exc}')

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
        """Cleanup RKNN resources."""
        if self._rknn_runtime is not None:
            self._rknn_runtime.release()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RemoteTrainedInference()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
