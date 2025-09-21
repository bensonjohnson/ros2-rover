#!/usr/bin/env python3
"""NPU exploration node consuming RTAB observation tensors.

This runtime node replaces the BEV-specific pipeline. It subscribes to the
`/exploration/observation` tensor, fuses proprioceptive scalars, and runs an
actor network (RKNN on-device or CPU fallback) to produce velocity commands. It
relies on the RTAB-based safety monitor for last-resort gating but also
respects external emergency/clearance signals.
"""

from __future__ import annotations

import math
import os
import time
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool, Float32, Float64MultiArray, Float32MultiArray, String
from std_srvs.srv import Trigger

try:
    from rknn.api import RKNN  # type: ignore
    RKNN_AVAILABLE = True
except ImportError:
    RKNN_AVAILABLE = False


class NPUExplorationRTAB(Node):
    def __init__(self) -> None:
        super().__init__('npu_exploration_rtab')

        # Parameters
        self.declare_parameter('max_speed', 0.20)
        self.declare_parameter('angular_speed', 0.8)
        self.declare_parameter('safety_distance', 0.25)
        self.declare_parameter('observation_topic', '/exploration/observation')
        self.declare_parameter('observation_timeout', 0.5)
        self.declare_parameter('cmd_vel_topic', 'cmd_vel_ai')
        self.declare_parameter('model_path', 'models/exploration_model_rtab.rknn')
        self.declare_parameter('cpu_fallback_speed', 0.12)
        self.declare_parameter('cpu_turn_rate', 0.6)
        self.declare_parameter('min_linear_cmd_mps', 0.03)
        self.declare_parameter('min_angular_cmd_rps', 0.05)
        self.declare_parameter('stuck_timeout_s', 6.0)

        self.max_speed = float(self.get_parameter('max_speed').value)
        self.angular_speed = float(self.get_parameter('angular_speed').value)
        self.safety_distance = float(self.get_parameter('safety_distance').value)
        self.observation_topic = str(self.get_parameter('observation_topic').value)
        self.observation_timeout = float(self.get_parameter('observation_timeout').value)
        self.cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        self.model_path = str(self.get_parameter('model_path').value)
        self.cpu_fallback_speed = float(self.get_parameter('cpu_fallback_speed').value)
        self.cpu_turn_rate = float(self.get_parameter('cpu_turn_rate').value)
        self.min_linear = float(self.get_parameter('min_linear_cmd_mps').value)
        self.min_angular = float(self.get_parameter('min_angular_cmd_rps').value)
        self.stuck_timeout = float(self.get_parameter('stuck_timeout_s').value)

        # State caches
        self.latest_observation: Optional[np.ndarray] = None  # (C,H,W)
        self.last_observation_time: float = 0.0
        self.observation_dims: Optional[Tuple[int, int, int]] = None
        self.external_emergency = False
        self.external_min_forward = 10.0
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.accel_mag = 0.0
        self.last_motion_time = time.time()
        self.wheel_velocities = np.zeros(2, dtype=np.float32)
        self.battery_percentage = 100.0

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.status_pub = self.create_publisher(String, '/npu_exploration_status', 10)
        self.debug_obs_pub = self.create_publisher(Float64MultiArray, '/exploration/observation_stats', 5)

        # Subscriptions
        self.create_subscription(Float32MultiArray, self.observation_topic, self.observation_callback, qos_profile_sensor_data)
        self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.create_subscription(Imu, '/lsm9ds1_imu_publisher/imu/data', self.imu_callback, qos_profile_sensor_data)
        self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 10)
        self.create_subscription(Float32, 'battery_percentage', self.battery_callback, 5)
        self.create_subscription(Bool, 'emergency_stop', self.emergency_callback, 10)
        self.create_subscription(Float32, 'min_forward_distance', self.min_forward_callback, 10)

        # Services
        self.create_service(Trigger, '/reload_rknn', self.reload_rknn_callback)

        # Control loop timer
        self.control_timer = self.create_timer(0.1, self.control_loop)
        self.status_timer = self.create_timer(1.0, self.publish_status)

        self.rknn: Optional[RKNN] = None
        self.rknn_ready = False
        if RKNN_AVAILABLE:
            self._init_rknn()
        else:
            self.get_logger().warn('RKNN library unavailable; using CPU fallback policy.')

        self.get_logger().info(
            f"NPU exploration (RTAB) initialized. model={self.model_path} max_speed={self.max_speed:.2f}"
        )

    # ------------------------------------------------------------------
    # Subscriptions
    def observation_callback(self, msg: Float32MultiArray) -> None:
        try:
            dims = msg.layout.dim
            if len(dims) >= 3:
                channels = dims[0].size
                height = dims[1].size
                width = dims[2].size
            elif len(dims) == 2:
                channels = dims[0].size
                height = dims[1].size
                width = len(msg.data) // (channels * height)
            else:
                self.get_logger().warn_throttle(5.0, 'Observation layout malformed.')
                return
            arr = np.array(msg.data, dtype=np.float32)
            if arr.size != channels * height * width:
                self.get_logger().warn_throttle(5.0, 'Observation size mismatch.')
                return
            tensor = arr.reshape((channels, height, width))
            self.latest_observation = tensor
            self.observation_dims = (channels, height, width)
            self.last_observation_time = time.time()
            if self.debug_obs_pub.get_subscription_count() > 0:
                stats = Float64MultiArray()
                stats.data = [
                    float(np.min(tensor)),
                    float(np.max(tensor)),
                    float(np.mean(tensor)),
                ]
                self.debug_obs_pub.publish(stats)
        except Exception as exc:
            self.get_logger().warn_throttle(5.0, f'Observation processing failed: {exc}')

    def odom_callback(self, msg: Odometry) -> None:
        self.linear_vel = float(msg.twist.twist.linear.x)
        self.angular_vel = float(msg.twist.twist.angular.z)
        speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        if speed > 0.02:
            self.last_motion_time = time.time()

    def imu_callback(self, msg: Imu) -> None:
        roll, pitch = self._roll_pitch_from_quaternion(
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        )
        ax, ay, az = msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
        self.roll = float(roll)
        self.pitch = float(pitch)
        self.accel_mag = float(math.sqrt(ax * ax + ay * ay + az * az))

    def joint_state_callback(self, msg: JointState) -> None:
        if len(msg.velocity) >= 2:
            self.wheel_velocities[0] = float(msg.velocity[0])
            self.wheel_velocities[1] = float(msg.velocity[1])

    def battery_callback(self, msg: Float32) -> None:
        self.battery_percentage = float(msg.data)

    def emergency_callback(self, msg: Bool) -> None:
        self.external_emergency = bool(msg.data)

    def min_forward_callback(self, msg: Float32) -> None:
        self.external_min_forward = float(msg.data)

    # ------------------------------------------------------------------
    def control_loop(self) -> None:
        now = time.time()
        if self.latest_observation is None:
            self._publish_idle('no_observation')
            return
        if (now - self.last_observation_time) > self.observation_timeout:
            self._publish_idle('stale_observation')
            return
        if self.external_emergency or (self.external_min_forward <= self.safety_distance):
            self._publish_idle('external_emergency')
            return
        if (now - self.last_motion_time) > self.stuck_timeout:
            # If stuck, encourage small turn
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = np.sign(np.random.randn()) * self.min_angular
            self.cmd_pub.publish(cmd)
            return

        obs = self.latest_observation
        proprio = self._proprio_vector()
        action = self._run_policy(obs, proprio)
        cmd = self._action_to_twist(action)
        self.cmd_pub.publish(cmd)

    def publish_status(self) -> None:
        msg = String()
        msg.data = (
            f"status | emergency={self.external_emergency} min_d={self.external_min_forward:.2f}m "
            f"battery={self.battery_percentage:.1f}% obs_age={time.time() - self.last_observation_time:.2f}s"
        )
        self.status_pub.publish(msg)

    # ------------------------------------------------------------------
    def _init_rknn(self) -> None:
        model_path = self.model_path
        if not model_path.startswith('/'):
            pkg_share = os.getenv('COLCON_PREFIX_PATH', '')
            if pkg_share:
                # Best effort: search beneath install space
                for path in pkg_share.split(':'):
                    candidate = os.path.join(path, 'tractor_bringup', model_path)
                    if os.path.isfile(candidate):
                        model_path = candidate
                        break
        if not os.path.isfile(model_path):
            self.get_logger().warn(f'RKNN model not found at {model_path}; using CPU fallback.')
            return
        try:
            self.rknn = RKNN()
            ret = self.rknn.load_rknn(model_path)
            if ret != 0:
                self.get_logger().error(f'Failed to load RKNN model (code {ret})')
                self.rknn = None
                return
            if self.rknn.init_runtime(target=None) != 0:
                self.get_logger().error('RKNN init_runtime failed.')
                self.rknn.release()
                self.rknn = None
                return
            self.rknn_ready = True
            self.get_logger().info(f'RKNN model loaded: {model_path}')
        except Exception as exc:
            self.get_logger().error(f'RKNN initialization error: {exc}')
            self.rknn = None
            self.rknn_ready = False

    def reload_rknn_callback(self, _req, _resp=None):
        self.get_logger().info('Reloading RKNN model on request...')
        if self.rknn:
            try:
                self.rknn.release()
            except Exception:
                pass
        self.rknn = None
        self.rknn_ready = False
        if RKNN_AVAILABLE:
            self._init_rknn()
        resp = Trigger.Response()
        resp.success = bool(self.rknn_ready)
        resp.message = 'RKNN reloaded' if self.rknn_ready else 'RKNN reload failed'
        return resp

    # ------------------------------------------------------------------
    def _run_policy(self, obs: np.ndarray, proprio: np.ndarray) -> np.ndarray:
        try:
            if self.rknn_ready and self.rknn is not None:
                input_tensor = obs.astype(np.float32)
                # Assume channels-first; convert to NHWC if required
                input_tensor = np.transpose(input_tensor, (1, 2, 0))
                input_tensor = input_tensor[np.newaxis, ...]
                outputs = self.rknn.inference(inputs=[input_tensor])
                if outputs and len(outputs) >= 1:
                    action = np.asarray(outputs[0], dtype=np.float32).flatten()
                    if action.size >= 2:
                        return action[:2]
            # CPU fallback: simple heuristic
            return self._cpu_policy(obs, proprio)
        except Exception as exc:
            self.get_logger().warn_throttle(5.0, f'Policy inference failed: {exc}')
            return self._cpu_policy(obs, proprio)

    def _cpu_policy(self, obs: np.ndarray, proprio: np.ndarray) -> np.ndarray:
        min_d = float(self.external_min_forward)
        if min_d > (self.safety_distance + 0.3):
            return np.array([self.cpu_fallback_speed / self.max_speed, 0.0], dtype=np.float32)
        turn = self.cpu_turn_rate / self.angular_speed
        # Prefer turning direction based on current angular velocity sign
        direction = -1.0 if self.angular_vel > 0 else 1.0
        return np.array([0.0, direction * turn], dtype=np.float32)

    def _action_to_twist(self, action: np.ndarray) -> Twist:
        lin = float(np.clip(action[0], -1.0, 1.0) * self.max_speed)
        ang = float(np.clip(action[1], -1.0, 1.0) * self.angular_speed)
        if 0.0 < abs(lin) < self.min_linear:
            lin = math.copysign(self.min_linear, lin)
        if 0.0 < abs(ang) < self.min_angular:
            ang = math.copysign(self.min_angular, ang)
        cmd = Twist()
        cmd.linear.x = lin
        cmd.angular.z = ang
        return cmd

    def _proprio_vector(self) -> np.ndarray:
        return np.array(
            [
                self.linear_vel,
                self.angular_vel,
                self.roll,
                self.pitch,
                self.accel_mag,
                self.wheel_velocities[0],
                self.wheel_velocities[1],
                self.external_min_forward,
            ],
            dtype=np.float32,
        )

    def _publish_idle(self, reason: str) -> None:
        cmd = Twist()
        self.cmd_pub.publish(cmd)
        if self.status_pub.get_subscription_count() > 0:
            msg = String()
            msg.data = f"idle | reason={reason}"
            self.status_pub.publish(msg)

    @staticmethod
    def _roll_pitch_from_quaternion(x: float, y: float, z: float, w: float) -> Tuple[float, float]:
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = max(min(t2, +1.0), -1.0)
        pitch = math.asin(t2)
        return roll, pitch


def main(args=None) -> None:
    rclpy.init(args=args)
    node = NPUExplorationRTAB()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
