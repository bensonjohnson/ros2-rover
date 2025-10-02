#!/usr/bin/env python3
"""RTAB observation builder.

Assembles an exploration observation tensor composed of:
- Cropped RTAB-Map occupancy window around the robot (float32 [-1,1]).
- Downsampled depth image aligned to color.
- Proprioceptive scalars (linear/Angular velocity, roll, pitch, accel magnitude).

The resulting tensor is published as a Float32MultiArray with layout metadata
(channels, height, width) so downstream policies can reshape efficiently.

Note: this is an initial scaffold; frontier masks and additional channels can be
layered in later phases.
"""

from typing import Optional, Tuple, List

import math
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import Image, Imu
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Float32, Float32MultiArray, MultiArrayDimension

from cv_bridge import CvBridge
import cv2


class RTABObservationNode(Node):
    def __init__(self) -> None:
        super().__init__('rtab_observation')

        # Parameters for topic names
        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('occupancy_topic', '/rtabmap/local_grid_map')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('imu_topic', '/lsm9ds1_imu_publisher/imu/data')
        self.declare_parameter('frontier_topic', '/rtabmap/frontiers')
        self.declare_parameter('output_topic', '/exploration/observation')
        self.declare_parameter('min_distance_topic', 'rtab_observation/min_forward_distance')
        self.declare_parameter('safety_hint_topic', 'rtab_observation/safety_hints')

        # Parameters for observation geometry
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('occupancy_window_m', 12.0)  # Square window length (meters)
        self.declare_parameter('observation_height', 128)  # Height
        self.declare_parameter('observation_width', 128)  # Width
        self.declare_parameter('depth_scale', 0.001)  # Depth to meters for uint16 frames
        self.declare_parameter('depth_clip_m', 6.0)
        self.declare_parameter('frontier_update_interval', 0.3)
        self.declare_parameter('safety_forward_width_m', 1.0)
        self.declare_parameter('safety_max_distance_m', 4.0)
        self.declare_parameter('enable_sample_logging', False)
        self.declare_parameter('sample_logging_interval', 30.0)
        self.declare_parameter('sample_logging_directory', 'log/observations')

        self.depth_topic = str(self.get_parameter('depth_topic').value)
        self.occupancy_topic = str(self.get_parameter('occupancy_topic').value)
        self.odom_topic = str(self.get_parameter('odom_topic').value)
        self.imu_topic = str(self.get_parameter('imu_topic').value)
        self.frontier_topic = str(self.get_parameter('frontier_topic').value)
        self.output_topic = str(self.get_parameter('output_topic').value)
        self.min_distance_topic = str(self.get_parameter('min_distance_topic').value)
        self.safety_hint_topic = str(self.get_parameter('safety_hint_topic').value)

        self.publish_rate = float(self.get_parameter('publish_rate_hz').value)
        self.window_m = float(self.get_parameter('occupancy_window_m').value)
        self.out_h = int(self.get_parameter('observation_height').value)
        self.out_w = int(self.get_parameter('observation_width').value)
        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.depth_clip = float(self.get_parameter('depth_clip_m').value)
        self.frontier_update_interval = float(self.get_parameter('frontier_update_interval').value)
        self.safety_forward_width = float(self.get_parameter('safety_forward_width_m').value)
        self.safety_max_distance = float(self.get_parameter('safety_max_distance_m').value)
        self.log_samples = bool(self.get_parameter('enable_sample_logging').value)
        self.sample_log_interval = float(self.get_parameter('sample_logging_interval').value)
        self.sample_log_dir = str(self.get_parameter('sample_logging_directory').value)

        if self.log_samples:
            os.makedirs(self.sample_log_dir, exist_ok=True)

        self.bridge = CvBridge()

        # Publishers for observation tensors and safety hints
        self.obs_pub = self.create_publisher(Float32MultiArray, self.output_topic, 10)
        self.min_distance_pub = self.create_publisher(Float32, self.min_distance_topic, 10)
        self.safety_hint_pub = self.create_publisher(Float32MultiArray, self.safety_hint_topic, 10)

        # Preallocated buffers for publishing/logging
        self._tensor_buffer: Optional[np.ndarray] = None
        self._flat_buffer: Optional[np.ndarray] = None
        self._frontier_canvas: Optional[np.ndarray] = np.zeros((self.out_h, self.out_w), dtype=np.float32)
        self._frontier_kernel: np.ndarray = np.ones((3, 3), dtype=np.uint8)
        self._pixel_u_grid: Optional[np.ndarray] = None
        self._pixel_v_grid: Optional[np.ndarray] = None
        self._next_frontier_update = 0.0

        # State caches
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_depth_ts = None
        self._occupancy: Optional[OccupancyGrid] = None
        self._latest_pose: Optional[Tuple[float, float, float]] = None  # x, y, yaw
        self._latest_vel: Optional[Tuple[float, float]] = None  # linear, angular
        self._latest_imu: Optional[Tuple[float, float, float]] = None  # roll, pitch, accel_mag
        self._frontier_points: Optional[np.ndarray] = None  # Nx2 world coords
        self._last_sample_time = time.time()

        # Subscribers
        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            qos_profile_sensor_data,
        )
        self.occ_sub = self.create_subscription(
            OccupancyGrid,
            self.occupancy_topic,
            self.occupancy_callback,
            10,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            20,
        )
        self.imu_sub = self.create_subscription(
            Imu,
            self.imu_topic,
            self.imu_callback,
            qos_profile_sensor_data,
        )
        if self.frontier_topic:
            self.frontier_sub = self.create_subscription(
                MarkerArray,
                self.frontier_topic,
                self.frontier_callback,
                5,
            )

        timer_period = 1.0 / max(self.publish_rate, 1e-3)
        self.timer = self.create_timer(timer_period, self.publish_observation)

        self.get_logger().info(
            f"RTAB observation node active: depth={self.depth_topic} occupancy={self.occupancy_topic}"
        )

    # --- Callbacks ---
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
            self._latest_depth_ts = msg.header.stamp
        except Exception as exc:
            self.get_logger().warn(f'Depth conversion failed: {exc}')

    def occupancy_callback(self, msg: OccupancyGrid) -> None:
        self._occupancy = msg

    def odom_callback(self, msg: Odometry) -> None:
        try:
            position = msg.pose.pose.position
            orientation = msg.pose.pose.orientation
            yaw = self._yaw_from_quaternion(orientation.x, orientation.y, orientation.z, orientation.w)
            self._latest_pose = (position.x, position.y, yaw)
            lin = float(msg.twist.twist.linear.x)
            ang = float(msg.twist.twist.angular.z)
            self._latest_vel = (lin, ang)
        except Exception as exc:
            self.get_logger().debug(f'Odom parsing failed: {exc}')

    def imu_callback(self, msg: Imu) -> None:
        try:
            roll, pitch = self._roll_pitch_from_quaternion(
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            )
            ax, ay, az = msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
            accel_mag = float(np.sqrt(ax * ax + ay * ay + az * az))
            self._latest_imu = (roll, pitch, accel_mag)
        except Exception as exc:
            self.get_logger().debug(f'IMU parsing failed: {exc}')

    # --- Observation publishing ---
    def publish_observation(self) -> None:
        if self._latest_depth is None or self._occupancy is None or self._latest_pose is None:
            return

        occ_patch, patch_bounds = self._extract_occupancy_patch(self._occupancy, self._latest_pose)
        if occ_patch is None or patch_bounds is None:
            return

        depth_frame = self._downsample_depth(self._latest_depth)

        # Frontier mask channel (optional)
        frontier_channel = self._build_frontier_channel(patch_bounds)
        now = time.time()
        if frontier_channel is not None and now < getattr(self, "_next_frontier_update", 0.0):
            frontier_channel = None
        else:
            self._next_frontier_update = now + max(self.frontier_update_interval, 0.1)

        proprio = self._compose_proprio() or []
        channel_count = 3 + len(proprio)

        if (
            self._tensor_buffer is None
            or self._tensor_buffer.shape[0] != channel_count
            or self._tensor_buffer.shape[1] != self.out_h
            or self._tensor_buffer.shape[2] != self.out_w
        ):
            self._tensor_buffer = np.zeros((channel_count, self.out_h, self.out_w), dtype=np.float32)
            self._flat_buffer = np.zeros(channel_count * self.out_h * self.out_w, dtype=np.float32)
            self._pixel_u_grid = None
            self._pixel_v_grid = None

        np.copyto(self._tensor_buffer[0], occ_patch, casting='unsafe')
        np.copyto(self._tensor_buffer[1], depth_frame, casting='unsafe')

        if frontier_channel is not None:
            np.copyto(self._tensor_buffer[2], frontier_channel, casting='unsafe')
        else:
            self._tensor_buffer[2].fill(0.0)

        for idx, value in enumerate(proprio, start=3):
            self._tensor_buffer[idx].fill(float(value))

        min_forward, left_clear, right_clear = self._compute_safety_hints(occ_patch, patch_bounds)

        self.min_distance_pub.publish(Float32(data=float(min_forward)))

        hints_msg = Float32MultiArray()
        hints_msg.layout.dim = [MultiArrayDimension(label='components', size=3, stride=3)]
        hints_msg.layout.data_offset = 0
        hints_msg.data = [float(min_forward), float(left_clear), float(right_clear)]
        self.safety_hint_pub.publish(hints_msg)

        if self.log_samples and (time.time() - self._last_sample_time) >= self.sample_log_interval:
            self._write_sample_to_disk(self._tensor_buffer.copy())
            self._last_sample_time = time.time()

        flat_view = self._tensor_buffer.reshape(-1)
        np.copyto(self._flat_buffer, flat_view, casting='unsafe')

        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label='channels', size=channel_count, stride=self.out_h * self.out_w),
            MultiArrayDimension(label='height', size=self.out_h, stride=self.out_w),
            MultiArrayDimension(label='width', size=self.out_w, stride=1),
        ]
        msg.layout.data_offset = 0
        msg.data = self._flat_buffer.tolist()
        self.obs_pub.publish(msg)

    # --- Helpers ---
    def _extract_occupancy_patch(
        self, grid: OccupancyGrid, pose: Tuple[float, float, float]
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float, float, float]]]:
        if grid.info.resolution <= 0.0 or grid.info.width == 0 or grid.info.height == 0:
            return None, None
        res = grid.info.resolution
        width = grid.info.width
        height = grid.info.height
        origin_x = grid.info.origin.position.x
        origin_y = grid.info.origin.position.y

        robot_x, robot_y, _ = pose

        # Convert occupancy data
        occ_data = np.array(grid.data, dtype=np.int16).reshape((height, width))
        # Map occupancy to [-1, 1]: unknown -> -1, free -> 0, occupied -> 1
        occ_norm = np.zeros_like(occ_data, dtype=np.float32)
        occ_norm[:] = -1.0
        occ_norm[occ_data == 0] = 0.0
        occ_norm[occ_data > 0] = 1.0

        window_cells = int(self.window_m / res)
        half = window_cells // 2

        center_c = int((robot_x - origin_x) / res)
        center_r = int((robot_y - origin_y) / res)

        r_min = max(0, center_r - half)
        r_max = min(height, center_r + half)
        c_min = max(0, center_c - half)
        c_max = min(width, center_c + half)

        if r_min >= r_max or c_min >= c_max:
            return None, None

        patch = occ_norm[r_min:r_max, c_min:c_max]
        if patch.size == 0:
            return None, None

        patch_resized = cv2.resize(patch, (self.out_w, self.out_h), interpolation=cv2.INTER_NEAREST)
        min_x = origin_x + (c_min * res)
        max_x = origin_x + (c_max * res)
        min_y = origin_y + (r_min * res)
        max_y = origin_y + (r_max * res)
        return patch_resized, (min_x, max_x, min_y, max_y)

    def _downsample_depth(self, depth: np.ndarray) -> np.ndarray:
        resized = cv2.resize(depth, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
        # Normalize depth to [0, 1] by clip value
        if self.depth_clip > 0:
            resized = resized / self.depth_clip
        return np.clip(resized, 0.0, 1.0)

    def _ensure_pixel_grids(self) -> None:
        if (
            self._pixel_u_grid is None
            or self._pixel_u_grid.shape != (self.out_h, self.out_w)
        ):
            width = max(self.out_w, 1)
            height = max(self.out_h, 1)
            u = (np.arange(width, dtype=np.float32) + 0.5) / float(width)
            v = (np.arange(height, dtype=np.float32) + 0.5) / float(height)
            self._pixel_u_grid, self._pixel_v_grid = np.meshgrid(u, v)

    def _compute_safety_hints(
        self,
        occ_patch: np.ndarray,
        bounds: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float]:
        self._ensure_pixel_grids()

        if occ_patch is None or self._pixel_u_grid is None or self._pixel_v_grid is None:
            return 10.0, 1.0, 1.0

        occupied_mask = occ_patch > 0.5
        if not np.any(occupied_mask):
            return 10.0, 1.0, 1.0

        min_x, max_x, min_y, max_y = bounds
        span_x = max(max_x - min_x, 1e-6)
        span_y = max(max_y - min_y, 1e-6)

        world_x = min_x + self._pixel_u_grid * span_x
        world_y = min_y + self._pixel_v_grid * span_y

        robot_x, robot_y, yaw = self._latest_pose
        dx = world_x - robot_x
        dy = world_y - robot_y

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        forward = cos_yaw * dx + sin_yaw * dy
        lateral = -sin_yaw * dx + cos_yaw * dy

        max_dist = max(self.safety_max_distance, 0.1)
        half_width = max(self.safety_forward_width * 0.5, 0.05)

        wedge_mask = (
            occupied_mask
            & (forward >= 0.0)
            & (forward <= max_dist)
            & (np.abs(lateral) <= half_width)
        )

        if np.any(wedge_mask):
            min_forward = float(np.min(forward[wedge_mask]))
        else:
            min_forward = max_dist

        left_mask = (
            occupied_mask
            & (forward >= 0.0)
            & (forward <= max_dist)
            & (lateral > 0.0)
        )
        right_mask = (
            occupied_mask
            & (forward >= 0.0)
            & (forward <= max_dist)
            & (lateral < 0.0)
        )

        left_forward = float(np.min(forward[left_mask])) if np.any(left_mask) else max_dist
        right_forward = float(np.min(forward[right_mask])) if np.any(right_mask) else max_dist

        left_clear = float(np.clip(left_forward, 0.0, max_dist) / max_dist)
        right_clear = float(np.clip(right_forward, 0.0, max_dist) / max_dist)

        return min_forward, left_clear, right_clear

    def _compose_proprio(self) -> Optional[List[float]]:
        if self._latest_vel is None or self._latest_imu is None:
            return None
        lin, ang = self._latest_vel
        roll, pitch, accel_mag = self._latest_imu
        values = [
            float(lin),
            float(ang),
            float(roll),
            float(pitch),
            float(accel_mag),
        ]
        return values

    def _build_frontier_channel(self, bounds: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        if self._frontier_points is None or self._frontier_points.size == 0:
            if self._frontier_canvas is not None:
                self._frontier_canvas.fill(0.0)
            return self._frontier_canvas
        min_x, max_x, min_y, max_y = bounds
        fx = self._frontier_points[:, 0]
        fy = self._frontier_points[:, 1]
        mask = (fx >= min_x) & (fx <= max_x) & (fy >= min_y) & (fy <= max_y)
        if self._frontier_canvas is None or self._frontier_canvas.shape != (self.out_h, self.out_w):
            self._frontier_canvas = np.zeros((self.out_h, self.out_w), dtype=np.float32)
        canvas = self._frontier_canvas
        canvas.fill(0.0)
        if not np.any(mask):
            return canvas
        fx = fx[mask]
        fy = fy[mask]
        # Normalize to [0,1] in patch coordinates
        u = (fx - min_x) / max(max_x - min_x, 1e-6)
        v = (fy - min_y) / max(max_y - min_y, 1e-6)
        # Map to pixel indices
        px = np.clip((u * (self.out_w - 1)).astype(np.int32), 0, self.out_w - 1)
        py = np.clip((v * (self.out_h - 1)).astype(np.int32), 0, self.out_h - 1)
        canvas[py, px] = 1.0
        cv2.dilate(canvas, self._frontier_kernel, dst=canvas, iterations=1)
        np.clip(canvas, 0.0, 1.0, out=canvas)
        return canvas

    def frontier_callback(self, msg: MarkerArray) -> None:
        points: List[Tuple[float, float]] = []
        for marker in msg.markers:
            for p in marker.points:
                points.append((p.x, p.y))
        if points:
            self._frontier_points = np.array(points, dtype=np.float32)
        else:
            self._frontier_points = None

    def _write_sample_to_disk(self, tensor: np.ndarray) -> None:
        try:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            path = os.path.join(self.sample_log_dir, f'observation_{timestamp}.npz')
            extras = {}
            if tensor.shape[0] > 0:
                extras['occupancy_channel'] = tensor[0]
            if tensor.shape[0] > 1:
                extras['depth_channel'] = tensor[1]
            np.savez_compressed(path, observation=tensor, **extras)
            self.get_logger().debug(f'Saved observation sample to {path}')
        except Exception as exc:
            self.get_logger().warn_throttle(5.0, f'Observation logging failed: {exc}')

    @staticmethod
    def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    @staticmethod
    def _roll_pitch_from_quaternion(x: float, y: float, z: float, w: float) -> Tuple[float, float]:
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch = np.arcsin(t2)
        return float(roll), float(pitch)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RTABObservationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
