#!/usr/bin/env python3
"""Safety monitor using RTAB-Map occupancy data.

Replaces the BEV-based guardian by evaluating the local RTAB occupancy grid
against commanded velocities. Provides conservative gating using a forward cone
check and publishes diagnostic topics similar to the BEV monitor.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool, Float32, String
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

import tf2_ros
from tf_transformations import quaternion_matrix


class SimpleSafetyMonitorRTAB(Node):
    def __init__(self) -> None:
        super().__init__('simple_safety_monitor_rtab')

        # Parameters
        self.declare_parameter('occupancy_topic', '/rtabmap/local_grid_map')
        self.declare_parameter('input_cmd_topic', 'cmd_vel_ai')
        self.declare_parameter('output_cmd_topic', 'cmd_vel_raw')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('emergency_stop_distance', 0.25)
        self.declare_parameter('hard_stop_distance', 0.10)
        self.declare_parameter('forward_width_m', 1.0)
        self.declare_parameter('max_eval_distance', 4.0)
        self.declare_parameter('occupancy_threshold', 50)
        self.declare_parameter('freshness_timeout', 0.5)

        self.occupancy_topic = str(self.get_parameter('occupancy_topic').value)
        self.input_cmd_topic = str(self.get_parameter('input_cmd_topic').value)
        self.output_cmd_topic = str(self.get_parameter('output_cmd_topic').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.emergency_distance = float(self.get_parameter('emergency_stop_distance').value)
        self.hard_stop_distance = float(self.get_parameter('hard_stop_distance').value)
        self.forward_width = float(self.get_parameter('forward_width_m').value)
        self.max_eval_distance = float(self.get_parameter('max_eval_distance').value)
        self.occupancy_threshold = int(self.get_parameter('occupancy_threshold').value)
        self.freshness_timeout = float(self.get_parameter('freshness_timeout').value)

        # State
        self.latest_cmd: Optional[Twist] = None
        self.commands_received = 0
        self.commands_blocked = 0
        self.emergency_stops = 0
        self.latest_grid: Optional[OccupancyGrid] = None
        self.latest_grid_stamp: Optional[Time] = None
        self.min_forward = 10.0
        self.left_clear = 1.0
        self.right_clear = 1.0

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscriptions
        self.create_subscription(OccupancyGrid, self.occupancy_topic, self.occupancy_callback, 10)
        self.create_subscription(Twist, self.input_cmd_topic, self.cmd_callback, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, self.output_cmd_topic, 10)
        self.estop_pub = self.create_publisher(Bool, 'emergency_stop', 10)
        self.status_pub = self.create_publisher(String, 'safety_monitor_status', 5)
        self.min_distance_pub = self.create_publisher(Float32, 'min_forward_distance', 5)
        self.diagnostics_pub = self.create_publisher(DiagnosticArray, 'diagnostics', 5)

        # Timers
        self.create_timer(0.1, self._publish_status)
        self.create_timer(1.0, self._publish_diagnostics)

        self.get_logger().info(
            f"Safety monitor (RTAB) active. occupancy={self.occupancy_topic} emergency={self.emergency_distance:.2f}m"
        )

    # ------------------------------------------------------------------
    def occupancy_callback(self, msg: OccupancyGrid) -> None:
        self.latest_grid = msg
        self.latest_grid_stamp = Time.from_msg(msg.header.stamp)
        self._recompute_distances()

    def cmd_callback(self, msg: Twist) -> None:
        self.commands_received += 1
        self.latest_cmd = msg
        self._recompute_distances()
        emergency = self.min_forward <= self.emergency_distance
        hard_stop = self.min_forward <= self.hard_stop_distance
        if emergency:
            self.emergency_stops += 1
        # Publish estop state
        self.estop_pub.publish(Bool(data=hard_stop))

        # Gate forward motion
        out = Twist()
        out.angular = msg.angular
        out.linear = msg.linear

        if self.min_forward > self.emergency_distance:
            self.cmd_pub.publish(msg)
            return

        # Too close: block forward component, allow reverse/turn
        blocked = False
        if msg.linear.x > 0.0:
            out.linear.x = 0.0
            blocked = True
        if msg.linear.x >= 0.0 and msg.angular.z == 0.0:
            # Encourage turning away from closer side
            out.angular.z = -0.5 if self.left_clear < self.right_clear else 0.5
        if blocked:
            self.commands_blocked += 1
        self.cmd_pub.publish(out)

    # ------------------------------------------------------------------
    def _recompute_distances(self) -> None:
        grid = self.latest_grid
        if grid is None:
            self.min_forward = 10.0
            return
        now = self.get_clock().now()
        if self.latest_grid_stamp is not None:
            age = (now - self.latest_grid_stamp).nanoseconds / 1e9
            if age > self.freshness_timeout:
                self.get_logger().warn_once('Occupancy grid stale, using safe defaults')
                self.min_forward = 10.0
                return

        mask, points_base = self._occupied_points_in_base(grid)
        if mask is None or points_base is None or not np.any(mask):
            self.min_forward = 10.0
            self.left_clear = self.right_clear = 1.0
            return
        pts = points_base[mask]
        forward = pts[:, 0]
        lateral = pts[:, 1]

        # Apply wedge filters
        forward_mask = (forward >= 0.0) & (forward <= self.max_eval_distance)
        lateral_mask = np.abs(lateral) <= self.forward_width * 0.5
        region = pts[forward_mask & lateral_mask]
        if region.size == 0:
            self.min_forward = 10.0
        else:
            self.min_forward = float(np.min(region[:, 0]))

        # Left/right heuristics for steering suggestions
        left_mask = (forward >= 0.0) & (forward <= self.max_eval_distance) & (lateral > 0.0)
        right_mask = (forward >= 0.0) & (forward <= self.max_eval_distance) & (lateral < 0.0)
        self.left_clear = float(np.clip(np.min(forward[left_mask]) if np.any(left_mask) else self.max_eval_distance, 0.0, self.max_eval_distance)) / self.max_eval_distance
        self.right_clear = float(np.clip(np.min(forward[right_mask]) if np.any(right_mask) else self.max_eval_distance, 0.0, self.max_eval_distance)) / self.max_eval_distance

        self.min_distance_pub.publish(Float32(data=float(self.min_forward)))

    def _occupied_points_in_base(self, grid: OccupancyGrid) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        data = np.asarray(grid.data, dtype=np.int16)
        mask = data >= self.occupancy_threshold
        if not np.any(mask):
            return mask, None

        res = grid.info.resolution
        width = grid.info.width
        height = grid.info.height

        cols = np.arange(width, dtype=np.float32)
        rows = np.arange(height, dtype=np.float32)
        cc, rr = np.meshgrid(cols, rows)
        # OccupancyGrid origin pose
        origin = grid.info.origin
        origin_rot = quaternion_matrix([
            origin.orientation.x,
            origin.orientation.y,
            origin.orientation.z,
            origin.orientation.w,
        ])
        origin_trans = np.array([
            origin.position.x,
            origin.position.y,
            origin.position.z,
        ], dtype=np.float32)

        # Cell centers in grid frame
        cell_x = (cc + 0.5) * res
        cell_y = (rr + 0.5) * res
        zeros = np.zeros_like(cell_x)
        points_grid = np.stack((cell_x, cell_y, zeros, np.ones_like(cell_x)), axis=-1)
        points_map = (points_grid @ origin_rot.T)[:, :, :3] + origin_trans
        points_map = points_map.reshape(-1, 3)

        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                grid.header.frame_id,
                Time.from_msg(grid.header.stamp),
                timeout=Duration(seconds=0.1),
            )
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as exc:
            self.get_logger().warn_throttle(5.0, f"TF lookup failed ({exc}); skipping occupancy safety")
            return mask, None

        rot = quaternion_matrix([
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ])
        trans = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ], dtype=np.float32)

        points_base = (points_map @ rot[:3, :3].T) + trans
        return mask.flatten(), points_base

    # ------------------------------------------------------------------
    def _publish_status(self) -> None:
        msg = String()
        msg.data = (
            f"SafetyRTAB min_forward={self.min_forward:.2f}m left={self.left_clear:.2f} right={self.right_clear:.2f}"
        )
        self.status_pub.publish(msg)

    def _publish_diagnostics(self) -> None:
        diag = DiagnosticArray()
        diag.status.append(self._build_status())
        diag.header.stamp = self.get_clock().now().to_msg()
        self.diagnostics_pub.publish(diag)

    def _build_status(self) -> DiagnosticStatus:
        st = DiagnosticStatus()
        st.name = 'safety_monitor_rtab'
        st.hardware_id = 'rtab_guardian'
        st.level = DiagnosticStatus.OK
        st.message = 'OK'
        st.values = []
        st.values.append(self._kv('commands_received', self.commands_received))
        st.values.append(self._kv('commands_blocked', self.commands_blocked))
        st.values.append(self._kv('emergency_stops', self.emergency_stops))
        st.values.append(self._kv('min_forward_m', f"{self.min_forward:.2f}"))
        return st

    @staticmethod
    def _kv(key: str, value) -> KeyValue:
        return KeyValue(key=str(key), value=str(value))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SimpleSafetyMonitorRTAB()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
