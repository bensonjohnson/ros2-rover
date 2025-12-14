#!/usr/bin/env python3
"""LIDAR-based Safety Monitor.

Processes LIDAR scan data to detect obstacles and compute minimum forward distance.
Monitors the front 180-degree arc for obstacles within safety thresholds.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32, String


class LidarSafetyMonitor(Node):
    """Safety monitor using LIDAR scan processing."""

    def __init__(self) -> None:
        super().__init__('lidar_safety_monitor')

        # Parameters
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('input_cmd_topic', 'cmd_vel_teleop')
        self.declare_parameter('output_cmd_topic', 'cmd_vel_raw')
        self.declare_parameter('emergency_stop_distance', 0.20)
        self.declare_parameter('hard_stop_distance', 0.15)
        self.declare_parameter('min_valid_range', 0.05)  # LIDAR minimum detection
        self.declare_parameter('max_eval_distance', 5.0)  # Ignore ranges > 5m

        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.input_cmd_topic = str(self.get_parameter('input_cmd_topic').value)
        self.output_cmd_topic = str(self.get_parameter('output_cmd_topic').value)
        self.emergency_distance = float(self.get_parameter('emergency_stop_distance').value)
        self.hard_stop_distance = float(self.get_parameter('hard_stop_distance').value)
        self.min_valid_range = float(self.get_parameter('min_valid_range').value)
        self.max_distance = float(self.get_parameter('max_eval_distance').value)

        # State
        self._min_forward_dist = 10.0
        self._latest_cmd = None
        self._commands_received = 0
        self._commands_blocked = 0
        self._emergency_stops = 0
        self._last_estop_state = False
        self._last_scan_time = self.get_clock().now()

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan, self.scan_topic, self.scan_callback, qos_profile_sensor_data
        )
        self.cmd_sub = self.create_subscription(
            Twist, self.input_cmd_topic, self.cmd_callback, 10
        )

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, self.output_cmd_topic, 10)
        self.estop_pub = self.create_publisher(Bool, 'emergency_stop', 10)
        self.status_pub = self.create_publisher(String, 'safety_monitor_status', 5)
        self.min_distance_pub = self.create_publisher(Float32, 'min_forward_distance', 5)

        # Timer for status updates
        self.create_timer(0.1, self._publish_status)

        self.get_logger().info('LIDAR safety monitor initialized')
        self.get_logger().info(f'Emergency stop: {self.emergency_distance}m, Hard stop: {self.hard_stop_distance}m')
        self.get_logger().info(f'Monitoring front 180° sector')

    def scan_callback(self, msg: LaserScan) -> None:
        """Process LIDAR scan and compute minimum forward distance."""
        self._last_scan_time = self.get_clock().now()
        try:
            # Extract ranges array
            ranges = np.array(msg.ranges)

            # Compute angles for each ray
            angles = np.arange(len(ranges)) * msg.angle_increment + msg.angle_min

            # Filter to front 180° (-π/2 to +π/2)
            front_mask = (angles >= -np.pi/2) & (angles <= np.pi/2)

            # Filter valid ranges (not NaN, Inf, or out of bounds)
            valid_mask = (
                np.isfinite(ranges) &
                (ranges >= self.min_valid_range) &
                (ranges <= self.max_distance) &
                front_mask
            )

            # Compute minimum distance in valid front sector
            if np.any(valid_mask):
                self._min_forward_dist = float(np.min(ranges[valid_mask]))
            else:
                self._min_forward_dist = self.max_distance

        except Exception as exc:
            self.get_logger().warn(f'LIDAR scan processing failed: {exc}')
            self._min_forward_dist = 0.0  # Assume danger if processing fails

    def cmd_callback(self, msg: Twist) -> None:
        """Process velocity command and apply safety gating."""
        self._latest_cmd = msg
        self._commands_received += 1

        # Create output command
        out_cmd = Twist()
        out_cmd.linear.x = msg.linear.x
        out_cmd.linear.y = msg.linear.y
        out_cmd.linear.z = msg.linear.z
        out_cmd.angular.x = msg.angular.x
        out_cmd.angular.y = msg.angular.y
        out_cmd.angular.z = msg.angular.z

        estop_active = False

        # Safety gating - only apply to forward motion
        if msg.linear.x > 0.01:  # Moving forward
            # Check for stale data
            time_since_scan = (self.get_clock().now() - self._last_scan_time).nanoseconds / 1e9

            if time_since_scan > 0.2:
                # Stale data - treat as hard stop
                out_cmd.linear.x = 0.0
                out_cmd.linear.y = 0.0
                estop_active = True
                self._commands_blocked += 1
                if self._commands_received % 20 == 0:
                    self.get_logger().warn(f'STALE DATA: Last scan {time_since_scan:.2f}s ago - stopping')

            elif self._min_forward_dist < self.hard_stop_distance:
                # Hard stop - stop forward motion, allow rotation and reverse to escape
                out_cmd.linear.x = 0.0
                out_cmd.linear.y = 0.0
                # Keep angular.z to allow rotation
                self._commands_blocked += 1
                self._emergency_stops += 1
                estop_active = True
                self.get_logger().warn(
                    f'HARD STOP: Obstacle at {self._min_forward_dist:.2f}m (threshold: {self.hard_stop_distance}m) - rotate or reverse to escape'
                )
            elif self._min_forward_dist < self.emergency_distance:
                # Soft stop - allow rotation but stop forward motion
                out_cmd.linear.x = 0.0
                out_cmd.linear.y = 0.0
                self._commands_blocked += 1
                self.get_logger().warn(
                    f'SOFT STOP: Obstacle at {self._min_forward_dist:.2f}m (threshold: {self.emergency_distance}m)'
                )

        # Allow reverse motion always (negative linear.x passes through)

        # Publish estop state changes only (not continuously)
        if estop_active != self._last_estop_state:
            self.estop_pub.publish(Bool(data=estop_active))
            self._last_estop_state = estop_active
            if not estop_active:
                self.get_logger().info('Emergency stop cleared - safe to proceed')

        # Always publish gated command (even if zero) to keep motor driver alive
        self.cmd_pub.publish(out_cmd)

        if estop_active:
             if self._commands_blocked % 10 == 0:
                 self.get_logger().info(
                     f'Commands blocked: {self._commands_blocked} | '
                     f'In: lin={msg.linear.x:.2f} ang={msg.angular.z:.2f} | '
                     f'Out: lin={out_cmd.linear.x:.2f} ang={out_cmd.angular.z:.2f}'
                 )

    def _publish_status(self) -> None:
        """Publish status and diagnostics."""
        # Publish minimum distance
        self.min_distance_pub.publish(Float32(data=self._min_forward_dist))

        # Publish status string
        if self._min_forward_dist < self.hard_stop_distance:
            status = f'HARD_STOP (dist: {self._min_forward_dist:.2f}m)'
        elif self._min_forward_dist < self.emergency_distance:
            status = f'SOFT_STOP (dist: {self._min_forward_dist:.2f}m)'
        else:
            status = f'CLEAR (dist: {self._min_forward_dist:.2f}m)'

        self.status_pub.publish(String(data=status))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LidarSafetyMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
