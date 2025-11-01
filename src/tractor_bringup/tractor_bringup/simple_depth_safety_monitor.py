#!/usr/bin/env python3
"""Simple Depth Image Safety Monitor.

Directly processes depth images to detect obstacles and compute minimum forward distance.
No point cloud or RTAB-Map required - just raw depth data.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32, String
from cv_bridge import CvBridge


class SimpleDepthSafetyMonitor(Node):
    """Safety monitor using direct depth image processing."""

    def __init__(self) -> None:
        super().__init__('simple_depth_safety_monitor')

        # Parameters
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('input_cmd_topic', 'cmd_vel_teleop')
        self.declare_parameter('output_cmd_topic', 'cmd_vel_raw')
        self.declare_parameter('emergency_stop_distance', 0.25)
        self.declare_parameter('hard_stop_distance', 0.12)
        self.declare_parameter('depth_scale', 0.001)  # For uint16 → meters
        self.declare_parameter('forward_roi_width_ratio', 0.6)  # Center 60% of image
        self.declare_parameter('forward_roi_height_ratio', 0.5)  # Bottom 50% of image
        self.declare_parameter('max_eval_distance', 5.0)  # Ignore depth > 5m

        self.depth_topic = str(self.get_parameter('depth_topic').value)
        self.input_cmd_topic = str(self.get_parameter('input_cmd_topic').value)
        self.output_cmd_topic = str(self.get_parameter('output_cmd_topic').value)
        self.emergency_distance = float(self.get_parameter('emergency_stop_distance').value)
        self.hard_stop_distance = float(self.get_parameter('hard_stop_distance').value)
        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.roi_width_ratio = float(self.get_parameter('forward_roi_width_ratio').value)
        self.roi_height_ratio = float(self.get_parameter('forward_roi_height_ratio').value)
        self.max_distance = float(self.get_parameter('max_eval_distance').value)

        # State
        self._min_forward_dist = 10.0
        self._latest_cmd = None
        self._commands_received = 0
        self._commands_blocked = 0
        self._emergency_stops = 0

        self.bridge = CvBridge()

        # Subscribers
        self.depth_sub = self.create_subscription(
            Image, self.depth_topic, self.depth_callback, qos_profile_sensor_data
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

        self.get_logger().info('Simple depth safety monitor initialized')
        self.get_logger().info(f'Emergency stop: {self.emergency_distance}m, Hard stop: {self.hard_stop_distance}m')

    def depth_callback(self, msg: Image) -> None:
        """Process depth image and compute minimum forward distance."""
        try:
            # Convert to numpy array
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # Convert to meters if uint16
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) * self.depth_scale
            elif depth.dtype != np.float32:
                depth = depth.astype(np.float32)

            # Clean up invalid values
            depth = np.nan_to_num(depth, nan=self.max_distance, posinf=self.max_distance, neginf=0.0)
            depth = np.clip(depth, 0.0, self.max_distance)

            # Define forward-looking ROI (center portion of image, bottom half)
            h, w = depth.shape
            roi_w = int(w * self.roi_width_ratio)
            roi_h = int(h * self.roi_height_ratio)
            x_start = (w - roi_w) // 2
            y_start = h - roi_h

            # Extract ROI
            roi = depth[y_start:, x_start:x_start + roi_w]

            # Find minimum distance in ROI (excluding zeros which are invalid readings)
            valid_depths = roi[roi > 0.01]  # Filter out near-zero invalid readings

            if len(valid_depths) > 0:
                self._min_forward_dist = float(np.min(valid_depths))
            else:
                self._min_forward_dist = self.max_distance

        except Exception as exc:
            self.get_logger().warn(f'Depth processing failed: {exc}')
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

        # Safety gating - only apply to forward motion
        if msg.linear.x > 0.01:  # Moving forward
            if self._min_forward_dist < self.hard_stop_distance:
                # Hard stop - zero all motion
                out_cmd.linear.x = 0.0
                out_cmd.linear.y = 0.0
                out_cmd.angular.z = 0.0
                self._commands_blocked += 1
                self._emergency_stops += 1
                self.estop_pub.publish(Bool(data=True))
                self.get_logger().warn(
                    f'HARD STOP: Obstacle at {self._min_forward_dist:.2f}m (threshold: {self.hard_stop_distance}m)'
                )
            elif self._min_forward_dist < self.emergency_distance:
                # Soft stop - allow rotation but stop forward motion
                out_cmd.linear.x = 0.0
                out_cmd.linear.y = 0.0
                self._commands_blocked += 1
                self.get_logger().warn(
                    f'SOFT STOP: Obstacle at {self._min_forward_dist:.2f}m (threshold: {self.emergency_distance}m)'
                )

        # Publish gated command
        self.cmd_pub.publish(out_cmd)

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
    node = SimpleDepthSafetyMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
