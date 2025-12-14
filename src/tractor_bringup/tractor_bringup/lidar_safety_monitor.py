#!/usr/bin/env python3
"""LIDAR-based Safety Monitor with Box Detection and Hysteresis.

Processes LIDAR scan data to detect obstacles using a rectangular footprint
("safety box") rather than simple radial distance, allowing navigation through
narrow gaps. Implements hysteresis for stability and linear speed scaling.
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import SetParametersResult

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32, String


class LidarSafetyMonitor(Node):
    """Safety monitor using LIDAR scan processing with box-based detection."""

    def __init__(self) -> None:
        super().__init__('lidar_safety_monitor')

        # Parameters
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('input_cmd_topic', 'cmd_vel_teleop')
        self.declare_parameter('output_cmd_topic', 'cmd_vel_raw')
        
        # Dimensions for "Safety Box"
        self.declare_parameter('robot_width', 0.6)        # Meters (physical width + small margin)
        self.declare_parameter('stop_distance', 0.25)     # Meters (hard stop)
        self.declare_parameter('slow_distance', 1.0)      # Meters (start slowing down)
        self.declare_parameter('hysteresis', 0.05)        # Meters (gap to resume)
        self.declare_parameter('max_eval_distance', 5.0)  # Ignore ranges > 5m

        # Load parameters
        self._load_parameters()
        self.add_on_set_parameters_callback(self._parameters_callback)

        # State
        self._min_forward_dist = 10.0
        self._closest_point_x = 10.0  # Forward distance in box
        self._closest_point_y = 0.0   # Lateral position of closest point
        
        self._latest_cmd = None
        self._commands_received = 0
        self._commands_blocked = 0
        self._emergency_stops = 0
        
        # Hysteresis state: True if we are currently in a stopped state
        self._is_stopped = False
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

        self.get_logger().info('LIDAR safety monitor initialized (Box Mode)')
        self._log_params()

    def _load_parameters(self):
        """Load parameter values from ROS system."""
        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.input_cmd_topic = str(self.get_parameter('input_cmd_topic').value)
        self.output_cmd_topic = str(self.get_parameter('output_cmd_topic').value)
        
        # Geometry
        self.robot_half_width = float(self.get_parameter('robot_width').value) / 2.0
        self.stop_dist = float(self.get_parameter('stop_distance').value)
        self.slow_dist = float(self.get_parameter('slow_distance').value)
        self.hysteresis = float(self.get_parameter('hysteresis').value)
        self.max_distance = float(self.get_parameter('max_eval_distance').value)

        # Computed thresholds
        self.resume_dist = self.stop_dist + self.hysteresis

    def _log_params(self):
        self.get_logger().info(
            f'Params: Stop={self.stop_dist:.2f}m (Resume={self.resume_dist:.2f}m), '
            f'Slow={self.slow_dist:.2f}m, Width={self.robot_half_width*2:.2f}m'
        )

    def _parameters_callback(self, params):
        """Handle parameter updates at runtime."""
        for param in params:
            if param.name == 'stop_distance':
                self.stop_dist = param.value
                self.resume_dist = self.stop_dist + self.hysteresis
            elif param.name == 'hysteresis':
                self.hysteresis = param.value
                self.resume_dist = self.stop_dist + self.hysteresis
            elif param.name == 'slow_distance':
                self.slow_dist = param.value
            elif param.name == 'robot_width':
                self.robot_half_width = param.value / 2.0
            
        self._log_params()
        return SetParametersResult(successful=True)

    def scan_callback(self, msg: LaserScan) -> None:
        """Process LIDAR scan and compute distance metrics."""
        self._last_scan_time = self.get_clock().now()
        
        try:
            # 1. Convert relevant ranges to Cartesian (x, y)
            ranges = np.array(msg.ranges)
            angles = np.arange(len(ranges)) * msg.angle_increment + msg.angle_min
            
            # Valid mask: reasonable range
            valid_mask = (np.isfinite(ranges) & (ranges > 0.05) & (ranges <= self.max_distance))
            
            if not np.any(valid_mask):
                # No valid data - assume clear (or handle as error?)
                # Usually better to stay safe, but for now we follow old logic: max distance
                self._min_forward_dist = self.max_distance
                self._closest_point_x = self.max_distance
                return

            r = ranges[valid_mask]
            theta = angles[valid_mask]

            x = r * np.cos(theta)  # Forward
            y = r * np.sin(theta)  # Left/Right

            # 2. Filter points within Robot's Width (Safety Box Width)
            # Check y coordinates against robot half-width
            in_path_mask = np.abs(y) <= self.robot_half_width
            
            # Also consider points *slightly* wider if they are very close? 
            # For now, stick to simple box: if it's within Y width, check X distance.
            
            points_in_path_x = x[in_path_mask]
            
            # Further filter: only points in front (x > 0)
            front_points = points_in_path_x[points_in_path_x > 0]
            
            if len(front_points) > 0:
                self._closest_point_x = float(np.min(front_points))
                # Store globally for status
                self._min_forward_dist = self._closest_point_x
            else:
                self._closest_point_x = self.max_distance
                self._min_forward_dist = self.max_distance

        except Exception as exc:
            self.get_logger().warn(f'LIDAR processing error: {exc}')
            # On error, don't stop blindly unless persistent, but reset dist just in case
            # self._min_forward_dist = 0.0 

    def cmd_callback(self, msg: Twist) -> None:
        """Process velocity command and apply safety gating."""
        self._latest_cmd = msg
        self._commands_received += 1

        # Copy command
        out_cmd = Twist()
        out_cmd.angular = msg.angular # Always allow rotation
        out_cmd.linear.y = msg.linear.y
        out_cmd.linear.z = msg.linear.z
        
        # Only gate positive x velocity (forward motion)
        target_speed = msg.linear.x
        
        estop_active = False
        scale_factor = 1.0

        if target_speed > 0.01:
            # Check for stale data
            time_since_scan = (self.get_clock().now() - self._last_scan_time).nanoseconds / 1e9
            
            if time_since_scan > 0.5:
                # Stale data safety
                target_speed = 0.0
                estop_active = True
                if self._commands_received % 30 == 0:
                    self.get_logger().warn(f'STALE DATA ({time_since_scan:.2f}s) - Stopping')
            else:
                # Hysteresis Logic
                dist = self._min_forward_dist
                
                if self._is_stopped:
                    # We are currently stopped. Must clear resume_dist to start.
                    if dist > self.resume_dist:
                        self._is_stopped = False
                        # We can proceed, but check scaling below
                    else:
                        # Still stopped
                        target_speed = 0.0
                        estop_active = True
                else:
                    # We are moving. Check if we need to stop.
                    if dist < self.stop_dist:
                        self._is_stopped = True
                        target_speed = 0.0
                        estop_active = True
        
                # Speed Scaling (Soft Stop)
                # If valid and not fully stopped, scale speed based on distance
                if not estop_active and dist < self.slow_dist:
                    # Linearly scale from 1.0 at slow_dist to 0.1 at stop_dist
                    # We don't go to 0.0 here because stop_dist handles the hard cut
                    
                    # Normalized factor (0 to 1) between stop and slow
                    range_span = self.slow_dist - self.stop_dist
                    if range_span > 0:
                        factor = (dist - self.stop_dist) / range_span
                        factor = np.clip(factor, 0.0, 1.0)
                        
                        # Scale speed: Min 10% or 0.05 m/s, Max 100%
                        # Helps maintain torque at low speeds if needed
                        scale_factor = 0.1 + (0.9 * factor)
                        estop_active = False # It is a warning, not a stop
                    
        # Apply speed
        out_cmd.linear.x = target_speed * scale_factor
        
        # Publish estop state changes
        if estop_active != self._last_estop_state:
            self.estop_pub.publish(Bool(data=estop_active))
            self._last_estop_state = estop_active
            if estop_active:
               self._emergency_stops += 1
               self.get_logger().warn(f'STOP ENGAGED: Dist {self._min_forward_dist:.2f}m < {self.stop_dist:.2f}m')
            else:
               self.get_logger().info('STOP CLEARED')

        if estop_active:
             self._commands_blocked += 1
             
        self.cmd_pub.publish(out_cmd)

    def _publish_status(self) -> None:
        """Publish status and diagnostics."""
        dist = self._min_forward_dist
        self.min_distance_pub.publish(Float32(data=dist))

        if self._is_stopped:
            status = f'HARD_STOP (dist: {dist:.2f}m)'
        elif dist < self.slow_dist:
            status = f'SLOW_ZONE (dist: {dist:.2f}m)'
        else:
            status = f'CLEAR (dist: {dist:.2f}m)'

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
