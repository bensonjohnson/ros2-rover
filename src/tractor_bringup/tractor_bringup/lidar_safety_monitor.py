#!/usr/bin/env python3
"""LIDAR-based Directional Safety Monitor.

Enhanced safety monitor that blocks movement in specific directions based on
where obstacles are detected:
- Front sector (±45°): Blocks forward movement
- Left sector (45° to 135°): Blocks left turns (negative angular.z)  
- Right sector (-45° to -135°): Blocks right turns (positive angular.z)
- Rear sector (±135° to ±180°): Blocks backward movement

For tank-steer rovers, this allows intelligent avoidance - the rover can
still turn away from an obstacle even when it can't go forward.
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import SetParametersResult

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32, Float32MultiArray, String


class LidarSafetyMonitor(Node):
    """Directional safety monitor using LIDAR sector-based detection."""

    def __init__(self) -> None:
        super().__init__('lidar_safety_monitor')

        # Parameters
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('input_cmd_topic', 'cmd_vel_teleop')
        self.declare_parameter('output_cmd_topic', 'cmd_vel_raw')
        
        # Safety thresholds
        self.declare_parameter('stop_distance', 0.20)     # Hard stop distance (200mm)
        self.declare_parameter('slow_distance', 0.30)     # Start slowing down (300mm)
        self.declare_parameter('hysteresis', 0.05)        # Gap to resume
        self.declare_parameter('max_eval_distance', 4.0)  # Ignore ranges > 4m
        self.declare_parameter('min_valid_range', 0.10)   # Ignore ranges < 10cm (noise/self-hits)

        # Load parameters
        self._load_parameters()
        self.add_on_set_parameters_callback(self._parameters_callback)

        # Sector distances (min distance in each 90° sector)
        # Sectors: Front (±45°), Left (45° to 135°), Right (-45° to -135°), Rear (±135° to ±180°)
        self._sector_dists = {
            'front': self.max_distance,
            'left': self.max_distance,
            'right': self.max_distance,
            'rear': self.max_distance
        }
        
        # Overall minimum for status
        self._min_overall_dist = self.max_distance
        
        # State
        self._latest_cmd = None
        self._commands_received = 0
        self._commands_blocked = 0
        self._emergency_stops = 0
        
        # Hysteresis state per sector
        self._sector_stopped = {
            'front': False,
            'left': False,
            'right': False,
            'rear': False
        }
        
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
        self.sector_dist_pub = self.create_publisher(Float32MultiArray, 'sector_distances', 5)

        # Timer for status updates
        self.create_timer(0.1, self._publish_status)

        self.get_logger().info('Directional LIDAR safety monitor initialized (Sector Mode)')
        self._log_params()

    def _load_parameters(self):
        """Load parameter values from ROS system."""
        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.input_cmd_topic = str(self.get_parameter('input_cmd_topic').value)
        self.output_cmd_topic = str(self.get_parameter('output_cmd_topic').value)
        
        self.stop_dist = float(self.get_parameter('stop_distance').value)
        self.slow_dist = float(self.get_parameter('slow_distance').value)
        self.hysteresis = float(self.get_parameter('hysteresis').value)
        self.max_distance = float(self.get_parameter('max_eval_distance').value)
        self.min_valid_range = float(self.get_parameter('min_valid_range').value)

        # Computed thresholds
        self.resume_dist = self.stop_dist + self.hysteresis

    def _log_params(self):
        self.get_logger().info(
            f'Params: Stop={self.stop_dist:.2f}m (Resume={self.resume_dist:.2f}m), '
            f'Slow={self.slow_dist:.2f}m, MaxRange={self.max_distance:.1f}m'
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
            
        self._log_params()
        return SetParametersResult(successful=True)

    def _get_sector(self, angle_rad: float) -> str:
        """Determine which sector an angle belongs to.
        
        Args:
            angle_rad: Angle in radians [-π, π], 0 = forward, positive = left
            
        Returns:
            Sector name: 'front', 'left', 'right', or 'rear'
        """
        angle_deg = math.degrees(angle_rad)
        
        # Normalize to [-180, 180]
        while angle_deg > 180:
            angle_deg -= 360
        while angle_deg < -180:
            angle_deg += 360
            
        if -45 <= angle_deg <= 45:
            return 'front'
        elif 45 < angle_deg <= 135:
            return 'left'
        elif -135 <= angle_deg < -45:
            return 'right'
        else:
            return 'rear'

    def scan_callback(self, msg: LaserScan) -> None:
        """Process LIDAR scan and compute distance metrics per sector."""
        self._last_scan_time = self.get_clock().now()
        
        try:
            # Convert to numpy arrays
            ranges = np.array(msg.ranges)
            angles = np.arange(len(ranges)) * msg.angle_increment + msg.angle_min
            
            # Wrap angles to [-π, π]
            angles = (angles + np.pi) % (2 * np.pi) - np.pi
            
            # Valid mask: reasonable range
            valid_mask = (
                np.isfinite(ranges) & 
                (ranges > self.min_valid_range) & 
                (ranges <= self.max_distance)
            )
            
            if not np.any(valid_mask):
                # No valid data - assume all clear  
                for sector in self._sector_dists:
                    self._sector_dists[sector] = self.max_distance
                self._min_overall_dist = self.max_distance
                return

            valid_ranges = ranges[valid_mask]
            valid_angles = angles[valid_mask]
            
            # Initialize sector minimums
            sector_mins = {
                'front': self.max_distance,
                'left': self.max_distance,
                'right': self.max_distance,
                'rear': self.max_distance
            }
            
            # Assign each point to a sector and track minimum
            for r, theta in zip(valid_ranges, valid_angles):
                sector = self._get_sector(theta)
                if r < sector_mins[sector]:
                    sector_mins[sector] = r
            
            self._sector_dists = sector_mins
            self._min_overall_dist = min(sector_mins.values())

        except Exception as exc:
            self.get_logger().warn(f'LIDAR processing error: {exc}')

    def cmd_callback(self, msg: Twist) -> None:
        """Process velocity command and apply directional safety gating."""
        self._latest_cmd = msg
        self._commands_received += 1

        # Copy command (will modify as needed)
        out_cmd = Twist()
        out_cmd.linear.x = msg.linear.x
        out_cmd.linear.y = msg.linear.y
        out_cmd.linear.z = msg.linear.z
        out_cmd.angular.x = msg.angular.x
        out_cmd.angular.y = msg.angular.y
        out_cmd.angular.z = msg.angular.z
        
        estop_active = False
        blocked_sectors = []

        # Check for stale data
        time_since_scan = (self.get_clock().now() - self._last_scan_time).nanoseconds / 1e9
        if time_since_scan > 0.5:
            # Stale data safety - block all movement
            out_cmd.linear.x = 0.0
            out_cmd.angular.z = 0.0
            estop_active = True
            if self._commands_received % 30 == 0:
                self.get_logger().warn(f'STALE DATA ({time_since_scan:.2f}s) - Stopping all')
        else:
            # === FRONT SECTOR: Gate forward movement ===
            if msg.linear.x > 0.01:  # Trying to go forward
                front_dist = self._sector_dists['front']
                
                if self._sector_stopped['front']:
                    # Currently stopped - need to clear resume threshold
                    if front_dist > self.resume_dist:
                        self._sector_stopped['front'] = False
                    else:
                        out_cmd.linear.x = 0.0
                        estop_active = True
                        blocked_sectors.append('front')
                else:
                    # Not stopped - check if should stop
                    if front_dist < self.stop_dist:
                        self._sector_stopped['front'] = True
                        out_cmd.linear.x = 0.0
                        estop_active = True
                        blocked_sectors.append('front')
                    elif front_dist < self.slow_dist:
                        # Scale speed based on distance
                        range_span = self.slow_dist - self.stop_dist
                        if range_span > 0:
                            factor = (front_dist - self.stop_dist) / range_span
                            factor = np.clip(factor, 0.1, 1.0)
                            out_cmd.linear.x = msg.linear.x * factor
            
            # === REAR SECTOR: Gate backward movement ===
            if msg.linear.x < -0.01:  # Trying to go backward
                rear_dist = self._sector_dists['rear']
                
                if self._sector_stopped['rear']:
                    if rear_dist > self.resume_dist:
                        self._sector_stopped['rear'] = False
                    else:
                        out_cmd.linear.x = 0.0
                        estop_active = True
                        blocked_sectors.append('rear')
                else:
                    if rear_dist < self.stop_dist:
                        self._sector_stopped['rear'] = True
                        out_cmd.linear.x = 0.0
                        estop_active = True
                        blocked_sectors.append('rear')
                    elif rear_dist < self.slow_dist:
                        range_span = self.slow_dist - self.stop_dist
                        if range_span > 0:
                            factor = (rear_dist - self.stop_dist) / range_span
                            factor = np.clip(factor, 0.1, 1.0)
                            out_cmd.linear.x = msg.linear.x * factor
            
            # === LEFT SECTOR: Gate left turns (positive angular.z for tank = turn left) ===
            if msg.angular.z > 0.05:  # Trying to turn left
                left_dist = self._sector_dists['left']
                
                if self._sector_stopped['left']:
                    if left_dist > self.resume_dist:
                        self._sector_stopped['left'] = False
                    else:
                        out_cmd.angular.z = 0.0
                        blocked_sectors.append('left')
                else:
                    if left_dist < self.stop_dist:
                        self._sector_stopped['left'] = True
                        out_cmd.angular.z = 0.0
                        blocked_sectors.append('left')
                    elif left_dist < self.slow_dist:
                        range_span = self.slow_dist - self.stop_dist
                        if range_span > 0:
                            factor = (left_dist - self.stop_dist) / range_span
                            factor = np.clip(factor, 0.2, 1.0)
                            out_cmd.angular.z = msg.angular.z * factor
            
            # === RIGHT SECTOR: Gate right turns (negative angular.z for tank = turn right) ===
            if msg.angular.z < -0.05:  # Trying to turn right
                right_dist = self._sector_dists['right']
                
                if self._sector_stopped['right']:
                    if right_dist > self.resume_dist:
                        self._sector_stopped['right'] = False
                    else:
                        out_cmd.angular.z = 0.0
                        blocked_sectors.append('right')
                else:
                    if right_dist < self.stop_dist:
                        self._sector_stopped['right'] = True
                        out_cmd.angular.z = 0.0
                        blocked_sectors.append('right')
                    elif right_dist < self.slow_dist:
                        range_span = self.slow_dist - self.stop_dist
                        if range_span > 0:
                            factor = (right_dist - self.stop_dist) / range_span
                            factor = np.clip(factor, 0.2, 1.0)
                            out_cmd.angular.z = msg.angular.z * factor

        # Publish estop state changes
        if estop_active != self._last_estop_state:
            self.estop_pub.publish(Bool(data=estop_active))
            self._last_estop_state = estop_active
            if estop_active:
                self._emergency_stops += 1
                sectors_str = ', '.join(blocked_sectors) if blocked_sectors else 'all'
                self.get_logger().warn(f'STOP ENGAGED: Blocked sectors: {sectors_str}')
            else:
                self.get_logger().info('STOP CLEARED')

        if blocked_sectors:
            self._commands_blocked += 1
             
        self.cmd_pub.publish(out_cmd)

    def _publish_status(self) -> None:
        """Publish status and diagnostics."""
        # Publish overall front distance for backward compatibility
        self.min_distance_pub.publish(Float32(data=self._sector_dists['front']))
        
        # Publish all sector distances [front, left, right, rear]
        sector_msg = Float32MultiArray()
        sector_msg.data = [
            self._sector_dists['front'],
            self._sector_dists['left'],
            self._sector_dists['right'],
            self._sector_dists['rear']
        ]
        self.sector_dist_pub.publish(sector_msg)

        # Build status string
        blocked = [s for s, stopped in self._sector_stopped.items() if stopped]
        if blocked:
            status = f'BLOCKED: {", ".join(blocked)} | F:{self._sector_dists["front"]:.2f} L:{self._sector_dists["left"]:.2f} R:{self._sector_dists["right"]:.2f} B:{self._sector_dists["rear"]:.2f}'
        elif self._min_overall_dist < self.slow_dist:
            status = f'SLOW_ZONE | F:{self._sector_dists["front"]:.2f} L:{self._sector_dists["left"]:.2f} R:{self._sector_dists["right"]:.2f} B:{self._sector_dists["rear"]:.2f}'
        else:
            status = f'CLEAR | F:{self._sector_dists["front"]:.2f} L:{self._sector_dists["left"]:.2f} R:{self._sector_dists["right"]:.2f} B:{self._sector_dists["rear"]:.2f}'

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
