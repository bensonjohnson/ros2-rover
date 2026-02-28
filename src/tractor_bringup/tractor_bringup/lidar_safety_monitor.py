#!/usr/bin/env python3
"""LIDAR-based Directional Safety Monitor.

Enhanced safety monitor that blocks movement in specific directions based on
where obstacles are detected:
- Front sector (±N°, configurable): Blocks forward movement + speed scaling
- Left sector: Blocks left turns when moving forward/backward
- Right sector: Blocks right turns when moving forward/backward
- Rear sector: Blocks backward movement

Features:
- Gradual speed scaling in the slow zone (slow_distance → stop_distance)
- robot_front_offset compensates for LiDAR-to-bumper distance
- Configurable front sector angle to cover robot corners
- Vectorized numpy scan processing for low latency
- Hysteresis-based gating to prevent oscillation

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
        self.declare_parameter('robot_front_offset', 0.06) # LiDAR to front bumper (m)
        self.declare_parameter('front_sector_half_angle', 55.0)  # Front sector half-width (degrees)
        self.declare_parameter('stale_timeout', 0.2)      # Stale scan timeout (s)

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
        self.robot_front_offset = float(self.get_parameter('robot_front_offset').value)
        self.front_sector_half_angle = float(self.get_parameter('front_sector_half_angle').value)
        self.stale_timeout = float(self.get_parameter('stale_timeout').value)

        # Computed thresholds
        self.resume_dist = self.stop_dist + self.hysteresis
        # Pre-compute sector boundaries in radians
        self._front_half_rad = math.radians(self.front_sector_half_angle)
        self._rear_half_rad = math.radians(180.0 - self.front_sector_half_angle)

    def _log_params(self):
        self.get_logger().info(
            f'Params: Stop={self.stop_dist:.2f}m (Resume={self.resume_dist:.2f}m), '
            f'Slow={self.slow_dist:.2f}m, FrontOffset={self.robot_front_offset:.3f}m, '
            f'FrontSector=±{self.front_sector_half_angle:.0f}°, '
            f'StaleTimeout={self.stale_timeout:.1f}s'
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
            elif param.name == 'robot_front_offset':
                self.robot_front_offset = param.value
            elif param.name == 'front_sector_half_angle':
                self.front_sector_half_angle = param.value
                self._front_half_rad = math.radians(param.value)
                self._rear_half_rad = math.radians(180.0 - param.value)
            elif param.name == 'stale_timeout':
                self.stale_timeout = param.value

        self._log_params()
        return SetParametersResult(successful=True)

    def scan_callback(self, msg: LaserScan) -> None:
        """Process LIDAR scan and compute distance metrics per sector (vectorized)."""
        self._last_scan_time = self.get_clock().now()

        try:
            ranges = np.array(msg.ranges)
            angles = np.arange(len(ranges)) * msg.angle_increment + msg.angle_min

            # Wrap angles to [-π, π]
            angles = (angles + np.pi) % (2 * np.pi) - np.pi

            # Valid mask: finite and within range
            valid_mask = (
                np.isfinite(ranges) &
                (ranges > self.min_valid_range) &
                (ranges <= self.max_distance)
            )

            if not np.any(valid_mask):
                for sector in self._sector_dists:
                    self._sector_dists[sector] = self.max_distance
                self._min_overall_dist = self.max_distance
                return

            valid_ranges = ranges[valid_mask]
            valid_angles = angles[valid_mask]
            abs_angles = np.abs(valid_angles)

            # Vectorized sector classification using configurable front sector angle
            front_mask = abs_angles <= self._front_half_rad
            rear_mask = abs_angles >= self._rear_half_rad
            side_mask = ~front_mask & ~rear_mask
            left_mask = side_mask & (valid_angles > 0)
            right_mask = side_mask & (valid_angles < 0)

            max_d = self.max_distance
            offset = self.robot_front_offset

            # Compute per-sector min distances, subtracting front offset for front sector
            front_min = float(np.min(valid_ranges[front_mask]) - offset) if np.any(front_mask) else max_d
            left_min = float(np.min(valid_ranges[left_mask])) if np.any(left_mask) else max_d
            right_min = float(np.min(valid_ranges[right_mask])) if np.any(right_mask) else max_d
            rear_min = float(np.min(valid_ranges[rear_mask])) if np.any(rear_mask) else max_d

            self._sector_dists = {
                'front': max(front_min, 0.0),
                'left': left_min,
                'right': right_min,
                'rear': rear_min,
            }
            self._min_overall_dist = min(self._sector_dists.values())

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
        if time_since_scan > self.stale_timeout:
            # Stale data safety - block all movement
            out_cmd.linear.x = 0.0
            out_cmd.angular.z = 0.0
            estop_active = True
            if self._commands_received % 30 == 0:
                self.get_logger().warn(f'STALE DATA ({time_since_scan:.2f}s) - Stopping all')
        else:
            # === FRONT SECTOR: Gate forward movement with speed scaling ===
            if msg.linear.x > 0.01:  # Trying to go forward
                front_dist = self._sector_dists['front']

                if self._sector_stopped['front']:
                    if front_dist > self.resume_dist:
                        self._sector_stopped['front'] = False
                    else:
                        out_cmd.linear.x = 0.0
                        out_cmd.angular.z = 0.0
                        estop_active = True
                        blocked_sectors.append('front')
                else:
                    if front_dist < self.stop_dist:
                        self._sector_stopped['front'] = True
                        out_cmd.linear.x = 0.0
                        out_cmd.angular.z = 0.0
                        estop_active = True
                        blocked_sectors.append('front')
                    elif front_dist < self.slow_dist:
                        # Linear speed scaling: 0% at stop_dist → 100% at slow_dist
                        scale = (front_dist - self.stop_dist) / (self.slow_dist - self.stop_dist)
                        out_cmd.linear.x = msg.linear.x * scale
                        out_cmd.angular.z = msg.angular.z * scale
            
            # === REAR SECTOR: Gate backward movement ===
            # Only blocks linear.x, keeps angular.z so rover can turn away from wall
            # Does NOT set estop_active to avoid log spam on noisy rear readings
            if msg.linear.x < -0.01:  # Trying to go backward
                rear_dist = self._sector_dists['rear']

                if self._sector_stopped['rear']:
                    if rear_dist > self.resume_dist:
                        self._sector_stopped['rear'] = False
                    else:
                        out_cmd.linear.x = 0.0
                        blocked_sectors.append('rear')
                else:
                    if rear_dist < self.stop_dist:
                        self._sector_stopped['rear'] = True
                        out_cmd.linear.x = 0.0
                        blocked_sectors.append('rear')
            
            # === LEFT/RIGHT SECTORS: Only gate turns when ALSO moving forward/backward ===
            # For tank-drive robots, allow zero-turns (pure rotation) to escape tight spaces
            # Only block turns when combined with forward/backward motion that could hit obstacles

            is_zero_turn = abs(msg.linear.x) < 0.05  # Pure rotation if not moving forward/back

            if not is_zero_turn:
                # Only check side sectors when moving forward/backward

                # LEFT SECTOR: Gate left turns when moving
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

                # RIGHT SECTOR: Gate right turns when moving
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
            else:
                # Zero-turn: always clear side sector stops to allow rotation
                self._sector_stopped['left'] = False
                self._sector_stopped['right'] = False

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
        self.min_distance_pub.publish(Float32(data=float(self._sector_dists['front'])))

        # Publish all sector distances [front, left, right, right, rear]
        sector_msg = Float32MultiArray()
        sector_msg.data = [
            float(self._sector_dists['front']),
            float(self._sector_dists['left']),
            float(self._sector_dists['right']),
            float(self._sector_dists['rear'])
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
