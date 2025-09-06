#!/usr/bin/env python3
"""
Simple Safety Monitor
- Direct point cloud processing without BEV complexity
- Prevents forward motion when obstacles are detected
- Based on obstacle detector but simplified for safety
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String, Float32
from sensor_msgs.msg import PointCloud2
from diagnostic_msgs.msg import DiagnosticStatus, DiagnosticArray, KeyValue
import numpy as np
import struct

try:
    from sensor_msgs_py import point_cloud2
    SENSOR_MSGS_PY_AVAILABLE = True
except Exception:
    SENSOR_MSGS_PY_AVAILABLE = False


class SimpleSafetyMonitor(Node):
    def __init__(self):
        super().__init__('simple_safety_monitor')

        # Parameters
        self.declare_parameter('emergency_stop_distance', 0.2)
        self.declare_parameter('hard_stop_distance', 0.08)
        self.declare_parameter('pointcloud_topic', '/camera/camera/depth/color/points')
        self.declare_parameter('input_cmd_topic', 'cmd_vel_ai')
        self.declare_parameter('output_cmd_topic', 'cmd_vel_raw')
        self.declare_parameter('detection_width', 1.0)  # meters left/right of center
        self.declare_parameter('min_obstacle_height', 0.25)  # minimum height for obstacle (25cm)
        self.declare_parameter('ground_tolerance', 0.3)  # tolerance for ground detection (30cm)

        self.safety_distance = float(self.get_parameter('emergency_stop_distance').value)
        self.hard_stop_distance = float(self.get_parameter('hard_stop_distance').value)
        self.pc_topic = str(self.get_parameter('pointcloud_topic').value)
        self.in_cmd_topic = str(self.get_parameter('input_cmd_topic').value)
        self.out_cmd_topic = str(self.get_parameter('output_cmd_topic').value)
        self.detection_width = float(self.get_parameter('detection_width').value)
        self.min_obstacle_height = float(self.get_parameter('min_obstacle_height').value)
        self.ground_tolerance = float(self.get_parameter('ground_tolerance').value)

        # State
        self.latest_pc = None
        self.min_forward = 10.0
        self.last_pc_time = self.get_clock().now()
        self.commands_received = 0
        self.commands_blocked = 0
        self.emergency_stops = 0
        self.sensor_failures = 0
        self.last_cmd_time = None

        # Subscribers - increase queue size for better real-time performance
        self.pc_sub = self.create_subscription(PointCloud2, self.pc_topic, self.pc_callback, 1)  # Keep only latest
        self.cmd_sub = self.create_subscription(Twist, self.in_cmd_topic, self.cmd_callback, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, self.out_cmd_topic, 10)
        self.estop_pub = self.create_publisher(Bool, 'emergency_stop', 10)
        
        # Diagnostic publishers
        self.safety_status_pub = self.create_publisher(String, 'safety_monitor_status', 5)
        self.min_distance_pub = self.create_publisher(Float32, 'min_forward_distance', 5)
        self.diagnostics_pub = self.create_publisher(DiagnosticArray, 'diagnostics', 5)

        # Timers - increase safety monitor rate for better responsiveness
        self.create_timer(0.05, self._update_min_distance)  # 20Hz for faster response
        self.create_timer(1.0, self._publish_diagnostics)
        self.create_timer(5.0, self._verify_connections)

        self.get_logger().info(f"Simple Safety Monitor active. safety_distance={self.safety_distance} hard_stop={self.hard_stop_distance}")
        self.get_logger().info(f"Input topic: {self.in_cmd_topic}, Output topic: {self.out_cmd_topic}")
        self.get_logger().info(f"Point cloud topic: {self.pc_topic}")

    def pc_callback(self, msg: PointCloud2):
        """Convert point cloud to simple numpy array"""
        try:
            if not SENSOR_MSGS_PY_AVAILABLE:
                self.sensor_failures += 1
                self.get_logger().error("sensor_msgs_py not available - using direct parsing")
                # Fall back to direct parsing
                points = self._parse_pointcloud2_direct(msg)
            else:
                # Use sensor_msgs_py for conversion - handle structured array properly
                points_gen = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
                points_list = list(points_gen)
                
                if len(points_list) > 0:
                    # Convert structured array to regular array
                    points_structured = np.array(points_list)
                    # Extract x, y, z fields into regular float32 array
                    points = np.column_stack((
                        points_structured['x'].astype(np.float32),
                        points_structured['y'].astype(np.float32), 
                        points_structured['z'].astype(np.float32)
                    ))
                else:
                    points = np.array([], dtype=np.float32).reshape(0, 3)

            self.latest_pc = points
            self.last_pc_time = self.get_clock().now()
            
        except Exception as e:
            self.latest_pc = None
            self.sensor_failures += 1
            self.get_logger().error(f"Point cloud processing failed: {e}")
            # Try fallback to direct parsing if sensor_msgs_py fails
            try:
                points = self._parse_pointcloud2_direct(msg)
                self.latest_pc = points
                self.last_pc_time = self.get_clock().now()
                self.get_logger().info("Fallback to direct parsing successful")
            except Exception as e2:
                self.get_logger().error(f"Direct parsing also failed: {e2}")

    def _parse_pointcloud2_direct(self, msg: PointCloud2):
        """Direct parsing of PointCloud2 message without sensor_msgs_py"""
        # Get point step and row step
        point_step = msg.point_step
        row_step = msg.row_step
        
        # Find x, y, z field offsets
        x_offset = y_offset = z_offset = None
        for field in msg.fields:
            if field.name == 'x':
                x_offset = field.offset
            elif field.name == 'y':
                y_offset = field.offset
            elif field.name == 'z':
                z_offset = field.offset
        
        if x_offset is None or y_offset is None or z_offset is None:
            raise ValueError("Point cloud missing x, y, or z fields")
        
        # Extract points
        points = []
        data = msg.data
        
        # Process each point
        for i in range(0, len(data), point_step):
            if i + 12 > len(data):  # Need at least 12 bytes for x,y,z (4 bytes each)
                break
                
            # Extract x, y, z as floats
            x = struct.unpack('<f', data[i + x_offset:i + x_offset + 4])[0]
            y = struct.unpack('<f', data[i + y_offset:i + y_offset + 4])[0]
            z = struct.unpack('<f', data[i + z_offset:i + z_offset + 4])[0]
            
            # Skip NaN values
            if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                points.append([x, y, z])
        
        return np.array(points, dtype=np.float32)

    def _update_min_distance(self):
        """Calculate minimum forward distance to obstacles"""
        current_time = self.get_clock().now()
        
        # Check data freshness first - be more strict for moving robot
        data_age_ns = 0
        if self.last_pc_time:
            data_age_ns = (current_time - self.last_pc_time).nanoseconds
            
        # If data is older than 200ms (0.2 seconds), be very conservative
        if data_age_ns > 2e8:  # 200 milliseconds
            self.min_forward = 0.1  # Assume obstacle very close when data is stale
            if data_age_ns > 1e9:  # 1 second - warn user
                self.get_logger().warn(f"Point cloud data is stale ({data_age_ns/1e9:.1f}s old) - assuming obstacles")
            return
        
        if self.latest_pc is None or len(self.latest_pc) == 0:
            self.min_forward = 0.1  # Conservative - assume obstacles when no data
            return

        try:
            points = self.latest_pc
            
            # Filter to detection area (forward cone in front of robot)
            # X: forward (0.2m to safety_distance*4) - skip immediate ground in front
            # Y: left/right (-detection_width/2 to +detection_width/2)  
            # Z: reasonable height range (-0.5m to 2.0m)
            
            forward_mask = (points[:, 0] > 0.2) & (points[:, 0] <= self.safety_distance * 4)  # Start 20cm ahead
            width_mask = (points[:, 1] >= -self.detection_width/2) & (points[:, 1] <= self.detection_width/2)
            height_mask = (points[:, 2] >= -0.5) & (points[:, 2] <= 2.0)  # Reasonable height range
            
            detection_mask = forward_mask & width_mask & height_mask
            detection_points = points[detection_mask]
            
            if len(detection_points) == 0:
                self.min_forward = 10.0
                return
            
            # Improved obstacle detection - points significantly above likely ground level
            # Estimate ground level from lowest 10% of points in detection area
            z_values = detection_points[:, 2]
            ground_level = np.percentile(z_values, 10)  # 10th percentile as ground estimate
            
            # Obstacles are points well above the estimated ground level
            obstacle_mask = detection_points[:, 2] > (ground_level + self.min_obstacle_height)
            obstacle_points = detection_points[obstacle_mask]
            
            if len(obstacle_points) == 0:
                self.min_forward = 10.0
                return
            
            # Find minimum forward distance to any obstacle
            forward_distances = obstacle_points[:, 0]
            self.min_forward = float(np.min(forward_distances))
            
            # Publish distance for monitoring
            distance_msg = Float32()
            distance_msg.data = self.min_forward
            self.min_distance_pub.publish(distance_msg)
            
        except Exception as e:
            self.min_forward = 10.0
            self.sensor_failures += 1
            self.get_logger().error(f"Distance calculation failed: {e}")

    def cmd_callback(self, msg: Twist):
        """Gate velocity commands based on obstacle detection"""
        # Track command reception
        self.commands_received += 1
        self.last_cmd_time = self.get_clock().now()
        
        # Create output command
        out = Twist()
        out.angular.z = msg.angular.z
        
        # Emergency stop handling - allow escape turns
        hard = (self.min_forward < self.hard_stop_distance)
        if hard:
            self.emergency_stops += 1
            out.linear.x = 0.0
            # Allow turning to escape, but reduce speed for safety
            if abs(msg.angular.z) > 0.1:  # Significant turn command
                out.angular.z = msg.angular.z * 0.5  # Reduce turn speed
                self.get_logger().warn(f"EMERGENCY: Obstacle at {self.min_forward:.2f}m - allowing escape turn")
            else:
                out.angular.z = 0.0
                self.get_logger().warn(f"EMERGENCY: Obstacle at {self.min_forward:.2f}m - full stop")
            self.estop_pub.publish(Bool(data=True))
            self.cmd_pub.publish(out)
            return
        else:
            # Clear emergency stop
            self.estop_pub.publish(Bool(data=False))
        
        # Soft safety - block forward motion only
        if msg.linear.x > 0.0 and self.min_forward < self.safety_distance:
            out.linear.x = 0.0
            self.commands_blocked += 1
            self.get_logger().info(f"Blocking forward motion - obstacle at {self.min_forward:.2f}m")
        else:
            out.linear.x = msg.linear.x
        
        self.cmd_pub.publish(out)

    def _publish_diagnostics(self):
        """Publish comprehensive safety diagnostics"""
        # Safety status message
        status = "ACTIVE"
        if self.latest_pc is None:
            status = "NO_SENSOR_DATA"
        elif self.min_forward < self.hard_stop_distance:
            status = "EMERGENCY_STOP"
        elif self.min_forward < self.safety_distance:
            status = "SAFETY_BLOCKED"
        
        status_msg = String()
        status_msg.data = f"Safety: {status}, Distance: {self.min_forward:.2f}m, Commands: {self.commands_received}, Blocked: {self.commands_blocked}"
        self.safety_status_pub.publish(status_msg)
        
        # Detailed diagnostics
        diag_array = DiagnosticArray()
        diag_array.header.stamp = self.get_clock().now().to_msg()
        
        safety_diag = DiagnosticStatus()
        safety_diag.name = "simple_safety_monitor"
        safety_diag.hardware_id = "simple_safety_monitor"
        
        if self.latest_pc is None:
            safety_diag.level = DiagnosticStatus.ERROR
            safety_diag.message = "No point cloud data available"
        elif self.sensor_failures > 0:
            safety_diag.level = DiagnosticStatus.WARN
            safety_diag.message = f"Sensor processing failures: {self.sensor_failures}"
        elif self.min_forward < self.safety_distance:
            safety_diag.level = DiagnosticStatus.WARN
            safety_diag.message = f"Obstacle detected at {self.min_forward:.2f}m"
        else:
            safety_diag.level = DiagnosticStatus.OK
            safety_diag.message = "Safety monitor operating normally"
        
        safety_diag.values = [
            KeyValue(key="min_forward_distance", value=f"{self.min_forward:.3f}"),
            KeyValue(key="safety_distance", value=f"{self.safety_distance:.3f}"),
            KeyValue(key="hard_stop_distance", value=f"{self.hard_stop_distance:.3f}"),
            KeyValue(key="commands_received", value=str(self.commands_received)),
            KeyValue(key="commands_blocked", value=str(self.commands_blocked)),
            KeyValue(key="emergency_stops", value=str(self.emergency_stops)),
            KeyValue(key="sensor_failures", value=str(self.sensor_failures)),
        ]
        
        # Check for stale commands
        if self.last_cmd_time:
            cmd_age = (self.get_clock().now() - self.last_cmd_time).nanoseconds / 1e9
            if cmd_age > 2.0:
                safety_diag.level = max(safety_diag.level, DiagnosticStatus.WARN)
                safety_diag.message += f" | No commands for {cmd_age:.1f}s"
        
        diag_array.status = [safety_diag]
        self.diagnostics_pub.publish(diag_array)

    def _verify_connections(self):
        """Verify topic connections"""
        # Check connections
        pc_info = self.get_subscriptions_info_by_topic(self.pc_topic)
        if not pc_info:
            self.get_logger().warn(f"Point cloud topic {self.pc_topic} has no publishers!")
        
        cmd_info = self.get_subscriptions_info_by_topic(self.in_cmd_topic)
        if not cmd_info:
            self.get_logger().warn(f"Input command topic {self.in_cmd_topic} has no publishers - AI may not be connected!")
        
        # Status logging
        if self.commands_received == 0:
            self.get_logger().warn(f"No commands received on {self.in_cmd_topic} yet - waiting for AI node...")
        elif self.commands_received % 100 == 0:
            self.get_logger().info(f"Safety monitor healthy: {self.commands_received} commands processed, {self.commands_blocked} blocked")


def main(args=None):
    rclpy.init(args=args)
    node = SimpleSafetyMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()