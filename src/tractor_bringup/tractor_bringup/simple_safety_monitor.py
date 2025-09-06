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
        self.declare_parameter('forward_min_distance', 0.05)  # start checking very close-in points (meters)

        self.safety_distance = float(self.get_parameter('emergency_stop_distance').value)
        self.hard_stop_distance = float(self.get_parameter('hard_stop_distance').value)
        self.pc_topic = str(self.get_parameter('pointcloud_topic').value)
        self.in_cmd_topic = str(self.get_parameter('input_cmd_topic').value)
        self.out_cmd_topic = str(self.get_parameter('output_cmd_topic').value)
        self.detection_width = float(self.get_parameter('detection_width').value)
        self.min_obstacle_height = float(self.get_parameter('min_obstacle_height').value)
        self.ground_tolerance = float(self.get_parameter('ground_tolerance').value)
        self.forward_min_distance = float(self.get_parameter('forward_min_distance').value)

        # State
        self.latest_pc = None
        self.min_forward = 10.0
        self.last_pc_time = self.get_clock().now()
        self.pc_frame_id = ""
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
                # Use sensor_msgs_py for conversion - handle both tuple and structured outputs
                points_gen = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
                pts_list = list(points_gen)
                if len(pts_list) == 0:
                    points = np.empty((0, 3), dtype=np.float32)
                else:
                    # Try fast path if elements are tuples/lists
                    first = pts_list[0]
                    if isinstance(first, (tuple, list)) and len(first) >= 3:
                        points = np.asarray(pts_list, dtype=np.float32).reshape(-1, 3)
                    else:
                        # Structured array path; extract named fields explicitly
                        structured = np.asarray(pts_list)
                        if getattr(structured, 'dtype', None) is not None and structured.dtype.names is not None:
                            x = structured['x'].astype(np.float32, copy=False)
                            y = structured['y'].astype(np.float32, copy=False)
                            z = structured['z'].astype(np.float32, copy=False)
                            points = np.stack((x, y, z), axis=1)
                        else:
                            # Fallback: build from iteration
                            points = np.array([[p[0], p[1], p[2]] for p in pts_list], dtype=np.float32)

            self.latest_pc = points
            self.last_pc_time = self.get_clock().now()
            try:
                self.pc_frame_id = getattr(msg.header, 'frame_id', '') or ''
            except Exception:
                self.pc_frame_id = ""
            
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
                try:
                    self.pc_frame_id = getattr(msg.header, 'frame_id', '') or ''
                except Exception:
                    self.pc_frame_id = ""
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

            # Determine axis mapping from frame id
            frame = (self.pc_frame_id or "").lower()
            optical = ('optical' in frame)
            f_idx = 2 if optical else 0  # forward: z for optical, x otherwise
            l_idx = 0 if optical else 1  # lateral: x for optical, y otherwise
            v_idx = 1 if optical else 2  # vertical raw: y (down) for optical, z (up) otherwise
            v_sign = -1.0 if optical else 1.0  # convert to "up" height

            # Masks for region of interest
            forward = points[:, f_idx]
            lateral = points[:, l_idx]
            height_up = v_sign * points[:, v_idx]

            # Start very close to robot; rely on height filtering to reject ground
            forward_mask = (forward >= self.forward_min_distance) & (forward <= self.safety_distance * 4.0)
            width_mask = (lateral >= -self.detection_width * 0.5) & (lateral <= self.detection_width * 0.5)
            height_mask = (height_up >= -0.5) & (height_up <= 2.0)

            detection_mask = forward_mask & width_mask & height_mask
            if not np.any(detection_mask):
                self.min_forward = 10.0
                return

            det_height = height_up[detection_mask]
            det_forward = forward[detection_mask]

            # Estimate ground level from lowest 10% height (in "up" coordinates)
            ground_level = np.percentile(det_height, 10)
            obstacle_mask = det_height > (ground_level + self.min_obstacle_height)

            if not np.any(obstacle_mask):
                self.min_forward = 10.0
                return

            self.min_forward = float(np.min(det_forward[obstacle_mask]))

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
        # Inclusive threshold to avoid off-by-one at the boundary
        if msg.linear.x > 0.0 and self.min_forward <= self.safety_distance:
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
