#!/usr/bin/env python3
"""
Simple BEV Safety Monitor
- Subscribes to point cloud and AI velocity commands (cmd_vel_ai)
- Prevents forward motion when obstacle is within safety distance
- Allows turning and backing up; optional hard E-stop if extremely close
- Publishes gated commands to cmd_vel_raw (for velocity controller)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String, Float32
from sensor_msgs.msg import PointCloud2, Image
from rclpy.qos import qos_profile_sensor_data
from diagnostic_msgs.msg import DiagnosticStatus, DiagnosticArray, KeyValue
import numpy as np

try:
    from sensor_msgs_py import point_cloud2
    SENSOR_MSGS_PY_AVAILABLE = True
except Exception:
    SENSOR_MSGS_PY_AVAILABLE = False

from .bev_generator import BEVGenerator


class SimpleSafetyMonitorBEV(Node):
    def __init__(self):
        super().__init__('simple_safety_monitor_bev')

        # Parameters
        self.declare_parameter('emergency_stop_distance', 0.2)
        self.declare_parameter('hard_stop_distance', 0.08)  # absolute stop threshold
        self.declare_parameter('pointcloud_topic', '/camera/camera/depth/color/points')
        self.declare_parameter('input_cmd_topic', 'cmd_vel_ai')
        self.declare_parameter('output_cmd_topic', 'cmd_vel_raw')
        self.declare_parameter('use_shared_bev', True)
        self.declare_parameter('bev_image_topic', '/bev/image')
        self.declare_parameter('bev_freshness_timeout', 0.5)

        self.safety_distance = float(self.get_parameter('emergency_stop_distance').value)
        self.hard_stop_distance = float(self.get_parameter('hard_stop_distance').value)
        self.pc_topic = str(self.get_parameter('pointcloud_topic').value)
        self.in_cmd_topic = str(self.get_parameter('input_cmd_topic').value)
        self.out_cmd_topic = str(self.get_parameter('output_cmd_topic').value)
        self.use_shared_bev = bool(self.get_parameter('use_shared_bev').value)
        self.bev_image_topic = str(self.get_parameter('bev_image_topic').value)
        self.bev_freshness_timeout = float(self.get_parameter('bev_freshness_timeout').value)

        # BEV for fast forward-distance checks
        self.bev = BEVGenerator(bev_size=(200, 200), bev_range=(10.0, 10.0), height_channels=(0.2, 1.0), enable_ground_removal=True)

        # State
        self.latest_pc = None
        self.min_forward = 10.0
        self.last_pc_time = self.get_clock().now()
        self.commands_received = 0
        self.commands_blocked = 0
        self.emergency_stops = 0
        self.sensor_failures = 0
        self.last_cmd_time = None
        # Shared BEV cache
        self.latest_bev = None
        self.last_bev_time = None

        # Subs/Pubs
        self.pc_sub = self.create_subscription(PointCloud2, self.pc_topic, self.pc_callback, qos_profile_sensor_data)
        self.cmd_sub = self.create_subscription(Twist, self.in_cmd_topic, self.cmd_callback, 10)
        if self.use_shared_bev:
            self.bev_sub = self.create_subscription(Image, self.bev_image_topic, self.bev_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, self.out_cmd_topic, 10)
        self.estop_pub = self.create_publisher(Bool, 'emergency_stop', 10)
        
        # Diagnostic publishers
        self.safety_status_pub = self.create_publisher(String, 'safety_monitor_status', 5)
        self.min_distance_pub = self.create_publisher(Float32, 'min_forward_distance', 5)
        self.diagnostics_pub = self.create_publisher(DiagnosticArray, 'diagnostics', 5)

        # Timer to recompute distance if needed
        self.create_timer(0.1, self._update_min_distance)
        # Timer for diagnostics
        self.create_timer(1.0, self._publish_diagnostics)
        # Timer for connection verification
        self.create_timer(5.0, self._verify_connections)

        self.get_logger().info(f"Safety monitor active. safety_distance={self.safety_distance} hard_stop={self.hard_stop_distance}")
        self.get_logger().info(f"Input topic: {self.in_cmd_topic}, Output topic: {self.out_cmd_topic}")
        self.get_logger().info(f"Point cloud topic: {self.pc_topic}")
        if self.use_shared_bev:
            self.get_logger().info(f"Shared BEV topic: {self.bev_image_topic}")

    def pc_callback(self, msg: PointCloud2):
        # Convert to Nx3 float32 points (x forward, y left, z up) using robust path
        try:
            if not SENSOR_MSGS_PY_AVAILABLE:
                self.sensor_failures += 1
                self.get_logger().warn("sensor_msgs_py not available - skipping point cloud fallback")
                return
            pts_list = list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
            if not pts_list:
                self.latest_pc = np.zeros((0, 3), dtype=np.float32)
                self.last_pc_time = self.get_clock().now()
                return
            first = pts_list[0]
            if isinstance(first, (tuple, list)) and len(first) >= 3:
                pts = np.asarray(pts_list, dtype=np.float32).reshape(-1, 3)
            else:
                structured = np.asarray(pts_list)
                if getattr(structured, 'dtype', None) is not None and structured.dtype.names is not None:
                    x = structured['x'].astype(np.float32, copy=False)
                    y = structured['y'].astype(np.float32, copy=False)
                    z = structured['z'].astype(np.float32, copy=False)
                    pts = np.stack((x, y, z), axis=1)
                else:
                    pts = np.array([[p[0], p[1], p[2]] for p in pts_list], dtype=np.float32)
            self.latest_pc = pts
            self.last_pc_time = self.get_clock().now()
        except Exception as e:
            self.latest_pc = None
            self.sensor_failures += 1
            self.get_logger().error(f"Point cloud processing failed: {e}")

    def _update_min_distance(self):
        current_time = self.get_clock().now()
        # Prefer shared BEV if available and fresh
        if self.use_shared_bev and self.latest_bev is not None and self.last_bev_time is not None:
            age = (current_time - self.last_bev_time).nanoseconds / 1e9
            if age <= self.bev_freshness_timeout:
                try:
                    bev_img = self.latest_bev
                    h, w, _ = bev_img.shape
                    conf = bev_img[:, :, 3]
                    low = bev_img[:, :, 2]
                    occ = (conf > 0.25) | (low > 0.1)
                    front_rows = slice(int(h*2/3), h)
                    center_cols = slice(int(w/3), int(2*w/3))
                    mask = occ[front_rows, center_cols]
                    ys, xs = np.where(mask)
                    if ys.size == 0:
                        self.min_forward = 10.0
                    else:
                        px = ys.astype(np.float32)
                        x_range = 10.0
                        x_m = (px / float(h)) * (2.0 * x_range) - x_range
                        x_m = x_m[x_m >= 0.0]
                        self.min_forward = float(np.min(x_m)) if x_m.size else 10.0
                    self.min_distance_pub.publish(Float32(data=self.min_forward))
                    return
                except Exception as e:
                    self.get_logger().debug(f"Shared BEV distance compute failed: {e}")

        if self.latest_pc is None or self.latest_pc.size == 0:
            self.min_forward = 10.0
            # Check if point cloud is too old
            if self.last_pc_time and (current_time - self.last_pc_time).nanoseconds > 2e9:
                self.get_logger().warn("Point cloud data is stale - safety may be compromised")
            return
        try:
            bev_img = self.bev.generate_bev(self.latest_pc)
            h, w, _ = bev_img.shape
            conf = bev_img[:, :, 3]
            low = bev_img[:, :, 2]
            occ = (conf > 0.25) | (low > 0.1)
            front_rows = slice(int(h*2/3), h)
            center_cols = slice(int(w/3), int(2*w/3))
            mask = occ[front_rows, center_cols]
            ys, xs = np.where(mask)
            if ys.size == 0:
                self.min_forward = 10.0
                return
            px = ys.astype(np.float32)
            x_range = float(self.bev.x_range)
            x_m = (px / float(h)) * (2.0 * x_range) - x_range
            x_m = x_m[x_m >= 0.0]
            self.min_forward = float(np.min(x_m)) if x_m.size else 10.0
            
            # Publish distance for monitoring
            distance_msg = Float32()
            distance_msg.data = self.min_forward
            self.min_distance_pub.publish(distance_msg)
            
        except Exception as e:
            self.min_forward = 10.0
            self.sensor_failures += 1
            self.get_logger().error(f"Distance calculation failed: {e}")

    def cmd_callback(self, msg: Twist):
        # Track command reception
        self.commands_received += 1
        self.last_cmd_time = self.get_clock().now()
        
        # Gate forward motion if obstacle is too close
        out = Twist()
        out.angular.z = msg.angular.z
        
        # Emergency stop handling - allow escape turns
        hard = (self.min_forward < self.hard_stop_distance)
        if hard:
            self.emergency_stops += 1
            out.linear.x = 0.0
            # Allow turning to escape, but reduce angular velocity for safety
            if abs(msg.angular.z) > 0.1:  # If significant turn command
                out.angular.z = msg.angular.z * 0.5  # Reduce turn speed by 50%
                self.get_logger().warn(f"EMERGENCY: Obstacle at {self.min_forward:.2f}m - allowing escape turn")
            else:
                out.angular.z = 0.0
                self.get_logger().warn(f"EMERGENCY: Obstacle at {self.min_forward:.2f}m - full stop")
            self.estop_pub.publish(Bool(data=True))
            self.cmd_pub.publish(out)
            return
        else:
            # Clear estop if previously set
            self.estop_pub.publish(Bool(data=False))
        
        # Soft safety - block forward motion only
        if msg.linear.x > 0.0 and self.min_forward < self.safety_distance:
            # Block forward, allow turning/backward
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
        if (self.latest_pc is None) and (not (self.use_shared_bev and self.latest_bev is not None)):
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
        
        # Safety monitor status
        safety_diag = DiagnosticStatus()
        safety_diag.name = "safety_monitor_bev"
        safety_diag.hardware_id = "simple_safety_monitor_bev"
        
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
        
        # Add diagnostic values
        safety_diag.values = [
            KeyValue(key="min_forward_distance", value=f"{self.min_forward:.3f}"),
            KeyValue(key="safety_distance", value=f"{self.safety_distance:.3f}"),
            KeyValue(key="hard_stop_distance", value=f"{self.hard_stop_distance:.3f}"),
            KeyValue(key="commands_received", value=str(self.commands_received)),
            KeyValue(key="commands_blocked", value=str(self.commands_blocked)),
            KeyValue(key="emergency_stops", value=str(self.emergency_stops)),
            KeyValue(key="sensor_failures", value=str(self.sensor_failures)),
        ]
        
        # Check for stale command data
        if self.last_cmd_time:
            cmd_age = (self.get_clock().now() - self.last_cmd_time).nanoseconds / 1e9
            if cmd_age > 2.0:
                safety_diag.level = max(safety_diag.level, DiagnosticStatus.WARN)
                safety_diag.message += f" | No commands for {cmd_age:.1f}s"
        
        diag_array.status = [safety_diag]
        self.diagnostics_pub.publish(diag_array)

    def bev_callback(self, msg: Image):
        try:
            from cv_bridge import CvBridge
            bridge = CvBridge()
            bev = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if isinstance(bev, np.ndarray) and bev.ndim == 3:
                self.latest_bev = bev.astype(np.float32, copy=False)
                self.last_bev_time = self.get_clock().now()
        except Exception as e:
            self.get_logger().debug(f"BEV image conversion failed: {e}")

    def _verify_connections(self):
        """Verify that required topics are connected"""
        # Check if we're receiving point cloud data
        pc_info = self.get_subscriptions_info_by_topic(self.pc_topic)
        if not pc_info:
            self.get_logger().warn(f"Point cloud topic {self.pc_topic} has no publishers - safety compromised!")
        
        # Check if we're receiving command data
        cmd_info = self.get_subscriptions_info_by_topic(self.in_cmd_topic)
        if not cmd_info:
            self.get_logger().warn(f"Input command topic {self.in_cmd_topic} has no publishers - AI may not be connected!")
        
        # Check if our output is being consumed
        out_info = self.get_publishers_info_by_topic(self.out_cmd_topic)
        if not any(pub.node_name == self.get_name() for pub in out_info):
            self.get_logger().warn(f"Output topic {self.out_cmd_topic} may not have subscribers!")
        
        # Log connection status periodically
        if self.commands_received == 0:
            self.get_logger().warn(f"No commands received on {self.in_cmd_topic} yet - waiting for AI node...")
        elif self.commands_received % 100 == 0:  # Every 100 commands
            self.get_logger().info(f"Safety monitor healthy: {self.commands_received} commands processed, {self.commands_blocked} blocked")


def main(args=None):
    rclpy.init(args=args)
    node = SimpleSafetyMonitorBEV()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
