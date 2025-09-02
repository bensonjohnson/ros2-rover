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
from std_msgs.msg import Bool
from sensor_msgs.msg import PointCloud2
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

        self.safety_distance = float(self.get_parameter('emergency_stop_distance').value)
        self.hard_stop_distance = float(self.get_parameter('hard_stop_distance').value)
        self.pc_topic = str(self.get_parameter('pointcloud_topic').value)
        self.in_cmd_topic = str(self.get_parameter('input_cmd_topic').value)
        self.out_cmd_topic = str(self.get_parameter('output_cmd_topic').value)

        # BEV for fast forward-distance checks
        self.bev = BEVGenerator(bev_size=(200, 200), bev_range=(10.0, 10.0), height_channels=(0.2, 1.0), enable_ground_removal=True)

        # State
        self.latest_pc = None
        self.min_forward = 10.0

        # Subs/Pubs
        self.pc_sub = self.create_subscription(PointCloud2, self.pc_topic, self.pc_callback, 5)
        self.cmd_sub = self.create_subscription(Twist, self.in_cmd_topic, self.cmd_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, self.out_cmd_topic, 10)
        self.estop_pub = self.create_publisher(Bool, 'emergency_stop', 10)

        # Timer to recompute distance if needed
        self.create_timer(0.1, self._update_min_distance)

        self.get_logger().info(f"Safety monitor active. safety_distance={self.safety_distance} hard_stop={self.hard_stop_distance}")

    def pc_callback(self, msg: PointCloud2):
        # Convert to Nx3 float32 points (x forward, y left, z up) using fast path
        try:
            if not SENSOR_MSGS_PY_AVAILABLE:
                return
            pts = np.array(list(point_cloud2.read_points(msg, field_names=("x","y","z"), skip_nans=True)), dtype=np.float32)
            self.latest_pc = pts
        except Exception:
            self.latest_pc = None

    def _update_min_distance(self):
        if self.latest_pc is None or self.latest_pc.size == 0:
            self.min_forward = 10.0
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
        except Exception:
            self.min_forward = 10.0

    def cmd_callback(self, msg: Twist):
        # Gate forward motion if obstacle is too close
        out = Twist()
        out.angular.z = msg.angular.z

        # Emergency stop handling
        hard = (self.min_forward < self.hard_stop_distance)
        if hard:
            out.linear.x = 0.0
            out.angular.z = 0.0
            self.estop_pub.publish(Bool(data=True))
            self.cmd_pub.publish(out)
            return
        else:
            # Clear estop if previously set
            self.estop_pub.publish(Bool(data=False))

        if msg.linear.x > 0.0 and self.min_forward < self.safety_distance:
            # Block forward, allow turning/backward
            out.linear.x = 0.0
        else:
            out.linear.x = msg.linear.x

        self.cmd_pub.publish(out)


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

