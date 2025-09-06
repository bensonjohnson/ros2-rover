#!/usr/bin/env python3
"""
BEV Processor Node
 - Subscribes to a PointCloud2 topic
 - Generates a Bird's Eye View (BEV) image using BEVGenerator
 - Publishes the BEV image as sensor_msgs/Image (encoding: 32FC4)
 - Publishes auxiliary metrics (bev_min_forward_distance)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, Image
from std_msgs.msg import Float32, String
import numpy as np
import struct
from cv_bridge import CvBridge

from .bev_generator import BEVGenerator

try:
    from sensor_msgs_py import point_cloud2
    SENSOR_MSGS_PY_AVAILABLE = True
except Exception:
    SENSOR_MSGS_PY_AVAILABLE = False


class BEVProcessorNode(Node):
    def __init__(self):
        super().__init__('bev_processor')

        # Parameters
        self.declare_parameter('pointcloud_topic', '/camera/camera/depth/color/points')
        self.declare_parameter('bev_image_topic', '/bev/image')
        self.declare_parameter('publish_rate_hz', 10.0)
        # BEV parameters
        self.declare_parameter('bev_size', [200, 200])
        self.declare_parameter('bev_range', [10.0, 10.0])
        self.declare_parameter('bev_height_channels', [0.2, 1.0])
        self.declare_parameter('enable_ground_removal', True)
        self.declare_parameter('ground_ransac_iterations', 100)
        self.declare_parameter('ground_ransac_threshold', 0.05)
        self.declare_parameter('bev_ground_update_interval', 10)
        self.declare_parameter('bev_enable_opencl', True)

        self.pc_topic = str(self.get_parameter('pointcloud_topic').value)
        self.bev_topic = str(self.get_parameter('bev_image_topic').value)
        self.publish_rate = float(self.get_parameter('publish_rate_hz').value)

        bev_size = self.get_parameter('bev_size').value
        bev_range = self.get_parameter('bev_range').value
        bev_height_channels = self.get_parameter('bev_height_channels').value
        enable_ground_removal = bool(self.get_parameter('enable_ground_removal').value)
        ground_ransac_iterations = int(self.get_parameter('ground_ransac_iterations').value)
        ground_ransac_threshold = float(self.get_parameter('ground_ransac_threshold').value)
        ground_update_interval = int(self.get_parameter('bev_ground_update_interval').value)
        enable_opencl = bool(self.get_parameter('bev_enable_opencl').value)

        # BEV generator
        self.bev = BEVGenerator(
            bev_size=(int(bev_size[0]), int(bev_size[1])),
            bev_range=(float(bev_range[0]), float(bev_range[1])),
            height_channels=tuple(float(x) for x in bev_height_channels),
            enable_ground_removal=enable_ground_removal,
            ground_ransac_iterations=ground_ransac_iterations,
            ground_ransac_threshold=ground_ransac_threshold,
            ground_update_interval=ground_update_interval,
            enable_opencl=enable_opencl,
        )

        self.bridge = CvBridge()
        self.last_pub_time = self.get_clock().now()
        self.latest_bev = None
        self.latest_bev_time = None
        self.last_pc_time = None

        # Publishers
        self.image_pub = self.create_publisher(Image, self.bev_topic, 2)
        self.min_dist_pub = self.create_publisher(Float32, '/bev_min_forward_distance', 5)
        self.status_pub = self.create_publisher(String, '/bev_processor_status', 2)

        # Subscriber
        self.pc_sub = self.create_subscription(PointCloud2, self.pc_topic, self.pc_callback, qos_profile_sensor_data)

        # Timer to publish status
        self.create_timer(1.0, self.publish_status)

        self.get_logger().info(f"BEV Processor active. Subscribing to {self.pc_topic}, publishing {self.bev_topic}")

    def pc_callback(self, msg: PointCloud2):
        # Rate limit by publish_rate
        now = self.get_clock().now()
        if (now - self.last_pub_time).nanoseconds < (1e9 / max(self.publish_rate, 1e-3)):
            return
        self.last_pub_time = now
        self.last_pc_time = now
        # Convert to Nx3
        pts = self._pc2_to_numpy(msg)
        if pts.size == 0:
            return
        # Generate BEV
        bev_img = self.bev.generate_bev(pts)
        self.latest_bev = bev_img
        self.latest_bev_time = now
        # Publish image (32FC4)
        try:
            img_msg = self.bridge.cv2_to_imgmsg(bev_img.astype(np.float32), encoding='32FC4')
            img_msg.header.stamp = now.to_msg()
            img_msg.header.frame_id = 'base_link'  # conceptual frame for BEV
            self.image_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().warn(f"BEV image publish failed: {e}")
        # Publish min forward distance derived from BEV occupancy
        try:
            min_d = self._compute_min_forward_distance(bev_img)
            self.min_dist_pub.publish(Float32(data=float(min_d)))
        except Exception as e:
            self.get_logger().debug(f"Min distance compute failed: {e}")

    def publish_status(self):
        status = String()
        age = float('inf')
        if self.latest_bev_time is not None:
            age = (self.get_clock().now() - self.latest_bev_time).nanoseconds / 1e9
        status.data = f"BEV: age={age:.2f}s, pub_rate={self.publish_rate}Hz"
        self.status_pub.publish(status)

    def _pc2_to_numpy(self, msg: PointCloud2) -> np.ndarray:
        try:
            if SENSOR_MSGS_PY_AVAILABLE:
                pts_list = list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
                if not pts_list:
                    return np.zeros((0, 3), dtype=np.float32)
                first = pts_list[0]
                if isinstance(first, (tuple, list)) and len(first) >= 3:
                    arr = np.asarray(pts_list, dtype=np.float32).reshape(-1, 3)
                else:
                    structured = np.asarray(pts_list)
                    if getattr(structured, 'dtype', None) is not None and structured.dtype.names is not None:
                        x = structured['x'].astype(np.float32, copy=False)
                        y = structured['y'].astype(np.float32, copy=False)
                        z = structured['z'].astype(np.float32, copy=False)
                        arr = np.stack((x, y, z), axis=1)
                    else:
                        arr = np.array([[p[0], p[1], p[2]] for p in pts_list], dtype=np.float32)
                mask = np.isfinite(arr).all(axis=1)
                return arr[mask]
            else:
                points = []
                step = msg.point_step
                data = msg.data
                for i in range(0, len(data), step):
                    if i + 12 <= len(data):
                        try:
                            x, y, z = struct.unpack('fff', data[i:i+12])
                            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                                points.append([x, y, z])
                        except Exception:
                            continue
                if not points:
                    return np.zeros((0, 3), dtype=np.float32)
                return np.asarray(points, dtype=np.float32)
        except Exception as e:
            self.get_logger().warn(f"Point cloud conversion failed: {e}")
            return np.zeros((0, 3), dtype=np.float32)

    def _compute_min_forward_distance(self, bev_img: np.ndarray) -> float:
        # Occupancy from channels: conf(3) and low(2)
        h, w, _ = bev_img.shape
        conf = bev_img[:, :, 3]
        low = bev_img[:, :, 2]
        occ = (conf > 0.25) | (low > 0.1)
        near_start = int(h * 0.50)
        near_end = int(h * 0.75)
        front_rows = slice(near_start, near_end)
        center_cols = slice(int(w / 3), int(2 * w / 3))
        mask = occ[front_rows, center_cols]
        ys, xs = np.where(mask)
        if ys.size == 0:
            return 10.0
        px = ys.astype(np.float32)
        x_range = float(self.bev.x_range)
        x_m = (px / float(h)) * (2.0 * x_range) - x_range
        x_m = x_m[x_m >= 0.0]
        return float(np.min(x_m)) if x_m.size else 10.0


def main(args=None):
    rclpy.init(args=args)
    node = BEVProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
