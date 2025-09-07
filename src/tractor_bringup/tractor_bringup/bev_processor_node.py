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
from sensor_msgs.msg import PointCloud2, Image, Imu
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
        # IMU-assisted ground removal
        self.declare_parameter('imu_topic', '/lsm9ds1_imu_publisher/imu/data')
        self.declare_parameter('sensor_height_m', 0.17)
        self.declare_parameter('imu_ransac_interval_s', 4.0)
        self.declare_parameter('imu_roll_pitch_threshold_deg', 3.0)
        # BEV parameters
        self.declare_parameter('bev_size', [200, 200])
        self.declare_parameter('bev_range', [10.0, 10.0])
        self.declare_parameter('bev_height_channels', [0.2, 1.0])
        self.declare_parameter('enable_ground_removal', True)
        self.declare_parameter('ground_ransac_iterations', 100)
        self.declare_parameter('ground_ransac_threshold', 0.05)
        self.declare_parameter('bev_ground_update_interval', 10)
        self.declare_parameter('bev_enable_opencl', True)
        # Obstacle/grass thresholds
        self.declare_parameter('min_obstacle_height_m', 0.25)
        self.declare_parameter('grass_height_tolerance_m', 0.15)

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
        imu_topic = str(self.get_parameter('imu_topic').value)
        sensor_height_m = float(self.get_parameter('sensor_height_m').value)
        imu_ransac_interval_s = float(self.get_parameter('imu_ransac_interval_s').value)
        imu_rp_thresh = float(self.get_parameter('imu_roll_pitch_threshold_deg').value)
        min_ob_h = float(self.get_parameter('min_obstacle_height_m').value)
        grass_tol = float(self.get_parameter('grass_height_tolerance_m').value)

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
            enable_grass_filtering=True,
            grass_height_tolerance=grass_tol,
            min_obstacle_height=min_ob_h,
        )
        # Configure IMU-assisted ground filtering
        self.bev.set_sensor_height(sensor_height_m)
        self.bev.set_imu_ground_params(ransac_interval_s=imu_ransac_interval_s, rp_threshold_deg=imu_rp_thresh)

        self.bridge = CvBridge()
        self.last_pub_time = self.get_clock().now()
        self.latest_bev = None
        self.latest_bev_time = None
        self.last_pc_time = None

        # Publishers
        self.image_pub = self.create_publisher(Image, self.bev_topic, 2)
        self.min_dist_pub = self.create_publisher(Float32, '/bev_min_forward_distance', 5)
        self.status_pub = self.create_publisher(String, '/bev_processor_status', 2)

        # Subscribers
        self.pc_sub = self.create_subscription(PointCloud2, self.pc_topic, self.pc_callback, qos_profile_sensor_data)
        self.imu_sub = self.create_subscription(Imu, imu_topic, self.imu_callback, qos_profile_sensor_data)

        # Timer to publish status
        self.create_timer(1.0, self.publish_status)

        self.get_logger().info(f"BEV Processor active. Subscribing to {self.pc_topic}, publishing {self.bev_topic}")

        # IMU cache
        self._last_up = None
        self._last_imu_time = None
        self.get_logger().info(f"IMU-assisted ground filter: imu_topic={imu_topic}, sensor_height={sensor_height_m:.2f}m")

    def imu_callback(self, msg: Imu):
        try:
            q = msg.orientation
            # Rotation from body to world; up in world is [0,0,1]
            # Body up vector = R^T * [0,0,1]
            w, x, y, z = q.w, q.x, q.y, q.z
            # Rotation matrix elements
            # Using standard quaternion to rotation matrix
            R = np.array([
                [1 - 2*(y*y + z*z),     2*(x*y - z*w),         2*(x*z + y*w)],
                [2*(x*y + z*w),         1 - 2*(x*x + z*z),     2*(y*z - x*w)],
                [2*(x*z - y*w),         2*(y*z + x*w),         1 - 2*(x*x + y*y)]
            ], dtype=np.float32)
            up_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            up_body = R.T @ up_world
            n = up_body / max(np.linalg.norm(up_body), 1e-6)
            self._last_up = n.astype(np.float32)
            self._last_imu_time = self.get_clock().now()
            # Provide to BEV generator
            self.bev.set_imu_up(self._last_up)
        except Exception as e:
            self.get_logger().debug(f"IMU processing failed: {e}")

    def pc_callback(self, msg: PointCloud2):
        # Rate limit by publish_rate
        now = self.get_clock().now()
        if (now - self.last_pub_time).nanoseconds < (1e9 / max(self.publish_rate, 1e-3)):
            return
        self.last_pub_time = now
        self.last_pc_time = now
        # Convert to Nx3
        pts = self._pc2_to_numpy(msg)
        # Map to forward-left-up if coming from an optical frame
        try:
            frame = (msg.header.frame_id or '').lower()
        except Exception:
            frame = ''
        if pts.size > 0 and 'optical' in frame:
            # RealSense optical: x=right, y=down, z=forward -> convert to forward-left-up
            x_fwd = pts[:, 2]
            y_left = -pts[:, 0]
            z_up = -pts[:, 1]
            pts = np.column_stack((x_fwd, y_left, z_up)).astype(np.float32)
        # Early ROI filter and voxel downsample to reduce load before BEV
        if pts.size > 0:
            pts = self._early_roi_and_voxel(pts)
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

    # TODO(organized-depth fast path): add direct depth->BEV projection to avoid PointCloud2 construction
    # This will compute per-pixel 3D rays and project directly into the BEV histogram, skipping point cloud builds.
    # Keep on roadmap to push BEV toward 10–12 Hz on CPU.

    def _early_roi_and_voxel(self, pts: np.ndarray) -> np.ndarray:
        """Apply an early ROI crop and voxel downsample in (x_fwd, y_left) to reduce compute.
        - ROI: x in [0, x_range], |y| <= y_range, z in [-0.5, 2.0]
        - Voxel grid: size ~6 cm (configurable here) on x/y plane; keep one point per cell.
        """
        try:
            if pts is None or pts.size == 0:
                return pts
            x = pts[:, 0]
            y = pts[:, 1]
            z = pts[:, 2]
            xr = float(getattr(self.bev, 'x_range', 6.0))
            yr = float(getattr(self.bev, 'y_range', 4.0))
            # ROI mask
            roi = (x >= 0.0) & (x <= xr) & (y >= -yr) & (y <= yr) & (z >= -0.5) & (z <= 2.0)
            if not np.any(roi):
                return np.zeros((0, 3), dtype=np.float32)
            pts_roi = pts[roi]
            # Voxel downsample
            vx = 0.06  # 6 cm voxel size
            # Compute voxel indices relative to ROI min (0 for x, -yr for y)
            ix = np.floor((pts_roi[:, 0] - 0.0) / vx).astype(np.int32)
            iy = np.floor((pts_roi[:, 1] + yr) / vx).astype(np.int32)
            key = ix.astype(np.int64) * 1000003 + iy.astype(np.int64)  # hash pair
            _, unique_idx = np.unique(key, return_index=True)
            return pts_roi[unique_idx]
        except Exception:
            return pts

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
        # Start slightly ahead of 0 to avoid self/noise
        start_x = 0.05  # 5 cm
        x_range = float(self.bev.x_range)
        near_start = int(((start_x + x_range) / (2.0 * x_range)) * h)
        near_end = int(h * 0.75)
        front_rows = slice(near_start, near_end)
        center_cols = slice(int(w / 3), int(2 * w / 3))
        mask = occ[front_rows, center_cols]
        ys, xs = np.where(mask)
        if ys.size == 0:
            return 10.0
        px = ys.astype(np.float32) + near_start
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
