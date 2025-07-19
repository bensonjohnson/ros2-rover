#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import Point
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header, ColorRGBA
import numpy as np
import cv2
from cv_bridge import CvBridge
import math


class ObstacleDetector(Node):
    def __init__(self):
        super().__init__("obstacle_detector")

        # Parameters
        self.declare_parameter("min_obstacle_height", 0.05)
        self.declare_parameter("max_obstacle_height", 2.0)
        self.declare_parameter("obstacle_threshold", 0.1)
        self.declare_parameter("detection_width", 2.0)
        self.declare_parameter("detection_distance", 3.0)
        self.declare_parameter("ground_plane_tolerance", 0.05)
        self.declare_parameter("min_points_for_plane", 1000)
        self.declare_parameter("cluster_tolerance", 0.1)
        self.declare_parameter("min_cluster_size", 50)
        self.declare_parameter("max_cluster_size", 5000)

        self.min_obstacle_height = self.get_parameter("min_obstacle_height").value
        self.max_obstacle_height = self.get_parameter("max_obstacle_height").value
        self.obstacle_threshold = self.get_parameter("obstacle_threshold").value
        self.detection_width = self.get_parameter("detection_width").value
        self.detection_distance = self.get_parameter("detection_distance").value
        self.ground_plane_tolerance = self.get_parameter("ground_plane_tolerance").value
        self.min_points_for_plane = self.get_parameter("min_points_for_plane").value
        self.cluster_tolerance = self.get_parameter("cluster_tolerance").value
        self.min_cluster_size = self.get_parameter("min_cluster_size").value
        self.max_cluster_size = self.get_parameter("max_cluster_size").value

        # CV Bridge for image processing
        self.bridge = CvBridge()

        # Ground plane parameters (will be estimated)
        self.ground_plane = None

        # Subscribers
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, "realsense_435i/depth/points", self.pointcloud_callback, 10
        )

        self.depth_sub = self.create_subscription(
            Image, "realsense_435i/depth/image_rect_raw", self.depth_callback, 10
        )

        # Publishers
        self.obstacle_markers_pub = self.create_publisher(
            MarkerArray, "obstacle_markers", 10
        )

        self.obstacle_mask_pub = self.create_publisher(Image, "obstacle_mask", 10)

        self.get_logger().info("Obstacle Detector initialized")

    def pointcloud_callback(self, msg):
        """Process point cloud for obstacle detection"""
        try:
            # Convert PointCloud2 to numpy array
            # Note: This is a simplified version - full implementation would
            # properly decode the PointCloud2 message format
            points = self.pointcloud2_to_array(msg)

            if points is None or len(points) < self.min_points_for_plane:
                return

            # Filter points within detection area
            valid_points = self.filter_detection_area(points)

            if len(valid_points) < self.min_points_for_plane:
                return

            # Estimate ground plane
            self.estimate_ground_plane(valid_points)

            # Detect obstacles
            obstacles = self.detect_obstacles(valid_points)

            # Publish obstacle markers
            self.publish_obstacle_markers(obstacles, msg.header)

        except Exception as e:
            self.get_logger().error(f"Point cloud processing error: {e}")

    def depth_callback(self, msg):
        """Process depth image for simple obstacle detection"""
        try:
            # Convert depth image
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")

            # Convert to meters
            depth_meters = depth_image.astype(np.float32) * 0.001

            # Create obstacle mask
            obstacle_mask = self.create_obstacle_mask(depth_meters)

            # Publish obstacle mask
            mask_msg = self.bridge.cv2_to_imgmsg(obstacle_mask, encoding="mono8")
            mask_msg.header = msg.header
            self.obstacle_mask_pub.publish(mask_msg)

        except Exception as e:
            self.get_logger().error(f"Depth image processing error: {e}")

    def pointcloud2_to_array(self, pointcloud_msg):
        """Convert PointCloud2 message to numpy array"""
        # This is a placeholder - full implementation would properly decode
        # the PointCloud2 message format based on fields and data layout
        # For now, return None to indicate this needs proper implementation
        return None

    def filter_detection_area(self, points):
        """Filter points to detection area in front of robot"""
        if points is None:
            return np.array([])

        # Filter by distance (forward direction is positive X)
        distance_mask = (points[:, 0] >= 0) & (points[:, 0] <= self.detection_distance)

        # Filter by width (Y direction)
        width_mask = (points[:, 1] >= -self.detection_width / 2) & (
            points[:, 1] <= self.detection_width / 2
        )

        # Filter by height (Z direction - above ground)
        height_mask = (points[:, 2] >= -0.5) & (
            points[:, 2] <= self.max_obstacle_height
        )

        combined_mask = distance_mask & width_mask & height_mask
        return points[combined_mask]

    def estimate_ground_plane(self, points):
        """Estimate ground plane using RANSAC-like approach"""
        if len(points) < 3:
            return

        # Simple ground plane estimation - assume ground is roughly at z=0
        # More sophisticated implementation would use RANSAC
        ground_points = points[points[:, 2] < self.ground_plane_tolerance]

        if len(ground_points) > self.min_points_for_plane:
            # Fit plane to ground points (simplified - assume horizontal plane)
            ground_height = np.median(ground_points[:, 2])
            self.ground_plane = {"height": ground_height, "normal": [0, 0, 1]}

    def detect_obstacles(self, points):
        """Detect obstacles above ground plane"""
        if self.ground_plane is None:
            return []

        # Points above ground plane
        ground_height = self.ground_plane["height"]
        obstacle_points = points[points[:, 2] > ground_height + self.obstacle_threshold]

        if len(obstacle_points) == 0:
            return []

        # Simple clustering - group nearby points
        obstacles = self.cluster_points(obstacle_points)

        return obstacles

    def cluster_points(self, points):
        """Simple clustering of obstacle points"""
        if len(points) == 0:
            return []

        obstacles = []
        remaining_points = points.copy()

        while len(remaining_points) > self.min_cluster_size:
            # Start with first point
            seed_point = remaining_points[0]
            cluster = [seed_point]

            # Find all points within cluster tolerance
            distances = np.linalg.norm(remaining_points - seed_point, axis=1)
            cluster_mask = distances <= self.cluster_tolerance
            cluster_points = remaining_points[cluster_mask]

            if (
                len(cluster_points) >= self.min_cluster_size
                and len(cluster_points) <= self.max_cluster_size
            ):
                # Create obstacle from cluster
                obstacle = {
                    "center": np.mean(cluster_points, axis=0),
                    "min_point": np.min(cluster_points, axis=0),
                    "max_point": np.max(cluster_points, axis=0),
                    "points": cluster_points,
                }
                obstacles.append(obstacle)

            # Remove clustered points
            remaining_points = remaining_points[~cluster_mask]

        return obstacles

    def create_obstacle_mask(self, depth_image):
        """Create obstacle mask from depth image"""
        h, w = depth_image.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        # Define detection region (center portion of image)
        roi_x1 = w // 4
        roi_x2 = 3 * w // 4
        roi_y1 = h // 2
        roi_y2 = h

        # Extract ROI
        roi_depth = depth_image[roi_y1:roi_y2, roi_x1:roi_x2]

        # Find obstacles (objects closer than threshold)
        obstacle_threshold_depth = 2.0  # meters
        obstacle_mask_roi = (roi_depth > 0.1) & (roi_depth < obstacle_threshold_depth)

        # Apply morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        obstacle_mask_roi = cv2.morphologyEx(
            obstacle_mask_roi.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel
        )

        # Put ROI back into full image
        mask[roi_y1:roi_y2, roi_x1:roi_x2] = obstacle_mask_roi

        return mask

    def publish_obstacle_markers(self, obstacles, header):
        """Publish obstacle markers for visualization"""
        marker_array = MarkerArray()

        for i, obstacle in enumerate(obstacles):
            marker = Marker()
            marker.header = header
            marker.ns = "obstacles"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            # Position at obstacle center
            marker.pose.position.x = float(obstacle["center"][0])
            marker.pose.position.y = float(obstacle["center"][1])
            marker.pose.position.z = float(obstacle["center"][2])

            # Orientation (identity quaternion)
            marker.pose.orientation.w = 1.0

            # Scale based on obstacle bounding box
            size = obstacle["max_point"] - obstacle["min_point"]
            marker.scale.x = max(float(size[0]), 0.1)
            marker.scale.y = max(float(size[1]), 0.1)
            marker.scale.z = max(float(size[2]), 0.1)

            # Color (red for obstacles)
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.7

            # Lifetime
            marker.lifetime.sec = 1

            marker_array.markers.append(marker)

        # Clear old markers if fewer obstacles
        for i in range(len(obstacles), 100):
            marker = Marker()
            marker.header = header
            marker.ns = "obstacles"
            marker.id = i
            marker.action = Marker.DELETE
            marker_array.markers.append(marker)

        self.obstacle_markers_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    detector = ObstacleDetector()

    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    finally:
        detector.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
