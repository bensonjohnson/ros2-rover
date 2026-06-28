"""Map Integrator — bridges RTAB-Map SLAM with the Deep Explorer Network.

Provides:
  - Local occupancy grid crops synchronized with /tf for the robot's pose
  - Frontier information as a simple occupancy layer in the crop
  - Map quality metrics for the explore manager
  - TF lookup for robot pose (vs simple odometry-only estimate)
"""

import math
import time
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from tf2_ros import TransformListener, Buffer
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Float32MultiArray, String


class MapIntegrator(Node):
    """Publishes local occupancy crops and map quality metrics.

    Subscribed topics:
      /map                    — RTAB-Map occupancy grid
      /explorer/goal          — current exploration goal (for reference)

    Published topics:
      /explorer/occ_crop      — Float32MultiArray [64×64] local crop
      /explorer/map_quality   — Float32MultiArray [coverage, frontiers, ...]
    """
    def __init__(self):
        super().__init__("map_integrator")

        p = self.declare_parameter
        p("crop_size", 64)
        p("crop_half_meters", 2.0)    # 4m × 4m crop
        p("update_rate_hz", 10.0)

        g = self.get_parameter
        self.crop_size = int(g("crop_size").value)
        self.crop_half_meters = float(g("crop_half_meters").value)
        self.update_rate = float(g("update_rate_hz").value)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # State
        self._map: Optional[OccupancyGrid] = None
        self._map_data: Optional[np.ndarray] = None

        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid, "/map", self._map_cb, 10)

        # Publishers
        self.crop_pub = self.create_publisher(
            Float32MultiArray, "/explorer/occ_crop", 10)
        self.quality_pub = self.create_publisher(
            Float32MultiArray, "/explorer/map_quality", 10)
        self.frontier_pub = self.create_publisher(
            Float32MultiArray, "/explorer/frontier_map", 10)

        self.timer = self.create_timer(1.0 / self.update_rate, self._update)

    def _map_cb(self, msg: OccupancyGrid):
        self._map = msg
        h, w = msg.info.height, msg.info.width
        self._map_data = np.array(msg.data, dtype=np.float32).reshape(h, w)

    def _get_pose(self) -> Optional[Tuple[float, float, float]]:
        """Look up robot pose via TF. Returns (x, y, yaw) or None."""
        try:
            t: TransformStamped = self.tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time())
            x = t.transform.translation.x
            y = t.transform.translation.y
            q = t.transform.rotation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            return (x, y, yaw)
        except Exception:
            return None

    def _extract_crop(self, pose: Tuple[float, float, float]) -> np.ndarray:
        """Extract local crop from the occupancy grid.

        Returns [crop_size × crop_size] float32:
          0.0 = free, 1.0 = occupied, 0.5 = unknown
        """
        if self._map_data is None or self._map is None:
            return np.full((self.crop_size, self.crop_size), 0.5, dtype=np.float32)

        info = self._map.info
        res = info.resolution
        ox = info.origin.position.x
        oy = info.origin.position.y
        rx, ry, _ = pose

        # Robot position in grid coords (rounded; fine enough for 64×64)
        gx = int((rx - ox) / res)
        gy = int((ry - oy) / res)
        half = self.crop_size // 2
        half_m = int(self.crop_half_meters / res)

        h, w = self._map_data.shape
        crop = np.full((self.crop_size, self.crop_size), 0.5, dtype=np.float32)

        # Map: map[y, x] → crop[row, col]
        y_start = max(0, gy - half_m)
        y_end = min(h, gy + half_m)
        x_start = max(0, gx - half_m)
        x_end = min(w, gx + half_m)

        # Where in the crop this patch goes
        row_start = half - (gy - y_start)
        col_start = half - (gx - x_start)

        patch = self._map_data[y_start:y_end, x_start:x_end]
        # Normalize: -1=unknown→0.5, 0=free→0.0, 100=occupied→1.0
        norm = np.where(patch < 0, 0.5, patch / 100.0)
        crop[row_start:row_start + norm.shape[0],
             col_start:col_start + norm.shape[1]] = norm

        return crop

    def _compute_quality(self) -> Tuple[float, float, float, float]:
        """Return (coverage_pct, n_frontiers, mean_openness, entropy)."""
        if self._map_data is None:
            return (0.0, 0.0, 0.0, 0.0)

        data = self._map_data.flatten()
        total = len(data)
        known_free = float(np.sum(data == 0))
        known_occ = float(np.sum(data == 100))
        unknown = total - known_free - known_occ
        coverage = (known_free + known_occ) / max(total, 1)

        # Count frontiers (free cells adjacent to unknown)
        free_mask = (data.reshape(self._map_data.shape) == 0)
        unknown_mask = (self._map_data < 0)
        from scipy.ndimage import binary_dilation
        unknown_dilated = binary_dilation(unknown_mask, structure=np.ones((3, 3)))
        frontiers_mask = free_mask & unknown_dilated[:self._map_data.shape[0],
                                                       :self._map_data.shape[1]]
        n_frontiers = int(np.sum(frontiers_mask))

        # Mean openness (for novelty calculation)
        mean_openness = known_free / max(total, 1)

        # Map entropy (Shannon): H = -Σ p log p over [free, occ, unknown]
        p_free = known_free / max(total, 1)
        p_occ = known_occ / max(total, 1)
        p_unk = unknown / max(total, 1)
        entropy = 0.0
        for p in (p_free, p_occ, p_unk):
            if p > 0:
                entropy -= p * math.log2(p)

        return (coverage, float(n_frontiers), mean_openness, entropy)

    def _update(self):
        pose = self._get_pose()
        if pose is None:
            return

        # Extract crop
        crop = self._extract_crop(pose)

        # Publish crop
        crop_msg = Float32MultiArray()
        crop_msg.data = crop.flatten().tolist()
        self.crop_pub.publish(crop_msg)

        # Quality
        coverage, n_frontiers, openness, entropy = self._compute_quality()
        quality_msg = Float32MultiArray()
        quality_msg.data = [coverage, float(n_frontiers), openness, entropy]
        self.quality_pub.publish(quality_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MapIntegrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
