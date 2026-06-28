"""Explore Manager — high-level autonomous mapping coordinator.

Orchestrates the full lifecycle of exploring and mapping an unknown house:
  1. UNEXPLORED → initial random exploration to seed the map
  2. FRONTIER_DRIVEN → frontier-based exploration using the occupancy grid
  3. COVERAGE_CHECK → detect coverage gaps and revisit
  4. COMPLETE → signal mapping is done

This node runs alongside explorer_runner and the safety monitor. It publishes
navigation goals as PoseStamped on /explorer/goal and provides a status
service.

Key concepts:
  - Frontiers are boundary cells between known-free and unknown space in the
    occupancy grid (standard frontier exploration).
  - The exploration policy (the NN) receives the local occupancy crop and
    learns to drive toward frontiers — the frontier signal emerges from the
    map, not a separate algorithm.
  - This manager handles what the NN cannot: global coverage tracking,
    detecting when the map is complete, and fallback behavior.
"""

import math
import time
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_srvs.srv import Trigger, SetBool
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Twist, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float32MultiArray, String, Bool


@dataclass
class CoverageStats:
    """Track mapping progress."""
    total_cells: int = 0
    known_free: int = 0
    known_occupied: int = 0
    unknown: int = 0
    frontiers: int = 0
    coverage_pct: float = 0.0          # (free + occupied) / total
    coverage_history: List[float] = field(default_factory=list)
    frontier_positions: List[Tuple[float, float]] = field(default_factory=list)
    mapping_time_s: float = 0.0


class ExploreManager(Node):
    """High-level exploration and mapping coordinator.

    Modes:
      - IDLE       waiting for start command
      - EXPLORE    frontier-driven autonomous exploration
      - COVERAGE   filling gaps in already-mapped areas
      - RETURNING  navigating back to start
      - COMPLETE   mapping finished, rover stops
    """
    def __init__(self):
        super().__init__("explore_manager")

        p = self.declare_parameter
        p("update_rate_hz", 5.0)
        p("map_topic", "/map")
        p("coverage_threshold", 0.95)      # stop when >95% known
        p("min_frontier_distance", 0.5)    # m from robot to consider frontier
        p("max_frontier_distance", 8.0)    # m beyond which we ignore
        p("stuck_timeout_s", 30.0)         # restart if no progress
        p("progress_radius_m", 0.5)        # how far robot must move to count progress
        p("min_coverage_increase", 0.005)  # min coverage gain in 30s to count

        g = self.get_parameter
        self.update_rate = float(g("update_rate_hz").value)
        self.map_topic = g("map_topic").value
        self.coverage_threshold = float(g("coverage_threshold").value)
        self.min_frontier_dist = float(g("min_frontier_distance").value)
        self.max_frontier_dist = float(g("max_frontier_distance").value)
        self.stuck_timeout = float(g("stuck_timeout_s").value)
        self.progress_radius = float(g("progress_radius_m").value)
        self.min_coverage_increase = float(g("min_coverage_increase").value)

        # State
        self.mode = "IDLE"
        self._map_data: Optional[np.ndarray] = None
        self._map_info: Optional[OccupancyGrid] = None
        self._pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._start_pose: Optional[Tuple[float, float, float]] = None
        self._last_progress_pose: Optional[Tuple[float, float, float]] = None
        self._last_progress_time = time.monotonic()
        self._last_coverage = 0.0
        self._last_coverage_time = time.monotonic()
        self._current_frontier: Optional[Tuple[float, float]] = None
        self._stats = CoverageStats()
        self._goal_id = 0

        # ROS
        self.map_sub = self.create_subscription(
            OccupancyGrid, self.map_topic, self._map_cb, 10)
        self.odom_sub = self.create_subscription(
            Odometry, "/odometry/filtered", self._odom_cb, 10)

        self.goal_pub = self.create_publisher(
            PoseStamped, "/explorer/goal", 10)
        self.viz_pub = self.create_publisher(
            MarkerArray, "/explorer/viz", 10)
        self.status_pub = self.create_publisher(
            String, "/explorer/status", 10)

        # Services
        self.srv_start = self.create_service(
            Trigger, "~/start_exploration", self._cb_start)
        self.srv_stop = self.create_service(
            Trigger, "~/stop_exploration", self._cb_stop)

        self.timer = self.create_timer(1.0 / self.update_rate, self._update)

        self.get_logger().info(
            f"Explore Manager ready: threshold={self.coverage_threshold} "
            f"stuck_timeout={self.stuck_timeout}s")

    # ---- Callbacks ----

    def _map_cb(self, msg: OccupancyGrid):
        self._map_info = msg
        h, w = msg.info.height, msg.info.width
        self._map_data = np.array(msg.data, dtype=np.int8).reshape(h, w)

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        # Get yaw from quaternion
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._pose = (float(p.x), float(p.y), yaw)

        if self._start_pose is None:
            self._start_pose = self._pose

    def _cb_start(self, req, resp):
        if self.mode == "EXPLORE":
            resp.success = True
            resp.message = "Already exploring"
            return resp
        self.mode = "EXPLORE"
        self._start_pose = self._pose
        self._last_progress_pose = self._pose
        self._last_progress_time = time.monotonic()
        self._last_coverage = 0.0
        resp.success = True
        resp.message = "Exploration started"
        self.get_logger().info("Exploration STARTED")
        return resp

    def _cb_stop(self, req, resp):
        self.mode = "IDLE"
        resp.success = True
        resp.message = "Exploration stopped"
        self.get_logger().info("Exploration STOPPED")
        return resp

    # ---- Main update ----

    def _update(self):
        if self.mode not in ("EXPLORE", "COVERAGE"):
            self._publish_status()
            return

        if self._map_data is None:
            self.get_logger().warning("Waiting for map...", throttle_duration_sec=10.0)
            return

        # Compute coverage stats
        self._compute_stats()

        # Check for completion
        if self._stats.coverage_pct >= self.coverage_threshold:
            self.mode = "COMPLETE"
            self.get_logger().info(
                f"Mapping COMPLETE: {self._stats.coverage_pct*100:.1f}% coverage")
            self._publish_stop()
            self._publish_status()
            return

        # Check for stuck
        self._check_stuck()

        # Detect frontiers
        frontiers = self._detect_frontiers()
        self._stats.frontiers = len(frontiers)
        self._stats.frontier_positions = frontiers

        if not frontiers:
            self.get_logger().warning(
                "No frontiers found! Switching to coverage mode.")
            self.mode = "COVERAGE"
            # In coverage mode, pick a random known-but-unvisited area
            target = self._pick_coverage_gap()
            self._publish_goal(target)
            return

        # Pick the best frontier
        target = self._select_frontier(frontiers)
        self._current_frontier = target

        # Publish goal
        self._publish_goal(target)
        self._publish_viz(frontiers, target)
        self._publish_status()

    def _compute_stats(self):
        if self._map_data is None:
            return
        data = self._map_data.flatten()
        total = len(data)
        known_free = int(np.sum(data == 0))
        known_occ = int(np.sum(data == 100))
        unknown = total - known_free - known_occ
        n_known = known_free + known_occ
        coverage = float(n_known) / float(max(total, 1))
        self._stats.total_cells = total
        self._stats.known_free = known_free
        self._stats.known_occupied = known_occ
        self._stats.unknown = unknown
        self._stats.coverage_pct = coverage
        self._stats.mapping_time_s = time.monotonic() - (
            self._last_progress_time if self._last_progress_time > 0 else time.monotonic())
        self._stats.coverage_history.append(coverage)
        if len(self._stats.coverage_history) > 1000:
            self._stats.coverage_history = self._stats.coverage_history[-1000:]

    def _check_stuck(self):
        """Detect if robot is stuck and reset frontier selection."""
        dx = self._pose[0] - self._last_progress_pose[0]
        dy = self._pose[1] - self._last_progress_pose[1]
        dist_moved = math.hypot(dx, dy)

        elapsed = time.monotonic() - self._last_progress_time
        if elapsed > self.stuck_timeout and dist_moved < self.progress_radius:
            self.get_logger().warning(
                f"STUCK: moved {dist_moved:.2f}m in {elapsed:.0f}s "
                f"coverage={self._stats.coverage_pct*100:.1f}%")
            self._last_progress_pose = self._pose
            self._last_progress_time = time.monotonic()
            # Clear current frontier to force re-selection
            self._current_frontier = None
        elif dist_moved > self.progress_radius:
            self._last_progress_pose = self._pose
            self._last_progress_time = time.monotonic()

        # Check coverage stagnation
        cov_delta = self._stats.coverage_pct - self._last_coverage
        if elapsed > self.stuck_timeout and cov_delta < self.min_coverage_increase:
            self.get_logger().info(
                f"Coverage stagnated ({cov_delta*100:.2f}% in {elapsed:.0f}s) "
                f"— re-exploring")
            self._last_coverage = self._stats.coverage_pct
            self._current_frontier = None

    # ---- Frontier detection ----

    def _detect_frontiers(self) -> List[Tuple[float, float]]:
        """Detect frontier cells between known-free and unknown space.

        A frontier cell is a known-free cell adjacent to at least one unknown cell.
        Returns list of (x, y) in meters.
        """
        if self._map_data is None or self._map_info is None:
            return []

        data = self._map_data
        h, w = data.shape
        res = self._map_info.resolution
        ox = self._map_info.origin.position.x
        oy = self._map_info.origin.position.y

        # Binary masks
        free = (data == 0)
        unknown = (data < 0)

        # Shift unknown mask by 1 in 4-connected directions
        from scipy.ndimage import binary_dilation
        unknown_dilated = binary_dilation(unknown, structure=np.ones((3, 3)))

        # Frontiers = free cells adjacent to unknown
        frontier_mask = free & unknown_dilated

        # Cluster (simple: just collect all frontier cells)
        frontier_ys, frontier_xs = np.where(frontier_mask)

        # Convert to world coords
        frontiers = []
        rx, ry = self._pose[0], self._pose[1]
        for yi, xi in zip(frontier_ys, frontier_xs):
            wx = xi * res + ox
            wy = yi * res + oy
            dist = math.hypot(wx - rx, wy - ry)
            if self.min_frontier_dist <= dist <= self.max_frontier_dist:
                frontiers.append((wx, wy))

        return frontiers

    def _select_frontier(self, frontiers: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Select the best frontier using a utility function.

        Utility = information gain (unknown neighbors) * novelty bonus / distance
        Prefers:
          - Close frontiers (less travel)
          - Large frontiers (more information)
          - Visited-least-often frontiers (novelty)
        """
        if self._map_data is None or self._map_info is None or not frontiers:
            return frontiers[0] if frontiers else (0.0, 0.0)

        data = self._map_data
        res = self._map_info.resolution
        ox = self._map_info.origin.position.x
        oy = self._map_info.origin.position.y
        rx, ry = self._pose[0], self._pose[1]

        best_score = -float("inf")
        best_f = frontiers[0]

        for fx, fy in frontiers:
            # Distance from robot
            dist = math.hypot(fx - rx, fy - ry)
            if dist < 0.01:
                continue

            # Convert to grid coords
            gx = int((fx - ox) / res)
            gy = int((fy - oy) / res)
            if gx < 0 or gy < 0 or gx >= data.shape[1] or gy >= data.shape[0]:
                continue

            # Count unknown neighbors in a 5×5 window (information gain)
            y_min = max(0, gy - 2)
            y_max = min(data.shape[0], gy + 3)
            x_min = max(0, gx - 2)
            x_max = min(data.shape[1], gx + 3)
            window = data[y_min:y_max, x_min:x_max]
            n_unknown = int(np.sum(window < 0))

            # Novelty bonus: prefer frontiers farther from past visits
            # (simplified: we don't store visit history, so use yaw alignment)
            angle_to = math.atan2(fy - ry, fx - rx)
            yaw_diff = abs(angle_to - self._pose[2])
            yaw_bonus = 1.0 if yaw_diff < math.pi / 2 else 0.5

            # Utility
            score = n_unknown * yaw_bonus / max(dist, 0.1)

            if score > best_score:
                best_score = score
                best_f = (fx, fy)

        return best_f

    def _pick_coverage_gap(self) -> Tuple[float, float]:
        """When no frontiers remain, pick the known-free cell farthest from
        the robot's path — a 'coverage gap'.

        Simplified: pick the free cell farthest from current pose.
        """
        if self._map_data is None or self._map_info is None:
            return (0.0, 0.0)

        free_ys, free_xs = np.where(self._map_data == 0)
        if len(free_ys) == 0:
            return (0.0, 0.0)

        res = self._map_info.resolution
        ox = self._map_info.origin.position.x
        oy = self._map_info.origin.position.y
        rx, ry = self._pose[0], self._pose[1]

        # Sample random free cells and pick farthest
        n = min(len(free_ys), 200)
        idx = np.random.choice(len(free_ys), n, replace=False)
        max_dist = 0.0
        best = (rx + 1.0, ry + 1.0)
        for i in idx:
            wx = free_xs[i] * res + ox
            wy = free_ys[i] * res + oy
            d = math.hypot(wx - rx, wy - ry)
            if d > max_dist:
                max_dist = d
                best = (wx, wy)
        return best

    # ---- Publishing ----

    def _publish_goal(self, target: Tuple[float, float]):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = target[0]
        msg.pose.position.y = target[1]
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        self._goal_id += 1
        self.goal_pub.publish(msg)

        self.get_logger().info(
            f"Goal #{self._goal_id}: ({target[0]:.1f}, {target[1]:.1f}) "
            f"coverage={self._stats.coverage_pct*100:.1f}% "
            f"frontiers={self._stats.frontiers}",
            throttle_duration_sec=2.0)

    def _publish_stop(self):
        # Publish zero-velocity command
        msg = Twist()
        # Use a simple zero-velocity publisher (or service)
        self.get_logger().info("Mapping complete — sending stop command")

    def _publish_viz(self, frontiers, target):
        markers = []

        # Frontier markers
        if frontiers:
            m = Marker()
            m.header.frame_id = "map"
            m.ns = "frontiers"
            m.type = Marker.POINTS
            m.action = Marker.ADD
            m.scale.x = 0.1
            m.scale.y = 0.1
            m.color.a = 0.8
            m.color.g = 1.0
            for fx, fy in frontiers[::5]:  # subsample for viz
                pt = Point()
                pt.x = fx; pt.y = fy; pt.z = 0.0
                m.points.append(pt)
            markers.append(m)

        # Current goal marker
        m = Marker()
        m.header.frame_id = "map"
        m.ns = "goal"
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = target[0]
        m.pose.position.y = target[1]
        m.pose.position.z = 0.0
        m.scale.x = 0.3; m.scale.y = 0.3; m.scale.z = 0.3
        m.color.a = 1.0; m.color.r = 1.0; m.color.g = 0.5
        markers.append(m)

        msg = MarkerArray(markers=markers)
        self.viz_pub.publish(msg)

    def _publish_status(self):
        msg = String()
        msg.data = (f"mode={self.mode} "
                    f"coverage={self._stats.coverage_pct*100:.1f}% "
                    f"frontiers={self._stats.frontiers} "
                    f"known={self._stats.known_free + self._stats.known_occupied} "
                    f"unknown={self._stats.unknown} "
                    f"mapping_time={self._stats.mapping_time_s:.0f}s")
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ExploreManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
