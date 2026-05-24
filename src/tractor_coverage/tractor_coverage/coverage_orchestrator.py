#!/usr/bin/env python3
"""Coverage orchestrator: map -> plan -> execute state machine.

Ties the classical coverage pipeline together:

    MAPPING ──► PLAN_COVERAGE ──► EXECUTING ──► DONE

- MAPPING:       wait until frontier exploration reports the map is closed
                 (``/exploration_status`` contains "COMPLETE"), or a manual
                 ``/start_coverage`` trigger arrives.
- PLAN_COVERAGE: take the latest ``/map``, convert it to a boundary + obstacle
                 polygons (:mod:`grid_to_polygon`), run the boustrophedon
                 :class:`CoveragePlanner`, publish the path for inspection.
- EXECUTING:     drive the waypoints through Nav2's ``NavigateThroughPoses``
                 action (global planning + RPP following + costmap avoidance +
                 recovery behaviours all come for free).

Run coverage on a freshly-explored map, or on a pre-existing latched ``/map`` by
calling the ``/start_coverage`` service directly.
"""

import math

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from nav2_msgs.action import NavigateThroughPoses
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .coverage_planner import CoveragePlanner
from .grid_to_polygon import from_occupancy_grid, points_to_ros


class CoverageOrchestrator(Node):
    def __init__(self):
        super().__init__("coverage_orchestrator")

        # --- parameters ---
        self.declare_parameter("tool_width", 0.30)        # robot/coverage width (m)
        self.declare_parameter("overlap", 0.10)           # sweep overlap fraction
        self.declare_parameter("border_offset", 0.25)     # inset from walls (m)
        self.declare_parameter("turn_radius", 0.30)       # min turn radius (m)
        self.declare_parameter("free_max", 25)            # occupancy: free threshold
        self.declare_parameter("occupied_min", 50)        # occupancy: occupied threshold
        self.declare_parameter("auto_start_on_complete", True)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("optimize_path", True)     # drop collinear waypoints

        self.tool_width = self.get_parameter("tool_width").value
        self.overlap = self.get_parameter("overlap").value
        self.border_offset = self.get_parameter("border_offset").value
        self.turn_radius = self.get_parameter("turn_radius").value
        self.free_max = int(self.get_parameter("free_max").value)
        self.occupied_min = int(self.get_parameter("occupied_min").value)
        self.auto_start = self.get_parameter("auto_start_on_complete").value
        self.map_frame = self.get_parameter("map_frame").value
        self.optimize = self.get_parameter("optimize_path").value

        self.planner = CoveragePlanner()

        # --- state ---
        self.state = "MAPPING"
        self.latest_map: OccupancyGrid | None = None
        self._busy = False  # guard against re-entrant planning/execution

        # Latched map from slam_toolbox needs transient-local durability.
        map_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(OccupancyGrid, "/map", self._map_cb, map_qos)
        self.create_subscription(
            String, "/exploration_status", self._exploration_cb, 10
        )

        self.path_pub = self.create_publisher(Path, "coverage_path", 10)
        self.status_pub = self.create_publisher(String, "coverage_status", 10)

        self.create_service(Trigger, "start_coverage", self._start_coverage_srv)

        self.nav_client = ActionClient(
            self, NavigateThroughPoses, "navigate_through_poses"
        )

        self.create_timer(2.0, self._publish_status)
        self.get_logger().info(
            "Coverage orchestrator up (state=MAPPING). "
            "Waiting for exploration to complete or /start_coverage."
        )

    # ------------------------------------------------------------------ inputs
    def _map_cb(self, msg: OccupancyGrid):
        self.latest_map = msg

    def _exploration_cb(self, msg: String):
        if (
            self.auto_start
            and self.state == "MAPPING"
            and "COMPLETE" in msg.data.upper()
        ):
            self.get_logger().info("Exploration complete — starting coverage.")
            self._begin_coverage()

    def _start_coverage_srv(self, request, response):
        if self._busy:
            response.success = False
            response.message = f"Coverage already running (state={self.state})"
            return response
        if self.latest_map is None:
            response.success = False
            response.message = "No /map received yet"
            return response
        self.get_logger().info("Manual /start_coverage trigger received.")
        self._begin_coverage()
        response.success = True
        response.message = "Coverage started"
        return response

    # ------------------------------------------------------------------ planning
    def _begin_coverage(self):
        if self._busy:
            return
        if self.latest_map is None:
            self.get_logger().warn("Cannot start coverage: no map available.")
            return
        self._busy = True
        self.state = "PLAN_COVERAGE"

        poses = self._plan(self.latest_map)
        if not poses:
            self.get_logger().error("Coverage planning produced no waypoints.")
            self.state = "MAPPING"
            self._busy = False
            return

        self._publish_path(poses)
        self.get_logger().info(f"Planned coverage: {len(poses)} waypoints.")
        self._execute(poses)

    def _plan(self, grid_msg: OccupancyGrid):
        boundary, obstacles = from_occupancy_grid(
            grid_msg, free_max=self.free_max, occupied_min=self.occupied_min
        )
        if len(boundary) < 3:
            self.get_logger().warn("No usable free region in map.")
            return []

        self.planner.set_tool_parameters(self.tool_width, self.overlap)
        self.planner.set_vehicle_parameters(self.turn_radius, self.border_offset)

        b_pts = points_to_ros(boundary)
        o_polys = [points_to_ros(o) for o in obstacles]
        path_pts = self.planner.plan_coverage_path(b_pts, o_polys)
        if self.optimize:
            path_pts = self.planner.optimize_path(path_pts)
        return self._to_pose_stamped(path_pts)

    def _to_pose_stamped(self, pts):
        """Build PoseStamped list with each pose facing the next waypoint."""
        now = self.get_clock().now().to_msg()
        poses = []
        prev_yaw = 0.0
        for i, p in enumerate(pts):
            nxt = pts[i + 1] if i + 1 < len(pts) else None
            if nxt is not None and (nxt.x != p.x or nxt.y != p.y):
                yaw = math.atan2(nxt.y - p.y, nxt.x - p.x)
            else:
                yaw = prev_yaw  # final waypoint keeps the approach heading
            ps = PoseStamped()
            ps.header.frame_id = self.map_frame
            ps.header.stamp = now
            ps.pose.position.x = float(p.x)
            ps.pose.position.y = float(p.y)
            ps.pose.orientation.z = math.sin(yaw / 2.0)
            ps.pose.orientation.w = math.cos(yaw / 2.0)
            poses.append(ps)
            prev_yaw = yaw
        return poses

    # ------------------------------------------------------------------ execution
    def _execute(self, poses):
        self.state = "EXECUTING"
        if not self.nav_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("navigate_through_poses server unavailable.")
            self.state = "MAPPING"
            self._busy = False
            return
        goal = NavigateThroughPoses.Goal()
        goal.poses = poses
        self.get_logger().info("Sending coverage path to Nav2…")
        future = self.nav_client.send_goal_async(
            goal, feedback_callback=self._nav_feedback
        )
        future.add_done_callback(self._goal_response)

    def _goal_response(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error("Nav2 rejected the coverage goal.")
            self.state = "MAPPING"
            self._busy = False
            return
        self.get_logger().info("Coverage goal accepted by Nav2.")
        handle.get_result_async().add_done_callback(self._goal_result)

    def _nav_feedback(self, feedback_msg):
        fb = feedback_msg.feedback
        remaining = getattr(fb, "number_of_poses_remaining", None)
        if remaining is not None:
            self._remaining = remaining

    def _goal_result(self, future):
        status = future.result().status
        # 4 == STATUS_SUCCEEDED in action_msgs/GoalStatus
        if status == 4:
            self.get_logger().info("Coverage COMPLETE.")
            self.state = "DONE"
        else:
            self.get_logger().warn(f"Coverage ended without success (status={status}).")
            self.state = "MAPPING"
        self._busy = False

    # ------------------------------------------------------------------ outputs
    def _publish_path(self, poses):
        path = Path()
        path.header.frame_id = self.map_frame
        path.header.stamp = self.get_clock().now().to_msg()
        path.poses = poses
        self.path_pub.publish(path)

    def _publish_status(self):
        msg = String()
        extra = ""
        if self.state == "EXECUTING" and hasattr(self, "_remaining"):
            extra = f" ({self._remaining} poses remaining)"
        msg.data = f"{self.state}{extra}"
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CoverageOrchestrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
