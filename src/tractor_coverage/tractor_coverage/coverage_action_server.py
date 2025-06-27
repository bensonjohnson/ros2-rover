#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Point, PoseStamped, Twist
from nav_msgs.msg import Path
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration
import tf2_ros
import tf2_geometry_msgs
import math
import time
from typing import List, Optional

# Import our custom action (will need to be generated)
# from tractor_coverage_msgs.action import CoveragePath
# For now, we'll create a placeholder structure

from .coverage_planner import CoveragePlanner


class CoverageActionServer(Node):
    """
    Action server for executing coverage patterns.
    Integrates with Nav2 for navigation and implement controllers.
    """
    
    def __init__(self):
        super().__init__('coverage_action_server')
        
        # Parameters
        self.declare_parameter('default_tool_width', 1.0)
        self.declare_parameter('default_overlap', 0.1)
        self.declare_parameter('default_work_speed', 0.5)
        self.declare_parameter('waypoint_tolerance', 0.25)
        self.declare_parameter('max_nav_timeout', 30.0)
        
        self.default_tool_width = self.get_parameter('default_tool_width').value
        self.default_overlap = self.get_parameter('default_overlap').value
        self.default_work_speed = self.get_parameter('default_work_speed').value
        self.waypoint_tolerance = self.get_parameter('waypoint_tolerance').value
        self.max_nav_timeout = self.get_parameter('max_nav_timeout').value
        
        # Coverage planner
        self.planner = CoveragePlanner()
        
        # Current operation state
        self.current_goal = None
        self.current_path = []
        self.current_waypoint_index = 0
        self.operation_start_time = None
        self.is_executing = False
        self.current_operation_type = ""
        
        # TF buffer for coordinate transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Publishers
        self.path_pub = self.create_publisher(Path, 'coverage_path', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.coverage_markers_pub = self.create_publisher(MarkerArray, 'coverage_markers', 10)
        self.status_pub = self.create_publisher(String, 'coverage_status', 10)
        
        # Implement control publishers
        self.mower_enable_pub = self.create_publisher(Bool, 'mower/enable', 10)
        self.sprayer_enable_pub = self.create_publisher(Bool, 'sprayer/enable', 10)
        
        # Subscribers
        self.nav_status_sub = self.create_subscription(
            String, '/navigation_status', self.nav_status_callback, 10)
        
        # Action server (placeholder - would use actual action definition)
        # self._action_server = ActionServer(
        #     self, CoveragePath, 'coverage_path',
        #     execute_callback=self.execute_callback,
        #     goal_callback=self.goal_callback,
        #     cancel_callback=self.cancel_callback
        # )
        
        # Timer for operation monitoring
        self.execution_timer = self.create_timer(0.5, self.execution_timer_callback)
        
        self.get_logger().info('Coverage Action Server initialized')
    
    def goal_callback(self, goal_request):
        """Accept or reject incoming goals"""
        if self.is_executing:
            self.get_logger().warn('Coverage operation already in progress')
            return GoalResponse.REJECT
        
        # Validate goal parameters
        if not self.validate_goal(goal_request):
            self.get_logger().error('Invalid goal parameters')
            return GoalResponse.REJECT
        
        self.get_logger().info(f'Accepting coverage goal: {goal_request.operation_type}')
        return GoalResponse.ACCEPT
    
    def cancel_callback(self, goal_handle):
        """Handle cancellation requests"""
        self.get_logger().info('Coverage operation cancelled')
        self.stop_operation()
        return CancelResponse.ACCEPT
    
    def execute_callback(self, goal_handle):
        """Main execution callback for coverage operations"""
        self.get_logger().info('Executing coverage operation...')
        
        request = goal_handle.request
        feedback_msg = goal_handle.feedback_msg
        
        try:
            # Setup operation
            self.setup_operation(request)
            
            # Plan coverage path
            path_points = self.plan_coverage_path(request)
            
            if not path_points:
                goal_handle.abort()
                result = goal_handle.result_msg
                result.success = False
                result.message = "Failed to generate valid coverage path"
                return result
            
            # Convert to navigation path
            self.current_path = self.create_nav_path(path_points)
            self.current_waypoint_index = 0
            
            # Publish path for visualization
            self.publish_coverage_path(self.current_path)
            self.publish_coverage_markers(path_points, request.operation_type)
            
            # Start operation
            self.is_executing = True
            self.operation_start_time = time.time()
            self.current_operation_type = request.operation_type
            
            # Enable appropriate implement
            self.enable_implement(request.operation_type, True)
            
            # Execute waypoint following
            success = self.execute_waypoint_following(goal_handle, feedback_msg)
            
            # Complete operation
            self.enable_implement(request.operation_type, False)
            self.is_executing = False
            
            # Create result
            result = goal_handle.result_msg
            result.success = success
            result.total_waypoints = len(path_points)
            result.total_distance = self.calculate_path_distance(path_points)
            result.area_covered = self.planner.calculate_coverage_area(
                list(request.boundary.points), 
                [list(obs.points) for obs in request.obstacles]
            )
            
            elapsed_time = time.time() - self.operation_start_time
            result.total_time = Duration(sec=int(elapsed_time), 
                                       nanosec=int((elapsed_time % 1) * 1e9))
            
            if success:
                result.message = "Coverage operation completed successfully"
                goal_handle.succeed()
            else:
                result.message = "Coverage operation failed or was interrupted"
                goal_handle.abort()
            
            return result
            
        except Exception as e:
            self.get_logger().error(f'Coverage execution error: {str(e)}')
            goal_handle.abort()
            result = goal_handle.result_msg
            result.success = False
            result.message = f"Execution error: {str(e)}"
            return result
    
    def setup_operation(self, request):
        """Setup coverage planner with goal parameters"""
        tool_width = request.tool_width if request.tool_width > 0 else self.default_tool_width
        overlap = request.overlap_percentage if request.overlap_percentage >= 0 else self.default_overlap
        
        self.planner.set_tool_parameters(tool_width, overlap)
        
        # Set vehicle parameters based on operation type
        if request.operation_type == "mowing":
            self.planner.set_vehicle_parameters(turn_radius=0.5, border_offset=0.3)
        elif request.operation_type == "spraying":
            self.planner.set_vehicle_parameters(turn_radius=0.8, border_offset=0.5)
        else:
            self.planner.set_vehicle_parameters(turn_radius=0.6, border_offset=0.4)
    
    def plan_coverage_path(self, request) -> List[Point]:
        """Generate coverage path from goal parameters"""
        # Convert boundary points
        boundary_points = [Point(x=p.x, y=p.y, z=0.0) for p in request.boundary.points]
        
        # Convert obstacle points
        obstacles = []
        for obstacle in request.obstacles:
            obs_points = [Point(x=p.x, y=p.y, z=0.0) for p in obstacle.points]
            obstacles.append(obs_points)
        
        # Plan path
        path_points = self.planner.plan_coverage_path(boundary_points, obstacles)
        
        # Optimize if requested
        if request.optimize_path:
            path_points = self.planner.optimize_path(path_points)
        
        return path_points
    
    def create_nav_path(self, points: List[Point]) -> Path:
        """Convert point list to nav_msgs/Path"""
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = "map"
        
        for point in points:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position = point
            pose.pose.orientation.w = 1.0  # Default orientation
            path.poses.append(pose)
        
        return path
    
    def execute_waypoint_following(self, goal_handle, feedback_msg) -> bool:
        """Execute waypoint following using Nav2"""
        total_waypoints = len(self.current_path.poses)
        
        while self.current_waypoint_index < total_waypoints and self.is_executing:
            # Check for cancellation
            if goal_handle.is_cancel_requested:
                self.get_logger().info('Coverage operation cancelled during execution')
                return False
            
            # Get current waypoint
            current_pose = self.current_path.poses[self.current_waypoint_index]
            
            # Publish navigation goal
            self.goal_pub.publish(current_pose)
            
            # Wait for waypoint completion
            if not self.wait_for_waypoint_completion(current_pose, timeout=self.max_nav_timeout):
                self.get_logger().error(f'Failed to reach waypoint {self.current_waypoint_index}')
                return False
            
            # Update progress
            self.current_waypoint_index += 1
            
            # Publish feedback
            self.publish_feedback(goal_handle, feedback_msg, total_waypoints)
            
            # Small delay between waypoints
            time.sleep(0.1)
        
        return self.current_waypoint_index >= total_waypoints
    
    def wait_for_waypoint_completion(self, target_pose: PoseStamped, timeout: float) -> bool:
        """Wait for robot to reach target waypoint"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if not self.is_executing:
                return False
            
            # Get current robot pose
            try:
                robot_pose = self.get_robot_pose()
                if robot_pose is None:
                    time.sleep(0.1)
                    continue
                
                # Calculate distance to target
                distance = self.calculate_distance(robot_pose, target_pose)
                
                if distance < self.waypoint_tolerance:
                    return True
                
            except Exception as e:
                self.get_logger().debug(f'Error getting robot pose: {e}')
            
            time.sleep(0.1)
        
        return False
    
    def get_robot_pose(self) -> Optional[PoseStamped]:
        """Get current robot pose in map frame"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time()
            )
            
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = 'map'
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z
            pose.pose.orientation = transform.transform.rotation
            
            return pose
            
        except Exception as e:
            self.get_logger().debug(f'Transform lookup failed: {e}')
            return None
    
    def calculate_distance(self, pose1: PoseStamped, pose2: PoseStamped) -> float:
        """Calculate 2D distance between two poses"""
        dx = pose1.pose.position.x - pose2.pose.position.x
        dy = pose1.pose.position.y - pose2.pose.position.y
        return math.sqrt(dx*dx + dy*dy)
    
    def publish_feedback(self, goal_handle, feedback_msg, total_waypoints: int):
        """Publish operation feedback"""
        feedback_msg.current_waypoint = self.current_waypoint_index
        feedback_msg.total_waypoints = total_waypoints
        feedback_msg.percent_complete = (self.current_waypoint_index / total_waypoints) * 100.0
        
        # Estimate remaining distance and time
        remaining_waypoints = total_waypoints - self.current_waypoint_index
        if remaining_waypoints > 0 and self.operation_start_time:
            elapsed_time = time.time() - self.operation_start_time
            time_per_waypoint = elapsed_time / max(1, self.current_waypoint_index)
            estimated_remaining = remaining_waypoints * time_per_waypoint
            
            feedback_msg.time_remaining = Duration(
                sec=int(estimated_remaining),
                nanosec=int((estimated_remaining % 1) * 1e9)
            )
        
        # Current robot position
        robot_pose = self.get_robot_pose()
        if robot_pose:
            feedback_msg.current_position = robot_pose.pose.position
        
        feedback_msg.current_status = f"Waypoint {self.current_waypoint_index}/{total_waypoints}"
        feedback_msg.implement_active = self.is_executing
        
        goal_handle.publish_feedback(feedback_msg)
    
    def enable_implement(self, operation_type: str, enable: bool):
        """Enable/disable appropriate implement"""
        msg = Bool()
        msg.data = enable
        
        if operation_type == "mowing":
            self.mower_enable_pub.publish(msg)
        elif operation_type == "spraying":
            self.sprayer_enable_pub.publish(msg)
        
        action = "Enabled" if enable else "Disabled"
        self.get_logger().info(f'{action} {operation_type} implement')
    
    def publish_coverage_path(self, path: Path):
        """Publish coverage path for visualization"""
        self.path_pub.publish(path)
    
    def publish_coverage_markers(self, points: List[Point], operation_type: str):
        """Publish visualization markers for coverage area"""
        marker_array = MarkerArray()
        
        # Path markers
        path_marker = Marker()
        path_marker.header.stamp = self.get_clock().now().to_msg()
        path_marker.header.frame_id = "map"
        path_marker.ns = "coverage_path"
        path_marker.id = 0
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD
        path_marker.scale.x = 0.1
        
        # Color based on operation type
        if operation_type == "mowing":
            path_marker.color.r = 0.0
            path_marker.color.g = 1.0
            path_marker.color.b = 0.0
        elif operation_type == "spraying":
            path_marker.color.r = 0.0
            path_marker.color.g = 0.0
            path_marker.color.b = 1.0
        else:
            path_marker.color.r = 1.0
            path_marker.color.g = 1.0
            path_marker.color.b = 0.0
        
        path_marker.color.a = 0.8
        
        for point in points:
            path_marker.points.append(point)
        
        marker_array.markers.append(path_marker)
        self.coverage_markers_pub.publish(marker_array)
    
    def calculate_path_distance(self, points: List[Point]) -> float:
        """Calculate total path distance"""
        if len(points) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(points)):
            dx = points[i].x - points[i-1].x
            dy = points[i].y - points[i-1].y
            total_distance += math.sqrt(dx*dx + dy*dy)
        
        return total_distance
    
    def validate_goal(self, request) -> bool:
        """Validate goal parameters"""
        # Check boundary has enough points
        if len(request.boundary.points) < 3:
            return False
        
        # Validate tool width
        if request.tool_width < 0:
            return False
        
        # Validate overlap percentage
        if request.overlap_percentage < 0 or request.overlap_percentage >= 1.0:
            return False
        
        # Validate operation type
        if request.operation_type not in ["mowing", "spraying", "general"]:
            return False
        
        return True
    
    def stop_operation(self):
        """Stop current coverage operation"""
        self.is_executing = False
        
        # Stop robot
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)
        
        # Disable implements
        if self.current_operation_type:
            self.enable_implement(self.current_operation_type, False)
        
        # Publish status
        status_msg = String()
        status_msg.data = "STOPPED"
        self.status_pub.publish(status_msg)
    
    def execution_timer_callback(self):
        """Timer callback for monitoring execution"""
        if self.is_executing:
            status_msg = String()
            status_msg.data = f"EXECUTING:{self.current_operation_type}:{self.current_waypoint_index}"
            self.status_pub.publish(status_msg)
    
    def nav_status_callback(self, msg):
        """Handle navigation status updates"""
        # Process navigation feedback if needed
        pass


def main(args=None):
    rclpy.init(args=args)
    
    coverage_server = CoverageActionServer()
    
    # Use multi-threaded executor for action server
    executor = MultiThreadedExecutor()
    executor.add_node(coverage_server)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        coverage_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()