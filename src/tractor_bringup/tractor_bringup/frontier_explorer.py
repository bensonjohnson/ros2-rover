#!/usr/bin/env python3
"""
Frontier-based Exploration Node for ROS2 Autonomous SLAM Mapping

This node implements frontier-based exploration to systematically map entire areas:
1. Analyzes the current SLAM map to find unexplored frontiers
2. Selects the best frontier based on distance and size
3. Sends navigation goals to explore new areas
4. Handles dynamic obstacles (people) through collision monitoring
5. Ensures complete area coverage by prioritizing unexplored regions

Key features:
- Uses occupancy grid map from SLAM Toolbox
- Implements frontier detection algorithm
- Smart goal selection (closest large frontiers first)
- Timeout handling for unreachable goals
- Integration with Nav2 for obstacle avoidance
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import numpy as np
import math
from collections import deque
from typing import List, Tuple, Optional

# ROS2 message types
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from nav2_msgs.action import NavigateToPose
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs

class FrontierExplorer(Node):
    """
    Frontier-based exploration node for autonomous mapping
    """
    
    def __init__(self):
        super().__init__('frontier_explorer')
        
        # Parameters
        self.declare_parameter('max_exploration_radius', 10.0)
        self.declare_parameter('frontier_search_radius', 2.0)
        self.declare_parameter('min_frontier_size', 10)
        self.declare_parameter('exploration_frequency', 1.0)
        self.declare_parameter('goal_timeout', 30.0)
        # use_sim_time is declared by the launch system
        
        self.max_exploration_radius = self.get_parameter('max_exploration_radius').get_parameter_value().double_value
        self.frontier_search_radius = self.get_parameter('frontier_search_radius').get_parameter_value().double_value
        self.min_frontier_size = self.get_parameter('min_frontier_size').get_parameter_value().integer_value
        self.exploration_frequency = self.get_parameter('exploration_frequency').get_parameter_value().double_value
        self.goal_timeout = self.get_parameter('goal_timeout').get_parameter_value().double_value
        
        # State variables
        self.current_map = None
        self.current_pose = None
        self.start_position = None
        self.frontiers = []
        self.current_goal = None
        self.goal_active = False
        self.goal_start_time = None
        self.exploration_complete = False
        self.explored_goals = set()  # Track previously explored goals
        
        # Use callback groups for parallel processing
        callback_group = ReentrantCallbackGroup()
        
        # TF2 setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Publishers
        self.status_pub = self.create_publisher(String, '/exploration_status', 10)
        self.frontier_markers_pub = self.create_publisher(MarkerArray, '/frontier_markers', 10)
        self.goal_marker_pub = self.create_publisher(Marker, '/exploration_goal', 10)
        
        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10,
            callback_group=callback_group
        )
        
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10,
            callback_group=callback_group
        )
        
        # Nav2 action client
        self.nav_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose',
            callback_group=callback_group
        )
        
        # Timers
        self.exploration_timer = self.create_timer(
            1.0 / self.exploration_frequency, self.exploration_loop,
            callback_group=callback_group
        )
        
        self.status_timer = self.create_timer(
            2.0, self.publish_status,
            callback_group=callback_group
        )
        
        self.get_logger().info("Frontier Explorer started")
        self.get_logger().info(f"Max exploration radius: {self.max_exploration_radius}m")
        self.get_logger().info(f"Min frontier size: {self.min_frontier_size} cells")
        
    def map_callback(self, msg):
        """Process new map data"""
        self.current_map = msg
        
    def odom_callback(self, msg):
        """Update current robot position"""
        self.current_pose = msg.pose.pose
        
        # Set start position on first odometry message
        if self.start_position is None:
            self.start_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
            self.get_logger().info(f"Start position set: {self.start_position}")
            
    def exploration_loop(self):
        """Main exploration control loop"""
        # Check if we have required data
        if not self.current_map or not self.current_pose:
            return
            
        # Check if we're too far from start position
        if self.start_position:
            current_pos = (self.current_pose.position.x, self.current_pose.position.y)
            distance_from_start = math.sqrt(
                (current_pos[0] - self.start_position[0])**2 + 
                (current_pos[1] - self.start_position[1])**2
            )
            
            if distance_from_start > self.max_exploration_radius:
                if not self.exploration_complete:
                    self.get_logger().info(f"Reached maximum exploration radius ({self.max_exploration_radius}m)")
                    self.exploration_complete = True
                    self.cancel_current_goal()
                return
        
        # Check if current goal has timed out
        if self.goal_active and self.goal_start_time:
            elapsed = self.get_clock().now().nanoseconds / 1e9 - self.goal_start_time
            if elapsed > self.goal_timeout:
                self.get_logger().warn(f"Goal timeout after {elapsed:.1f}s, finding new goal")
                self.cancel_current_goal()
                
        # Find and explore frontiers if no active goal
        if not self.goal_active and not self.exploration_complete:
            self.find_and_explore_frontiers()
            
    def find_and_explore_frontiers(self):
        """Find frontiers in the map and select the best one to explore"""
        if not self.current_map:
            return
            
        # Find all frontiers in the map
        self.frontiers = self.detect_frontiers()
        
        if not self.frontiers:
            if not self.exploration_complete:
                self.get_logger().info("No more frontiers found - exploration complete!")
                self.exploration_complete = True
            return
            
        # Filter out previously explored goals
        new_frontiers = []
        for frontier in self.frontiers:
            frontier_key = (round(frontier[0], 1), round(frontier[1], 1))
            if frontier_key not in self.explored_goals:
                new_frontiers.append(frontier)
                
        if not new_frontiers:
            if not self.exploration_complete:
                self.get_logger().info("All frontiers have been explored - exploration complete!")
                self.exploration_complete = True
            return
            
        # Select best frontier (closest large frontier)
        best_frontier = self.select_best_frontier(new_frontiers)
        
        if best_frontier:
            self.send_exploration_goal(best_frontier)
            
        # Publish frontier markers for visualization
        self.publish_frontier_markers()
        
    def detect_frontiers(self) -> List[Tuple[float, float, int]]:
        """
        Detect frontiers in the occupancy grid map
        Returns list of (x, y, size) tuples in world coordinates
        """
        if not self.current_map:
            return []
            
        # Convert occupancy grid to numpy array
        width = self.current_map.info.width
        height = self.current_map.info.height
        resolution = self.current_map.info.resolution
        origin_x = self.current_map.info.origin.position.x
        origin_y = self.current_map.info.origin.position.y
        
        # Reshape map data
        map_data = np.array(self.current_map.data).reshape((height, width))
        
        # Find frontier cells using flood fill
        frontiers = []
        visited = np.zeros((height, width), dtype=bool)
        
        for y in range(height):
            for x in range(width):
                if visited[y, x] or map_data[y, x] != -1:  # Skip known cells
                    continue
                    
                # Check if this unknown cell is adjacent to free space
                if self.is_frontier_cell(map_data, x, y):
                    # Flood fill to find connected frontier
                    frontier_cells = self.flood_fill_frontier(map_data, visited, x, y)
                    
                    if len(frontier_cells) >= self.min_frontier_size:
                        # Calculate centroid of frontier
                        centroid_x = sum(cell[0] for cell in frontier_cells) / len(frontier_cells)
                        centroid_y = sum(cell[1] for cell in frontier_cells) / len(frontier_cells)
                        
                        # Convert to world coordinates
                        world_x = origin_x + centroid_x * resolution
                        world_y = origin_y + centroid_y * resolution
                        
                        # Check if frontier is within exploration radius
                        if self.start_position:
                            distance = math.sqrt(
                                (world_x - self.start_position[0])**2 + 
                                (world_y - self.start_position[1])**2
                            )
                            if distance <= self.max_exploration_radius:
                                frontiers.append((world_x, world_y, len(frontier_cells)))
        
        return frontiers
        
    def is_frontier_cell(self, map_data, x, y) -> bool:
        """Check if a cell is a frontier (unknown cell adjacent to free space)"""
        height, width = map_data.shape
        
        # Check 8-connected neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                    
                nx, ny = x + dx, y + dy
                
                # Check bounds
                if 0 <= nx < width and 0 <= ny < height:
                    # If neighbor is free space (0), this is a frontier
                    if map_data[ny, nx] == 0:
                        return True
                        
        return False
        
    def flood_fill_frontier(self, map_data, visited, start_x, start_y) -> List[Tuple[int, int]]:
        """Flood fill to find connected frontier cells"""
        height, width = map_data.shape
        frontier_cells = []
        queue = deque([(start_x, start_y)])
        
        while queue:
            x, y = queue.popleft()
            
            if visited[y, x]:
                continue
                
            visited[y, x] = True
            
            # Check if this is still a frontier cell
            if map_data[y, x] == -1 and self.is_frontier_cell(map_data, x, y):
                frontier_cells.append((x, y))
                
                # Add 4-connected neighbors to queue
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    
                    if (0 <= nx < width and 0 <= ny < height and 
                        not visited[ny, nx] and map_data[ny, nx] == -1):
                        queue.append((nx, ny))
                        
        return frontier_cells
        
    def select_best_frontier(self, frontiers) -> Optional[Tuple[float, float, int]]:
        """Select the best frontier to explore based on distance and size"""
        if not frontiers or not self.current_pose:
            return None
            
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        
        best_frontier = None
        best_score = float('-inf')
        
        for frontier_x, frontier_y, frontier_size in frontiers:
            # Calculate distance to frontier
            distance = math.sqrt(
                (frontier_x - current_x)**2 + (frontier_y - current_y)**2
            )
            
            # Skip frontiers that are too close (likely noise)
            if distance < 0.5:
                continue
                
            # Score based on size/distance ratio (prefer large close frontiers)
            score = frontier_size / (distance + 0.1)  # Add small value to avoid division by zero
            
            if score > best_score:
                best_score = score
                best_frontier = (frontier_x, frontier_y, frontier_size)
                
        return best_frontier
        
    def send_exploration_goal(self, frontier):
        """Send navigation goal to explore a frontier"""
        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("Nav2 action server not available")
            return
            
        frontier_x, frontier_y, frontier_size = frontier
        
        # Create navigation goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = frontier_x
        goal_msg.pose.pose.position.y = frontier_y
        goal_msg.pose.pose.position.z = 0.0
        
        # Set orientation towards the frontier
        if self.current_pose:
            dx = frontier_x - self.current_pose.position.x
            dy = frontier_y - self.current_pose.position.y
            yaw = math.atan2(dy, dx)
            
            # Convert to quaternion
            goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
            goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        else:
            goal_msg.pose.pose.orientation.w = 1.0
            
        # Send goal
        self.get_logger().info(f"Exploring frontier at ({frontier_x:.2f}, {frontier_y:.2f}) - size: {frontier_size}")
        
        send_goal_future = self.nav_client.send_goal_async(
            goal_msg, feedback_callback=self.nav_feedback_callback
        )
        send_goal_future.add_done_callback(self.nav_response_callback)
        
        self.current_goal = frontier
        self.goal_active = True
        self.goal_start_time = self.get_clock().now().nanoseconds / 1e9
        
        # Mark this goal as explored
        goal_key = (round(frontier_x, 1), round(frontier_y, 1))
        self.explored_goals.add(goal_key)
        
        # Publish goal marker
        self.publish_goal_marker(frontier_x, frontier_y)
        
    def nav_response_callback(self, future):
        """Handle navigation goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Navigation goal rejected")
            self.goal_active = False
            return
            
        self.get_logger().info("Navigation goal accepted")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.nav_result_callback)
        
    def nav_feedback_callback(self, feedback_msg):
        """Handle navigation feedback"""
        # Optional: could log progress or check for obstacles
        pass
        
    def nav_result_callback(self, future):
        """Handle navigation result"""
        result = future.result().result
        status = future.result().status
        
        if status == 4:  # SUCCEEDED
            self.get_logger().info("Successfully reached frontier!")
        elif status == 3:  # CANCELED
            self.get_logger().info("Navigation to frontier was canceled")
        else:
            self.get_logger().warn(f"Failed to reach frontier (status: {status})")
            
        self.goal_active = False
        self.goal_start_time = None
        self.current_goal = None
        
    def cancel_current_goal(self):
        """Cancel the current navigation goal"""
        if self.goal_active:
            self.nav_client.cancel_all_goals()
            self.goal_active = False
            self.goal_start_time = None
            self.current_goal = None
            
    def publish_frontier_markers(self):
        """Publish visualization markers for frontiers"""
        marker_array = MarkerArray()
        
        for i, (x, y, size) in enumerate(self.frontiers):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "frontiers"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.1
            marker.pose.orientation.w = 1.0
            
            # Size based on frontier size
            scale = 0.1 + min(size / 100.0, 0.4)
            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale
            
            # Color based on size (red = small, green = large)
            marker.color.r = max(0.0, 1.0 - size / 50.0)
            marker.color.g = min(1.0, size / 50.0)
            marker.color.b = 0.0
            marker.color.a = 0.7
            
            marker.lifetime.sec = 5
            marker_array.markers.append(marker)
            
        self.frontier_markers_pub.publish(marker_array)
        
    def publish_goal_marker(self, x, y):
        """Publish visualization marker for current exploration goal"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "exploration_goal"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.2
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.5
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        marker.lifetime.sec = 30
        self.goal_marker_pub.publish(marker)
        
    def publish_status(self):
        """Publish exploration status"""
        status_msg = String()
        
        if self.exploration_complete:
            status_msg.data = "Exploration COMPLETE - All areas mapped"
        elif self.goal_active and self.current_goal:
            elapsed = self.get_clock().now().nanoseconds / 1e9 - self.goal_start_time
            remaining = max(0, self.goal_timeout - elapsed)
            status_msg.data = (f"Exploring frontier at ({self.current_goal[0]:.1f}, {self.current_goal[1]:.1f}) - "
                             f"Size: {self.current_goal[2]}, Time remaining: {remaining:.0f}s")
        else:
            status_msg.data = f"Searching for frontiers - Found: {len(self.frontiers)}"
            
        self.status_pub.publish(status_msg)
        self.get_logger().info(status_msg.data)


def main(args=None):
    rclpy.init(args=args)
    
    explorer = FrontierExplorer()
    executor = MultiThreadedExecutor()
    
    try:
        rclpy.spin(explorer, executor=executor)
    except KeyboardInterrupt:
        explorer.get_logger().info("Shutting down frontier explorer")
    finally:
        explorer.cancel_current_goal()
        explorer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()