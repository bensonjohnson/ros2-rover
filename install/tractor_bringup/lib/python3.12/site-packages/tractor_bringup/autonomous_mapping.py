#!/usr/bin/env python3
"""
Autonomous Mapping Script for ROS2 Tractor
Drives around avoiding obstacles while creating a map using RealSense camera
"""
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import String
from nav2_msgs.action import NavigateToPose
import math
import random
import time
import numpy as np
from threading import Lock


class AutonomousMapper(Node):
    def __init__(self):
        super().__init__('autonomous_mapper')
        
        # Parameters
        self.declare_parameter('max_speed', 0.5)  # m/s
        self.declare_parameter('exploration_radius', 5.0)  # meters
        self.declare_parameter('safety_distance', 0.8)  # meters
        self.declare_parameter('mapping_duration', 600)  # seconds (10 minutes)
        self.declare_parameter('waypoint_timeout', 30.0)  # seconds
        
        # Get parameters
        self.max_speed = self.get_parameter('max_speed').get_parameter_value().double_value
        self.exploration_radius = self.get_parameter('exploration_radius').get_parameter_value().double_value
        self.safety_distance = self.get_parameter('safety_distance').get_parameter_value().double_value
        self.mapping_duration = self.get_parameter('mapping_duration').get_parameter_value().integer_value
        self.waypoint_timeout = self.get_parameter('waypoint_timeout').get_parameter_value().double_value
        
        # State variables
        self.current_pose = None
        self.map_data = None
        self.obstacle_detected = False
        self.emergency_stop = False
        self.mapping_complete = False
        self.start_time = time.time()
        self.lock = Lock()
        
        # Exploration state
        self.current_goal = None
        self.goal_start_time = None
        self.visited_points = []
        self.exploration_points = []
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/mapping_status', 10)
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/odometry/filtered', self.odom_callback, 10
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.realsense_scan_sub = self.create_subscription(
            LaserScan, '/realsense/scan', self.realsense_scan_callback, 10
        )
        
        # Action client for navigation
        self.nav_action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # Timers
        self.main_timer = self.create_timer(1.0, self.main_control_loop)
        self.safety_timer = self.create_timer(0.1, self.safety_check)
        self.status_timer = self.create_timer(5.0, self.publish_status)
        
        # Initialize exploration
        self.generate_exploration_points()
        
        self.get_logger().info("Autonomous Mapper initialized")
        self.get_logger().info(f"Mapping for {self.mapping_duration} seconds")
        self.get_logger().info(f"Max speed: {self.max_speed} m/s")
        self.get_logger().info(f"Safety distance: {self.safety_distance} m")
    
    def odom_callback(self, msg):
        """Update current pose from odometry"""
        with self.lock:
            self.current_pose = msg.pose.pose
    
    def map_callback(self, msg):
        """Update map data"""
        with self.lock:
            self.map_data = msg
    
    def scan_callback(self, msg):
        """Process laser scan for obstacle detection"""
        self.process_scan(msg, "laser")
    
    def realsense_scan_callback(self, msg):
        """Process RealSense scan for obstacle detection"""
        self.process_scan(msg, "realsense")
    
    def process_scan(self, msg, source):
        """Process scan data for obstacle detection"""
        if not msg.ranges:
            return
        
        # Check for close obstacles
        min_distance = float('inf')
        for i, distance in enumerate(msg.ranges):
            if msg.range_min <= distance <= msg.range_max:
                # Check front 180 degrees
                angle = msg.angle_min + i * msg.angle_increment
                if -math.pi/2 <= angle <= math.pi/2:  # Front 180 degrees
                    min_distance = min(min_distance, distance)
        
        # Update obstacle status
        with self.lock:
            if min_distance < self.safety_distance:
                self.obstacle_detected = True
                if min_distance < 0.3:  # Very close obstacle
                    self.emergency_stop = True
                    self.get_logger().warn(f"EMERGENCY STOP: Obstacle at {min_distance:.2f}m ({source})")
            else:
                self.obstacle_detected = False
    
    def generate_exploration_points(self):
        """Generate random exploration waypoints around the starting area"""
        self.exploration_points = []
        
        # Generate points in a grid pattern with some randomization
        for x in range(-2, 3):  # -10m to +10m
            for y in range(-2, 3):  # -10m to +10m
                if x == 0 and y == 0:  # Skip origin
                    continue
                
                # Add some randomization
                rand_x = x * 2.5 + random.uniform(-1.0, 1.0)
                rand_y = y * 2.5 + random.uniform(-1.0, 1.0)
                
                # Limit to exploration radius
                distance = math.sqrt(rand_x**2 + rand_y**2)
                if distance <= self.exploration_radius:
                    self.exploration_points.append((rand_x, rand_y))
        
        # Shuffle for random exploration order
        random.shuffle(self.exploration_points)
        
        self.get_logger().info(f"Generated {len(self.exploration_points)} exploration points")
    
    def get_next_waypoint(self):
        """Get the next exploration waypoint"""
        if not self.current_pose:
            return None
        
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        
        # Find unvisited exploration point
        for point in self.exploration_points:
            point_x, point_y = point
            
            # Check if already visited
            visited = False
            for visited_point in self.visited_points:
                dist = math.sqrt((point_x - visited_point[0])**2 + (point_y - visited_point[1])**2)
                if dist < 1.0:  # Consider visited if within 1m
                    visited = True
                    break
            
            if not visited:
                return (point_x, point_y)
        
        # If all points visited, generate new ones further out
        self.exploration_radius += 2.0
        self.get_logger().info(f"Expanding exploration radius to {self.exploration_radius}m")
        self.generate_exploration_points()
        
        if self.exploration_points:
            return self.exploration_points[0]
        
        return None
    
    def send_goal(self, x, y):
        """Send navigation goal to Nav2"""
        if not self.nav_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Nav2 action server not available")
            return False
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation.w = 1.0
        
        self.get_logger().info(f"Sending goal: ({x:.2f}, {y:.2f})")
        
        future = self.nav_action_client.send_goal_async(goal_msg)
        self.current_goal = (x, y)
        self.goal_start_time = time.time()
        
        return True
    
    def emergency_stop_robot(self):
        """Stop the robot immediately"""
        stop_cmd = Twist()
        for _ in range(5):  # Send multiple stop commands
            self.cmd_vel_pub.publish(stop_cmd)
            time.sleep(0.1)
        
        self.get_logger().warn("Robot emergency stopped")
    
    def main_control_loop(self):
        """Main control loop for autonomous mapping"""
        # Check if mapping duration exceeded
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.mapping_duration:
            if not self.mapping_complete:
                self.mapping_complete = True
                self.emergency_stop_robot()
                self.get_logger().info("Mapping duration completed. Stopping robot.")
            return
        
        # Check emergency stop
        if self.emergency_stop:
            self.emergency_stop_robot()
            time.sleep(1.0)  # Wait before clearing emergency
            self.emergency_stop = False
            return
        
        # Check if we need a new goal
        if self.current_goal is None:
            waypoint = self.get_next_waypoint()
            if waypoint:
                self.send_goal(waypoint[0], waypoint[1])
            else:
                self.get_logger().info("No more waypoints to explore")
                return
        
        # Check goal timeout
        if self.goal_start_time and (time.time() - self.goal_start_time) > self.waypoint_timeout:
            self.get_logger().warn("Goal timeout, selecting new waypoint")
            if self.current_goal:
                self.visited_points.append(self.current_goal)
            self.current_goal = None
            self.goal_start_time = None
        
        # Check if we've reached the current goal
        if self.current_goal and self.current_pose:
            current_x = self.current_pose.position.x
            current_y = self.current_pose.position.y
            goal_x, goal_y = self.current_goal
            
            distance_to_goal = math.sqrt((current_x - goal_x)**2 + (current_y - goal_y)**2)
            
            if distance_to_goal < 1.0:  # Within 1 meter of goal
                self.get_logger().info(f"Reached waypoint: ({goal_x:.2f}, {goal_y:.2f})")
                self.visited_points.append(self.current_goal)
                self.current_goal = None
                self.goal_start_time = None
    
    def safety_check(self):
        """High-frequency safety monitoring"""
        with self.lock:
            if self.emergency_stop:
                self.emergency_stop_robot()
    
    def publish_status(self):
        """Publish mapping status"""
        elapsed_time = time.time() - self.start_time
        remaining_time = max(0, self.mapping_duration - elapsed_time)
        
        status_msg = String()
        status_msg.data = f"Mapping Status: {len(self.visited_points)} waypoints visited, " \
                         f"{remaining_time:.0f}s remaining, " \
                         f"Emergency: {'YES' if self.emergency_stop else 'NO'}, " \
                         f"Obstacle: {'YES' if self.obstacle_detected else 'NO'}"
        
        self.status_pub.publish(status_msg)
        self.get_logger().info(status_msg.data)
    
    def shutdown_sequence(self):
        """Clean shutdown sequence"""
        self.get_logger().info("Shutting down autonomous mapper...")
        self.emergency_stop_robot()
        
        # Cancel any active navigation goals
        if self.nav_action_client.server_is_ready():
            self.nav_action_client.cancel_all_goals()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        mapper = AutonomousMapper()
        
        def signal_handler(sig, frame):
            mapper.shutdown_sequence()
            rclpy.shutdown()
        
        import signal
        signal.signal(signal.SIGINT, signal_handler)
        
        rclpy.spin(mapper)
        
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()