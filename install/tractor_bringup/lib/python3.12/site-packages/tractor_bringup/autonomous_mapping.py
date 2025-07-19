#!/usr/bin/env python3
"""
Fresh Autonomous Mapping Script for ROS2 Tractor
Simple forward exploration using Nav2 for obstacle avoidance
Works in odom frame without SLAM - uses point cloud directly for costmaps
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from nav2_msgs.action import NavigateToPose
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import math
import time


class SimpleAutonomousMapper(Node):
    """
    Simple autonomous mapping node that:
    1. Drives forward in straight lines
    2. Uses Nav2 for obstacle avoidance
    3. Works in odom frame (no SLAM required)
    4. Uses RealSense IMU for orientation
    """
    
    def __init__(self):
        super().__init__('simple_autonomous_mapper')
        
        # Parameters
        self.declare_parameter('max_speed', 0.3)
        self.declare_parameter('exploration_distance', 3.0)  # How far to drive forward
        self.declare_parameter('mapping_duration', 600)  # 10 minutes
        
        self.max_speed = self.get_parameter('max_speed').get_parameter_value().double_value
        self.exploration_distance = self.get_parameter('exploration_distance').get_parameter_value().double_value
        self.mapping_duration = self.get_parameter('mapping_duration').get_parameter_value().integer_value
        
        # State variables
        self.current_pose = None
        self.current_heading = 0.0
        self.start_time = time.time()
        self.mapping_complete = False
        self.current_goal = None
        self.nav_goal_active = False
        self.goal_start_time = None
        self.goal_timeout = 30.0  # Force new goal after 30 seconds
        self.last_goal_fail_time = None
        self.goal_retry_delay = 10.0  # Wait 10 seconds after goal failure
        
        # Use callback group for parallel processing
        callback_group = ReentrantCallbackGroup()
        
        # Publishers
        self.status_pub = self.create_publisher(String, '/mapping_status', 10)
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10,
            callback_group=callback_group
        )
        
        self.imu_sub = self.create_subscription(
            Imu, '/camera/camera/imu', self.imu_callback, 10,
            callback_group=callback_group
        )
        
        # Nav2 action client
        self.nav_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose',
            callback_group=callback_group
        )
        
        # Timers
        self.control_timer = self.create_timer(
            5.0, self.control_loop, callback_group=callback_group
        )
        self.status_timer = self.create_timer(
            5.0, self.publish_status, callback_group=callback_group
        )
        
        self.get_logger().info("Simple Autonomous Mapper started")
        self.get_logger().info(f"Will map for {self.mapping_duration} seconds")
        self.get_logger().info(f"Forward exploration distance: {self.exploration_distance}m")
        
    def odom_callback(self, msg):
        """Update current pose from odometry"""
        self.current_pose = msg.pose.pose
        
    def imu_callback(self, msg):
        """Update current heading from RealSense IMU"""
        # Convert quaternion to yaw
        q = msg.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_heading = math.atan2(siny_cosp, cosy_cosp)
        
    def control_loop(self):
        """Main control loop - simple forward exploration"""
        # Check if mapping time is up
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.mapping_duration:
            if not self.mapping_complete:
                self.mapping_complete = True
                self.get_logger().info("Mapping duration complete - stopping")
                self.cancel_navigation()
            return
            
        # Skip if we don't have pose data yet
        if not self.current_pose:
            self.get_logger().info("Waiting for odometry data...")
            return
            
        # Skip if Nav2 is not ready
        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("Nav2 action server not available")
            return
            
        # Check if current goal has been active too long
        if self.nav_goal_active and self.goal_start_time:
            goal_elapsed = time.time() - self.goal_start_time
            if goal_elapsed > self.goal_timeout:
                self.get_logger().warn(f"Goal timeout after {goal_elapsed:.1f}s - forcing new goal")
                self.nav_goal_active = False
                self.goal_start_time = None
        
        # Check if enough time has passed since last goal failure
        if self.last_goal_fail_time:
            fail_elapsed = time.time() - self.last_goal_fail_time
            if fail_elapsed < self.goal_retry_delay:
                return  # Wait before trying again
        
        # If no active goal, send a new forward exploration goal
        if not self.nav_goal_active:
            self.send_forward_goal()
            
    def send_forward_goal(self):
        """Send a goal to drive forward in current direction"""
        if not self.current_pose:
            return
            
        # Get current position
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        
        # Calculate goal position based on current heading
        goal_x = current_x + self.exploration_distance * math.cos(self.current_heading)
        goal_y = current_y + self.exploration_distance * math.sin(self.current_heading)
        
        # Create navigation goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'odom'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = goal_x
        goal_msg.pose.pose.position.y = goal_y
        goal_msg.pose.pose.position.z = 0.0
        
        # Keep current orientation
        goal_msg.pose.pose.orientation = self.current_pose.orientation
        
        # Send goal
        self.get_logger().info(f"Sending forward goal: ({goal_x:.2f}, {goal_y:.2f})")
        send_goal_future = self.nav_client.send_goal_async(
            goal_msg, feedback_callback=self.nav_feedback_callback
        )
        send_goal_future.add_done_callback(self.nav_response_callback)
        
        self.current_goal = (goal_x, goal_y)
        self.nav_goal_active = True
        self.goal_start_time = time.time()
        
    def nav_response_callback(self, future):
        """Handle Nav2 goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Navigation goal rejected")
            self.nav_goal_active = False
            return
            
        self.get_logger().info("Navigation goal accepted")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.nav_result_callback)
        
    def nav_feedback_callback(self, feedback_msg):
        """Handle Nav2 feedback"""
        # Optional: could log progress here
        pass
        
    def nav_result_callback(self, future):
        """Handle Nav2 result"""
        result = future.result().result
        status = future.result().status
        
        if status == 4:  # SUCCEEDED
            self.get_logger().info("Navigation goal completed successfully")
            self.last_goal_fail_time = None  # Clear failure time on success
        else:
            self.get_logger().warn(f"Navigation goal failed with status: {status}")
            self.last_goal_fail_time = time.time()  # Record failure time
            
        self.nav_goal_active = False
        self.goal_start_time = None
        
        # After completing/failing a goal, wait a bit then continue exploring
        time.sleep(1.0)
        
    def cancel_navigation(self):
        """Cancel any active navigation goals"""
        if self.nav_goal_active:
            self.nav_client.cancel_all_goals()
            self.nav_goal_active = False
            
    def publish_status(self):
        """Publish current mapping status"""
        elapsed_time = time.time() - self.start_time
        remaining_time = max(0, self.mapping_duration - elapsed_time)
        
        status_msg = String()
        status_msg.data = (
            f"Simple Autonomous Mapping - "
            f"Elapsed: {elapsed_time:.0f}s, "
            f"Remaining: {remaining_time:.0f}s, "
            f"Goal Active: {'Yes' if self.nav_goal_active else 'No'}"
        )
        
        self.status_pub.publish(status_msg)
        self.get_logger().info(status_msg.data)
        

def main(args=None):
    rclpy.init(args=args)
    
    # Create node with multithreaded executor
    mapper = SimpleAutonomousMapper()
    executor = MultiThreadedExecutor()
    
    try:
        rclpy.spin(mapper, executor=executor)
    except KeyboardInterrupt:
        mapper.get_logger().info("Shutting down autonomous mapper")
    finally:
        mapper.cancel_navigation()
        mapper.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()