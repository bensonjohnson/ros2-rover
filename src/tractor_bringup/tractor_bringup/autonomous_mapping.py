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
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import math
import time
import numpy as np

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
        self.declare_parameter('exploration_distance', 1.5)  # Shorter distance for straighter paths
        self.declare_parameter('mapping_duration', 600)  # 10 minutes
        self.declare_parameter('obstacle_avoidance_distance', 0.025)  # 25mm trigger distance
        self.declare_parameter('turn_around_distance', 0.025)  # 25mm - distance to trigger 180 turn
        self.declare_parameter('camera_fov_angle', 65.0)  # Camera horizontal FOV in degrees
        self.declare_parameter('camera_max_range', 4.0)  # Maximum range of camera in meters
        self.declare_parameter('wall_follow_distance', 0.3)  # 30cm from wall
        self.declare_parameter('wall_detection_range', 2.0)  # 2m range to detect walls

        self.max_speed = self.get_parameter('max_speed').get_parameter_value().double_value
        self.exploration_distance = self.get_parameter('exploration_distance').get_parameter_value().double_value
        self.mapping_duration = self.get_parameter('mapping_duration').get_parameter_value().integer_value
        self.obstacle_avoidance_distance = self.get_parameter('obstacle_avoidance_distance').get_parameter_value().double_value
        self.turn_around_distance = self.get_parameter('turn_around_distance').get_parameter_value().double_value
        self.camera_fov_angle = self.get_parameter('camera_fov_angle').get_parameter_value().double_value
        self.camera_max_range = self.get_parameter('camera_max_range').get_parameter_value().double_value
        self.wall_follow_distance = self.get_parameter('wall_follow_distance').get_parameter_value().double_value
        self.wall_detection_range = self.get_parameter('wall_detection_range').get_parameter_value().double_value

        # State variables
        self.current_pose = None
        self.current_heading = 0.0
        self.start_time = time.time()
        self.mapping_complete = False
        self.current_goal = None
        self.nav_goal_active = False
        self.goal_start_time = None
        self.goal_timeout = 25.0  # Force new goal after 25 seconds
        self.last_goal_fail_time = None
        self.goal_retry_delay = 3.0  # Reduced wait time after goal failure
        self.obstacle_detected = False
        self.need_to_turn_around = False
        self.turning_around = False
        self.wall_following_mode = False
        self.wall_on_right = True  # Track which side the wall is on
        self.wall_detected = False
        self.wall_points = []
        self.seeking_wall = True  # Start by looking for a wall

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

        self.pointcloud_sub = self.create_subscription(
            PointCloud2, '/camera/camera/depth/color/points', self.pointcloud_callback, 10,
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
        """Main control loop - simple forward exploration with obstacle avoidance"""
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

        # Wall following behavior
        if self.seeking_wall and not self.wall_following_mode:
            # Look for walls first
            if not self.nav_goal_active and not self.turning_around:
                self.search_for_wall()
                return
        elif self.wall_following_mode:
            # Follow detected wall
            if not self.nav_goal_active and not self.turning_around:
                self.follow_wall()
                return
        else:
            # Fallback to original behavior
            # Check if we need to turn around (180 degrees)
            if self.need_to_turn_around and not self.turning_around:
                self.execute_180_turn()
                return

            # Check if enough time has passed since last goal failure
            if self.last_goal_fail_time:
                fail_elapsed = time.time() - self.last_goal_fail_time
                if fail_elapsed < self.goal_retry_delay:
                    return  # Wait before trying again

            # If no active goal, send a new forward exploration goal
            if not self.nav_goal_active and not self.turning_around:
                self.send_forward_goal()
                # Wait for the goal to be reached before sending a new one
                time.sleep(1.0)
                return

    def pointcloud_callback(self, msg):
        """Process point cloud data to detect walls"""
        try:
            # Convert point cloud to list of points
            points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
            
            # Filter points within wall detection range
            wall_points = []
            for point in points:
                x, y, z = point
                # Filter points above track height (robot can climb over anything below 9cm)
                if (0.10 < z < 2.0 and  # Height filter: ignore grass/small obstacles, detect fences/walls
                    math.sqrt(x*x + y*y) < self.wall_detection_range):
                    wall_points.append((x, y, z))
            
            self.wall_points = wall_points
            self.wall_detected = len(wall_points) > 50  # Threshold for wall detection
            
        except Exception as e:
            self.get_logger().warn(f"Point cloud processing error: {e}")

    def search_for_wall(self):
        """Search for a wall by rotating and moving forward"""
        if not self.current_pose:
            return
            
        if self.wall_detected:
            self.get_logger().info("Wall detected! Switching to wall following mode")
            self.seeking_wall = False
            self.wall_following_mode = True
            self.follow_wall()
            return
        
        # Continue forward to search for walls
        self.get_logger().info("Searching for walls...")
        self.send_forward_goal()

    def follow_wall(self):
        """Follow the detected wall at a safe distance"""
        if not self.current_pose or not self.wall_points:
            return
        
        # Analyze wall points to determine wall direction
        wall_direction = self.analyze_wall_direction()
        if wall_direction is None:
            self.get_logger().warn("Cannot determine wall direction")
            return
        
        # Calculate goal position to follow wall
        goal_x, goal_y = self.calculate_wall_following_goal(wall_direction)
        
        # Create navigation goal - using odom frame for consistency with costmaps
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'odom'  # Consistent with SLAM and Nav2 costmap configuration
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = goal_x
        goal_msg.pose.pose.position.y = goal_y
        goal_msg.pose.pose.position.z = 0.0
        
        # Set orientation to follow wall
        wall_following_heading = wall_direction
        q = self.euler_to_quaternion(0, 0, wall_following_heading)
        goal_msg.pose.pose.orientation.x = q[0]
        goal_msg.pose.pose.orientation.y = q[1]
        goal_msg.pose.pose.orientation.z = q[2]
        goal_msg.pose.pose.orientation.w = q[3]
        
        self.get_logger().info(f"Following wall: goal ({goal_x:.2f}, {goal_y:.2f})")
        send_goal_future = self.nav_client.send_goal_async(
            goal_msg, feedback_callback=self.nav_feedback_callback
        )
        send_goal_future.add_done_callback(self.nav_response_callback)
        
        self.nav_goal_active = True
        self.goal_start_time = time.time()

    def analyze_wall_direction(self):
        """Analyze wall points to determine the direction to follow the wall"""
        if len(self.wall_points) < 10:
            return None
        
        # Find the average position of wall points
        avg_x = sum(p[0] for p in self.wall_points) / len(self.wall_points)
        avg_y = sum(p[1] for p in self.wall_points) / len(self.wall_points)
        
        # Determine if wall is more on the left or right
        wall_angle = math.atan2(avg_y, avg_x)
        
        # Calculate direction to follow wall (perpendicular to wall direction)
        if avg_y > 0:  # Wall on left
            self.wall_on_right = False
            follow_direction = wall_angle - math.pi/2  # Follow with wall on left
        else:  # Wall on right
            self.wall_on_right = True
            follow_direction = wall_angle + math.pi/2  # Follow with wall on right
        
        # Normalize angle
        while follow_direction > math.pi:
            follow_direction -= 2 * math.pi
        while follow_direction < -math.pi:
            follow_direction += 2 * math.pi
            
        return follow_direction

    def calculate_wall_following_goal(self, wall_direction):
        """Calculate goal position for wall following"""
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        
        # Move forward along the wall direction
        move_distance = 1.0  # 1 meter forward along wall
        goal_x = current_x + move_distance * math.cos(wall_direction)
        goal_y = current_y + move_distance * math.sin(wall_direction)
        
        return goal_x, goal_y

    def execute_180_turn(self):
        """Execute a 180-degree turn by setting a goal with opposite heading"""
        if not self.current_pose:
            return

        self.turning_around = True
        self.get_logger().info("Executing 180-degree turn...")

        # Calculate opposite heading (add π radians = 180 degrees)
        opposite_heading = self.current_heading + math.pi
        # Normalize to [-π, π]
        while opposite_heading > math.pi:
            opposite_heading -= 2 * math.pi
        while opposite_heading < -math.pi:
            opposite_heading += 2 * math.pi

        # Create goal at same position but with opposite orientation
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'odom'  # Consistent with SLAM and Nav2 costmap configuration
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = self.current_pose.position.x
        goal_msg.pose.pose.position.y = self.current_pose.position.y
        goal_msg.pose.pose.position.z = 0.0

        # Set opposite orientation
        q = self.euler_to_quaternion(0, 0, opposite_heading)
        goal_msg.pose.pose.orientation.x = q[0]
        goal_msg.pose.pose.orientation.y = q[1]
        goal_msg.pose.pose.orientation.z = q[2]
        goal_msg.pose.pose.orientation.w = q[3]

        # Send turn goal
        self.get_logger().info(f"Turning from {self.current_heading:.2f} to {opposite_heading:.2f} radians")
        send_goal_future = self.nav_client.send_goal_async(
            goal_msg, feedback_callback=self.nav_feedback_callback
        )
        send_goal_future.add_done_callback(self.turn_response_callback)

        self.nav_goal_active = True
        self.goal_start_time = time.time()

    def turn_response_callback(self, future):
        """Handle 180-degree turn response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("180-degree turn goal rejected")
            self.turning_around = False
            self.need_to_turn_around = False
            self.nav_goal_active = False
            return

        self.get_logger().info("180-degree turn goal accepted")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.turn_result_callback)

    def turn_result_callback(self, future):
        """Handle 180-degree turn completion"""
        result = future.result().result
        status = future.result().status

        if status == 4:  # SUCCEEDED
            self.get_logger().info("180-degree turn completed successfully")
            self.current_heading += math.pi  # Update our heading
            # Normalize heading
            while self.current_heading > math.pi:
                self.current_heading -= 2 * math.pi
            while self.current_heading < -math.pi:
                self.current_heading += 2 * math.pi
        else:
            self.get_logger().warn(f"180-degree turn failed with status: {status}")

        # Reset turn state
        self.turning_around = False
        self.need_to_turn_around = False
        self.nav_goal_active = False
        self.goal_start_time = None
        self.last_goal_fail_time = None  # Clear failure time after turn

    def send_forward_goal(self):
        """Send a goal to drive forward in current direction, limited to camera FOV"""
        if not self.current_pose:
            return

        # Get current position
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y

        # Calculate goal position based on current heading
        # Limit distance to camera's maximum range
        safe_distance = min(self.exploration_distance, self.camera_max_range)

        # Calculate goal position
        goal_x = current_x + safe_distance * math.cos(self.current_heading)
        goal_y = current_y + safe_distance * math.sin(self.current_heading)

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
        self.get_logger().info(f"Goal limited to camera FOV ({self.camera_fov_angle}°) and max range ({self.camera_max_range}m)")
        send_goal_future = self.nav_client.send_goal_async(
            goal_msg, feedback_callback=self.nav_feedback_callback
        )
        send_goal_future.add_done_callback(self.nav_response_callback)

        self.current_goal = (goal_x, goal_y)
        self.nav_goal_active = True
        self.goal_start_time = time.time()
        self.get_logger().info(f"Waiting for goal to be reached before setting new goal")

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
            # Wait at the goal for a moment to ensure proper stopping
            time.sleep(2.0)
        elif status == 3:  # CANCELED
            self.get_logger().info("Navigation goal was canceled")
            self.last_goal_fail_time = None
        else:
            self.get_logger().warn(f"Navigation goal failed with status: {status}")
            self.last_goal_fail_time = time.time()  # Record failure time
            # Trigger 180-degree turn when goal fails (likely due to obstacle)
            self.need_to_turn_around = True
            self.get_logger().info("Goal failed due to obstacle - will turn 180 degrees")

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

    def euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles to quaternion (x, y, z, w)"""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return [x, y, z, w]

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
