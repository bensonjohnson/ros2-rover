#!/usr/bin/env python3
"""
NPU Point Cloud Exploration Node
Uses RKNN for real-time obstacle avoidance and exploration
Integrates with existing Hiwonder motor controller for odometry
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import numpy as np
import struct
import time

try:
    from rknn.api import RKNN
    RKNN_AVAILABLE = True
except ImportError:
    RKNN_AVAILABLE = False
    print("RKNN not available - using CPU fallback")

class NPUExplorationNode(Node):
    def __init__(self):
        super().__init__('npu_exploration')
        
        # Parameters
        self.declare_parameter('max_speed', 0.15)
        self.declare_parameter('exploration_time', 300)
        self.declare_parameter('safety_distance', 0.2)
        self.declare_parameter('max_points', 512)
        self.declare_parameter('npu_inference_rate', 5.0)
        
        self.max_speed = self.get_parameter('max_speed').value
        self.exploration_time = self.get_parameter('exploration_time').value
        self.safety_distance = self.get_parameter('safety_distance').value
        self.max_points = self.get_parameter('max_points').value
        self.inference_rate = self.get_parameter('npu_inference_rate').value
        
        # State tracking
        self.current_velocity = np.array([0.0, 0.0])  # [linear, angular]
        self.position = np.array([0.0, 0.0])  # [x, y] from odometry
        self.orientation = 0.0  # yaw from odometry
        self.start_time = time.time()
        self.step_count = 0
        self.last_inference_time = 0.0
        
        # Exploration state
        self.exploration_mode = "forward_explore"  # forward_explore, turn_explore, retreat
        self.stuck_counter = 0
        self.last_position = np.array([0.0, 0.0])
        self.movement_threshold = 0.05  # meters
        
        # Initialize NPU or fallback
        self.init_inference_engine()
        
        # ROS2 interfaces
        self.pc_sub = self.create_subscription(
            PointCloud2, 'point_cloud',
            self.pointcloud_callback, 10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry, 'odom',
            self.odom_callback, 10
        )
        
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/npu_exploration_status', 10)
        
        # Control timer
        self.control_timer = self.create_timer(0.1, self.control_loop)  # 10Hz control
        
        self.get_logger().info(f"NPU Exploration Node initialized")
        self.get_logger().info(f"  Max Speed: {self.max_speed} m/s")
        self.get_logger().info(f"  Exploration Time: {self.exploration_time} s")
        self.get_logger().info(f"  NPU Available: {RKNN_AVAILABLE}")
        
    def init_inference_engine(self):
        """Initialize RKNN NPU or CPU fallback"""
        if RKNN_AVAILABLE:
            try:
                self.rknn = RKNN(verbose=False)
                # Try to load pre-trained model
                model_path = "/home/ubuntu/ros2-rover/models/exploration_model.rknn"
                # For now, use fallback since model doesn't exist yet
                self.use_npu = False
                self.get_logger().info("Using CPU fallback - NPU model not found")
            except Exception as e:
                self.use_npu = False
                self.get_logger().warn(f"NPU initialization failed: {e}")
        else:
            self.use_npu = False
            
    def pointcloud_callback(self, msg):
        """Process point cloud and update internal state"""
        current_time = time.time()
        
        # Rate limit inference
        if current_time - self.last_inference_time < (1.0 / self.inference_rate):
            return
            
        self.last_inference_time = current_time
        
        # Process point cloud
        self.latest_pointcloud = self.preprocess_pointcloud(msg)
        self.step_count += 1
        
        # Publish status
        self.publish_status()
        
    def odom_callback(self, msg):
        """Update position and velocity from motor controller"""
        # Extract position
        self.position[0] = msg.pose.pose.position.x
        self.position[1] = msg.pose.pose.position.y
        
        # Extract orientation (yaw)
        orientation_q = msg.pose.pose.orientation
        self.orientation = self.quaternion_to_yaw(orientation_q)
        
        # Extract velocity
        self.current_velocity[0] = msg.twist.twist.linear.x
        self.current_velocity[1] = msg.twist.twist.angular.z
        
    def preprocess_pointcloud(self, pc_msg):
        """Convert ROS PointCloud2 to numpy array for processing"""
        try:
            # Extract point cloud data
            points = self.ros_pc2_to_numpy(pc_msg)
            
            # Filter valid points (remove NaN and infinite values)
            valid_mask = np.isfinite(points).all(axis=1)
            points = points[valid_mask]
            
            # Filter by distance (remove points too far or too close)
            distances = np.linalg.norm(points, axis=1)
            range_mask = (distances > 0.1) & (distances < 5.0)
            points = points[range_mask]
            
            # Downsample to max_points for efficiency
            if len(points) > self.max_points:
                indices = np.random.choice(len(points), self.max_points, replace=False)
                points = points[indices]
            elif len(points) < self.max_points:
                # Pad with distant points if needed
                padding = np.full((self.max_points - len(points), 3), [0, 0, 5.0])
                points = np.vstack([points, padding]) if len(points) > 0 else padding
                
            return points
            
        except Exception as e:
            self.get_logger().warn(f"Point cloud processing failed: {e}")
            # Return safe fallback
            return np.full((self.max_points, 3), [0, 0, 5.0])
            
    def control_loop(self):
        """Main control loop - runs at 10Hz"""
        # Check if exploration time is up
        if time.time() - self.start_time > self.exploration_time:
            self.stop_robot()
            self.get_logger().info("Exploration time completed")
            return
            
        # Check if we have recent point cloud data
        if not hasattr(self, 'latest_pointcloud'):
            return
            
        # Generate control command
        cmd = self.generate_control_command()
        
        # Publish command
        self.cmd_pub.publish(cmd)
        
    def generate_control_command(self):
        """Generate control command using simple reactive navigation or NPU"""
        cmd = Twist()
        
        if self.use_npu:
            # Use NPU inference (when model is available)
            action = self.npu_inference()
            cmd.linear.x = float(action[0]) * self.max_speed
            cmd.angular.z = float(action[1]) * 2.0  # Max 2 rad/s angular
        else:
            # Use simple reactive navigation
            cmd = self.reactive_navigation()
            
        return cmd
        
    def reactive_navigation(self):
        """Simple reactive navigation without SLAM"""
        cmd = Twist()
        
        # Analyze point cloud for obstacles
        points = self.latest_pointcloud
        
        # Find minimum distance in front (forward cone)
        forward_mask = (points[:, 0] > 0) & (np.abs(points[:, 1]) < 0.5)  # Forward 0.5m wide
        
        if np.any(forward_mask):
            forward_points = points[forward_mask]
            min_forward_distance = np.min(np.linalg.norm(forward_points, axis=1))
        else:
            min_forward_distance = 5.0  # Assume clear if no points
            
        # Check movement progress
        position_change = np.linalg.norm(self.position - self.last_position)
        if position_change < self.movement_threshold:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.last_position = self.position.copy()
            
        # State machine for exploration
        if min_forward_distance < self.safety_distance or self.stuck_counter > 20:
            # Obstacle ahead or stuck - turn
            cmd.linear.x = 0.0
            cmd.angular.z = 1.0 if np.random.random() > 0.5 else -1.0  # Random turn direction
            self.exploration_mode = "turn_explore"
        elif min_forward_distance < self.safety_distance * 2:
            # Slow approach
            cmd.linear.x = self.max_speed * 0.3
            cmd.angular.z = 0.0
            self.exploration_mode = "slow_approach"
        else:
            # Clear ahead - move forward
            cmd.linear.x = self.max_speed
            cmd.angular.z = 0.0
            self.exploration_mode = "forward_explore"
            
        return cmd
        
    def npu_inference(self):
        """Run NPU inference (placeholder for when model is ready)"""
        # Prepare inputs
        proprioceptive = np.array([
            self.current_velocity[0],
            self.current_velocity[1], 
            1.0,  # Battery level (placeholder)
            float(self.step_count % 100) / 100.0
        ]).astype(np.float32)
        
        point_cloud = self.latest_pointcloud.T.reshape(1, 3, self.max_points).astype(np.float32)
        
        # TODO: Replace with actual RKNN inference
        # outputs = self.rknn.inference(inputs=[point_cloud, proprioceptive])
        # action = outputs[0]  # [linear_x, angular_z]
        
        # Placeholder: return random valid action
        action = np.array([np.random.uniform(0, 1), np.random.uniform(-1, 1)])
        return action
        
    def ros_pc2_to_numpy(self, pc_msg):
        """Convert ROS PointCloud2 to numpy array"""
        # Basic point cloud conversion
        points = []
        point_step = pc_msg.point_step
        
        for i in range(0, len(pc_msg.data), point_step):
            # Extract x, y, z (assuming first 12 bytes are xyz as float32)
            if i + 12 <= len(pc_msg.data):
                x, y, z = struct.unpack('fff', pc_msg.data[i:i+12])
                points.append([x, y, z])
                
        return np.array(points) if points else np.zeros((0, 3))
        
    def quaternion_to_yaw(self, q):
        """Convert quaternion to yaw angle"""
        # Extract yaw from quaternion
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)
        
    def stop_robot(self):
        """Send stop command"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)
        
    def publish_status(self):
        """Publish exploration status"""
        elapsed_time = time.time() - self.start_time
        status_msg = String()
        status_msg.data = f"NPU Exploration | Mode: {self.exploration_mode} | Time: {elapsed_time:.1f}s/{self.exploration_time}s | Steps: {self.step_count}"
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = NPUExplorationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("NPU exploration interrupted")
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()