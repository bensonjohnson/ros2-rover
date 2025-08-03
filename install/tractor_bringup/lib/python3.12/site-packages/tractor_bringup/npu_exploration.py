#!/usr/bin/env python3
"""
NPU Point Cloud Exploration Node
Uses RKNN for real-time obstacle avoidance and exploration
Integrates with existing Hiwonder motor controller for odometry
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image, Imu
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import numpy as np
import struct
import time
from cv_bridge import CvBridge
import cv2

try:
    from rknn.api import RKNN
    RKNN_AVAILABLE = True
    print("RKNN library successfully imported")
except ImportError as e:
    RKNN_AVAILABLE = False
    print(f"RKNN not available - using CPU fallback. Error: {e}")

try:
    from .rknn_trainer import RKNNTrainer
    TRAINER_AVAILABLE = True
    print("RKNN Trainer successfully imported")
except ImportError as e:
    TRAINER_AVAILABLE = False
    print(f"RKNN Trainer not available - using simple reactive navigation. Error: {e}")
except Exception as e:
    TRAINER_AVAILABLE = False
    print(f"RKNN Trainer initialization failed - using simple reactive navigation. Error: {e}")

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
        
        # Sensor data storage
        self.latest_pointcloud = None
        self.latest_rgb_image = None
        self.latest_imu_data = np.zeros(6)  # [ax, ay, az, gx, gy, gz]
        self.bridge = CvBridge()
        
        # Previous state for reward calculation
        self.prev_position = np.array([0.0, 0.0])
        self.prev_pointcloud = None
        self.collision_detected = False
        
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
        
        # RGB disabled for bandwidth reasons
        # self.rgb_sub = self.create_subscription(
        #     Image, '/camera/camera/color/image_raw',
        #     self.rgb_callback, 10
        # )
        
        self.imu_sub = self.create_subscription(
            Imu, '/camera/camera/imu',
            self.imu_callback, 10
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
        """Initialize RKNN NPU training system"""
        self.use_npu = False
        self.trainer = None
        
        if TRAINER_AVAILABLE:
            try:
                self.trainer = RKNNTrainer()
                self.use_npu = True
                self.get_logger().info("RKNN Trainer initialized - Learning enabled!")
                self.get_logger().info(f"Training stats: {self.trainer.get_training_stats()}")
            except Exception as e:
                self.get_logger().warn(f"RKNN Trainer initialization failed: {e}")
                self.use_npu = False
        else:
            self.get_logger().info("Using reactive navigation - no learning")
            
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
        
        # Train neural network if available
        if self.use_npu and self.trainer and self.step_count > 10:
            self.train_from_experience()
        
        # Publish status
        self.publish_status()
        
    def rgb_callback(self, msg):
        """Process RGB image from camera"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # Resize for NPU processing (smaller for efficiency)
            self.latest_rgb_image = cv2.resize(cv_image, (224, 224))
        except Exception as e:
            self.get_logger().warn(f"RGB image processing failed: {e}")
            
    def imu_callback(self, msg):
        """Process IMU data from RealSense"""
        try:
            # Store accelerometer and gyroscope data
            self.latest_imu_data[0] = msg.linear_acceleration.x
            self.latest_imu_data[1] = msg.linear_acceleration.y  
            self.latest_imu_data[2] = msg.linear_acceleration.z
            self.latest_imu_data[3] = msg.angular_velocity.x
            self.latest_imu_data[4] = msg.angular_velocity.y
            self.latest_imu_data[5] = msg.angular_velocity.z
        except Exception as e:
            self.get_logger().warn(f"IMU processing failed: {e}")
        
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
                # Use uniform sampling for better distribution
                indices = np.linspace(0, len(points) - 1, self.max_points, dtype=int)
                points = points[indices]
            elif len(points) < self.max_points and len(points) > 0:
                # Pad with distant points if needed
                padding = np.full((self.max_points - len(points), 3), [0, 0, 5.0])
                points = np.vstack([points, padding])
            elif len(points) == 0:
                # Return safe fallback if no valid points
                return np.full((self.max_points, 3), [0, 0, 5.0])
                
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
        """Generate control command using neural network or reactive navigation"""
        cmd = Twist()
        
        if self.use_npu and self.trainer and self.all_sensors_ready():
            # Use neural network inference
            action, confidence = self.npu_inference()
            
            # Use neural network if confident, otherwise fall back to reactive
            if confidence > 0.3:  # Confidence threshold
                cmd.linear.x = float(action[0]) * self.max_speed
                cmd.angular.z = float(action[1]) * 2.0  # Max 2 rad/s angular
                self.exploration_mode = f"neural_net (conf: {confidence:.2f})"
            else:
                # Low confidence - use reactive navigation
                cmd = self.reactive_navigation()
                self.exploration_mode += f" (fallback, conf: {confidence:.2f})"
                
            # Store action for training
            self.last_action = np.array([cmd.linear.x / self.max_speed, cmd.angular.z / 2.0])
        else:
            # Use simple reactive navigation
            cmd = self.reactive_navigation()
            self.last_action = np.array([cmd.linear.x / self.max_speed, cmd.angular.z / 2.0])
            
        return cmd
        
    def reactive_navigation(self):
        """Simple reactive navigation without SLAM"""
        cmd = Twist()
        
        # Analyze point cloud for obstacles
        points = self.latest_pointcloud
        
        # Safety check for None point cloud
        if points is None or len(points) == 0:
            # No point cloud data - conservative approach
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5  # Slow turn to find obstacles
            self.exploration_mode = "searching"
            return cmd
        
        # Find minimum distance in different sectors
        # Forward sector (0.5m wide)
        forward_mask = (points[:, 0] > 0) & (np.abs(points[:, 1]) < 0.25)
        # Left sector
        left_mask = (points[:, 1] > 0) & (np.abs(points[:, 0]) < 0.25)
        # Right sector
        right_mask = (points[:, 1] < 0) & (np.abs(points[:, 0]) < 0.25)
        
        min_forward_distance = 5.0
        min_left_distance = 5.0
        min_right_distance = 5.0
        
        if np.any(forward_mask):
            forward_points = points[forward_mask]
            min_forward_distance = np.min(np.linalg.norm(forward_points, axis=1))
            
        if np.any(left_mask):
            left_points = points[left_mask]
            min_left_distance = np.min(np.linalg.norm(left_points, axis=1))
            
        if np.any(right_mask):
            right_points = points[right_mask]
            min_right_distance = np.min(np.linalg.norm(right_points, axis=1))
            
        # Check movement progress
        position_change = np.linalg.norm(self.position - self.last_position)
        if position_change < self.movement_threshold:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.last_position = self.position.copy()
            
        # State machine for exploration with better obstacle avoidance
        if min_forward_distance < self.safety_distance or self.stuck_counter > 20:
            # Obstacle ahead or stuck - turn away from obstacles
            cmd.linear.x = 0.0
            # Turn toward the direction with more space
            if min_left_distance > min_right_distance:
                cmd.angular.z = 1.0  # Turn left
                self.exploration_mode = "turn_left"
            else:
                cmd.angular.z = -1.0  # Turn right
                self.exploration_mode = "turn_right"
        elif min_forward_distance < self.safety_distance * 2:
            # Slow approach with cautious turning
            cmd.linear.x = self.max_speed * 0.3
            # Slight turn away from closer side
            if min_left_distance < min_right_distance:
                cmd.angular.z = -0.3  # Slight right turn
                self.exploration_mode = "cautious_right"
            else:
                cmd.angular.z = 0.3   # Slight left turn
                self.exploration_mode = "cautious_left"
        else:
            # Clear ahead - move forward
            cmd.linear.x = self.max_speed
            cmd.angular.z = 0.0
            self.exploration_mode = "forward_explore"
            
        return cmd
        
    def npu_inference(self):
        """Run neural network inference"""
        if not self.all_sensors_ready():
            return np.array([0.0, 0.0]), 0.0
            
        try:
            # Prepare point cloud input (3 x N points)
            pc_input = self.latest_pointcloud.T.astype(np.float32)
            if pc_input.shape[1] < self.max_points:
                # Pad with zeros
                padding = np.zeros((3, self.max_points - pc_input.shape[1]))
                pc_input = np.concatenate([pc_input, padding], axis=1)
            elif pc_input.shape[1] > self.max_points:
                # Downsample
                pc_input = pc_input[:, :self.max_points]
                
            # Prepare proprioceptive input
            proprioceptive = np.array([
                self.current_velocity[0],
                self.current_velocity[1], 
                1.0,  # Battery level (placeholder)
                float(self.step_count % 100) / 100.0
            ]).astype(np.float32)
            
            # Run inference
            action, confidence = self.trainer.inference(
                pc_input, self.latest_imu_data, proprioceptive
            )
            
            return action, confidence
            
        except Exception as e:
            self.get_logger().warn(f"Neural inference failed: {e}")
            return np.array([0.0, 0.0]), 0.0
            
    def all_sensors_ready(self):
        """Check if all sensor data is available (no RGB)"""
        return (self.latest_pointcloud is not None and
                self.latest_imu_data is not None)
                
    def train_from_experience(self):
        """Add experience to training buffer and perform training step"""
        if not self.all_sensors_ready() or not hasattr(self, 'last_action'):
            return
            
        try:
            # Calculate reward based on current state
            progress = np.linalg.norm(self.position - self.prev_position) 
            exploration_bonus = self.calculate_exploration_bonus()
            
            reward = self.trainer.calculate_reward(
                action=self.last_action,
                collision=self.collision_detected,
                progress=progress,
                exploration_bonus=exploration_bonus
            )
            
            # Add experience to buffer
            if self.prev_pointcloud is not None:
                self.trainer.add_experience(
                    pointcloud=self.prev_pointcloud.T.astype(np.float32),
                    imu_data=self.latest_imu_data,
                    proprioceptive=np.array([
                        self.current_velocity[0], self.current_velocity[1], 1.0, 
                        float(self.step_count % 100) / 100.0
                    ]),
                    action=self.last_action,
                    reward=reward,
                    next_pointcloud=self.latest_pointcloud.T.astype(np.float32)
                )
                
            # Perform training step
            training_stats = self.trainer.train_step()
            
            # Log training progress occasionally
            if self.step_count % 50 == 0:
                self.get_logger().info(
                    f"Training: Loss={training_stats['loss']:.4f}, "
                    f"Reward={training_stats['avg_reward']:.2f}, "
                    f"Samples={training_stats['samples']}"
                )
                
            # Update previous state
            self.prev_position = self.position.copy()
            self.prev_pointcloud = self.latest_pointcloud.copy()
            
        except Exception as e:
            self.get_logger().warn(f"Training step failed: {e}")
            
    def calculate_exploration_bonus(self):
        """Calculate bonus for exploring new areas"""
        # Simple exploration bonus based on movement
        if np.linalg.norm(self.position - self.prev_position) > 0.1:
            return 1.0
        return 0.0
        
    def ros_pc2_to_numpy(self, pc_msg):
        """Convert ROS PointCloud2 to numpy array"""
        try:
            # More robust point cloud conversion
            points = []
            point_step = pc_msg.point_step
            height = pc_msg.height
            width = pc_msg.width
            
            # Handle both organized and unorganized point clouds
            if height > 1 and width > 1:
                # Organized point cloud
                for v in range(height):
                    for u in range(width):
                        i = (v * width + u) * point_step
                        if i + 12 <= len(pc_msg.data):
                            x, y, z = struct.unpack('fff', pc_msg.data[i:i+12])
                            # Only add valid points (not NaN or Inf)
                            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                                points.append([x, y, z])
            else:
                # Unorganized point cloud
                for i in range(0, len(pc_msg.data), point_step):
                    if i + 12 <= len(pc_msg.data):
                        x, y, z = struct.unpack('fff', pc_msg.data[i:i+12])
                        # Only add valid points (not NaN or Inf)
                        if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                            points.append([x, y, z])
                
            return np.array(points) if points else np.zeros((0, 3))
        except Exception as e:
            self.get_logger().warn(f"Point cloud conversion failed: {e}")
            return np.zeros((0, 3))
        
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
        
        if self.use_npu and self.trainer:
            training_stats = self.trainer.get_training_stats()
            status_msg.data = (
                f"NPU Learning | Mode: {self.exploration_mode} | "
                f"Time: {elapsed_time:.1f}s/{self.exploration_time}s | "
                f"Steps: {self.step_count} | "
                f"Training: {training_stats['training_steps']} | "
                f"Buffer: {training_stats['buffer_size']}/10000 | "
                f"Avg Reward: {training_stats['avg_reward']:.2f}"
            )
        else:
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