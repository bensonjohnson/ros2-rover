#!/usr/bin/env python3
"""
NPU Depth Image Exploration Node
Uses RKNN for real-time obstacle avoidance and exploration
Integrates with existing Hiwonder motor controller for odometry
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, JointState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32
import numpy as np
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
    from .rknn_trainer_depth import RKNNTrainerDepth
    TRAINER_AVAILABLE = True
    print("RKNN Trainer successfully imported")
except ImportError as e:
    TRAINER_AVAILABLE = False
    print(f"RKNN Trainer not available - using simple reactive navigation. Error: {e}")
except Exception as e:
    TRAINER_AVAILABLE = False
    print(f"RKNN Trainer initialization failed - using simple reactive navigation. Error: {e}")

class NPUExplorationDepthNode(Node):
    def __init__(self):
        super().__init__('npu_exploration_depth')
        
        # Parameters
        self.declare_parameter('max_speed', 0.15)
        self.declare_parameter('min_battery_percentage', 30.0)
        self.declare_parameter('safety_distance', 0.2)
        self.declare_parameter('npu_inference_rate', 5.0)
        self.declare_parameter('stacked_frames', 1)
        
        self.max_speed = self.get_parameter('max_speed').value
        self.min_battery_percentage = self.get_parameter('min_battery_percentage').value
        self.safety_distance = self.get_parameter('safety_distance').value
        self.inference_rate = self.get_parameter('npu_inference_rate').value
        self.stacked_frames = self.get_parameter('stacked_frames').value
        
        # State tracking
        self.current_velocity = np.array([0.0, 0.0])  # [linear, angular]
        self.position = np.array([0.0, 0.0])  # [x, y] from odometry
        self.orientation = 0.0  # yaw from odometry
        self.wheel_velocities = (0.0, 0.0)  # [left, right] wheel velocities from encoders
        self.start_time = time.time()
        self.step_count = 0
        self.last_inference_time = 0.0
        
        # Battery monitoring
        self.current_battery_percentage = 100.0  # Start optimistic
        self.low_battery_shutdown = False
        
        # Sensor data storage
        self.latest_depth_image = None
        self.bridge = CvBridge()
        
        # Previous state for reward calculation
        self.prev_position = np.array([0.0, 0.0])
        self.prev_depth_image = None
        self.collision_detected = False
        
        # Exploration state
        self.exploration_mode = "forward_explore"  # forward_explore, turn_explore, retreat
        self.stuck_counter = 0
        self.last_position = np.array([0.0, 0.0])
        self.movement_threshold = 0.05  # meters
        
        # Simple tracking for movement detection
        self.movement_check_counter = 0
        
        # Initialize NPU or fallback
        self.init_inference_engine()
        
        # ROS2 interfaces
        self.depth_sub = self.create_subscription(
            Image, 'depth_image',
            self.depth_callback, 10
        )
        
        
        self.odom_sub = self.create_subscription(
            Odometry, 'odom',
            self.odom_callback, 10
        )
        
        # Battery monitoring subscription
        self.battery_sub = self.create_subscription(
            Float32, '/battery_percentage',
            self.battery_callback, 10
        )
        
        # Joint state subscription for wheel velocities
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states',
            self.joint_state_callback, 10
        )
        
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/npu_exploration_status', 10)
        
        # Control timer
        self.control_timer = self.create_timer(0.1, self.control_loop)  # 10Hz control
        
        self.get_logger().info(f"NPU Depth Exploration Node initialized")
        self.get_logger().info(f"  Max Speed: {self.max_speed} m/s")
        self.get_logger().info(f"  Min Battery: {self.min_battery_percentage}%")
        self.get_logger().info(f"  NPU Available: {RKNN_AVAILABLE}")
        
    def init_inference_engine(self):
        self.use_npu = False
        self.trainer = None
        
        if TRAINER_AVAILABLE:
            try:
                self.trainer = RKNNTrainerDepth(stacked_frames=self.stacked_frames)
                self.use_npu = True
                self.get_logger().info("RKNN Trainer initialized (Phase1 preprocessing)")
            except Exception as e:
                self.get_logger().warn(f"RKNN Trainer init failed: {e}")
        else:
            self.get_logger().info("Trainer not available")
            
    def depth_callback(self, msg):
        """Process depth image and update internal state"""
        current_time = time.time()
        
        # Rate limit inference
        if current_time - self.last_inference_time < (1.0 / self.inference_rate):
            return
            
        self.last_inference_time = current_time
        
        # Process depth image
        self.latest_depth_image = self.preprocess_depth_image(msg)
        self.step_count += 1
        
        # Train neural network if available
        if self.use_npu and self.trainer and self.step_count > 10:
            self.train_from_experience()
        
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
        
    def battery_callback(self, msg):
        """Update current battery percentage"""
        self.current_battery_percentage = msg.data
        
        # Check for low battery condition
        if self.current_battery_percentage <= self.min_battery_percentage and not self.low_battery_shutdown:
            self.get_logger().warn(f"Low battery detected: {self.current_battery_percentage:.1f}% <= {self.min_battery_percentage}% - Initiating safe shutdown")
            self.low_battery_shutdown = True
    
    def joint_state_callback(self, msg):
        """Extract wheel velocities from joint states"""
        try:
            # Find left and right wheel velocity indices
            if 'left_viz_wheel_joint' in msg.name and 'right_viz_wheel_joint' in msg.name:
                left_idx = msg.name.index('left_viz_wheel_joint')
                right_idx = msg.name.index('right_viz_wheel_joint')
                
                if len(msg.velocity) > max(left_idx, right_idx):
                    left_vel = msg.velocity[left_idx]
                    right_vel = msg.velocity[right_idx]
                    self.wheel_velocities = (left_vel, right_vel)
                    
                    # Log wheel velocities for debugging differential drive issues
                    if abs(left_vel) > 0.01 or abs(right_vel) > 0.01:
                        self.get_logger().debug(f"Wheel velocities: L={left_vel:.2f}, R={right_vel:.2f}")
        except Exception as e:
            self.get_logger().warn(f"Joint state processing failed: {e}")
        
    def preprocess_depth_image(self, depth_msg):
        """Convert ROS Image to numpy array for processing"""
        try:
            # Convert ROS image to OpenCV format
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            
            if cv_depth is None or cv_depth.size == 0:
                self.get_logger().warn("Empty or None depth image received")
                # Return safe fallback if no valid depth data
                return np.zeros((240, 424), dtype=np.float32)
                
            # Log some information about the depth image
            self.get_logger().debug(f"Raw depth image shape: {cv_depth.shape}, dtype: {cv_depth.dtype}")
            
            # Convert to meters if needed (assuming depth is in millimeters)
            if cv_depth.dtype == np.uint16:
                depth_meters = cv_depth.astype(np.float32) / 1000.0
            else:
                depth_meters = cv_depth.astype(np.float32)
                
            # Log some statistics about the depth values
            valid_depths = depth_meters[(depth_meters > 0.01) & (depth_meters < 10.0)]
            if len(valid_depths) > 0:
                self.get_logger().debug(f"Depth range: {np.min(valid_depths):.2f}m - {np.max(valid_depths):.2f}m")
                
            # Resize to standard size for NPU processing
            depth_resized = cv2.resize(depth_meters, (424, 240))  # Width x Height
            
            # Filter out invalid depth values
            depth_resized = np.clip(depth_resized, 0.0, 10.0)  # Clip to 0-10 meters
            
            self.get_logger().debug(f"Processed depth image shape: {depth_resized.shape}")
            
            return depth_resized
            
        except Exception as e:
            self.get_logger().warn(f"Depth image processing failed: {e}")
            # Return safe fallback
            return np.zeros((240, 424), dtype=np.float32)
            
    def control_loop(self):
        """Main control loop - runs at 10Hz"""
        # Check if battery is too low
        if self.low_battery_shutdown:
            self.stop_robot()
            self.get_logger().info(f"Exploration stopped - Battery at {self.current_battery_percentage:.1f}%")
            self.get_logger().info("Initiating graceful shutdown due to low battery...")
            
            # Signal ROS2 to shutdown this node
            self.destroy_node()
            rclpy.shutdown()
            return
            
        # Check if we have recent depth image data
        if not hasattr(self, 'latest_depth_image'):
            return
            
        # Generate control command
        cmd = self.generate_control_command()
        
        # Publish command
        self.cmd_pub.publish(cmd)
        
    def generate_control_command(self):
        """Generate control command - NPU drives, safety only intervenes for emergency stop"""
        cmd = Twist()
        
        # Check for emergency collision risk
        emergency_stop = self.check_emergency_collision()
        
        if self.use_npu and self.trainer and self.all_sensors_ready():
            # Always use neural network when available
            action, confidence = self.npu_inference()
            
            # Convert NPU output to command
            cmd.linear.x = float(action[0]) * self.max_speed
            cmd.angular.z = float(action[1]) * 2.0  # Max 2 rad/s angular
            
            # Emergency override only for imminent collision
            if emergency_stop:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.exploration_mode = f"EMERGENCY_STOP (NPU conf: {confidence:.2f})"
                self.collision_detected = True
            else:
                self.exploration_mode = f"NPU_DRIVING (conf: {confidence:.2f})"
                self.collision_detected = False
                
            # Store action for training (original NPU decision, not emergency override)
            self.last_action = np.array([float(action[0]), float(action[1])])
            
        else:
            # Fallback: stop and wait for NPU
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.exploration_mode = "WAITING_FOR_NPU"
            self.last_action = np.array([0.0, 0.0])
            
        return cmd
        
    def check_emergency_collision(self):
        """Check for imminent collision requiring emergency stop"""
        if self.latest_depth_image is None or self.latest_depth_image.size == 0:
            return False
            
        try:
            # Check for very close obstacles in front (emergency zone)
            height, width = self.latest_depth_image.shape
            
            # Define region of interest (front center area)
            roi_height_start = int(height * 0.3)  # Top 30%
            roi_height_end = int(height * 0.7)    # Bottom 70%
            roi_width_start = int(width * 0.4)    # Left 40%
            roi_width_end = int(width * 0.6)      # Right 60%
            
            front_roi = self.latest_depth_image[roi_height_start:roi_height_end, roi_width_start:roi_width_end]
            
            # Filter out invalid depth values
            valid_depths = front_roi[(front_roi > 0.01) & (front_roi < 10.0)]  # 1cm to 10m range
            
            if len(valid_depths) > 0:
                min_distance = np.min(valid_depths)
                # Log the minimum distance for debugging
                self.get_logger().debug(f"Min distance to obstacle: {min_distance:.2f}m")
                return min_distance < 0.15  # 15cm emergency zone (reduced for better learning)
                
        except Exception as e:
            self.get_logger().warn(f"Emergency collision check failed: {e}")
            
        return False
        
    def reactive_navigation(self):
        """Fallback navigation - not used in NPU mode"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.exploration_mode = "NPU_FALLBACK_STOP"
        return cmd
        
    def npu_inference(self):
        if not self.all_sensors_ready():
            return np.array([0.0, 0.0]), 0.0
        try:
            depth_input = self.latest_depth_image.astype(np.float32)
            # Depth stats for proprio features
            valid = depth_input[(depth_input > 0.05) & (depth_input < 4.0)]
            min_d = float(np.min(valid)) if valid.size else 0.0
            mean_d = float(np.mean(valid)) if valid.size else 0.0
            near_collision_flag = 1.0 if (valid.size and np.percentile(valid,5) < 0.25) else 0.0
            wheel_diff = self.wheel_velocities[0] - self.wheel_velocities[1]
            proprioceptive = np.array([
                self.current_velocity[0],
                self.current_velocity[1],
                float(self.step_count % 100) / 100.0,
                self.last_action[0],
                self.last_action[1],
                wheel_diff,
                min_d,
                mean_d,
                near_collision_flag
            ], dtype=np.float32)
            action, confidence = self.trainer.inference(depth_input, proprioceptive)
            return action, confidence
        except Exception as e:
            self.get_logger().warn(f"Inference failed: {e}")
            return np.array([0.0,0.0]), 0.0
            
    def all_sensors_ready(self):
        """Check if all sensor data is available"""
        return self.latest_depth_image is not None
                
    def train_from_experience(self):
        if not self.all_sensors_ready() or not hasattr(self, 'last_action'):
            return
        try:
            progress = np.linalg.norm(self.position - self.prev_position)
            reward = self.trainer.calculate_reward(
                action=self.last_action,
                collision=self.collision_detected,
                progress=progress,
                exploration_bonus=0.0,
                position=self.position,
                depth_data=self.latest_depth_image,
                wheel_velocities=self.wheel_velocities
            )
            if self.prev_depth_image is not None:
                # Build extended proprio (must match inference): duplicate logic
                depth_input_prev = self.prev_depth_image
                valid_prev = depth_input_prev[(depth_input_prev > 0.05) & (depth_input_prev < 4.0)]
                min_d_prev = float(np.min(valid_prev)) if valid_prev.size else 0.0
                mean_d_prev = float(np.mean(valid_prev)) if valid_prev.size else 0.0
                near_collision_prev = 1.0 if (valid_prev.size and np.percentile(valid_prev,5) < 0.25) else 0.0
                wheel_diff_prev = self.wheel_velocities[0] - self.wheel_velocities[1]
                proprio_prev = np.array([
                    self.current_velocity[0],
                    self.current_velocity[1],
                    float(self.step_count % 100) / 100.0,
                    self.last_action[0],
                    self.last_action[1],
                    wheel_diff_prev,
                    min_d_prev,
                    mean_d_prev,
                    near_collision_prev
                ], dtype=np.float32)
                self.trainer.add_experience(
                    depth_image=self.prev_depth_image.astype(np.float32),
                    proprioceptive=proprio_prev,
                    action=self.last_action,
                    reward=reward,
                    next_depth_image=self.latest_depth_image.astype(np.float32)
                )
            training_stats = self.trainer.train_step()
            if self.step_count % 50 == 0 and 'loss' in training_stats:
                self.get_logger().info(f"Training: Loss={training_stats['loss']:.4f} AvgR={training_stats.get('avg_reward',0):.2f} Samples={training_stats.get('samples',0)}")
            self.prev_position = self.position.copy()
            self.prev_depth_image = self.latest_depth_image.copy()
        except Exception as e:
            self.get_logger().warn(f"Training step failed: {e}")
        
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
                f"Battery: {self.current_battery_percentage:.1f}% | "
                f"Steps: {self.step_count} | "
                f"Training: {training_stats['training_steps']} | "
                f"Buffer: {training_stats['buffer_size']}/10000 | "
                f"Avg Reward: {training_stats['avg_reward']:.2f}"
            )
        else:
            status_msg.data = f"NPU Exploration | Mode: {self.exploration_mode} | Battery: {self.current_battery_percentage:.1f}% | Steps: {self.step_count}"
            
        self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = NPUExplorationDepthNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("NPU depth exploration interrupted")
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
