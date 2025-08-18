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
    from .es_trainer_depth import EvolutionaryStrategyTrainer
    TRAINER_AVAILABLE = True
    print("RKNN Trainer successfully imported")
    print("ES Trainer successfully imported")
except ImportError as e:
    TRAINER_AVAILABLE = False
    print(f"RKNN/ES Trainer not available - using simple reactive navigation. Error: {e}")
except Exception as e:
    TRAINER_AVAILABLE = False
    print(f"RKNN/ES Trainer initialization failed - using simple reactive navigation. Error: {e}")

class NPUExplorationDepthNode(Node):
    def __init__(self):
        super().__init__('npu_exploration_depth')
        
        # Parameters
        self.declare_parameter('max_speed', 0.15)
        self.declare_parameter('min_battery_percentage', 30.0)
        self.declare_parameter('safety_distance', 0.2)
        self.declare_parameter('npu_inference_rate', 5.0)
        self.declare_parameter('stacked_frames', 1)
        self.declare_parameter('operation_mode', 'cpu_training')  # cpu_training | hybrid | inference
        self.declare_parameter('train_every_n_frames', 3)  # NEW: train interval to reduce CPU load
        # Initialize critical attributes BEFORE subscriptions / inference
        self.last_action = np.array([0.0, 0.0])
        self.exploration_warmup_steps = 300  # steps of forced exploration
        self.random_action_prob = 0.3        # probability to inject random action during warmup
        self.min_forward_bias = 0.25          # bias for forward movement (scaled later)
        self.forward_bias_extension_steps = 800  # extend bias period
        
        self.max_speed = self.get_parameter('max_speed').value
        self.min_battery_percentage = self.get_parameter('min_battery_percentage').value
        self.safety_distance = self.get_parameter('safety_distance').value
        self.inference_rate = self.get_parameter('npu_inference_rate').value
        self.stacked_frames = self.get_parameter('stacked_frames').value
        self.operation_mode = self.get_parameter('operation_mode').value
        self.train_every_n_frames = int(self.get_parameter('train_every_n_frames').value)
        
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
        # Separate status timer (1 Hz) to reduce coupling to depth rate
        self.status_timer = self.create_timer(1.0, self.publish_status)
        self.get_logger().info(f"NPU Depth Exploration Node initialized")
        self.get_logger().info(f"  Max Speed: {self.max_speed} m/s")
        self.get_logger().info(f"  Min Battery: {self.min_battery_percentage}%")
        self.get_logger().info(f"  NPU Available: {RKNN_AVAILABLE}")
        self.get_logger().info(f"  Inference target rate: {self.inference_rate} Hz")
        self.get_logger().info(f"  Operation Mode: {self.operation_mode}")
        
        self.angular_scale = 0.8  # reduced from 2.0 to lessen spin dominance
        self.spin_penalty = 3.0
        self.forward_free_bonus_scale = 1.5
        
        # Action post-processing parameters to encourage forward movement
        self.forward_bias_factor = 1.8  # Increase forward movement (increased from 1.2)
        self.angular_dampening = 0.6   # Reduce angular velocity (increased dampening from 0.7)
        self.backward_penalty_factor = 0.3  # Strongly discourage backward movement
        
        # Recovery features
        self.recovery_active = False
        self.last_min_distance = None
        # Scripted recovery state
        self.recovery_phase = 0          # 0=reverse,1=rotate,2=probe
        self.recovery_phase_ticks = 0
        self.recovery_phase_target = 0
        self.recovery_total_ticks = 0
        self.recovery_direction = 0      # +1 CCW, -1 CW
        self.recovery_no_progress_ticks = 0
        self.recovery_last_min_d = None
        self.recovery_clear_ticks = 0
        # Adjusted recovery parameters to prevent excessive backward movement
        self.recovery_reverse_speed = -0.3  # Reduced from -0.5
        self.recovery_max_duration = 80     # Reduced from 120

    def init_inference_engine(self):
        self.use_npu = False
        self.trainer = None
        
        if TRAINER_AVAILABLE:
            try:
                enable_debug = (self.get_parameter('operation_mode').value != 'inference')
                mode = self.get_parameter('operation_mode').value
                
                # Initialize appropriate trainer based on mode
                if mode in ['es_training', 'es_hybrid', 'es_inference', 'safe_es_training']:
                    # Use Evolutionary Strategy trainer
                    self.trainer = EvolutionaryStrategyTrainer(
                        stacked_frames=self.stacked_frames, 
                        enable_debug=enable_debug,
                        population_size=10,
                        sigma=0.1,
                        learning_rate=0.01
                    )
                    self.get_logger().info("ES Trainer initialized")
                else:
                    # Use Reinforcement Learning trainer
                    self.trainer = RKNNTrainerDepth(stacked_frames=self.stacked_frames, enable_debug=enable_debug)
                    self.get_logger().info("RKNN Trainer initialized")
                
                if mode == 'cpu_training':
                    # Training on CPU only; no RKNN runtime inference
                    self.use_npu = True  # still use trainer inference (PyTorch)
                    self.get_logger().info("Mode: CPU training (PyTorch inference, periodic RKNN export)")
                elif mode == 'hybrid':
                    # Try to enable RKNN runtime; fall back to PyTorch if missing
                    if self.trainer.enable_rknn_inference():
                        self.use_npu = True
                        self.get_logger().info("Mode: Hybrid (RKNN runtime inference + ongoing training)")
                    else:
                        self.use_npu = True
                        self.get_logger().warn("Hybrid mode requested but RKNN runtime not available - using PyTorch")
                elif mode == 'inference':
                    # Pure inference: load RKNN and disable training logic
                    if self.trainer.enable_rknn_inference():
                        self.use_npu = True
                        # Disable optimizer to avoid accidental training
                        if hasattr(self.trainer, 'optimizer'):
                            self.trainer.optimizer = None
                        self.get_logger().info("Mode: Pure RKNN inference (no training)")
                    else:
                        self.use_npu = True
                        self.get_logger().warn("Pure inference mode requested but RKNN file/runtime not available - falling back to PyTorch")
                elif mode == 'safe_training':
                    # Safe training with anti-overtraining measures
                    self.use_npu = True
                    self.get_logger().info("Mode: Safe training (anti-overtraining protection)")
                elif mode == 'es_training':
                    # Evolutionary Strategy training on CPU
                    self.use_npu = True
                    self.get_logger().info("Mode: ES training (Evolutionary Strategy on CPU)")
                elif mode == 'es_hybrid':
                    # ES training with RKNN inference
                    if self.trainer.enable_rknn_inference():
                        self.use_npu = True
                        self.get_logger().info("Mode: ES Hybrid (RKNN runtime inference + ES training)")
                    else:
                        self.use_npu = True
                        self.get_logger().warn("ES Hybrid mode requested but RKNN runtime not available - using PyTorch")
                elif mode == 'es_inference':
                    # Pure ES inference: load RKNN and disable training logic
                    if self.trainer.enable_rknn_inference():
                        self.use_npu = True
                        self.get_logger().info("Mode: ES Pure RKNN inference (no training)")
                    else:
                        self.use_npu = True
                        self.get_logger().warn("ES Pure inference mode requested but RKNN file/runtime not available - falling back to PyTorch")
                elif mode == 'safe_es_training':
                    # Safe ES training with anti-overtraining measures
                    self.use_npu = True
                    self.get_logger().info("Mode: Safe ES training (anti-overtraining protection with ES)")
                else:
                    self.use_npu = True
                    self.get_logger().warn(f"Unknown operation_mode '{mode}' - defaulting to cpu_training")
            except Exception as e:
                self.get_logger().warn(f"Trainer init failed: {e}")
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
            if self.operation_mode != 'inference' and (self.step_count % self.train_every_n_frames == 0):
                self.train_from_experience()
        
        # status now handled by status timer
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
            if self.trainer:
                self.trainer.safe_save()
            self.get_logger().info("Initiating graceful shutdown due to low battery...")
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
        emergency_stop, min_d, left_free, right_free, center_free = self.check_emergency_collision()
        # Decide if in recovery context
        in_recovery_context = emergency_stop or self.recovery_active
        if in_recovery_context:
            if not self.recovery_active:
                # Initialize recovery session
                self.recovery_active = True
                self.recovery_phase = 0
                self.recovery_phase_ticks = 0
                self.recovery_total_ticks = 0
                self.recovery_no_progress_ticks = 0
                self.recovery_last_min_d = min_d
                # Choose direction based on freer side (positive angular = left/CCW)
                self.recovery_direction = 1 if left_free >= right_free else -1
                # Phase target durations (set first)
                self.recovery_phase_target = np.random.randint(8, 13)  # reverse phase length
            # Obtain network suggestion if sensors ready
            net_action = np.array([0.0, 0.0])
            confidence = 0.0
            if self.use_npu and self.trainer and self.all_sensors_ready():
                net_action, confidence = self.npu_inference(emergency_stop, min_d, left_free, right_free, center_free)
            scripted_action = self.compute_recovery_action(min_d, left_free, right_free, center_free)
            # Blend: if net action weak or low confidence use scripted
            if (abs(net_action[0]) < 0.05 and abs(net_action[1]) < 0.05) or confidence < 0.3:
                final_action = scripted_action
            else:
                final_action = 0.5 * scripted_action + 0.5 * net_action
            # Scale to cmd
            cmd.linear.x = float(np.clip(final_action[0], -1.0, 1.0)) * self.max_speed
            cmd.angular.z = float(np.clip(final_action[1], -1.0, 1.0)) * self.angular_scale
            self.exploration_mode = f"RECOVERY P{self.recovery_phase} d={min_d:.2f}"
            self.last_action = np.array([cmd.linear.x / self.max_speed if self.max_speed>0 else 0.0,
                                         cmd.angular.z / self.angular_scale if self.angular_scale!=0 else 0.0])
            # Clear condition: stable clear distance
            if not emergency_stop and min_d > 0.28:
                self.recovery_clear_ticks += 1
            else:
                self.recovery_clear_ticks = 0
            if self.recovery_clear_ticks >= 5:
                self.reset_recovery_state()
        else:
            if self.use_npu and self.trainer and self.all_sensors_ready():
                action, confidence = self.npu_inference(emergency_stop, min_d, left_free, right_free, center_free)
                
                # Post-process action to encourage forward movement and reduce spinning
                processed_action = self.post_process_action(action)
                
                cmd.linear.x = float(processed_action[0]) * self.max_speed
                cmd.angular.z = float(processed_action[1]) * self.angular_scale
                if self.recovery_active and min_d and min_d > 0.25:
                    self.reset_recovery_state()
                self.exploration_mode = f"NPU_DRIVING (conf: {confidence:.2f})"
                self.collision_detected = False
                self.last_action = np.array([float(processed_action[0]), float(processed_action[1])])
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
            return False, 0.0, 0.0, 0.0, 0.0
            
        try:
            di = self.latest_depth_image
            h, w = di.shape
            # front center region
            ch0 = int(h*0.3); ch1 = int(h*0.7); cw0 = int(w*0.4); cw1 = int(w*0.6)
            front_roi = di[ch0:ch1, cw0:cw1]
            valid_front = front_roi[(front_roi > 0.05) & (front_roi < 4.0)]
            min_distance = float(np.min(valid_front)) if valid_front.size else 10.0
            # side bands for steering guidance
            left_band = di[ch0:ch1, int(w*0.15):int(w*0.35)]
            right_band = di[ch0:ch1, int(w*0.65):int(w*0.85)]
            center_band = di[ch0:ch1, int(w*0.45):int(w*0.55)]
            def free_metric(b):
                v = b[(b > 0.05) & (b < 4.0)]
                return float(np.mean(v)) if v.size else 0.0
            left_free = free_metric(left_band)
            right_free = free_metric(right_band)
            center_free = free_metric(center_band)
            emergency = min_distance < 0.18  # slightly higher to start recovery earlier
            return emergency, min_distance, left_free, right_free, center_free
        except Exception:
            return False, 0.0, 0.0, 0.0, 0.0

    def npu_inference(self, emergency_flag=None, min_d=0.0, left_free=0.0, right_free=0.0, center_free=0.0):
        if not self.all_sensors_ready():
            return np.array([0.0, 0.0]), 0.0
        try:
            depth_input = self.latest_depth_image.astype(np.float32)
            valid = depth_input[(depth_input > 0.05) & (depth_input < 4.0)]
            min_d_global = float(np.min(valid)) if valid.size else 0.0
            mean_d_global = float(np.mean(valid)) if valid.size else 0.0
            near_collision_flag = 1.0 if (valid.size and np.percentile(valid,5) < 0.25) else 0.0
            wheel_diff = self.wheel_velocities[0] - self.wheel_velocities[1]
            emergency_numeric = 1.0 if emergency_flag else 0.0
            
            # Enhanced obstacle awareness - calculate gradient of free space
            depth_gradient = self._calculate_depth_gradient(depth_input)
            
            proprioceptive = np.array([
                self.current_velocity[0],
                self.current_velocity[1],
                float(self.step_count % 100) / 100.0,
                self.last_action[0],
                self.last_action[1],
                wheel_diff,
                min_d_global,
                mean_d_global,
                near_collision_flag,
                emergency_numeric,
                left_free,
                right_free,
                center_free,
                depth_gradient[0],  # left gradient
                depth_gradient[1],  # center gradient
                depth_gradient[2]   # right gradient
            ], dtype=np.float32)
            action, confidence = self.trainer.inference(depth_input, proprioceptive)
            
            # Post-process action for smoother, more proactive behavior
            action = self.post_process_action(action)
            
            # Exploration warmup adjustments (unchanged):
            if self.step_count < self.exploration_warmup_steps and not emergency_flag:
                if np.random.rand() < self.random_action_prob:
                    action = np.array([
                        np.random.uniform(self.min_forward_bias, 1.0),
                        np.random.uniform(-0.6, 0.6)
                    ], dtype=np.float32)
                else:
                    if abs(action[0]) < 0.1:
                        action[0] = self.min_forward_bias
                decay = 1.0 - (self.step_count / self.exploration_warmup_steps)
                action[0] = np.clip(action[0] + 0.1 * decay, -1.0, 1.0)
            # If in emergency and network still outputs forward, bias slightly backward (curriculum)
            if emergency_flag and action[0] > 0.0:
                action[0] = -0.2  # gentle corrective nudge early training
            return action, confidence
        except Exception as e:
            self.get_logger().warn(f"Inference failed: {e}")
            return np.array([0.0,0.0]), 0.0

    def compute_recovery_action(self, min_d, left_free, right_free, center_free):
        """Scripted multi-phase recovery policy.
        Phases:
          0: Reverse to create space
          1: Rotate toward freer side
          2: Forward probe
        Transitions based on ticks and distance improvement.
        """
        # Safety immediate reverse if extremely close
        if min_d < 0.10:
            return np.array([self.recovery_reverse_speed, 0.0], dtype=np.float32)
        # Track distance improvement
        improved = False
        if self.recovery_last_min_d is not None and min_d > self.recovery_last_min_d + 0.015:
            improved = True
        # Phase logic
        if self.recovery_phase == 0:
            # Reverse phase
            action = np.array([self.recovery_reverse_speed, 0.0], dtype=np.float32)
            if self.recovery_phase_ticks >= self.recovery_phase_target or improved:
                self.recovery_phase = 1
                self.recovery_phase_ticks = 0
                # Set rotation duration
                self.recovery_phase_target = np.random.randint(18, 26)
        elif self.recovery_phase == 1:
            # Rotate toward freer side (direction chosen at start)
            # Re-evaluate direction mid-way if large disparity
            if self.recovery_phase_ticks == 0 or (self.recovery_phase_ticks % 10 == 0):
                self.recovery_direction = 1 if left_free >= right_free else -1
            action = np.array([0.0, 0.9 * self.recovery_direction], dtype=np.float32)
            if self.recovery_phase_ticks >= self.recovery_phase_target or improved:
                self.recovery_phase = 2
                self.recovery_phase_ticks = 0
                self.recovery_phase_target = np.random.randint(8, 13)  # probe duration
        else:
            # Forward probe
            forward_speed = 0.4 if min_d > 0.22 else 0.25
            action = np.array([forward_speed, 0.0], dtype=np.float32)
            # If still blocked quickly, go back to rotate with opposite direction
            if min_d < 0.20 and self.recovery_phase_ticks > 3:
                self.recovery_phase = 1
                self.recovery_phase_ticks = 0
                self.recovery_direction *= -1
                self.recovery_phase_target = np.random.randint(18, 26)
            elif self.recovery_phase_ticks >= self.recovery_phase_target or improved:
                # Loop rotation-probe cycle until clear
                self.recovery_phase = 1
                self.recovery_phase_ticks = 0
                self.recovery_phase_target = np.random.randint(18, 26)
        # Update counters
        self.recovery_phase_ticks += 1
        self.recovery_total_ticks += 1
        if not improved:
            self.recovery_no_progress_ticks += 1
        else:
            self.recovery_no_progress_ticks = 0
        self.recovery_last_min_d = min_d
        # Abort recovery if taking too long
        if self.recovery_total_ticks > self.recovery_max_duration:
            self.reset_recovery_state()
        return action

    def _calculate_depth_gradient(self, depth_image):
        """Calculate gradient of free space in left, center, and right regions"""
        if depth_image is None or depth_image.size == 0:
            return [0.0, 0.0, 0.0]
        
        try:
            h, w = depth_image.shape
            
            # Define regions
            left_region = depth_image[int(h*0.3):int(h*0.7), int(w*0.1):int(w*0.3)]
            center_region = depth_image[int(h*0.3):int(h*0.7), int(w*0.4):int(w*0.6)]
            right_region = depth_image[int(h*0.3):int(h*0.7), int(w*0.7):int(w*0.9)]
            
            # Calculate gradients (difference between near and far pixels)
            def calculate_region_gradient(region):
                valid_pixels = region[(region > 0.1) & (region < 4.0)]
                if valid_pixels.size < 10:  # Not enough valid pixels
                    return 0.0
                
                # Sort pixels by distance and calculate gradient
                sorted_pixels = np.sort(valid_pixels)
                near_pixels = sorted_pixels[:len(sorted_pixels)//3]
                far_pixels = sorted_pixels[-len(sorted_pixels)//3:]
                
                if len(near_pixels) > 0 and len(far_pixels) > 0:
                    gradient = np.mean(far_pixels) - np.mean(near_pixels)
                    return float(gradient)
                return 0.0
            
            left_gradient = calculate_region_gradient(left_region)
            center_gradient = calculate_region_gradient(center_region)
            right_gradient = calculate_region_gradient(right_region)
            
            return [left_gradient, center_gradient, right_gradient]
        except Exception:
            return [0.0, 0.0, 0.0]

    def reset_recovery_state(self):
        self.recovery_active = False
        self.recovery_phase = 0
        self.recovery_phase_ticks = 0
        self.recovery_total_ticks = 0
        self.recovery_no_progress_ticks = 0
        self.recovery_last_min_d = None
        self.recovery_clear_ticks = 0
        self.exploration_mode = "NPU_DRIVING"

    def train_from_experience(self):
        if self.operation_mode == 'inference' or self.operation_mode == 'es_inference':
            return
        if not self.all_sensors_ready() or not hasattr(self, 'last_action'):
            return
        try:
            depth_input = self.latest_depth_image
            valid = depth_input[(depth_input > 0.05) & (depth_input < 4.0)]
            min_d_global = float(np.min(valid)) if valid.size else 0.0
            mean_d_global = float(np.mean(valid)) if valid.size else 0.0
            # Improvement signal for recovery
            distance_improved = 0.0
            if self.last_min_distance is not None and min_d_global > self.last_min_distance + 0.02:
                distance_improved = min( (min_d_global - self.last_min_distance), 0.3)
            self.last_min_distance = min_d_global
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
            # Recovery shaping
            if self.recovery_active:
                reward += 4.0 * distance_improved  # positive when clearing space
                if distance_improved == 0.0:
                    reward -= 0.5  # small penalty for ineffective recovery step
                # Extra penalty if many steps with no progress
                if hasattr(self, 'recovery_no_progress_ticks') and self.recovery_no_progress_ticks > 25:
                    reward -= 1.0
            reward = float(np.clip(reward, -10.0, 15.0))
            # Prepare previous frame experience (unchanged core logic)
            if self.prev_depth_image is not None:
                prev = self.prev_depth_image
                pv = prev[(prev > 0.05) & (prev < 4.0)]
                min_prev = float(np.min(pv)) if pv.size else 0.0
                mean_prev = float(np.mean(pv)) if pv.size else 0.0
                near_prev = 1.0 if (pv.size and np.percentile(pv,5) < 0.25) else 0.0
                wheel_diff_prev = self.wheel_velocities[0] - self.wheel_velocities[1]
                emergency_prev = 1.0 if self.recovery_active else 0.0
                # reuse bands for prev if desired (simplify: zeros)
                proprio_prev = np.array([
                    self.current_velocity[0],
                    self.current_velocity[1],
                    float(self.step_count % 100) / 100.0,
                    self.last_action[0],
                    self.last_action[1],
                    wheel_diff_prev,
                    min_prev,
                    mean_prev,
                    near_prev,
                    emergency_prev,
                    0.0,0.0,0.0
                ], dtype=np.float32)
                self.trainer.add_experience(
                    depth_image=prev.astype(np.float32),
                    proprioceptive=proprio_prev,
                    action=self.last_action,
                    reward=reward,
                    next_depth_image=self.latest_depth_image.astype(np.float32),
                    done=False,
                    collision=self.collision_detected,
                    in_recovery=self.recovery_active
                )
            
            # Train based on mode
            if self.operation_mode in ['es_training', 'es_hybrid', 'safe_es_training']:
                # For ES, we evolve every N generations (based on buffer size)
                if self.trainer.buffer_size >= 50 and self.step_count % 50 == 0:
                    training_stats = self.trainer.evolve_population()
                    if self.step_count % 250 == 0:
                        self.get_logger().info(f"ES Training: Gen={training_stats.get('generation',0)} AvgFit={training_stats.get('avg_fitness',0):.4f} BestFit={training_stats.get('best_fitness',0):.4f} Samples={training_stats.get('samples',0)}")
            else:
                # For RL, we train every step
                if self.trainer.buffer_size >= max(32, self.trainer.batch_size):
                    training_stats = self.trainer.train_step()
                    if self.step_count % 50 == 0 and 'loss' in training_stats:
                        self.get_logger().info(f"RL Training: Loss={training_stats['loss']:.4f} AvgR={training_stats.get('avg_reward',0):.2f} Samples={training_stats.get('samples',0)}")
            self.prev_position = self.position.copy()
            self.prev_depth_image = self.latest_depth_image.copy()
        except Exception as e:
            self.get_logger().warn(f"Training step failed: {e}")
            import traceback
            self.get_logger().error(f"Full traceback: {traceback.format_exc()}")
        
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
        if self.trainer:
            self.trainer.safe_save()
        
    def publish_status(self):
        """Publish exploration status"""
        elapsed_time = time.time() - self.start_time
        status_msg = String()
        
        if self.use_npu and self.trainer:
            training_stats = self.trainer.get_training_stats()
            # Check if this is ES or RL training
            if self.operation_mode in ['es_training', 'es_hybrid', 'es_inference', 'safe_es_training']:
                # ES training stats
                status_msg.data = (
                    f"NPU Learning | Mode: {self.exploration_mode} | "
                    f"Battery: {self.current_battery_percentage:.1f}% | "
                    f"Steps: {self.step_count} | "
                    f"Generation: {training_stats.get('generation', 0)} | "
                    f"Buffer: {training_stats['buffer_size']}/10000 | "
                    f"Best Fitness: {training_stats.get('best_fitness', 0):.2f}"
                )
            else:
                # RL training stats
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
    
    def post_process_action(self, action):
        """Post-process action to encourage forward movement and reduce spinning"""
        processed_action = action.copy()
        
        # Apply forward bias factor to encourage forward movement
        if processed_action[0] > 0:
            processed_action[0] = np.clip(processed_action[0] * self.forward_bias_factor, 0, 1.0)
        # Apply backward penalty to discourage backward movement
        elif processed_action[0] < 0:
            processed_action[0] = np.clip(processed_action[0] * self.backward_penalty_factor, -1.0, 0)
        
        # Apply angular dampening to reduce spinning
        processed_action[1] = processed_action[1] * self.angular_dampening
        
        # Additional anti-spinning logic
        linear_speed = abs(processed_action[0])
        angular_speed = abs(processed_action[1])
        
        # If spinning without forward movement, reduce angular velocity further
        if angular_speed > 0.3 and linear_speed < 0.1:
            processed_action[1] = processed_action[1] * 0.5
            
        # If mostly spinning, add a small forward component
        if angular_speed > 0.5 and linear_speed < 0.2:
            processed_action[0] = max(processed_action[0], 0.1)
        
        return processed_action

    def all_sensors_ready(self):
        """Check minimal sensor readiness for depth-based inference.
        Requirements:
          - A processed depth frame received
          - Odometry updated at least once (position defaults change) OR step_count threshold
          - Wheel velocities tuple populated (len==2)
        """
        if self.latest_depth_image is None:
            return False
        # Basic odom evidence: position not both zeros after some steps OR we have advanced step_count
        odom_ok = (self.step_count > 5) or not np.allclose(self.position, [0.0, 0.0])
        wheels_ok = isinstance(self.wheel_velocities, tuple) and len(self.wheel_velocities) == 2
        return odom_ok and wheels_ok

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
