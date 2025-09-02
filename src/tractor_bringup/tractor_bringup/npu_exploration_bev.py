#!/usr/bin/env python3
"""
NPU Bird's Eye View Exploration Node
Uses RKNN for real-time obstacle avoidance and exploration with BEV maps
Integrates with existing Hiwonder motor controller for odometry
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Imu, JointState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32
from std_srvs.srv import Trigger
import numpy as np
import time
import struct
from cv_bridge import CvBridge
import cv2

try:
    from sensor_msgs_py import point_cloud2
    SENSOR_MSGS_PY_AVAILABLE = True
except ImportError:
    SENSOR_MSGS_PY_AVAILABLE = False
    print("sensor_msgs_py not available - using slower point cloud conversion")

try:
    from rknn.api import RKNN
    RKNN_AVAILABLE = True
    print("RKNN library successfully imported")
except ImportError as e:
    RKNN_AVAILABLE = False
    print(f"RKNN not available - using CPU fallback. Error: {e}")

try:
    from .rknn_trainer_bev import RKNNTrainerBEV
    from .es_trainer_bev import EvolutionaryStrategyTrainerBEV
    from .pbt_es_rl_trainer import PBT_ES_RL_Trainer
    from .bev_generator import BEVGenerator
    TRAINER_AVAILABLE = True
    print("Core BEV trainers imported")
except Exception as e:
    TRAINER_AVAILABLE = False
    print(f"Core trainer imports failed: {e}")

class NPUExplorationBEVNode(Node):
    def __init__(self):
        super().__init__('npu_exploration_bev')
        
        # Parameters
        self.declare_parameter('max_speed', 0.15)
        self.declare_parameter('min_battery_percentage', 30.0)
        self.declare_parameter('safety_distance', 0.2)
        self.declare_parameter('npu_inference_rate', 10.0)  # Higher rate for BEV processing
        self.declare_parameter('operation_mode', 'cpu_training')  # cpu_training | hybrid | inference
        self.declare_parameter('train_every_n_frames', 3)  # NEW: train interval to reduce CPU load
        self.declare_parameter('enable_bayesian_optimization', True)  # Enable Bayesian optimization for ES modes
        self.declare_parameter('optimization_level', 'standard')  # basic | standard | full | research
        self.declare_parameter('enable_training_optimization', True)  # Enable training parameter optimization
        self.declare_parameter('enable_reward_optimization', False)  # Enable reward parameter optimization
        self.declare_parameter('enable_multi_metric_evaluation', True)  # Enable multi-metric fitness evaluation
        self.declare_parameter('enable_optimization_monitoring', True)  # Enable comprehensive monitoring
        # IMU integration (LSM9DS1) parameters
        self.declare_parameter('enable_lsm_imu_proprio', False)
        self.declare_parameter('lsm_imu_topic', '/lsm9ds1_imu_publisher/imu/data')
        self.declare_parameter('imu_lpf_alpha', 0.2)  # low-pass filter smoothing factor
        # PBT parameters (for ES-RL hybrid)
        self.declare_parameter('pbt_population_size', 4)
        self.declare_parameter('pbt_update_interval', 1000)
        self.declare_parameter('pbt_perturb_prob', 0.25)
        self.declare_parameter('pbt_resample_prob', 0.25)
        # Distillation parameters (reward-based auto distill)
        self.declare_parameter('enable_reward_based_distill', True)
        self.declare_parameter('distill_check_interval_sec', 300)
        self.declare_parameter('distill_cooldown_sec', 600)
        self.declare_parameter('distill_plateau_sec', 900)
        self.declare_parameter('distill_min_improvement', 0.5)
        self.declare_parameter('distill_min_reward', 0.0)
        
        # Phase 2 parameters - Multi-objective and Architecture Optimization
        self.declare_parameter('enable_multi_objective_optimization', False)  # Enable Pareto frontier optimization
        self.declare_parameter('enable_safety_constraints', True)  # Enable safety constraint handling  
        self.declare_parameter('enable_architecture_optimization', False)  # Enable neural architecture search
        self.declare_parameter('enable_progressive_architecture', False)  # Enable progressive architecture refinement
        self.declare_parameter('enable_sensor_fusion_optimization', False)  # Enable sensor fusion optimization
        
        # BEV parameters
        self.declare_parameter('bev_size', [200, 200])  # [height, width] in pixels
        self.declare_parameter('bev_range', [10.0, 10.0])  # [x_range, y_range] in meters
        self.declare_parameter('bev_height_channels', [0.2, 1.0])  # Height thresholds for channels
        self.declare_parameter('enable_ground_removal', True)  # Enable ground plane removal
        self.declare_parameter('ground_ransac_iterations', 100)  # RANSAC iterations for ground removal
        self.declare_parameter('ground_ransac_threshold', 0.05)  # Distance threshold for ground points
        
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
        self.operation_mode = self.get_parameter('operation_mode').value
        self.train_every_n_frames = int(self.get_parameter('train_every_n_frames').value)
        self.enable_bayesian_optimization = self.get_parameter('enable_bayesian_optimization').value
        self.optimization_level = self.get_parameter('optimization_level').value
        self.enable_training_optimization = self.get_parameter('enable_training_optimization').value
        self.enable_reward_optimization = self.get_parameter('enable_reward_optimization').value
        self.enable_multi_metric_evaluation = self.get_parameter('enable_multi_metric_evaluation').value
        self.enable_optimization_monitoring = self.get_parameter('enable_optimization_monitoring').value
        # IMU integration values
        self.enable_lsm_imu_proprio = bool(self.get_parameter('enable_lsm_imu_proprio').value)
        self.lsm_imu_topic = str(self.get_parameter('lsm_imu_topic').value)
        self.imu_lpf_alpha = float(self.get_parameter('imu_lpf_alpha').value)
        
        # Phase 2 parameter values
        self.enable_multi_objective_optimization = self.get_parameter('enable_multi_objective_optimization').value
        self.enable_safety_constraints = self.get_parameter('enable_safety_constraints').value
        self.enable_architecture_optimization = self.get_parameter('enable_architecture_optimization').value
        self.enable_progressive_architecture = self.get_parameter('enable_progressive_architecture').value
        self.enable_sensor_fusion_optimization = self.get_parameter('enable_sensor_fusion_optimization').value

        # Adjust training cadence for PBT mode to reduce CPU load
        if self.operation_mode == 'es_rl_hybrid':
            self.train_every_n_frames = max(self.train_every_n_frames, 5)
        
        # BEV parameter values
        bev_size_param = self.get_parameter('bev_size').value
        self.bev_height = int(bev_size_param[0]) if isinstance(bev_size_param, list) else 200
        self.bev_width = int(bev_size_param[1]) if isinstance(bev_size_param, list) else 200
        
        bev_range_param = self.get_parameter('bev_range').value
        self.bev_x_range = float(bev_range_param[0]) if isinstance(bev_range_param, list) else 10.0
        self.bev_y_range = float(bev_range_param[1]) if isinstance(bev_range_param, list) else 10.0
        
        height_channels_param = self.get_parameter('bev_height_channels').value
        self.bev_height_channels = tuple(float(x) for x in height_channels_param) if isinstance(height_channels_param, list) else (0.2, 1.0)
        
        self.enable_ground_removal = self.get_parameter('enable_ground_removal').value
        self.ground_ransac_iterations = self.get_parameter('ground_ransac_iterations').value
        self.ground_ransac_threshold = self.get_parameter('ground_ransac_threshold').value
        
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
        self.latest_pointcloud = None
        self.bridge = CvBridge()
        
        # Previous state for reward calculation
        self.prev_position = np.array([0.0, 0.0])
        self.prev_bev_image = None
        self.collision_detected = False
        
        # Exploration state
        self.exploration_mode = "forward_explore"  # forward_explore, turn_explore, retreat
        self.stuck_counter = 0
        self.last_position = np.array([0.0, 0.0])
        self.movement_threshold = 0.05  # meters
        
        # Simple tracking for movement detection
        self.movement_check_counter = 0
        
        # Initialize BEV generator with grass-aware filtering
        self.bev_generator = BEVGenerator(
            bev_size=(self.bev_height, self.bev_width),
            bev_range=(self.bev_x_range, self.bev_y_range),
            height_channels=self.bev_height_channels,
            enable_ground_removal=self.enable_ground_removal,
            ground_ransac_iterations=self.ground_ransac_iterations,
            ground_ransac_threshold=self.ground_ransac_threshold,
            # Grass-aware filtering parameters
            enable_grass_filtering=True,
            grass_height_tolerance=0.15,  # 15cm grass tolerance
            min_obstacle_height=0.25      # 25cm minimum obstacle height
        )
        
        # Initialize NPU or fallback
        self.init_inference_engine()
        
        # Disable heavy optimization components in simplified setup
        self.multi_metric_evaluator = None
        self.optimization_monitor = None
        self.bayesian_reward_wrapper = None
        
        # ROS2 interfaces
        self.pc_sub = self.create_subscription(
            PointCloud2, 'point_cloud',
            self.pointcloud_callback, 10
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
        # Expose services to save and distill models
        self.save_srv = self.create_service(Trigger, '/save_models', self.handle_save_models)
        self.distill_srv = self.create_service(Trigger, '/distill_best', self.handle_distill_best)
        
        # Control timer
        self.control_timer = self.create_timer(0.1, self.control_loop)  # 10Hz control
        # Separate status timer (1 Hz) to reduce coupling to point cloud rate
        self.status_timer = self.create_timer(1.0, self.publish_status)
        # Reward-based distillation tracking and timer
        self.enable_reward_based_distill = bool(self.get_parameter('enable_reward_based_distill').value)
        self.last_distill_time = 0.0
        self.last_distill_metric = None
        if self.enable_reward_based_distill:
            check_int = float(self.get_parameter('distill_check_interval_sec').value)
            self.distill_timer = self.create_timer(max(30.0, check_int), self._maybe_distill)
        self.get_logger().info(f"NPU BEV Exploration Node initialized")
        self.get_logger().info(f"  Max Speed: {self.max_speed} m/s")
        self.get_logger().info(f"  Min Battery: {self.min_battery_percentage}%")
        self.get_logger().info(f"  NPU Available: {RKNN_AVAILABLE}")
        self.get_logger().info(f"  Inference target rate: {self.inference_rate} Hz")
        self.get_logger().info(f"  Operation Mode: {self.operation_mode}")
        self.get_logger().info(f"  Optimization Level: {self.optimization_level}")
        self.get_logger().info(f"  Training Opt: {self.enable_training_optimization}")
        self.get_logger().info(f"  Reward Opt: {self.enable_reward_optimization}")
        self.get_logger().info(f"  Multi-Metric: {self.enable_multi_metric_evaluation}")
        self.get_logger().info(f"  Monitoring: {self.enable_optimization_monitoring}")
        self.get_logger().info(f"  Multi-Objective: {self.enable_multi_objective_optimization}")
        self.get_logger().info(f"  Safety Constraints: {self.enable_safety_constraints}")
        self.get_logger().info(f"  Architecture Opt: {self.enable_architecture_optimization}")
        self.get_logger().info(f"  Progressive Arch: {self.enable_progressive_architecture}")
        self.get_logger().info(f"  Sensor Fusion Opt: {self.enable_sensor_fusion_optimization}")
        self.get_logger().info(f"  BEV Size: {self.bev_height}x{self.bev_width}")
        self.get_logger().info(f"  BEV Range: {self.bev_x_range}x{self.bev_y_range}m")
        self.get_logger().info(f"  Height Channels: {self.bev_height_channels}")
        self.get_logger().info(f"  Ground Removal: {self.enable_ground_removal}")
        self.get_logger().info(f"  LSM IMU Proprio: {self.enable_lsm_imu_proprio}")
        
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

        # IMU state and subscription (optional)
        self.imu_ready = False
        self.imu_state = {
            'yaw_rate': 0.0,
            'roll': 0.0,
            'pitch': 0.0,
            'accel_forward': 0.0,
            'accel_mag': 0.0,
        }
        if self.enable_lsm_imu_proprio:
            try:
                self.imu_sub = self.create_subscription(Imu, self.lsm_imu_topic, self.imu_callback, 50)
            except Exception as e:
                self.get_logger().warn(f"IMU subscription failed: {e}")

    def handle_save_models(self, request, response):
        try:
            if self.trainer:
                self.trainer.safe_save()
            response.success = True
            response.message = "Models saved"
        except Exception as e:
            response.success = False
            response.message = f"Save failed: {e}"
        return response

    def handle_distill_best(self, request, response):
        try:
            if not self.trainer:
                response.success = False
                response.message = "No trainer available"
                return response
            # Prefer PBT distillation if available
            if hasattr(self.trainer, 'distill_best'):
                out = self.trainer.distill_best()
            elif hasattr(self.trainer, 'distill_to_student'):
                out = self.trainer.distill_to_student()
            else:
                out = ""
            if out:
                response.success = True
                response.message = f"Distilled to {out}.pth/.rknn"
            else:
                response.success = False
                response.message = "Distillation not supported or failed"
        except Exception as e:
            response.success = False
            response.message = f"Distillation error: {e}"
        return response

    def init_inference_engine(self):
        self.use_npu = False
        self.trainer = None
        
        if TRAINER_AVAILABLE:
            try:
                enable_debug = (self.get_parameter('operation_mode').value != 'inference')
                mode = self.get_parameter('operation_mode').value
                
                # Calculate number of BEV channels (height slices + max height + density)
                bev_channels = len(self.bev_height_channels) + 2
                
                # Initialize appropriate trainer based on mode
                # For now, always use BEV trainer since we're working with BEV images
                # TODO: Create ES trainer that supports BEV images
                # Determine extra proprio size (standardize: base extras 13 + 5 IMU features)
                # IMU features default to zeros if IMU data is disabled/unavailable
                extra_proprio = 13 + 5
                self.trainer = RKNNTrainerBEV(
                    bev_channels=bev_channels,
                    enable_debug=enable_debug,
                    enable_bayesian_training_optimization=self.enable_training_optimization,
                    extra_proprio=extra_proprio
                )
                
                if mode in ['es_training', 'es_hybrid', 'es_inference', 'safe_es_training']:
                    # Switch to BEV ES trainer for ES modes
                    self.trainer = EvolutionaryStrategyTrainerBEV(
                        bev_channels=bev_channels,
                        enable_debug=enable_debug,
                        enable_bayesian_optimization=self.enable_bayesian_optimization,
                        extra_proprio=extra_proprio
                    )
                    self.get_logger().info("ES BEV Trainer initialized")
                else:
                    training_opt_status = "with Bayesian training optimization" if self.enable_training_optimization else "with fixed training parameters"
                    self.get_logger().info(f"RKNN BEV Trainer initialized {training_opt_status}")
                
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
                    if hasattr(self.trainer, 'enable_rknn_inference') and self.trainer.enable_rknn_inference():
                        self.use_npu = True
                        self.get_logger().info("Mode: ES Hybrid (RKNN runtime inference + ES training)")
                    else:
                        self.use_npu = True
                        self.get_logger().warn("ES Hybrid mode requested but RKNN runtime not available - using PyTorch")
                elif mode == 'es_inference':
                    # Pure ES inference: load RKNN and disable training logic
                    if hasattr(self.trainer, 'enable_rknn_inference') and self.trainer.enable_rknn_inference():
                        self.use_npu = True
                        self.get_logger().info("Mode: ES Pure RKNN inference (no training)")
                    else:
                        self.use_npu = True
                        self.get_logger().warn("ES Pure inference mode requested but RKNN file/runtime not available - falling back to PyTorch")
                elif mode == 'safe_es_training':
                    # Safe ES training with anti-overtraining measures
                    self.use_npu = True
                    self.get_logger().info("Mode: Safe ES training (anti-overtraining protection with ES)")
                elif mode == 'es_rl_hybrid':
                    # PBT ES-RL Hybrid training
                    self.use_npu = True
                    # Read PBT params
                    pbt_pop = int(self.get_parameter('pbt_population_size').value)
                    pbt_interval = int(self.get_parameter('pbt_update_interval').value)
                    pbt_perturb = float(self.get_parameter('pbt_perturb_prob').value)
                    pbt_resample = float(self.get_parameter('pbt_resample_prob').value)
                    self.trainer = PBT_ES_RL_Trainer(
                        population_size=pbt_pop,
                        bev_channels=bev_channels,
                        pbt_interval=pbt_interval,
                        perturb_prob=pbt_perturb,
                        resample_prob=pbt_resample,
                        enable_debug=enable_debug,
                        extra_proprio=extra_proprio
                    )
                    # Use the same interval for agent switching in this node
                    self.pbt_update_interval = pbt_interval
                    self.get_logger().info("Mode: PBT ES-RL Hybrid training")
                else:
                    self.use_npu = True
                    self.get_logger().warn(f"Unknown operation_mode '{mode}' - defaulting to cpu_training")
            except Exception as e:
                self.get_logger().warn(f"Trainer init failed: {e}")
        else:
            self.get_logger().info("Trainer not available")
            
    def init_optimization_components(self):
        """Initialize advanced optimization components based on configuration level"""
        
        # Initialize optimization components based on availability and settings
        self.multi_metric_evaluator = None
        self.optimization_monitor = None
        self.bayesian_reward_wrapper = None
        
        # Phase 2 components
        self.multi_objective_optimizer = None
        self.safety_constraint_handler = None
        self.architecture_optimizer = None
        self.progressive_architecture = None
        self.sensor_fusion_optimizer = None
        
        if not OPTIMIZATION_AVAILABLE:
            self.get_logger().info("Optimization components not available")
            return
            
        try:
            # Initialize multi-metric evaluator
            if self.enable_multi_metric_evaluation:
                # Set objective weights based on optimization level
                if self.optimization_level == 'basic':
                    weights = ObjectiveWeights(performance=0.6, safety=0.4, efficiency=0.0, robustness=0.0)
                elif self.optimization_level == 'standard':
                    weights = ObjectiveWeights(performance=0.4, safety=0.3, efficiency=0.2, robustness=0.1)
                elif self.optimization_level == 'full':
                    weights = ObjectiveWeights(performance=0.3, safety=0.3, efficiency=0.25, robustness=0.15)
                elif self.optimization_level == 'research':
                    weights = ObjectiveWeights(performance=0.25, safety=0.25, efficiency=0.25, robustness=0.25)
                else:
                    weights = ObjectiveWeights()  # Default
                
                self.multi_metric_evaluator = MultiMetricEvaluator(
                    evaluation_window=200,
                    enable_debug=False,  # Disable debug to reduce log noise
                    objective_weights=weights
                )
                self.get_logger().info(f"Multi-metric evaluator initialized with {self.optimization_level} level weights")
            
            # Initialize optimization monitoring
            if self.enable_optimization_monitoring:
                self.optimization_monitor = OptimizationMonitor(
                    log_dir="logs/optimization",
                    enable_visualization=False,  # Disable visualization on robot
                    enable_debug=False
                )
                self.get_logger().info("Optimization monitoring initialized")
            
            # Initialize reward parameter optimization wrapper
            if self.enable_reward_optimization and hasattr(self.trainer, 'reward_calculator') and self.trainer.reward_calculator is not None:
                self.bayesian_reward_wrapper = AdaptiveBayesianRewardWrapper(
                    reward_calculator=self.trainer.reward_calculator,
                    optimization_interval=500,  # Every 500 training steps
                    min_observations=5,
                    enable_debug=False
                )
                self.get_logger().info("Bayesian reward optimization wrapper initialized")
            
            # Configure trainer with Bayesian training optimization if available
            if self.enable_training_optimization and hasattr(self.trainer, 'enable_bayesian_training_optimization'):
                # This was already set in trainer initialization
                self.get_logger().info("Bayesian training optimization enabled in trainer")
                
            # Initialize Phase 2 components
            self._initialize_phase2_components()
            
            self.get_logger().info(f"Optimization components initialized for {self.optimization_level} level")
            
        except Exception as e:
            self.get_logger().warn(f"Optimization component initialization failed: {e}")
            # Disable optimization components to prevent crashes
            self.multi_metric_evaluator = None
            self.optimization_monitor = None
            self.bayesian_reward_wrapper = None
            
    def _initialize_phase2_components(self):
        """Initialize Phase 2 multi-objective and architecture optimization components"""
        
        if not PHASE2_AVAILABLE:
            self.get_logger().info("Phase 2 optimization components not available")
            return
            
        try:
            # Initialize safety constraint handler
            if self.enable_safety_constraints:
                self.safety_constraint_handler = SafetyConstraintHandler(
                    enable_debug=False  # Disable debug to reduce log noise
                )
                self.get_logger().info("Safety constraint handler initialized")
            
            # Initialize multi-objective optimizer
            if self.enable_multi_objective_optimization:
                # Define parameter bounds for multi-objective optimization
                parameter_bounds = {
                    'max_speed': (0.05, 0.5),
                    'safety_distance': (0.1, 0.5),
                    'exploration_bonus': (0.0, 5.0),
                    'collision_penalty': (5.0, 50.0),
                    'learning_rate': (1e-6, 0.1),
                    'batch_size': (8, 256)
                }
                
                # Set objective names and reference point based on optimization level
                if self.optimization_level == 'research':
                    objective_names = ['performance', 'safety', 'efficiency', 'robustness']
                    reference_point = [0.0, 0.0, 0.0, 0.0]
                else:
                    objective_names = ['performance', 'safety']
                    reference_point = [0.0, 0.0]
                
                self.multi_objective_optimizer = MultiObjectiveBayesianOptimizer(
                    parameter_bounds=parameter_bounds,
                    objective_names=objective_names,
                    reference_point=reference_point,
                    enable_debug=False
                )
                self.get_logger().info(f"Multi-objective optimizer initialized with {len(objective_names)} objectives")
            
            # Initialize architecture optimizer
            if self.enable_architecture_optimization:
                # Only enable for research level due to computational cost
                if self.optimization_level == 'research':
                    # Create architecture for BEV input
                    from .bayesian_nas import ConvLayerConfig, FusionLayerConfig, NetworkArchitecture, ActivationType
                    
                    initial_conv_layers = [
                        ConvLayerConfig(out_channels=32, kernel_size=5, stride=2),
                        ConvLayerConfig(out_channels=64, kernel_size=3, stride=2),
                        ConvLayerConfig(out_channels=128, kernel_size=3, stride=2),
                        ConvLayerConfig(out_channels=256, kernel_size=3, stride=2)
                    ]
                    
                    initial_fusion_config = FusionLayerConfig(
                        hidden_sizes=[512, 256],
                        activation=ActivationType.RELU,
                        dropout_rate=0.3
                    )
                    
                    initial_architecture = NetworkArchitecture(
                        depth_conv_layers=initial_conv_layers,  # Reuse for BEV
                        depth_fc_size=512,
                        sensor_fc_layers=[128, 64],
                        sensor_activation=ActivationType.RELU,
                        fusion_config=initial_fusion_config,
                        output_size=3
                    )
                    
                    self.architecture_optimizer = BayesianArchitectureOptimizer(
                        input_shape=(len(self.bev_height_channels) + 2, self.bev_height, self.bev_width),  # BEV shape
                        sensor_input_size=16,  # Proprioceptive input size
                        output_size=3,  # [forward, angular, confidence]
                        max_parameters=1000000,  # 1M parameter budget
                        enable_debug=False
                    )
                    self.get_logger().info("Bayesian architecture optimizer initialized for BEV")
                else:
                    self.get_logger().info("Architecture optimization requires research level - skipping")
            
            # Initialize progressive architecture refinement
            if self.enable_progressive_architecture and self.architecture_optimizer:
                # Create a basic initial architecture for BEV
                from .bayesian_nas import ConvLayerConfig, FusionLayerConfig, NetworkArchitecture, ActivationType
                
                initial_conv_layers = [
                    ConvLayerConfig(out_channels=32, kernel_size=3, stride=2),
                    ConvLayerConfig(out_channels=64, kernel_size=3, stride=2),
                    ConvLayerConfig(out_channels=128, kernel_size=3, stride=2)
                ]
                
                initial_fusion_config = FusionLayerConfig(
                    hidden_sizes=[256, 128],
                    activation=ActivationType.RELU,
                    dropout_rate=0.3
                )
                
                initial_architecture = NetworkArchitecture(
                    depth_conv_layers=initial_conv_layers,
                    depth_fc_size=512,
                    sensor_fc_layers=[64, 32],
                    sensor_activation=ActivationType.RELU,
                    fusion_config=initial_fusion_config,
                    output_size=3
                )
                
                progression_config = ProgressionConfig(
                    min_stage_duration=200,  # Shorter for robot training
                    max_stage_duration=1000,
                    convergence_patience=100
                )
                
                self.progressive_architecture = ProgressiveArchitectureRefinement(
                    initial_architecture=initial_architecture,
                    progression_config=progression_config,
                    enable_debug=False
                )
                self.get_logger().info("Progressive architecture refinement initialized for BEV")
            
            # Initialize sensor fusion optimizer
            if self.enable_sensor_fusion_optimization:
                # Enable for full and research optimization levels
                if self.optimization_level in ['full', 'research']:
                    self.sensor_fusion_optimizer = BayesianSensorFusionOptimizer(
                        depth_image_shape=(self.bev_height, self.bev_width),  # BEV image shape
                        max_frame_stack=1,  # No frame stacking for BEV
                        enable_debug=False
                    )
                    
                    # Suggest initial configuration and apply it
                    initial_config = self.sensor_fusion_optimizer.suggest_configuration()
                    self._current_sensor_fusion_config = initial_config
                    
                    self.get_logger().info("Bayesian sensor fusion optimizer initialized for BEV")
                    self.get_logger().info(f"Initial sensor fusion config: "
                                         f"preprocessing={initial_config.preprocessing_method.value}, "
                                         f"frame_stack={initial_config.frame_stack_size}")
                else:
                    self.get_logger().info("Sensor fusion optimization requires full/research level - skipping")
            
            phase2_components_count = sum([
                1 if self.safety_constraint_handler else 0,
                1 if self.multi_objective_optimizer else 0,
                1 if self.architecture_optimizer else 0,
                1 if self.progressive_architecture else 0,
                1 if self.sensor_fusion_optimizer else 0
            ])
            
            if phase2_components_count > 0:
                self.get_logger().info(f"Phase 2: Initialized {phase2_components_count} advanced optimization components")
            else:
                self.get_logger().info("Phase 2: No advanced optimization components enabled")
                
        except Exception as e:
            self.get_logger().warn(f"Phase 2 component initialization failed: {e}")
            # Reset Phase 2 components on failure
            self.multi_objective_optimizer = None
            self.safety_constraint_handler = None
            self.architecture_optimizer = None
            self.progressive_architecture = None
            self.sensor_fusion_optimizer = None
            
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
        
        # For PBT, select an active agent periodically (configurable interval)
        if self.operation_mode == 'es_rl_hybrid':
            interval = getattr(self, 'pbt_update_interval', 1000)
            if interval > 0 and self.step_count % interval == 0:
                self.trainer.select_active_agent()
                self.get_logger().info(f"Switched to PBT agent {self.trainer.active_agent_idx}")

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
        
    def preprocess_pointcloud(self, pc_msg):
        """Convert ROS PointCloud2 to numpy array for processing"""
        try:
            # Extract point cloud data using existing method
            points = self.ros_pc2_to_numpy(pc_msg)
            
            if points is None or points.size == 0:
                self.get_logger().warn("Empty or None point cloud received")
                # Return empty array
                return np.zeros((0, 3), dtype=np.float32)
                
            # Log some information about the point cloud
            self.get_logger().debug(f"Raw point cloud shape: {points.shape}")
            
            # Filter out invalid points
            valid_mask = np.isfinite(points).all(axis=1)
            points = points[valid_mask]
            
            # Filter by distance range
            distances = np.linalg.norm(points[:, :2], axis=1)  # Only x,y for distance
            range_mask = (distances > 0.1) & (distances < self.bev_x_range * 1.5)
            points = points[range_mask]
            
            self.get_logger().debug(f"Processed point cloud shape: {points.shape}")
            
            return points
            
        except Exception as e:
            self.get_logger().warn(f"Point cloud processing failed: {e}")
            # Return empty array
            return np.zeros((0, 3), dtype=np.float32)
            
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
            
        # Check if we have recent point cloud data
        if not hasattr(self, 'latest_pointcloud'):
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
        
        # Update multi-metric evaluation
        if self.multi_metric_evaluator and self.step_count % 10 == 0:  # Every 10 steps
            try:
                # Calculate current reward (simplified)
                current_reward = 0.0
                if hasattr(self, 'last_action') and len(self.last_action) > 0:
                    # Simple reward: forward movement good, collision bad
                    current_reward = self.last_action[0] * 5.0  # Forward motion bonus
                    if self.collision_detected:
                        current_reward -= 20.0  # Collision penalty
                
                # Update multi-metric evaluator
                self.multi_metric_evaluator.update_metrics(
                    reward=current_reward,
                    position=self.position,
                    action=self.last_action if hasattr(self, 'last_action') else np.array([0.0, 0.0]),
                    collision=self.collision_detected,
                    near_collision=emergency_stop and not self.collision_detected,
                    safety_margin=min_d if 'min_d' in locals() else 1.0,
                    energy_used=abs(self.last_action[0]) + abs(self.last_action[1]) if hasattr(self, 'last_action') else 0.0,
                    training_loss=None  # Would be filled in during training
                )
                
                # Calculate comprehensive fitness every 100 steps
                if self.step_count % 100 == 0:
                    overall_fitness, metrics, objective_scores = self.multi_metric_evaluator.calculate_comprehensive_fitness()
                    
                    # Log multi-metric evaluation
                    if self.optimization_monitor:
                        detailed_metrics = {
                            'avg_reward': metrics.avg_reward,
                            'exploration_coverage': metrics.exploration_coverage,
                            'collision_rate': metrics.collision_rate,
                            'behavioral_diversity': metrics.behavioral_diversity,
                            'movement_efficiency': metrics.movement_efficiency
                        }
                        self.optimization_monitor.log_multi_metric_evaluation(
                            step=self.step_count,
                            overall_fitness=overall_fitness,
                            objective_scores=objective_scores,
                            detailed_metrics=detailed_metrics
                        )
                    
                    # Optional: Log summary periodically
                    if self.step_count % 500 == 0:
                        self.get_logger().info(f"Multi-Metric Fitness: {overall_fitness:.3f} "
                                             f"(P:{objective_scores.get('performance', 0):.2f} "
                                             f"S:{objective_scores.get('safety', 0):.2f} "
                                             f"E:{objective_scores.get('efficiency', 0):.2f} "
                                             f"R:{objective_scores.get('robustness', 0):.2f})")
                        
            except Exception as e:
                self.get_logger().warn(f"Multi-metric evaluation failed: {e}")
        
        return cmd
        
    def check_emergency_collision(self):
        """Guardian: compute nearest forward obstacle distance in meters from BEV and enforce safety stop.
        Uses obstacle confidence channel and correct BEV orientation (bottom = forward)."""
        if self.latest_pointcloud is None or self.latest_pointcloud.size == 0:
            return False, 10.0, 0.0, 0.0, 0.0

        try:
            bev_image = self.bev_generator.generate_bev(self.latest_pointcloud)
            h, w, _ = bev_image.shape
            # Channels
            conf = bev_image[:, :, 3]  # obstacle confidence [0,1]
            low = bev_image[:, :, 2]   # low obstacle presence
            # Occupancy mask (confidence or low obstacles)
            occ = (conf > 0.25) | (low > 0.1)

            # Define regions. In our BEV, pixel_x maps x in [-range, +range] to [0..H-1].
            # Front (forward) is at larger pixel_x (bottom of image).
            front_rows = slice(int(h*2/3), h)
            center_cols = slice(int(w/3), int(2*w/3))
            left_cols = slice(0, int(w/3))
            right_cols = slice(int(2*w/3), w)

            # Helper to compute nearest forward distance (meters) in a region
            def nearest_forward_distance(mask_region):
                ys, xs = np.where(mask_region)
                if ys.size == 0:
                    return 10.0
                # Convert row index to meters: x = -x_range .. +x_range
                # px = ((x + x_range) / (2*x_range)) * H  => x = (px/H)*2*x_range - x_range
                px = ys.astype(np.float32)
                x_m = (px / float(h)) * (2.0 * self.bev_generator.x_range) - self.bev_generator.x_range
                # Forward only
                x_m = x_m[x_m >= 0.0]
                if x_m.size == 0:
                    return 10.0
                return float(np.min(x_m))

            # Compose regional masks
            front_center_mask = occ[front_rows, center_cols]
            left_mask = occ[front_rows, left_cols]
            right_mask = occ[front_rows, right_cols]
            center_mask = occ[front_rows, center_cols]

            min_d = nearest_forward_distance(front_center_mask)
            left_d = nearest_forward_distance(left_mask)
            right_d = nearest_forward_distance(right_mask)
            center_d = nearest_forward_distance(center_mask)

            # Convert distances into free metrics (larger distance => more free)
            # Normalize by x_range to [0,1]
            xr = float(self.bev_generator.x_range)
            left_free = float(np.clip(left_d / xr, 0.0, 1.0))
            right_free = float(np.clip(right_d / xr, 0.0, 1.0))
            center_free = float(np.clip(center_d / xr, 0.0, 1.0))

            # Emergency if nearest forward obstacle is closer than safety_distance
            sd = float(self.safety_distance)
            emergency = (min_d < sd)
            return emergency, float(min_d), left_free, right_free, center_free
        except Exception:
            return False, 10.0, 0.0, 0.0, 0.0

    def npu_inference(self, emergency_flag=None, min_d=0.0, left_free=0.0, right_free=0.0, center_free=0.0):
        if not self.all_sensors_ready():
            return np.array([0.0, 0.0]), 0.0
        try:
            # Generate BEV from point cloud
            bev_image = self.bev_generator.generate_bev(self.latest_pointcloud)
            
            # Calculate BEV-based metrics
            max_height_channel = bev_image[:, :, 0]
            density_channel = bev_image[:, :, 1]
            
            valid_heights = max_height_channel[max_height_channel > 0.05]
            min_d_global = float(np.min(valid_heights)) if valid_heights.size else 0.0
            mean_d_global = float(np.mean(valid_heights)) if valid_heights.size else 0.0
            near_collision_flag = 1.0 if (valid_heights.size and np.percentile(valid_heights, 5) < 0.25) else 0.0
            
            wheel_diff = self.wheel_velocities[0] - self.wheel_velocities[1]
            emergency_numeric = 1.0 if emergency_flag else 0.0
            
            # Enhanced obstacle awareness - calculate gradient of free space in BEV
            bev_gradient = self._calculate_bev_gradient(bev_image)
            
            proprioceptive = self.build_proprio_vector(
                wheel_diff, min_d_global, mean_d_global,
                near_collision_flag, emergency_numeric,
                left_free, right_free, center_free,
                bev_gradient
            )
            
            action, confidence = self.trainer.inference(bev_image, proprioceptive)
            
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

    def _calculate_bev_gradient(self, bev_image):
        """Calculate gradient of free space in left, center, and right regions of BEV"""
        if bev_image is None or bev_image.size == 0:
            return [0.0, 0.0, 0.0]
        
        try:
            h, w, c = bev_image.shape
            
            # Use max height channel (channel 0) for gradient calculation
            height_channel = bev_image[:, :, 0]
            
            # Define regions
            left_region = height_channel[:2*h//3, :w//3]
            center_region = height_channel[:2*h//3, w//3:2*w//3]
            right_region = height_channel[:2*h//3, 2*w//3:]
            
            # Calculate gradients (difference between near and far pixels)
            def calculate_region_gradient(region):
                valid_pixels = region[region > 0.1]
                if valid_pixels.size < 10:  # Not enough valid pixels
                    return 0.0
                
                # Sort pixels by height and calculate gradient
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
            # Generate BEV from point cloud
            bev_image = self.bev_generator.generate_bev(self.latest_pointcloud)
            
            # Calculate BEV-based metrics
            max_height_channel = bev_image[:, :, 0]
            valid_heights = max_height_channel[max_height_channel > 0.05]
            min_d_global = float(np.min(valid_heights)) if valid_heights.size else 0.0
            mean_d_global = float(np.mean(valid_heights)) if valid_heights.size else 0.0
            
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
                bev_data=bev_image,
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
            if self.prev_bev_image is not None:
                prev_max_height = self.prev_bev_image[:, :, 0]
                pv = prev_max_height[prev_max_height > 0.05]
                min_prev = float(np.min(pv)) if pv.size else 0.0
                mean_prev = float(np.mean(pv)) if pv.size else 0.0
                near_prev = 1.0 if (pv.size and np.percentile(pv, 5) < 0.25) else 0.0
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
                    0.0, 0.0, 0.0
                ], dtype=np.float32)
                self.trainer.add_experience(
                    bev_image=self.prev_bev_image.astype(np.float32),
                    proprioceptive=proprio_prev,
                    action=self.last_action,
                    reward=reward,
                    next_bev_image=bev_image.astype(np.float32),
                    done=False,
                    collision=self.collision_detected,
                    in_recovery=self.recovery_active
                )
            
            # Train based on mode
            if self.operation_mode in ['es_training', 'es_hybrid', 'safe_es_training']:
                # Use ES trainer when available
                if hasattr(self.trainer, 'should_evolve') and hasattr(self.trainer, 'evolve_population'):
                    if self.trainer.should_evolve(self.step_count):
                        training_stats = self.trainer.evolve_population()
                        if self.step_count % 50 == 0:
                            self.get_logger().info(
                                f"ES: Gen={training_stats.get('generation',0)} AvgFit={training_stats.get('avg_fitness',0):.3f} "
                                f"Best={training_stats.get('best_fitness',0):.3f} Samples={training_stats.get('samples',0)}"
                            )
                        if self.optimization_monitor:
                            self.optimization_monitor.log_training_optimization(
                                step=self.step_count,
                                fitness_score=training_stats.get('avg_fitness', 0.0),
                                loss=None,
                                details=training_stats
                            )
                else:
                    # Fallback to RL training if ES trainer not available
                    if self.trainer.buffer_size >= max(32, getattr(self.trainer, 'batch_size', 32)):
                        training_stats = self.trainer.train_step()
                        if self.step_count % 50 == 0 and 'loss' in training_stats:
                            self.get_logger().info(
                                f"ES Mode (RL fallback): Loss={training_stats['loss']:.4f} AvgR={training_stats.get('avg_reward',0):.2f} Samples={training_stats.get('samples',0)}"
                            )
                        if self.optimization_monitor:
                            self.optimization_monitor.log_training_optimization(
                                step=self.step_count,
                                fitness_score=-training_stats.get('loss', 0),
                                loss=training_stats.get('loss', None),
                                details=training_stats
                            )
            else:
                # For RL, we train every step
                if self.trainer.buffer_size >= max(32, self.trainer.batch_size):
                    training_stats = self.trainer.train_step()
                    if self.step_count % 50 == 0 and 'loss' in training_stats:
                        self.get_logger().info(f"RL Training: Loss={training_stats['loss']:.4f} AvgR={training_stats.get('avg_reward',0):.2f} Samples={training_stats.get('samples',0)}")
                        
                        # Update optimization monitoring for training parameters
                        if self.optimization_monitor and hasattr(self.trainer, 'get_training_optimization_stats'):
                            train_opt_stats = self.trainer.get_training_optimization_stats()
                            if train_opt_stats.get('bayesian_training_optimization') != 'disabled':
                                self.optimization_monitor.log_training_optimization(
                                    step=self.step_count,
                                    fitness_score=train_opt_stats.get('current_fitness', 0),
                                    best_params={
                                        'learning_rate': train_opt_stats.get('best_learning_rate', 0),
                                        'batch_size': train_opt_stats.get('best_batch_size', 0),
                                        'dropout_rate': train_opt_stats.get('best_dropout_rate', 0)
                                    },
                                    training_stats=training_stats
                                )
                        
                        # Update reward parameter optimization
                        if self.bayesian_reward_wrapper:
                            performance_metrics = {
                                'avg_reward': training_stats.get('avg_reward', 0),
                                'behavioral_diversity': 0.5,  # Would need to calculate this
                                'collision_rate': 0.1 if self.collision_detected else 0.0,
                                'exploration_progress': min(1.0, self.step_count / 1000.0)
                            }
                            self.bayesian_reward_wrapper.apply_bayesian_optimization(
                                self.step_count, performance_metrics
                            )
            self.prev_position = self.position.copy()
            self.prev_bev_image = bev_image.copy()
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

    def imu_callback(self, msg: Imu):
        """Low-pass filter IMU signals and cache for proprio usage"""
        try:
            a = self.imu_lpf_alpha
            # Yaw rate from gyro z
            self.imu_state['yaw_rate'] = (1-a) * self.imu_state['yaw_rate'] + a * float(msg.angular_velocity.z)
            # Roll/pitch from orientation (if provided)
            try:
                # Convert quaternion to roll/pitch (radians)
                x, y, z, w = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
                # Roll (x-axis rotation)
                sinr_cosp = 2 * (w * x + y * z)
                cosr_cosp = 1 - 2 * (x * x + y * y)
                roll = np.arctan2(sinr_cosp, cosr_cosp)
                # Pitch (y-axis rotation)
                sinp = 2 * (w * y - z * x)
                pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
            except Exception:
                roll = 0.0
                pitch = 0.0
            self.imu_state['roll'] = (1-a) * self.imu_state['roll'] + a * float(roll)
            self.imu_state['pitch'] = (1-a) * self.imu_state['pitch'] + a * float(pitch)
            # Accel forward (x) and magnitude
            ax, ay, az = float(msg.linear_acceleration.x), float(msg.linear_acceleration.y), float(msg.linear_acceleration.z)
            accel_mag = float(np.sqrt(ax*ax + ay*ay + az*az))
            self.imu_state['accel_forward'] = (1-a) * self.imu_state['accel_forward'] + a * ax
            self.imu_state['accel_mag'] = (1-a) * self.imu_state['accel_mag'] + a * accel_mag
            self.imu_ready = True
        except Exception as e:
            self.get_logger().debug(f"IMU processing failed: {e}")

    def build_proprio_vector(self, wheel_diff, min_d_global, mean_d_global,
                              near_collision_flag, emergency_numeric,
                              left_free, right_free, center_free,
                              bev_gradient):
        """Assemble proprio vector; append IMU features if enabled"""
        vec = [
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
            bev_gradient[0],
            bev_gradient[1],
            bev_gradient[2]
        ]
        # Always append IMU features (zeros by default when not enabled/ready)
        imu_feats = [
            self.imu_state['yaw_rate'],
            self.imu_state['roll'],
            self.imu_state['pitch'],
            self.imu_state['accel_forward'],
            self.imu_state['accel_mag']
        ]
        vec.extend(imu_feats)
        return np.array(vec, dtype=np.float32)
        if self.trainer:
            self.trainer.safe_save()
        
        # Cleanup optimization components
        if self.optimization_monitor:
            try:
                final_report = self.optimization_monitor.cleanup()
                self.get_logger().info(f"Optimization monitoring final report: {final_report}")
            except Exception as e:
                self.get_logger().warn(f"Optimization monitor cleanup failed: {e}")
        
        if self.multi_metric_evaluator and self.step_count > 100:
            try:
                final_metrics = self.multi_metric_evaluator.get_current_metrics_summary()
                self.get_logger().info(f"Final multi-metric summary: {final_metrics}")
            except Exception as e:
                self.get_logger().warn(f"Multi-metric final summary failed: {e}")
        
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
                    f"Buffer: {training_stats['buffer_size']}/{training_stats.get('buffer_capacity', 50000)} | "
                    f"Best Fitness: {training_stats.get('best_fitness', 0):.2f} | "
                    f"Freq: {training_stats.get('evolution_frequency', 50)} | "
                    f"Sigma: {training_stats.get('sigma', 0.1):.4f}"
                )
            elif self.operation_mode == 'es_rl_hybrid':
                # PBT training stats
                status_msg.data = (
                    f"PBT ES-RL | Agent: {training_stats.get('active_agent', 0)}/{training_stats.get('population_size', 0)} | "
                    f"Battery: {self.current_battery_percentage:.1f}% | "
                    f"Steps: {self.step_count} | "
                    f"Buffer: {training_stats['buffer_size']}/{training_stats.get('buffer_capacity', 50000)} | "
                    f"Avg Pop Fitness: {training_stats.get('avg_population_fitness', 0):.2f} | "
                    f"Max Pop Fitness: {training_stats.get('max_population_fitness', 0):.2f}"
                )
            else:
                # RL training stats
                status_msg.data = (
                    f"NPU Learning | Mode: {self.exploration_mode} | "
                    f"Battery: {self.current_battery_percentage:.1f}% | "
                    f"Steps: {self.step_count} | "
                    f"Training: {training_stats['training_steps']} | "
                    f"Buffer: {training_stats['buffer_size']}/{training_stats.get('buffer_capacity', 50000)} | "
                    f"Avg Reward: {training_stats['avg_reward']:.2f}"
                )
        else:
            status_msg.data = f"NPU Exploration | Mode: {self.exploration_mode} | Battery: {self.current_battery_percentage:.1f}% | Steps: {self.step_count}"
            
        self.status_pub.publish(status_msg)

    def _current_reward_metric(self) -> float:
        try:
            if not self.trainer:
                return float('-inf')
            stats = self.trainer.get_training_stats()
            if self.operation_mode == 'es_rl_hybrid':
                return float(stats.get('max_population_fitness', 0.0))
            elif self.operation_mode in ['es_training', 'es_hybrid', 'es_inference', 'safe_es_training']:
                return float(stats.get('best_fitness', -999.0))
            else:
                return float(stats.get('avg_reward', 0.0))
        except Exception:
            return float('-inf')

    def _maybe_distill(self):
        if not self.enable_reward_based_distill or not self.trainer:
            return
        now = time.time()
        cooldown = float(self.get_parameter('distill_cooldown_sec').value)
        if now - self.last_distill_time < cooldown:
            return
        metric = self._current_reward_metric()
        if metric == float('-inf'):
            return
        min_reward = float(self.get_parameter('distill_min_reward').value)
        if metric < min_reward:
            return
        improved = False
        min_impr = float(self.get_parameter('distill_min_improvement').value)
        if self.last_distill_metric is None:
            improved = True  # allow first distill once cooldown passed
        else:
            if (metric - self.last_distill_metric) >= min_impr:
                improved = True
        plateau_ok = False
        plateau_sec = float(self.get_parameter('distill_plateau_sec').value)
        if (now - self.last_distill_time) >= plateau_sec:
            plateau_ok = True
        if not (improved or plateau_ok):
            return
        # Trigger distillation
        try:
            if hasattr(self.trainer, 'distill_best'):
                out = self.trainer.distill_best()
            elif hasattr(self.trainer, 'distill_to_student'):
                out = self.trainer.distill_to_student()
            else:
                out = ""
            if out:
                self.last_distill_time = now
                self.last_distill_metric = metric
                self.get_logger().info(f"Distilled student at metric={metric:.3f} -> {out}")
            else:
                self.get_logger().warn("Distillation requested but no output produced")
        except Exception as e:
            self.get_logger().warn(f"Distillation failed: {e}")
    
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
        """Check minimal sensor readiness for BEV-based inference.
        Requirements:
          - A processed point cloud received
          - Odometry updated at least once (position defaults change) OR step_count threshold
          - Wheel velocities tuple populated (len==2)
        """
        if self.latest_pointcloud is None:
            return False
        # Basic odom evidence: position not both zeros after some steps OR we have advanced step_count
        odom_ok = (self.step_count > 5) or not np.allclose(self.position, [0.0, 0.0])
        wheels_ok = isinstance(self.wheel_velocities, tuple) and len(self.wheel_velocities) == 2
        return odom_ok and wheels_ok

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
                            try:
                                x, y, z = struct.unpack('fff', pc_msg.data[i:i+12])
                                # Only add valid points (not NaN or Inf)
                                if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                                    points.append([x, y, z])
                            except:
                                continue
            else:
                # Unorganized point cloud
                for i in range(0, len(pc_msg.data), point_step):
                    if i + 12 <= len(pc_msg.data):
                        try:
                            x, y, z = struct.unpack('fff', pc_msg.data[i:i+12])
                            # Only add valid points (not NaN or Inf)
                            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                                points.append([x, y, z])
                        except:
                            continue
                    
            return np.array(points) if points else np.zeros((0, 3))
        except Exception as e:
            self.get_logger().warn(f"Point cloud conversion failed: {e}")
            return np.zeros((0, 3))

def main(args=None):
    rclpy.init(args=args)
    node = NPUExplorationBEVNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("NPU BEV exploration interrupted")
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
