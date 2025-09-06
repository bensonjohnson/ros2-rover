#!/usr/bin/env python3
"""
NPU Bird's Eye View Exploration Launch File
Clean architecture: Motor control + RealSense + NPU AI only
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Package directories
    pkg_tractor_bringup = get_package_share_directory("tractor_bringup")

    # Launch arguments
    max_speed = LaunchConfiguration("max_speed")
    min_battery_percentage = LaunchConfiguration("min_battery_percentage")
    safety_distance = LaunchConfiguration("safety_distance")
    operation_mode = LaunchConfiguration("operation_mode")
    anti_overtraining = LaunchConfiguration("anti_overtraining")
    exploration_time = LaunchConfiguration("exploration_time")
    enable_bayesian_optimization = LaunchConfiguration("enable_bayesian_optimization")

    declare_max_speed_cmd = DeclareLaunchArgument(
        "max_speed",
        default_value="0.15",
        description="Maximum exploration speed (m/s)",
    )

    declare_min_battery_cmd = DeclareLaunchArgument(
        "min_battery_percentage",
        default_value="30.0",
        description="Minimum battery percentage before shutdown",
    )

    declare_safety_distance_cmd = DeclareLaunchArgument(
        "safety_distance",
        default_value="0.2",
        description="Safety distance for obstacle avoidance (meters)",
    )

    declare_operation_mode_cmd = DeclareLaunchArgument(
        "operation_mode",
        default_value="cpu_training",
        description="Operation mode: cpu_training | hybrid | inference | safe_training | es_training | es_hybrid | es_inference | safe_es_training | es_rl_hybrid",
    )

    declare_anti_overtraining_cmd = DeclareLaunchArgument(
        "anti_overtraining",
        default_value="false",
        description="Enable anti-overtraining measures (true/false)",
    )

    declare_exploration_time_cmd = DeclareLaunchArgument(
        "exploration_time",
        default_value="300",
        description="Maximum exploration time in seconds",
    )

    declare_bayesian_optimization_cmd = DeclareLaunchArgument(
        "enable_bayesian_optimization",
        default_value="true",
        description="Enable Bayesian optimization for ES hyperparameters (true/false)",
    )

    declare_optimization_level_cmd = DeclareLaunchArgument(
        "optimization_level",
        default_value="standard",
        description="Optimization level: basic | standard | full | research",
    )

    declare_enable_training_optimization_cmd = DeclareLaunchArgument(
        "enable_training_optimization", 
        default_value="true",
        description="Enable Bayesian optimization for training hyperparameters (true/false)",
    )

    declare_enable_reward_optimization_cmd = DeclareLaunchArgument(
        "enable_reward_optimization",
        default_value="false", 
        description="Enable Bayesian optimization for reward parameters (true/false)",
    )

    declare_enable_multi_metric_cmd = DeclareLaunchArgument(
        "enable_multi_metric_evaluation",
        default_value="true",
        description="Enable multi-objective fitness evaluation (true/false)",
    )

    declare_enable_optimization_monitoring_cmd = DeclareLaunchArgument(
        "enable_optimization_monitoring",
        default_value="true",
        description="Enable comprehensive optimization monitoring and logging (true/false)",
    )

    # PBT (ES-RL hybrid) launch arguments
    declare_pbt_population_size_cmd = DeclareLaunchArgument(
        "pbt_population_size",
        default_value="4",
        description="Population size for PBT ES-RL hybrid",
    )
    declare_pbt_update_interval_cmd = DeclareLaunchArgument(
        "pbt_update_interval",
        default_value="1000",
        description="Update interval (steps) for PBT exploit/explore and agent switching",
    )
    declare_pbt_perturb_prob_cmd = DeclareLaunchArgument(
        "pbt_perturb_prob",
        default_value="0.25",
        description="Probability to perturb hyperparameters/weights during PBT explore",
    )
    declare_pbt_resample_prob_cmd = DeclareLaunchArgument(
        "pbt_resample_prob",
        default_value="0.25",
        description="Probability to resample (reinitialize) an agent during PBT explore",
    )

    # Phase 2 launch arguments - Multi-objective and Architecture Optimization
    declare_enable_multi_objective_optimization_cmd = DeclareLaunchArgument(
        "enable_multi_objective_optimization",
        default_value="false",
        description="Enable multi-objective Bayesian optimization with Pareto frontier (true/false)",
    )

    declare_enable_safety_constraints_cmd = DeclareLaunchArgument(
        "enable_safety_constraints",
        default_value="true", 
        description="Enable safety constraint handling in optimization (true/false)",
    )

    declare_enable_architecture_optimization_cmd = DeclareLaunchArgument(
        "enable_architecture_optimization",
        default_value="false",
        description="Enable Bayesian neural architecture search (research level only) (true/false)",
    )

    declare_enable_progressive_architecture_cmd = DeclareLaunchArgument(
        "enable_progressive_architecture",
        default_value="false",
        description="Enable progressive architecture refinement (true/false)",
    )

    declare_enable_sensor_fusion_optimization_cmd = DeclareLaunchArgument(
        "enable_sensor_fusion_optimization",
        default_value="false",
        description="Enable sensor fusion parameter optimization (full/research level only) (true/false)",
    )

    # BEV-specific launch arguments
    declare_bev_size_cmd = DeclareLaunchArgument(
        "bev_size",
        default_value="[200, 200]",
        description="BEV image size [height, width] in pixels",
    )

    declare_bev_range_cmd = DeclareLaunchArgument(
        "bev_range",
        default_value="[10.0, 10.0]",
        description="BEV range [x_range, y_range] in meters",
    )

    declare_bev_height_channels_cmd = DeclareLaunchArgument(
        "bev_height_channels",
        default_value="[0.2, 1.0]",
        description="Height thresholds for BEV channels",
    )

    declare_enable_ground_removal_cmd = DeclareLaunchArgument(
        "enable_ground_removal",
        default_value="true",
        description="Enable ground plane removal (true/false)",
    )

    # BEV performance tuning
    declare_bev_ground_update_interval_cmd = DeclareLaunchArgument(
        "bev_ground_update_interval",
        default_value="10",
        description="Update cached ground plane every N frames (int)",
    )
    declare_bev_enable_opencl_cmd = DeclareLaunchArgument(
        "bev_enable_opencl",
        default_value="true",
        description="Enable OpenCL offload for BEV histogramming (true/false)",
    )

    # 1. Robot Description (TF only)
    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tractor_bringup, "launch", "robot_description.launch.py")
        ),
    )

    # 2. Hiwonder Motor Control (encoders and odometry)
    hiwonder_motor_node = Node(
        package="tractor_control",
        executable="hiwonder_motor_driver",
        name="hiwonder_motor_driver",
        output="screen",
        parameters=[
            os.path.join(pkg_tractor_bringup, "config", "hiwonder_motor_params.yaml")
        ]
    )

    # 2.1 LSM9DS1 IMU (external IMU)
    lsm9ds1_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("tractor_sensors"), "launch", "lsm9ds1_imu.launch.py")
        )
    )

    # 3. RealSense Camera (optimized for point cloud)
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("realsense2_camera"), "launch", "rs_launch.py")
        ),
        launch_arguments={
            "enable_pointcloud": "true",  # Enable pointcloud for BEV
            "align_depth": "true",  # Align depth for better accuracy
            "enable_color": "false",  # Disable color to reduce processing overhead
            "enable_depth": "true",
            "enable_sync": "false",  # Disable sync for faster processing
            "device_type": "435i",
            "depth_module.depth_profile": "424x240x30",  # 30 FPS to reduce CPU load
            "rgb_camera.color_profile": "424x240x30",  # Match depth profile
            "enable_imu": "false",
            "enable_gyro": "false",
            "enable_accel": "false",
            # Post-processing filters (decimation reduces point cloud density significantly)
            "decimation_filter.enable": "true",
            "decimation_filter.filter_magnitude": "2",
            "pointcloud.stream_filter": "2",  # Filter to depth only 
            "config_file": os.path.join(get_package_share_directory("tractor_bringup"), "config", "realsense_config.yaml"),
        }.items(),
    )

    # 4. NPU Exploration Node (Main AI controller)
    npu_exploration_node = Node(
        package="tractor_bringup",
        executable="npu_exploration_bev.py",
        name="npu_exploration",
        output="screen",
        parameters=[
            {
                "max_speed": LaunchConfiguration("max_speed"),
                "min_battery_percentage": LaunchConfiguration("min_battery_percentage"),
                "safety_distance": LaunchConfiguration("safety_distance"),
                "npu_inference_rate": 10.0,
                "operation_mode": LaunchConfiguration("operation_mode"),
                "anti_overtraining": LaunchConfiguration("anti_overtraining"),
                "exploration_time": LaunchConfiguration("exploration_time"),
                "enable_bayesian_optimization": LaunchConfiguration("enable_bayesian_optimization"),
                "optimization_level": LaunchConfiguration("optimization_level"),
                "enable_training_optimization": LaunchConfiguration("enable_training_optimization"),
                "enable_reward_optimization": LaunchConfiguration("enable_reward_optimization"),
                "enable_multi_metric_evaluation": LaunchConfiguration("enable_multi_metric_evaluation"),
                "enable_optimization_monitoring": LaunchConfiguration("enable_optimization_monitoring"),
                # PBT parameters (used in es_rl_hybrid mode)
                "pbt_population_size": LaunchConfiguration("pbt_population_size"),
                "pbt_update_interval": LaunchConfiguration("pbt_update_interval"),
                "pbt_perturb_prob": LaunchConfiguration("pbt_perturb_prob"),
                "pbt_resample_prob": LaunchConfiguration("pbt_resample_prob"),
                # Phase 2 parameters
                "enable_multi_objective_optimization": LaunchConfiguration("enable_multi_objective_optimization"),
                "enable_safety_constraints": LaunchConfiguration("enable_safety_constraints"),
                "enable_architecture_optimization": LaunchConfiguration("enable_architecture_optimization"),
                "enable_progressive_architecture": LaunchConfiguration("enable_progressive_architecture"),
                "enable_sensor_fusion_optimization": LaunchConfiguration("enable_sensor_fusion_optimization"),
                # BEV parameters
                "bev_size": LaunchConfiguration("bev_size"),
                "bev_range": LaunchConfiguration("bev_range"),
                "bev_height_channels": LaunchConfiguration("bev_height_channels"),
                "enable_ground_removal": LaunchConfiguration("enable_ground_removal"),
                "bev_ground_update_interval": LaunchConfiguration("bev_ground_update_interval"),
                "bev_enable_opencl": LaunchConfiguration("bev_enable_opencl"),
            }
        ],
        remappings=[
            ("point_cloud", "/camera/camera/depth/color/points"),  # Point cloud topic
            ("odom", "/odom"),
        ]
    )

    # 5. Velocity feedback controller (reads cmd_vel_raw, publishes cmd_vel)
    vfc_node = Node(
        package="tractor_control",
        executable="velocity_feedback_controller",
        name="velocity_feedback_controller",
        output="screen",
        parameters=[
            {"control_frequency": 50.0}
        ]
    )

    # 6. Simple safety monitor (gates forward motion) - direct point cloud processing
    safety_monitor_node = Node(
        package="tractor_bringup",
        executable="simple_safety_monitor.py",
        name="simple_safety_monitor",
        output="screen",
        parameters=[
            {
                "emergency_stop_distance": LaunchConfiguration("safety_distance"),
                "pointcloud_topic": "/camera/camera/depth/color/points",
                "input_cmd_topic": "cmd_vel_ai",
                "output_cmd_topic": "cmd_vel_raw"
            }
        ]
    )

    # Build launch description
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_max_speed_cmd)
    ld.add_action(declare_min_battery_cmd)
    ld.add_action(declare_safety_distance_cmd)
    ld.add_action(declare_operation_mode_cmd)
    ld.add_action(declare_anti_overtraining_cmd)
    ld.add_action(declare_exploration_time_cmd)
    ld.add_action(declare_bayesian_optimization_cmd)
    ld.add_action(declare_optimization_level_cmd)
    ld.add_action(declare_enable_training_optimization_cmd)
    ld.add_action(declare_enable_reward_optimization_cmd)
    ld.add_action(declare_enable_multi_metric_cmd)
    ld.add_action(declare_enable_optimization_monitoring_cmd)
    # Add PBT arguments
    ld.add_action(declare_pbt_population_size_cmd)
    ld.add_action(declare_pbt_update_interval_cmd)
    ld.add_action(declare_pbt_perturb_prob_cmd)
    ld.add_action(declare_pbt_resample_prob_cmd)
    
    # Add Phase 2 launch arguments
    ld.add_action(declare_enable_multi_objective_optimization_cmd)
    ld.add_action(declare_enable_safety_constraints_cmd)
    ld.add_action(declare_enable_architecture_optimization_cmd)
    ld.add_action(declare_enable_progressive_architecture_cmd)
    ld.add_action(declare_enable_sensor_fusion_optimization_cmd)
    
    # Add BEV-specific launch arguments
    ld.add_action(declare_bev_size_cmd)
    ld.add_action(declare_bev_range_cmd)
    ld.add_action(declare_bev_height_channels_cmd)
    ld.add_action(declare_enable_ground_removal_cmd)
    ld.add_action(declare_bev_ground_update_interval_cmd)
    ld.add_action(declare_bev_enable_opencl_cmd)

    # Core system - immediate start
    ld.add_action(robot_description_launch)
    ld.add_action(hiwonder_motor_node)

    # Camera - start with delay for stability
    ld.add_action(TimerAction(period=3.0, actions=[realsense_launch]))
    # IMU - start shortly after
    ld.add_action(TimerAction(period=4.0, actions=[lsm9ds1_launch]))

    # Safety monitor - start early to ensure it's ready before AI
    ld.add_action(TimerAction(period=6.0, actions=[safety_monitor_node]))
    # Velocity controller - start after safety monitor
    ld.add_action(TimerAction(period=7.0, actions=[vfc_node]))
    # NPU system - start last after safety chain is established
    ld.add_action(TimerAction(period=8.0, actions=[npu_exploration_node]))

    return ld
