#!/usr/bin/env python3
"""
NPU Depth Image Exploration Launch File
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
        description="Operation mode: cpu_training | hybrid | inference | safe_training | es_training | es_hybrid | es_inference | safe_es_training",
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

    # 1. Robot Description (TF only)
    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tractor_bringup, "launch", "robot_description.launch.py")
        ),
    )

    # 2. Hiwonder Motor Control (REUSE EXISTING - has encoders and odometry)
    hiwonder_motor_node = Node(
        package="tractor_control",
        executable="hiwonder_motor_driver",
        name="hiwonder_motor_driver",
        output="screen",
        parameters=[
            os.path.join(pkg_tractor_bringup, "config", "hiwonder_motor_params.yaml")
        ],
        remappings=[
            ("cmd_vel", "cmd_vel_safe")  # Will receive commands from NPU node
        ]
    )

    # 3. RealSense Camera (optimized for depth images)
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("realsense2_camera"), "launch", "rs_launch.py")
        ),
        launch_arguments={
            "pointcloud.enable": "false",  # Disable pointcloud for bandwidth
            "align_depth.enable": "true",
            "enable_color": "false",  # Disable color for bandwidth
            "enable_depth": "true",
            "enable_sync": "true",
            "device_type": "435i",
            "depth_module.depth_profile": "424x240x30",  # Higher FPS for better data collection
            "enable_imu": "false",
            "enable_gyro": "false",
            "enable_accel": "false",
        }.items(),
    )

    # 4. NPU Exploration Node (Main AI controller)
    npu_exploration_node = Node(
        package="tractor_bringup",
        executable="npu_exploration_depth.py",
        name="npu_exploration",
        output="screen",
        parameters=[
            {
                "max_speed": LaunchConfiguration("max_speed"),
                "min_battery_percentage": LaunchConfiguration("min_battery_percentage"),
                "safety_distance": LaunchConfiguration("safety_distance"),
                "npu_inference_rate": 30.0,
                "operation_mode": LaunchConfiguration("operation_mode"),
                "anti_overtraining": LaunchConfiguration("anti_overtraining"),
                "exploration_time": LaunchConfiguration("exploration_time"),
                "enable_bayesian_optimization": LaunchConfiguration("enable_bayesian_optimization"),
                "optimization_level": LaunchConfiguration("optimization_level"),
                "enable_training_optimization": LaunchConfiguration("enable_training_optimization"),
                "enable_reward_optimization": LaunchConfiguration("enable_reward_optimization"),
                "enable_multi_metric_evaluation": LaunchConfiguration("enable_multi_metric_evaluation"),
                "enable_optimization_monitoring": LaunchConfiguration("enable_optimization_monitoring"),
                # Phase 2 parameters
                "enable_multi_objective_optimization": LaunchConfiguration("enable_multi_objective_optimization"),
                "enable_safety_constraints": LaunchConfiguration("enable_safety_constraints"),
                "enable_architecture_optimization": LaunchConfiguration("enable_architecture_optimization"),
                "enable_progressive_architecture": LaunchConfiguration("enable_progressive_architecture"),
                "enable_sensor_fusion_optimization": LaunchConfiguration("enable_sensor_fusion_optimization"),
            }
        ],
        remappings=[
            ("cmd_vel", "cmd_vel_raw"),
            ("depth_image", "/camera/camera/depth/image_rect_raw"),
            ("odom", "/odom"),
        ]
    )

    # 4.1. Training Monitor Node (only when anti-overtraining is enabled)
    training_monitor_node = Node(
        package="tractor_bringup",
        executable="training_monitor_node.py",
        name="training_monitor",
        output="screen",
        parameters=[
            os.path.join(pkg_tractor_bringup, "config", "anti_overtraining_params.yaml"),
            {
                "monitor_frequency": 1.0,  # Check every second
                "diversity_window": 20,
                "alert_threshold": 0.7,
                "enable_plotting": False,  # Disable plotting on robot
            }
        ],
        condition=IfCondition(LaunchConfiguration("anti_overtraining"))
    )

    # 5. Simple Safety Monitor (Emergency stop only)
    safety_monitor_node = Node(
        package="tractor_bringup",
        executable="simple_safety_monitor_depth.py",
        name="simple_safety_monitor",
        output="screen",
        parameters=[
            {
                "emergency_stop_distance": LaunchConfiguration("safety_distance"),  # Use the passed safety distance
                "max_speed_limit": LaunchConfiguration("max_speed"),
            }
        ],
        remappings=[
            ("cmd_vel_in", "cmd_vel_raw"),
            ("cmd_vel_out", "cmd_vel_safe"),
            ("depth_image", "/camera/camera/depth/image_rect_raw"),  # Depth image topic
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
    
    # Add Phase 2 launch arguments
    ld.add_action(declare_enable_multi_objective_optimization_cmd)
    ld.add_action(declare_enable_safety_constraints_cmd)
    ld.add_action(declare_enable_architecture_optimization_cmd)
    ld.add_action(declare_enable_progressive_architecture_cmd)
    ld.add_action(declare_enable_sensor_fusion_optimization_cmd)

    # Core system - immediate start
    ld.add_action(robot_description_launch)
    ld.add_action(hiwonder_motor_node)

    # Camera - start with delay for stability
    ld.add_action(TimerAction(period=3.0, actions=[realsense_launch]))

    # NPU system - start after camera is ready
    ld.add_action(TimerAction(period=8.0, actions=[npu_exploration_node]))
    ld.add_action(TimerAction(period=10.0, actions=[safety_monitor_node]))
    
    # Training monitor - start after NPU system (only if anti-overtraining enabled)
    ld.add_action(TimerAction(period=12.0, actions=[training_monitor_node]))

    return ld
