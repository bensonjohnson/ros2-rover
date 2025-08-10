#!/usr/bin/env python3
"""
NPU Depth Image Exploration Launch File
Clean architecture: Motor control + RealSense + NPU AI only
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.substitutions import LaunchConfiguration
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
        description="Operation mode: cpu_training | hybrid | inference",
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
            "depth_module.depth_profile": "424x240x15",  # Good balance of resolution and performance
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
                "npu_inference_rate": 5.0,
                "operation_mode": LaunchConfiguration("operation_mode"),
            }
        ],
        remappings=[
            ("cmd_vel", "cmd_vel_raw"),
            ("depth_image", "/camera/camera/depth/image_rect_raw"),
            ("odom", "/odom"),
        ]
    )

    # 5. Simple Safety Monitor (Emergency stop only)
    safety_monitor_node = Node(
        package="tractor_bringup",
        executable="simple_safety_monitor_depth.py",
        name="simple_safety_monitor",
        output="screen",
        parameters=[
            {
                "emergency_stop_distance": 0.1,  # 10cm emergency stop
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

    # Core system - immediate start
    ld.add_action(robot_description_launch)
    ld.add_action(hiwonder_motor_node)

    # Camera - start with delay for stability
    ld.add_action(TimerAction(period=3.0, actions=[realsense_launch]))

    # NPU system - start after camera is ready
    ld.add_action(TimerAction(period=8.0, actions=[npu_exploration_node]))
    ld.add_action(TimerAction(period=10.0, actions=[safety_monitor_node]))

    return ld
