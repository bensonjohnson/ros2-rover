#!/usr/bin/env python3
"""Launch file for remote-trained model inference.

This launch file runs the rover with a model trained on the V620 server,
using the RK3588 NPU for fast inference.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_tractor_bringup = get_package_share_directory("tractor_bringup")

    # Launch arguments
    declare_model_path_cmd = DeclareLaunchArgument(
        "model_path",
        default_value="/home/benson/Documents/ros2-rover/models/remote_trained.rknn",
        description="Path to RKNN model file",
    )

    declare_max_speed_cmd = DeclareLaunchArgument(
        "max_speed",
        default_value="0.18",
        description="Maximum linear speed",
    )

    declare_safety_distance_cmd = DeclareLaunchArgument(
        "safety_distance",
        default_value="0.25",
        description="Emergency stop distance (m)",
    )

    declare_inference_rate_cmd = DeclareLaunchArgument(
        "inference_rate_hz",
        default_value="10.0",
        description="NPU inference rate (Hz)",
    )

    declare_use_npu_cmd = DeclareLaunchArgument(
        "use_npu",
        default_value="true",
        description="Use NPU acceleration (true/false)",
    )

    # Robot description
    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tractor_bringup, "launch", "robot_description.launch.py")
        )
    )

    # Motor driver
    hiwonder_motor_node = Node(
        package="tractor_control",
        executable="hiwonder_motor_driver",
        name="hiwonder_motor_driver",
        output="screen",
        parameters=[os.path.join(pkg_tractor_bringup, "config", "hiwonder_motor_params.yaml")]
    )

    # RealSense camera
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("realsense2_camera"), "launch", "rs_launch.py")
        ),
        launch_arguments={
            "pointcloud.enable": "false",
            "align_depth.enable": "true",
            "enable_color": "true",
            "enable_depth": "true",
            "enable_sync": "true",
            "device_type": "435i",
            "depth_module.depth_profile": "424x240x30",
            "rgb_camera.color_profile": "424x240x30",
            "enable_gyro": "false",
            "enable_accel": "false",
            "enable_imu": "false",
        }.items()
    )

    # IMU
    lsm9ds1_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("tractor_sensors"), "launch", "lsm9ds1_imu.launch.py")
        )
    )

    # Safety monitor
    safety_monitor_node = Node(
        package="tractor_bringup",
        executable="simple_safety_monitor.py",
        name="safety_monitor",
        output="screen",
        parameters=[{
            "max_speed_limit": LaunchConfiguration("max_speed"),
            "emergency_stop_distance": LaunchConfiguration("safety_distance"),
            "warning_distance": 0.5,
        }],
        remappings=[
            ("cmd_vel_in", "cmd_vel_ai"),
            ("cmd_vel_out", "cmd_vel_raw"),
        ],
    )

    # Remote trained inference node
    inference_node = Node(
        package="tractor_bringup",
        executable="remote_trained_inference.py",
        name="remote_trained_inference",
        output="screen",
        parameters=[{
            "model_path": LaunchConfiguration("model_path"),
            "rgb_topic": "/camera/camera/color/image_raw",
            "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
            "imu_topic": "/lsm9ds1_imu_publisher/imu/data",
            "odom_topic": "/odom",
            "min_distance_topic": "/min_forward_distance",
            "cmd_vel_topic": "cmd_vel_ai",
            "max_linear_speed": LaunchConfiguration("max_speed"),
            "max_angular_speed": 1.0,
            "inference_rate_hz": LaunchConfiguration("inference_rate_hz"),
            "use_npu": LaunchConfiguration("use_npu"),
            "npu_core_mask": 0,  # Auto-select NPU core
        }]
    )

    # Velocity feedback controller
    vfc_node = Node(
        package="tractor_control",
        executable="velocity_feedback_controller",
        name="velocity_feedback_controller",
        output="screen",
        parameters=[{"control_frequency": 50.0}]
    )

    # Build launch description
    ld = LaunchDescription()

    # Arguments
    ld.add_action(declare_model_path_cmd)
    ld.add_action(declare_max_speed_cmd)
    ld.add_action(declare_safety_distance_cmd)
    ld.add_action(declare_inference_rate_cmd)
    ld.add_action(declare_use_npu_cmd)

    # Core system
    ld.add_action(robot_description_launch)
    ld.add_action(hiwonder_motor_node)

    # Sensors
    ld.add_action(TimerAction(period=2.0, actions=[lsm9ds1_launch]))
    ld.add_action(TimerAction(period=5.0, actions=[realsense_launch]))

    # Control and inference
    ld.add_action(TimerAction(period=8.0, actions=[vfc_node]))
    ld.add_action(TimerAction(period=9.0, actions=[safety_monitor_node]))
    ld.add_action(TimerAction(period=10.0, actions=[inference_node]))

    return ld
