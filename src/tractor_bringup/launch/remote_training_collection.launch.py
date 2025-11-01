#!/usr/bin/env python3
"""Launch file for remote training data collection.

This launch file starts the rover in data collection mode,
streaming RGB-D + proprioceptive data to the V620 training server.
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
    declare_server_addr_cmd = DeclareLaunchArgument(
        "server_address",
        default_value="tcp://192.168.1.100:5555",
        description="V620 training server address (ZMQ)",
    )

    declare_collection_rate_cmd = DeclareLaunchArgument(
        "collection_rate_hz",
        default_value="10.0",
        description="Data collection rate (Hz)",
    )

    declare_max_speed_cmd = DeclareLaunchArgument(
        "max_speed",
        default_value="0.18",
        description="Maximum speed during collection",
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

    # Safety monitor (computes min forward distance)
    safety_monitor_node = Node(
        package="tractor_bringup",
        executable="simple_safety_monitor.py",
        name="safety_monitor",
        output="screen",
        parameters=[{
            "max_speed_limit": LaunchConfiguration("max_speed"),
            "emergency_stop_distance": 0.15,
            "warning_distance": 0.3,
        }],
        remappings=[
            ("cmd_vel_in", "cmd_vel_teleop"),
            ("cmd_vel_out", "cmd_vel_raw"),
        ],
    )

    # Remote training data collector
    remote_collector_node = Node(
        package="tractor_bringup",
        executable="remote_training_collector.py",
        name="remote_training_collector",
        output="screen",
        parameters=[{
            "server_address": LaunchConfiguration("server_address"),
            "collection_rate_hz": LaunchConfiguration("collection_rate_hz"),
            "rgb_topic": "/camera/camera/color/image_raw",
            "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
            "imu_topic": "/lsm9ds1_imu_publisher/imu/data",
            "odom_topic": "/odom",
            "cmd_vel_topic": "cmd_vel_ai",
            "min_distance_topic": "/min_forward_distance",
            "enable_compression": True,
            "jpeg_quality": 85,
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
    ld.add_action(declare_server_addr_cmd)
    ld.add_action(declare_collection_rate_cmd)
    ld.add_action(declare_max_speed_cmd)

    # Core system
    ld.add_action(robot_description_launch)
    ld.add_action(hiwonder_motor_node)

    # Sensors
    ld.add_action(TimerAction(period=2.0, actions=[lsm9ds1_launch]))
    ld.add_action(TimerAction(period=5.0, actions=[realsense_launch]))

    # Control and safety
    ld.add_action(TimerAction(period=8.0, actions=[vfc_node]))
    ld.add_action(TimerAction(period=9.0, actions=[safety_monitor_node]))

    # Data collector
    ld.add_action(TimerAction(period=10.0, actions=[remote_collector_node]))

    return ld
