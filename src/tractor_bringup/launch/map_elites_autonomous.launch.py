#!/usr/bin/env python3
"""Launch file for MAP-Elites autonomous episode running.

Runs rover autonomously, evaluating models from V620 MAP-Elites server.
No teleoperation required!
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
        "server_addr",
        default_value="tcp://10.0.0.200:5556",
        description="V620 MAP-Elites server address",
    )

    declare_episode_duration_cmd = DeclareLaunchArgument(
        "episode_duration",
        default_value="60.0",
        description="Episode duration in seconds",
    )

    declare_max_speed_cmd = DeclareLaunchArgument(
        "max_speed",
        default_value="0.18",
        description="Maximum linear speed",
    )

    declare_collision_distance_cmd = DeclareLaunchArgument(
        "collision_distance",
        default_value="0.12",
        description="Collision detection distance (m)",
    )

    declare_inference_rate_cmd = DeclareLaunchArgument(
        "inference_rate_hz",
        default_value="10.0",
        description="Inference rate (Hz)",
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
            "depth_module.depth_profile": "640x480x30",
            "rgb_camera.color_profile": "640x480x30",
            "enable_gyro": "false",
            "enable_accel": "false",
            "enable_imu": "false",
        }.items()
    )

    # Safety monitor (depth-based, no point cloud required)
    safety_monitor_node = Node(
        package="tractor_bringup",
        executable="simple_depth_safety_monitor.py",
        name="simple_depth_safety_monitor",
        output="screen",
        parameters=[{
            "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
            "input_cmd_topic": "cmd_vel_ai",
            "output_cmd_topic": "cmd_vel_raw",
            "emergency_stop_distance": 0.25,
            "hard_stop_distance": LaunchConfiguration("collision_distance"),
            "depth_scale": 0.001,
            "forward_roi_width_ratio": 0.6,
            "forward_roi_height_ratio": 0.5,
            "max_eval_distance": 5.0,
        }],
    )

    # MAP-Elites episode runner
    episode_runner_node = Node(
        package="tractor_bringup",
        executable="map_elites_episode_runner.py",
        name="map_elites_episode_runner",
        output="screen",
        parameters=[{
            "server_addr": LaunchConfiguration("server_addr"),
            "episode_duration": LaunchConfiguration("episode_duration"),
            "max_linear_speed": LaunchConfiguration("max_speed"),
            "max_angular_speed": 1.0,
            "collision_distance": LaunchConfiguration("collision_distance"),
            "inference_rate_hz": LaunchConfiguration("inference_rate_hz"),
            "use_npu": LaunchConfiguration("use_npu"),
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
    ld.add_action(declare_episode_duration_cmd)
    ld.add_action(declare_max_speed_cmd)
    ld.add_action(declare_collision_distance_cmd)
    ld.add_action(declare_inference_rate_cmd)
    ld.add_action(declare_use_npu_cmd)

    # Core system
    ld.add_action(robot_description_launch)
    ld.add_action(hiwonder_motor_node)

    # Sensors
    ld.add_action(TimerAction(period=5.0, actions=[realsense_launch]))

    # Control and episode runner
    ld.add_action(TimerAction(period=8.0, actions=[vfc_node]))
    ld.add_action(TimerAction(period=9.0, actions=[safety_monitor_node]))
    ld.add_action(TimerAction(period=10.0, actions=[episode_runner_node]))

    return ld
