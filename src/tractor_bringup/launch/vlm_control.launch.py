#!/usr/bin/env python3
"""
VLM Control launch file for autonomous rover operation using vision language model.

This launch file sets up:
1. Robot description and basic sensors
2. RealSense camera for visual input
3. Motor driver for movement
4. VLM controller for AI-based navigation
5. Safety monitoring
6. Optional teleop backup
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_tractor_bringup = get_package_share_directory("tractor_bringup")

    # Launch arguments
    use_sim_time = LaunchConfiguration("use_sim_time")
    rkllama_url = LaunchConfiguration("rkllama_url")
    model_name = LaunchConfiguration("model_name")
    max_linear_speed = LaunchConfiguration("max_linear_speed")
    max_angular_speed = LaunchConfiguration("max_angular_speed")
    with_teleop = LaunchConfiguration("with_teleop")
    with_motor = LaunchConfiguration("with_motor")
    with_safety = LaunchConfiguration("with_safety")
    with_vlm = LaunchConfiguration("with_vlm")
    inference_interval = LaunchConfiguration("inference_interval")
    num_ctx = LaunchConfiguration("num_ctx")

    # Declare launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        "use_sim_time", default_value="false", description="Use sim time"
    )
    declare_rkllama_url_cmd = DeclareLaunchArgument(
        "rkllama_url",
        default_value="https://ollama.gokickrocks.org",
        description="URL of the Ollama server"
    )
    declare_model_name_cmd = DeclareLaunchArgument(
        "model_name",
        default_value="mistral-small3.2:24b",
        description="Name of the VLM model to use in Ollama"
    )
    declare_max_linear_speed_cmd = DeclareLaunchArgument(
        "max_linear_speed", default_value="0.15", description="Maximum linear speed for VLM control"
    )
    declare_max_angular_speed_cmd = DeclareLaunchArgument(
        "max_angular_speed", default_value="0.3", description="Maximum angular speed for VLM control"
    )
    declare_with_teleop_cmd = DeclareLaunchArgument(
        "with_teleop", default_value="false", description="Start Xbox teleop as backup"
    )
    declare_with_motor_cmd = DeclareLaunchArgument(
        "with_motor", default_value="true", description="Start motor driver"
    )
    declare_with_safety_cmd = DeclareLaunchArgument(
        "with_safety", default_value="true", description="Start safety monitor"
    )
    declare_with_vlm_cmd = DeclareLaunchArgument(
        "with_vlm", default_value="true", description="Start VLM controller"
    )
    declare_inference_interval_cmd = DeclareLaunchArgument(
        "inference_interval", default_value="5.0", description="Seconds between VLM inferences"
    )
    declare_num_ctx_cmd = DeclareLaunchArgument(
        "num_ctx", default_value="16384", description="Context window size for Ollama (16K for 2-frame mode)"
    )

    # 1) Robot description
    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tractor_bringup, "launch", "robot_description.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    # 2) Motor driver
    hiwonder_motor_node = Node(
        package="tractor_control",
        executable="hiwonder_motor_driver",
        name="hiwonder_motor_driver",
        output="screen",
        parameters=[
            os.path.join(pkg_tractor_bringup, "config", "hiwonder_motor_params.yaml"),
            {"use_sim_time": use_sim_time},
        ],
        remappings=[("cmd_vel", "cmd_vel_mux")],
        condition=IfCondition(with_motor),
    )

    # 3) Essential sensors for VLM operation
    lsm9ds1_imu_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("tractor_sensors"), "launch", "lsm9ds1_imu.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    # 4) RealSense camera with optimized settings for VLM
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("realsense2_camera"), "launch", "rs_launch.py")
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "camera_name": "camera",
            "camera_namespace": "camera",
            # Enable color stream for VLM
            "enable_color": "true",
            "enable_depth": "true",
            "enable_pointcloud": "true",
            "align_depth": "true",
            "device_type": "435i",
            # Optimized resolution for VLM processing
            # 640x480 = 307K pixels (balanced), 848x480 = 407K pixels (wider FOV)
            "rgb_camera.color_profile": "640x480x15",  # Lower FPS for VLM
            "depth_module.depth_profile": "640x480x15",
            # Disable camera IMU (using LSM9DS1)
            "enable_imu": "false",
            "enable_gyro": "false",
            "enable_accel": "false",
            # Basic filters
            "decimation_filter.enable": "true",
            "spatial_filter.enable": "true",
            "temporal_filter.enable": "true",
            # Connection settings
            "wait_for_device_timeout": "10.0",
            "reconnect_timeout": "5.0",
        }.items(),
    )

    # 5) Pointcloud to laserscan for safety monitoring
    pointcloud_to_laserscan_node = Node(
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        name="pointcloud_to_laserscan",
        output="screen",
        parameters=[
            os.path.join(pkg_tractor_bringup, "config", "pointcloud_to_laserscan.yaml"),
            {"use_sim_time": use_sim_time},
        ],
        remappings=[
            ("cloud_in", "/camera/camera/depth/color/points"),
            ("scan", "/scan"),
        ],
    )

    # 6) VLM Controller - the main AI navigation node
    vlm_controller_node = Node(
        package="tractor_bringup",
        executable="vlm_rover_controller.py",
        name="vlm_rover_controller",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "rkllama_url": rkllama_url,
                "model_name": model_name,
                "camera_topic": "/camera/camera/color/image_raw",
                "max_linear_speed": max_linear_speed,
                "max_angular_speed": max_angular_speed,
                "inference_interval": inference_interval,
                "command_timeout": 6.0,
                "request_timeout": 30.0,
                "simulation_mode": False,  # Will auto-detect if rkllama server is available
                "num_ctx": num_ctx,
            }
        ],
        remappings=[("cmd_vel_vlm", "cmd_vel_vlm")],
        condition=IfCondition(with_vlm),
    )

    # 7) Command multiplexer to prioritize safety over VLM
    cmd_mux_node = Node(
        package="tractor_bringup",
        executable="simple_cmd_mux.py",
        name="cmd_mux",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "timeout_teleop": 1.0,
                "timeout_vlm": 5.0,
                "priority_order": ["teleop", "vlm", "autonomous"],  # teleop > vlm > autonomous
            }
        ],
        remappings=[
            ("cmd_vel_teleop", "cmd_vel_teleop"),
            ("cmd_vel_vlm", "cmd_vel_vlm"),
            ("cmd_vel_autonomous", "cmd_vel_autonomous"),
            ("cmd_vel_out", "cmd_vel_mux"),
        ],
    )

    # 8) Safety monitor
    safety_monitor_node = Node(
        package="tractor_bringup",
        executable="simple_safety_monitor.py",
        name="safety_monitor",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "max_speed_limit": max_linear_speed,
                "emergency_stop_distance": 0.3,
                "hard_stop_distance": 0.15,
                "warning_distance": 0.5,
                "pointcloud_topic": "/camera/camera/depth/color/points",
                "input_cmd_topic": "cmd_vel_mux",
                "output_cmd_topic": "cmd_vel_safe",
            }
        ],
        condition=IfCondition(with_safety),
    )

    # 9) Foxglove bridge for visualization
    foxglove_bridge_node = Node(
        package="foxglove_bridge",
        executable="foxglove_bridge",
        name="foxglove_bridge",
        output="screen",
        parameters=[{
            "port": 8765, 
            "address": "0.0.0.0", 
            "tls": False, 
            "topic_whitelist": [".*"], 
            "use_sim_time": use_sim_time
        }],
    )

    # Build launch description
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_rkllama_url_cmd)
    ld.add_action(declare_model_name_cmd)
    ld.add_action(declare_max_linear_speed_cmd)
    ld.add_action(declare_max_angular_speed_cmd)
    ld.add_action(declare_with_teleop_cmd)
    ld.add_action(declare_with_motor_cmd)
    ld.add_action(declare_with_safety_cmd)
    ld.add_action(declare_with_vlm_cmd)
    ld.add_action(declare_inference_interval_cmd)
    ld.add_action(declare_num_ctx_cmd)

    # Core components
    ld.add_action(robot_description_launch)
    ld.add_action(hiwonder_motor_node)

    # Sensors with timing
    ld.add_action(TimerAction(period=1.0, actions=[lsm9ds1_imu_launch]))
    ld.add_action(TimerAction(period=2.0, actions=[realsense_launch]))
    ld.add_action(TimerAction(period=4.0, actions=[pointcloud_to_laserscan_node]))

    # VLM and control system
    ld.add_action(TimerAction(period=6.0, actions=[vlm_controller_node]))
    ld.add_action(TimerAction(period=7.0, actions=[cmd_mux_node]))
    ld.add_action(TimerAction(period=8.0, actions=[safety_monitor_node]))

    # Visualization
    ld.add_action(TimerAction(period=3.0, actions=[foxglove_bridge_node]))

    # Optional teleop backup
    xbox_teleop_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tractor_bringup, "launch", "xbox_teleop.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
        condition=IfCondition(with_teleop),
    )
    ld.add_action(TimerAction(period=5.0, actions=[xbox_teleop_launch]))

    return ld