#!/usr/bin/env python3
"""PPO live-training launch focused on the RTAB observation pipeline.

This launch brings up the motor stack, RealSense, RTAB-Map, the RTAB
observation builder, safety monitor, velocity controller, NPU runtime node and
its PPO manager.  All BEV-specific components have been removed so the system
relies solely on the RTAB occupancy/frontier observations.
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

    declare_max_speed_cmd = DeclareLaunchArgument("max_speed", default_value="0.15")
    declare_safety_distance_cmd = DeclareLaunchArgument("safety_distance", default_value="0.2")
    declare_obs_height_cmd = DeclareLaunchArgument("observation_height", default_value="128")
    declare_obs_width_cmd = DeclareLaunchArgument("observation_width", default_value="128")
    declare_ppo_update_cmd = DeclareLaunchArgument("ppo_update_interval_sec", default_value="20.0")
    declare_ppo_min_export_cmd = DeclareLaunchArgument("ppo_min_export_interval_sec", default_value="120.0")

    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tractor_bringup, "launch", "robot_description.launch.py")
        )
    )

    hiwonder_motor_node = Node(
        package="tractor_control",
        executable="hiwonder_motor_driver",
        name="hiwonder_motor_driver",
        output="screen",
        parameters=[os.path.join(pkg_tractor_bringup, "config", "hiwonder_motor_params.yaml")]
    )

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("realsense2_camera"), "launch", "rs_launch.py")
        ),
        launch_arguments={
            "pointcloud.enable": "false",
            "align_depth.enable": "false",
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

    rtabmap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("rtabmap_launch"), "launch", "rtabmap.launch.py")
        ),
        launch_arguments={
            "use_sim_time": "false",
            "frame_id": "base_link",
            "odom_topic": "odom",
            "subscribe_depth": "true",
            "subscribe_rgb": "true",
            "approx_sync": "false",
            "queue_size": "30",
            "rgb_topic": "/camera/camera/color/image_raw",
            "depth_topic": "/camera/camera/depth/image_rect_raw",
            "camera_info_topic": "/camera/camera/depth/camera_info",
            "rtabmap": "true",
            "rtabmapviz": "false",
            "rviz": "false",
            "args": "--delete_db_on_start --Mem/IncrementalMemory true --Grid/Sensor 0 --Grid/FromDepth true --subscribe_scan false --subscribe_imu true --RGBD/CreateOccupancyGrid true",
        }.items()
    )

    rtab_observation_node = Node(
        package="tractor_bringup",
        executable="rtab_observation_node.py",
        name="rtab_observation",
        output="screen",
        parameters=[{
            "depth_topic": "/camera/camera/depth/image_rect_raw",
            "occupancy_topic": "/rtabmap/local_grid_map",
            "frontier_topic": "/rtabmap/frontiers",
            "odom_topic": "/odom",
            "imu_topic": "/lsm9ds1_imu_publisher/imu/data",
            "publish_rate_hz": 10.0,
            "occupancy_window_m": 12.0,
            "observation_height": LaunchConfiguration("observation_height"),
            "observation_width": LaunchConfiguration("observation_width"),
            "depth_clip_m": 6.0,
        }]
    )

    safety_monitor_rtab = Node(
        package="tractor_bringup",
        executable="simple_safety_monitor_rtab.py",
        name="simple_safety_monitor_rtab",
        output="screen",
        parameters=[{
            "occupancy_topic": "/rtabmap/local_grid_map",
            "input_cmd_topic": "cmd_vel_ai",
            "output_cmd_topic": "cmd_vel_raw",
            "emergency_stop_distance": LaunchConfiguration("safety_distance"),
            "hard_stop_distance": 0.12,
            "forward_width_m": 1.2,
        }]
    )

    vfc_node = Node(
        package="tractor_control",
        executable="velocity_feedback_controller",
        name="velocity_feedback_controller",
        output="screen",
        parameters=[{"control_frequency": 50.0}]
    )

    npu_rtab_node = Node(
        package="tractor_bringup",
        executable="npu_exploration_rtab.py",
        name="npu_exploration_rtab",
        output="screen",
        parameters=[{
            "max_speed": LaunchConfiguration("max_speed"),
            "safety_distance": LaunchConfiguration("safety_distance"),
            "observation_topic": "/exploration/observation",
        }]
    )

    ppo_rtab_node = Node(
        package="tractor_bringup",
        executable="ppo_manager_rtab.py",
        name="ppo_manager_rtab",
        output="screen",
        parameters=[{
            "observation_topic": "/exploration/observation",
            "cmd_topic": "cmd_vel_ai",
            "update_interval_sec": LaunchConfiguration("ppo_update_interval_sec"),
            "min_export_interval_sec": LaunchConfiguration("ppo_min_export_interval_sec"),
            "rollout_capacity": 4096,
            "minibatch_size": 128,
            "update_epochs": 3,
            "reward_forward_scale": 4.0,
            "reward_emergency_penalty": -5.0,
            "reward_block_penalty": -1.0,
            "reward_coverage_scale": 2.0,
            "max_speed": LaunchConfiguration("max_speed"),
            "min_forward_threshold": LaunchConfiguration("safety_distance"),
        }]
    )

    lsm9ds1_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("tractor_sensors"), "launch", "lsm9ds1_imu.launch.py")
        )
    )

    ld = LaunchDescription()
    ld.add_action(declare_max_speed_cmd)
    ld.add_action(declare_safety_distance_cmd)
    ld.add_action(declare_obs_height_cmd)
    ld.add_action(declare_obs_width_cmd)
    ld.add_action(declare_ppo_update_cmd)
    ld.add_action(declare_ppo_min_export_cmd)

    ld.add_action(robot_description_launch)
    ld.add_action(hiwonder_motor_node)
    ld.add_action(TimerAction(period=5.0, actions=[realsense_launch]))
    ld.add_action(TimerAction(period=8.0, actions=[rtabmap_launch]))
    ld.add_action(TimerAction(period=9.0, actions=[rtab_observation_node]))
    ld.add_action(TimerAction(period=6.0, actions=[lsm9ds1_launch]))
    ld.add_action(TimerAction(period=10.0, actions=[safety_monitor_rtab]))
    ld.add_action(TimerAction(period=11.0, actions=[vfc_node]))
    ld.add_action(TimerAction(period=12.0, actions=[npu_rtab_node]))
    ld.add_action(TimerAction(period=13.0, actions=[ppo_rtab_node]))

    return ld
