#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Package directories
    pkg_tractor_bringup = get_package_share_directory("tractor_bringup")

    # Launch arguments
    use_sim_time = LaunchConfiguration("use_sim_time")

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation (Gazebo) clock if true",
    )

    # Configuration file
    robot_localization_config = os.path.join(
        pkg_tractor_bringup, "config", "robot_localization.yaml"
    )

    # Local EKF node (odom frame)
    ekf_local_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[robot_localization_config, {"use_sim_time": use_sim_time}],
        remappings=[
            ("odometry/filtered", "odometry/filtered"),
        ],
    )

    # Global EKF node (map frame)
    ekf_global_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node_map",
        output="screen",
        parameters=[robot_localization_config, {"use_sim_time": use_sim_time}],
        remappings=[
            ("odometry/filtered", "odometry/filtered_map"),
        ],
    )

    # NavSat transform node
    navsat_transform_node = Node(
        package="robot_localization",
        executable="navsat_transform_node",
        name="navsat_transform_node",
        output="screen",
        parameters=[robot_localization_config, {"use_sim_time": use_sim_time}],
        remappings=[
            ("imu/data", "hglrc_gps/imu"),
            ("gps/fix", "hglrc_gps/fix"),
            ("odometry/filtered", "odometry/filtered"),
            ("odometry/gps", "odometry/gps"),
        ],
    )

    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)

    # Add localization nodes
    ld.add_action(ekf_local_node)
    ld.add_action(ekf_global_node)
    ld.add_action(navsat_transform_node)

    return ld
