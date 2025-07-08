#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Package Share
    # pkg_tractor_bringup = FindPackageShare('tractor_bringup').find('tractor_bringup')
    # Corrected way to get package share directory for path joining
    pkg_tractor_bringup_share = FindPackageShare('tractor_bringup')

    # Launch arguments
    use_sim_time = LaunchConfiguration("use_sim_time")

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation (Gazebo) clock if true",
    )

    # Parameters file
    motor_params_file = PathJoinSubstitution(
        [pkg_tractor_bringup, "config", "hiwonder_motor_params.yaml"]
    )

    # Hiwonder motor driver node (includes battery publishing)
    hiwonder_motor_node = Node(
        package="tractor_control",
        executable="hiwonder_motor_driver",
        name="hiwonder_motor_driver",
        output="screen",
        parameters=[
            motor_params_file,  # Load parameters from YAML file
            # Override use_sim_time from launch argument
            {"use_sim_time": use_sim_time},
        ],
    )

    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)

    # Add control nodes
    ld.add_action(hiwonder_motor_node)

    return ld
