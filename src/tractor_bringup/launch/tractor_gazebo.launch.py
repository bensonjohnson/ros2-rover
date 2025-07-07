#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    ExecuteProcess,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration("use_sim_time")
    headless = LaunchConfiguration("headless")
    world_file = LaunchConfiguration("world_file")

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (Gazebo) clock",
    )

    declare_headless_cmd = DeclareLaunchArgument(
        "headless", default_value="false", description="Run Gazebo in headless mode"
    )

    declare_world_file_cmd = DeclareLaunchArgument(
        "world_file", default_value="empty.sdf", description="Gazebo world file to load"
    )

    # Package paths
    pkg_tractor_bringup = FindPackageShare("tractor_bringup")

    # URDF/Robot description
    urdf_file = PathJoinSubstitution(
        [pkg_tractor_bringup, "urdf", "tractor.urdf.xacro"]
    )
    robot_description = Command(["xacro ", urdf_file, " use_sim:=true"])

    # Robot state publisher
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {"robot_description": robot_description, "use_sim_time": use_sim_time}
        ],
    )

    # Start Gazebo with GUI
    start_gazebo_cmd = ExecuteProcess(
        cmd=["gz", "sim", "-v", "4", "-r", world_file],
        output="screen",
        condition=UnlessCondition(headless),
    )

    # Start Gazebo headless
    start_gazebo_headless_cmd = ExecuteProcess(
        cmd=["gz", "sim", "-v", "4", "-r", "-s", world_file],
        output="screen",
        condition=IfCondition(headless),
    )

    # Spawn robot in Gazebo using ros_gz_sim
    spawn_robot_node = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-topic",
            "robot_description",
            "-name",
            "tractor",
            "-x",
            "0.0",
            "-y",
            "0.0",
            "-z",
            "0.1",
        ],
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    # Bridge between Gazebo and ROS2
    gz_ros_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
            "/model/tractor/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry",
            "/world/empty/model/tractor/joint_state@sensor_msgs/msg/JointState@gz.msgs.Model",
        ],
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    # Static transform from map to odom (for navigation)
    map_to_odom_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["0", "0", "0", "0", "0", "0", "map", "odom"],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    # Static transform from odom to base_link (temporary fix for missing
    # odometry)
    odom_to_base_link_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["0", "0", "0", "0", "0", "0", "odom", "base_link"],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    # Joint state publisher (for simulation)
    joint_state_publisher_node = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_headless_cmd)
    ld.add_action(declare_world_file_cmd)

    # Add actions
    ld.add_action(robot_state_publisher_node)
    ld.add_action(start_gazebo_cmd)
    ld.add_action(start_gazebo_headless_cmd)
    ld.add_action(spawn_robot_node)
    ld.add_action(gz_ros_bridge)
    ld.add_action(map_to_odom_tf)
    ld.add_action(odom_to_base_link_tf)
    ld.add_action(joint_state_publisher_node)

    return ld
