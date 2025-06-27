#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock'
    )
    
    # Package paths
    pkg_tractor_bringup = FindPackageShare('tractor_bringup')
    
    # Robot description
    urdf_file = PathJoinSubstitution([pkg_tractor_bringup, 'urdf', 'tractor.urdf.xacro'])
    robot_description = Command(['xacro ', urdf_file, ' use_sim:=true'])
    
    # Robot state publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': use_sim_time
        }]
    )
    
    # Note: joint_state_publisher not needed in Gazebo simulation
    # Gazebo provides joint states through the gz_ros_bridge
    
    # Note: GPS and odometry will come from Gazebo simulation
    # through the gz_ros_bridge in the gazebo launch file
    
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    
    # Add nodes (only simulation-compatible ones)
    ld.add_action(robot_state_publisher_node)
    # Skip hardware-dependent nodes in simulation
    
    return ld