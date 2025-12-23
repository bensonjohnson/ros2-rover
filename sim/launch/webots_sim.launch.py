#!/usr/bin/env python3
"""
Webots simulation launch file for SAC Rover training.

Launches:
- Webots simulator with training_arena world
- Robot state publisher with rover URDF
- Optionally: sac_episode_runner for local testing
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Paths
    pkg_sim = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    world_file = os.path.join(pkg_sim, 'worlds', 'training_arena.wbt')
    
    # Try to get tractor_bringup for URDF
    try:
        pkg_tractor_bringup = get_package_share_directory('tractor_bringup')
        urdf_file = os.path.join(pkg_tractor_bringup, 'urdf', 'tractor.urdf.xacro')
    except:
        urdf_file = None
    
    # Launch arguments
    use_gui = LaunchConfiguration('gui')
    fast_mode = LaunchConfiguration('fast')
    
    declare_gui = DeclareLaunchArgument(
        'gui',
        default_value='true',
        description='Launch Webots with GUI')
    
    declare_fast = DeclareLaunchArgument(
        'fast',
        default_value='false',
        description='Run simulation faster than real-time')
    
    # Webots command
    # Note: Webots snap path may differ, adjust if needed
    webots_cmd = [
        'webots',
        '--mode=realtime',
        world_file
    ]
    
    webots_process = ExecuteProcess(
        cmd=webots_cmd,
        output='screen',
        shell=False
    )
    
    # Robot state publisher (if URDF available)
    nodes = []
    if urdf_file and os.path.exists(urdf_file):
        robot_description_content = ParameterValue(
            Command(['xacro ', urdf_file]),
            value_type=str
        )
        
        robot_state_publisher = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description_content,
                'use_sim_time': True
            }]
        )
        nodes.append(robot_state_publisher)
    
    # Static TF for odom -> map (simple identity for sim)
    static_tf_map = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_map_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
    )
    nodes.append(static_tf_map)
    
    ld = LaunchDescription()
    ld.add_action(declare_gui)
    ld.add_action(declare_fast)
    ld.add_action(webots_process)
    
    for node in nodes:
        ld.add_action(node)
    
    return ld
