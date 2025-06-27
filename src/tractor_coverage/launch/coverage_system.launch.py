#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    enable_visualizer = LaunchConfiguration('enable_visualizer')
    
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )
    
    declare_enable_visualizer_cmd = DeclareLaunchArgument(
        'enable_visualizer',
        default_value='true',
        description='Enable coverage visualizer'
    )
    
    # Coverage action server
    coverage_server_node = Node(
        package='tractor_coverage',
        executable='coverage_action_server',
        name='coverage_action_server',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'default_tool_width': 1.0,
            'default_overlap': 0.1,
            'default_work_speed': 0.5,
            'waypoint_tolerance': 0.25,
            'max_nav_timeout': 30.0
        }]
    )
    
    # Coverage visualizer (optional)
    coverage_visualizer_node = Node(
        package='tractor_coverage',
        executable='coverage_visualizer',
        name='coverage_visualizer',
        output='screen',
        condition=IfCondition(enable_visualizer),
        parameters=[{
            'use_sim_time': use_sim_time
        }]
    )
    
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_enable_visualizer_cmd)
    
    # Add nodes
    ld.add_action(coverage_server_node)
    ld.add_action(coverage_visualizer_node)
    
    return ld