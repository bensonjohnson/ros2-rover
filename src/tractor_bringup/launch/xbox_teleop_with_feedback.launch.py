#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    max_linear_speed = LaunchConfiguration('max_linear_speed')
    max_angular_speed = LaunchConfiguration('max_angular_speed')
    
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )
    
    declare_max_linear_speed_cmd = DeclareLaunchArgument(
        'max_linear_speed',
        default_value='4.0',
        description='Maximum linear speed in m/s'
    )
    
    declare_max_angular_speed_cmd = DeclareLaunchArgument(
        'max_angular_speed',
        default_value='2.0',
        description='Maximum angular speed in rad/s'
    )
    
    # Xbox controller teleop with feedback control enabled
    xbox_teleop_node = Node(
        package='tractor_control',
        executable='xbox_controller_teleop',
        name='xbox_controller_teleop',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'max_linear_speed': max_linear_speed,
            'max_angular_speed': max_angular_speed,
            'deadzone': 0.15,
            'tank_drive_mode': True,
            'controller_index': 0,
            'use_feedback_control': True  # Enable encoder feedback
        }],
        remappings=[
            # Xbox controller publishes to cmd_vel_raw, feedback controller converts to cmd_vel
        ]
    )
    
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_max_linear_speed_cmd)
    ld.add_action(declare_max_angular_speed_cmd)
    
    # Add teleop node
    ld.add_action(xbox_teleop_node)
    
    return ld