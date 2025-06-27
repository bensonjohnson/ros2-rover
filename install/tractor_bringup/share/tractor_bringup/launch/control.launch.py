#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )
    
    # Tank steering controller node
    tank_steering_node = Node(
        package='tractor_control',
        executable='tank_steering_controller',
        name='tank_steering_controller',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'i2c_bus': 1,
            'motor_controller_address': 0x60,
            'wheel_separation': 0.5,
            'max_motor_speed': 255,
            'deadband': 0.05
        }]
    )
    
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    
    # Add control nodes
    ld.add_action(tank_steering_node)
    
    return ld