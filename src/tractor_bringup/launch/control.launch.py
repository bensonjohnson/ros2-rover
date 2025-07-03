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
    
    # Hiwonder motor driver node (includes battery publishing)
    hiwonder_motor_node = Node(
        package='tractor_control',
        executable='hiwonder_motor_driver',
        name='hiwonder_motor_driver',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'i2c_bus': 5,
            'motor_controller_address': 0x34,
            'wheel_separation': 0.5,
            'wheel_radius': 0.15,
            'max_motor_speed': 100,
            'deadband': 0.05,
            'encoder_ppr': 1980,
            'publish_rate': 100.0,
            'use_pwm_control': True,
            'motor_type': 3,
            'min_samples_for_estimation': 10,
            'max_history_minutes': 60
        }]
    )
    
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    
    # Add control nodes
    ld.add_action(hiwonder_motor_node)
    
    return ld