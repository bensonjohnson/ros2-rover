#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    enable_mower = LaunchConfiguration('enable_mower')
    enable_sprayer = LaunchConfiguration('enable_sprayer')
    
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )
    
    declare_enable_mower_cmd = DeclareLaunchArgument(
        'enable_mower',
        default_value='false',
        description='Enable mower controller'
    )
    
    declare_enable_sprayer_cmd = DeclareLaunchArgument(
        'enable_sprayer',
        default_value='false',
        description='Enable sprayer controller'
    )
    
    # Mower controller node
    mower_controller_node = Node(
        package='tractor_implements',
        executable='mower_controller',
        name='mower_controller',
        output='screen',
        condition=IfCondition(enable_mower),
        parameters=[{
            'use_sim_time': use_sim_time,
            'mower_enable_pin': 22,
            'mower_pwm_pin': 23,
            'blade_rpm_sensor_pin': 24,
            'safety_stop_pin': 25,
            'max_blade_rpm': 3000,
            'min_blade_rpm': 1500,
            'mower_height_i2c_addr': 0x40,
            'default_cut_height': 25,
            'safety_timeout': 5.0
        }]
    )
    
    # Sprayer controller node
    sprayer_controller_node = Node(
        package='tractor_implements',
        executable='sprayer_controller',
        name='sprayer_controller',
        output='screen',
        condition=IfCondition(enable_sprayer),
        parameters=[{
            'use_sim_time': use_sim_time,
            'pump_enable_pin': 26,
            'pump_pwm_pin': 27,
            'nozzle_control_pins': [5, 6, 7, 8],
            'flow_sensor_pin': 9,
            'tank_level_sensor_i2c_addr': 0x41,
            'pressure_sensor_i2c_addr': 0x42,
            'max_pump_speed': 100,
            'target_pressure': 2.0,
            'min_tank_level': 10,
            'spray_width': 2.0,
            'default_flow_rate': 1.0
        }]
    )
    
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_enable_mower_cmd)
    ld.add_action(declare_enable_sprayer_cmd)
    
    # Add implement nodes
    ld.add_action(mower_controller_node)
    ld.add_action(sprayer_controller_node)
    
    return ld