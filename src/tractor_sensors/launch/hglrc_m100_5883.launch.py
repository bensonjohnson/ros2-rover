#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Package directories
    pkg_tractor_sensors = get_package_share_directory('tractor_sensors')
    
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    gps_port = LaunchConfiguration('gps_port')
    i2c_bus = LaunchConfiguration('i2c_bus')
    
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )
    
    declare_gps_port_cmd = DeclareLaunchArgument(
        'gps_port',
        default_value='/dev/ttyS6',
        description='GPS serial port device'
    )
    
    declare_i2c_bus_cmd = DeclareLaunchArgument(
        'i2c_bus',
        default_value='5',
        description='I2C bus number for QMC5883 compass'
    )
    
    # Config file path
    config_file = os.path.join(pkg_tractor_sensors, 'config', 'hglrc_m100_5883_config.yaml')
    
    # HGLRC M100-5883 GPS and Compass node
    hglrc_node = Node(
        package='tractor_sensors',
        executable='hglrc_m100_5883',
        name='hglrc_m100_5883_publisher',
        output='screen',
        parameters=[
            config_file,
            {
                'use_sim_time': use_sim_time,
                'gps_port': gps_port,
                'i2c_bus': int(i2c_bus.perform_substitution(context=None)) if hasattr(i2c_bus, 'perform_substitution') else 5
            }
        ]
    )
    
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_gps_port_cmd)
    ld.add_action(declare_i2c_bus_cmd)
    
    # Add sensor node
    ld.add_action(hglrc_node)
    
    return ld