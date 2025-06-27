#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Package directories
    pkg_tractor_sensors = get_package_share_directory('tractor_sensors')
    
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )
    
    # Encoder publisher node
    encoder_publisher_node = Node(
        package='tractor_sensors',
        executable='encoder_publisher',
        name='encoder_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'left_encoder_pin_a': 18,
            'left_encoder_pin_b': 19,
            'right_encoder_pin_a': 20,
            'right_encoder_pin_b': 21,
            'encoder_ppr': 1440,
            'wheel_radius': 0.15,
            'wheel_separation': 0.5,
            'publish_rate': 50.0
        }]
    )
    
    # GPS and compass publisher node
    gps_compass_node = Node(
        package='tractor_sensors',
        executable='gps_compass_publisher',
        name='gps_compass_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'gps_port': '/dev/ttyUSB0',
            'gps_baudrate': 9600,
            'compass_port': '/dev/ttyUSB1',
            'compass_baudrate': 9600,
            'magnetic_declination': 0.0,
            'gps_frame_id': 'gps_link',
            'compass_frame_id': 'compass_link'
        }]
    )
    
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    
    # Add sensor nodes
    ld.add_action(encoder_publisher_node)
    ld.add_action(gps_compass_node)
    
    return ld