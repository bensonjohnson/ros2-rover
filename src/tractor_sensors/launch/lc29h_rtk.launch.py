#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def launch_setup(context, *args, **kwargs):
    # Get parameters
    config_file = LaunchConfiguration('config_file')
    
    # LC29H RTK GPS node
    lc29h_node = Node(
        package='tractor_sensors',
        executable='lc29h_rtk_gps',
        name='lc29h_rtk_gps_publisher',
        output='screen',
        parameters=[config_file],
        remappings=[
            ('gps/fix', 'gps/fix'),
            ('gps/filtered', 'gps/filtered'), 
            ('gps/velocity', 'gps/velocity'),
        ]
    )

    return [
        lc29h_node
    ]


def generate_launch_description():
    # Default config file
    default_config_file = PathJoinSubstitution([
        FindPackageShare('tractor_sensors'),
        'config',
        'lc29h_rtk_config.yaml'
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config_file,
            description='Path to LC29H RTK configuration file'
        ),
        
        OpaqueFunction(function=launch_setup)
    ])