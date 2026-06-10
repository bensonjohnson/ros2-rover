#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config_file = PathJoinSubstitution([
        FindPackageShare('tractor_sensors'),
        'config',
        'bno085_config.yaml'
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config_file,
            description='Path to BNO085 IMU configuration file'
        ),
        Node(
            package='tractor_sensors',
            executable='bno085_imu',
            name='bno085_imu_publisher',
            output='screen',
            parameters=[LaunchConfiguration('config_file')],
        ),
    ])
