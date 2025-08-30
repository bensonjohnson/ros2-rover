#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    # Get parameters
    config_file = LaunchConfiguration('config_file')
    
    # LSM9DS1 IMU node
    lsm9ds1_node = Node(
        package='tractor_sensors',
        executable='lsm9ds1_imu',
        name='lsm9ds1_imu_publisher',
        output='screen',
        parameters=[config_file],
        remappings=[
            ('imu/data', 'imu/data'),
            ('imu/mag', 'imu/mag'),
            ('imu/temperature', 'imu/temperature'),
            ('imu/accel_raw', 'imu/accel_raw'),
            ('imu/gyro_raw', 'imu/gyro_raw'),
        ]
    )

    return [
        lsm9ds1_node
    ]


def generate_launch_description():
    # Default config file
    default_config_file = PathJoinSubstitution([
        FindPackageShare('tractor_sensors'),
        'config',
        'lsm9ds1_config.yaml'
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config_file,
            description='Path to LSM9DS1 IMU configuration file'
        ),
        
        OpaqueFunction(function=launch_setup)
    ])