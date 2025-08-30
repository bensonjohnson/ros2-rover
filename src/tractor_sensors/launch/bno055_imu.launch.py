#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    # Get parameters
    config_file = LaunchConfiguration('config_file')
    
    # BNO055 IMU node
    bno055_node = Node(
        package='tractor_sensors',
        executable='bno055_imu',
        name='bno055_imu_publisher',
        output='screen',
        parameters=[config_file],
        remappings=[
            ('imu/data', 'imu/data'),
            ('imu/mag', 'imu/mag'),
            ('imu/temperature', 'imu/temperature'),
            ('imu/euler', 'imu/euler'),
            ('imu/quaternion', 'imu/quaternion'),
            ('imu/calibration_status', 'imu/calibration_status'),
            ('imu/status', 'imu/status'),
        ]
    )

    return [
        bno055_node
    ]


def generate_launch_description():
    # Default config file
    default_config_file = PathJoinSubstitution([
        FindPackageShare('tractor_sensors'),
        'config',
        'bno055_config.yaml'
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config_file,
            description='Path to BNO055 IMU configuration file'
        ),
        
        OpaqueFunction(function=launch_setup)
    ])