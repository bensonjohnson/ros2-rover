#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    # Get parameters
    config_file = LaunchConfiguration('config_file')
    use_gps = LaunchConfiguration('use_gps')
    
    nodes = []
    
    # Local EKF node - fuses wheel odometry and IMU
    local_ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[config_file],
        remappings=[
            ('odometry/filtered', 'odometry/filtered'),
            ('/diagnostics', 'diagnostics'),
        ]
    )
    nodes.append(local_ekf_node)
    
    # Add GPS-based global localization if GPS is enabled
    use_gps_str = use_gps.perform(context)
    if use_gps_str.lower() == 'true':
        # Global EKF node - fuses GPS with local EKF
        global_ekf_node = Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node_map',
            output='screen',
            parameters=[config_file],
            remappings=[
                ('odometry/filtered', 'odometry/filtered_map'),
                ('/diagnostics', 'diagnostics'),
            ]
        )
        nodes.append(global_ekf_node)
        
        # NavSat transform node - converts GPS to odometry
        navsat_node = Node(
            package='robot_localization',
            executable='navsat_transform_node',
            name='navsat_transform_node',
            output='screen',
            parameters=[config_file],
            remappings=[
                ('imu/data', 'imu/data'),
                ('gps/fix', 'gps/fix'),
                ('odometry/filtered', 'odometry/filtered'),
                ('odometry/gps', 'odometry/gps'),
                ('gps/filtered', 'gps/filtered_navsat'),
            ]
        )
        nodes.append(navsat_node)

    return nodes


def generate_launch_description():
    # Default config file
    default_config_file = PathJoinSubstitution([
        FindPackageShare('tractor_bringup'),
        'config',
        'robot_localization.yaml'
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config_file,
            description='Path to robot_localization configuration file'
        ),
        
        DeclareLaunchArgument(
            'use_gps',
            default_value='true',
            description='Enable GPS-based global localization'
        ),
        
        OpaqueFunction(function=launch_setup)
    ])