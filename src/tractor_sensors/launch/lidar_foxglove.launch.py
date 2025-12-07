#!/usr/bin/env python3
"""
Launch LD19 LiDAR with Foxglove Bridge for visualization
Includes robot URDF publishing for 3D model visualization
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Package directories
    pkg_tractor_bringup = get_package_share_directory('tractor_bringup')
    pkg_tractor_sensors = get_package_share_directory('tractor_sensors')

    # Launch arguments
    port = LaunchConfiguration('port')
    address = LaunchConfiguration('address')
    lidar_port = LaunchConfiguration('lidar_port')
    frame_id = LaunchConfiguration('frame_id')

    declare_port_cmd = DeclareLaunchArgument(
        'port',
        default_value='8765',
        description='Foxglove Bridge WebSocket port'
    )

    declare_address_cmd = DeclareLaunchArgument(
        'address',
        default_value='0.0.0.0',
        description='Foxglove Bridge address (0.0.0.0 for all interfaces)'
    )

    declare_lidar_port_cmd = DeclareLaunchArgument(
        'lidar_port',
        default_value='/dev/ttyUSB0',
        description='Serial port name of the lidar'
    )

    declare_frame_id_cmd = DeclareLaunchArgument(
        'frame_id',
        default_value='laser_link',
        description='Frame ID for the lidar'
    )

    # URDF file path
    urdf_file = os.path.join(pkg_tractor_bringup, 'urdf', 'tractor.urdf.xacro')

    # Robot description - process xacro to get URDF
    robot_description_content = ParameterValue(
        Command(['xacro ', urdf_file]),
        value_type=str
    )

    # Robot state publisher - publishes robot_description and TF
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {
                'robot_description': robot_description_content,
                'use_sim_time': False,
            }
        ]
    )

    # LDROBOT LiDAR Publisher
    ldlidar_node = Node(
        package='ldlidar_stl_ros2',
        executable='ldlidar_stl_ros2_node',
        name='ldlidar_publisher_ld19',
        output='screen',
        parameters=[
            {'product_name': 'LDLiDAR_LD19'},
            {'topic_name': 'scan'},
            {'frame_id': frame_id},
            {'port_name': lidar_port},
            {'port_baudrate': 230400},
            {'laser_scan_dir': True},  # Counterclockwise - arrow points forward, standard orientation
            {'enable_angle_crop_func': False},
            {'angle_crop_min': 135.0},
            {'angle_crop_max': 225.0}
        ]
    )

    # Static Transform (Base -> Laser)
    # LiDAR mounted flat above RealSense camera
    # - 38mm BEHIND RealSense front (camera front at 133.35mm, LiDAR front at 95.35mm)
    # - 200mm from ground to LiDAR bottom
    # - Arrow points FORWARD (standard orientation)
    # - Counterclockwise scan with no rotation for correct orientation
    # x=0.07635m, y=0m, z=0.1915m, roll=0, pitch=0, yaw=0
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_laser_tf',
        arguments=['0.07635', '0.0', '0.1915', '0.0', '0.0', '0.0', 'base_link', 'laser_link']
    )

    # Foxglove Bridge
    foxglove_bridge_node = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        output='screen',
        parameters=[
            {
                'port': port,
                'address': address,
                'tls': False,
                'topic_whitelist': ['.*'],
                'send_buffer_limit': 10000000,
                'use_compression': True,
            }
        ]
    )

    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_port_cmd)
    ld.add_action(declare_address_cmd)
    ld.add_action(declare_lidar_port_cmd)
    ld.add_action(declare_frame_id_cmd)

    # Add nodes
    ld.add_action(robot_state_publisher_node)
    ld.add_action(ldlidar_node)
    ld.add_action(static_tf_node)
    ld.add_action(foxglove_bridge_node)

    return ld
