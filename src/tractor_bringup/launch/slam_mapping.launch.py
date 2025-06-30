#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Package directories
    pkg_tractor_bringup = get_package_share_directory('tractor_bringup')
    
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    slam_params_file = LaunchConfiguration('slam_params_file')
    foxglove_port = LaunchConfiguration('foxglove_port')
    
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )
    
    declare_slam_params_file_cmd = DeclareLaunchArgument(
        'slam_params_file',
        default_value=os.path.join(pkg_tractor_bringup, 'config', 'slam_toolbox_params.yaml'),
        description='Full path to slam toolbox parameters file'
    )
    
    declare_foxglove_port_cmd = DeclareLaunchArgument(
        'foxglove_port',
        default_value='8765',
        description='Port for Foxglove bridge WebSocket server'
    )
    
    # Start robot localization for GPS fusion
    robot_localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('tractor_bringup'),
                'launch',
                'robot_localization.launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time,
        }.items()
    )
    
    # Start vision processing for RealSense
    vision_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('tractor_bringup'),
                'launch',
                'vision.launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': use_sim_time,
        }.items()
    )
    
    # Convert RealSense depth image to laser scan
    # This creates a 2D laser scan from the 3D depth data
    depthimage_to_laserscan_node = Node(
        package='depthimage_to_laserscan',
        executable='depthimage_to_laserscan_node',
        name='depthimage_to_laserscan',
        remappings=[
            ('depth', '/realsense_435i/depth/image_rect_raw'),
            ('depth_camera_info', '/realsense_435i/depth/camera_info'),
            ('scan', '/realsense_435i/scan')
        ],
        parameters=[{
            'use_sim_time': use_sim_time,
            'scan_height': 10,  # Number of pixel rows to use for scan
            'scan_time': 0.033,  # Time between scans (30 Hz)
            'range_min': 0.45,   # Minimum scan range (meters)
            'range_max': 10.0,   # Maximum scan range (meters)
            'output_frame_id': 'realsense_depth_frame'
        }]
    )
    
    # SLAM Toolbox for mapping
    slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[
            slam_params_file,
            {'use_sim_time': use_sim_time}
        ]
    )
    
    # Foxglove bridge for web-based visualization
    foxglove_bridge_node = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        parameters=[{
            'port': foxglove_port,
            'address': '0.0.0.0',
            'tls': False,
            'use_sim_time': use_sim_time
        }],
        output='screen'
    )
    
    # Map saver service (to save maps during runtime)
    map_saver_server_node = Node(
        package='nav2_map_server',
        executable='map_saver_server',
        name='map_saver_server',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )
    
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_slam_params_file_cmd)
    ld.add_action(declare_foxglove_port_cmd)
    
    # Add nodes and launches
    ld.add_action(robot_localization_launch)
    ld.add_action(vision_launch)
    ld.add_action(depthimage_to_laserscan_node)
    ld.add_action(slam_toolbox_node)
    ld.add_action(foxglove_bridge_node)
    ld.add_action(map_saver_server_node)
    
    return ld