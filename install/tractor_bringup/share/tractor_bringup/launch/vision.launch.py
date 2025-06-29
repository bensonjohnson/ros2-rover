#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Package directories
    pkg_tractor_vision = get_package_share_directory('tractor_vision')
    
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )
    
    # Config file path
    config_file = os.path.join(pkg_tractor_vision, 'config', 'realsense_config.yaml')
    
    # Official RealSense camera node
    realsense_camera_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='realsense2_camera',
        output='screen',
        parameters=[
            {
                'camera_name': 'realsense',
                'camera_namespace': 'realsense',
                'enable_color': True,
                'enable_depth': True,
                'enable_infra1': False,
                'enable_infra2': False,
                'color_width': 640,
                'color_height': 480,
                'color_fps': 30,
                'depth_width': 640,
                'depth_height': 480,
                'depth_fps': 30,
                'pointcloud.enable': True,
                'align_depth.enable': True,
                'use_sim_time': use_sim_time
            }
        ]
    )

    # Custom RealSense processor node (for custom processing)
    realsense_processor_node = Node(
        package='tractor_vision',
        executable='realsense_processor',
        name='realsense_processor',
        output='screen',
        parameters=[
            config_file,
            {'use_sim_time': use_sim_time}
        ]
    )
    
    # Obstacle detector node
    obstacle_detector_node = Node(
        package='tractor_vision',
        executable='obstacle_detector',
        name='obstacle_detector',
        output='screen',
        parameters=[
            config_file,
            {'use_sim_time': use_sim_time}
        ]
    )
    
    ld = LaunchDescription()
    
    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    
    # Add vision nodes
    ld.add_action(realsense_camera_node)
    ld.add_action(realsense_processor_node)
    ld.add_action(obstacle_detector_node)
    
    return ld