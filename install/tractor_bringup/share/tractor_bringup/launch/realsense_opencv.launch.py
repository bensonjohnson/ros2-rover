#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Package directories
    pkg_tractor_vision = get_package_share_directory("tractor_vision")

    # Launch arguments
    use_sim_time = LaunchConfiguration("use_sim_time")
    camera_device = LaunchConfiguration("camera_device")

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation (Gazebo) clock if true",
    )
    
    declare_camera_device_cmd = DeclareLaunchArgument(
        "camera_device",
        default_value="2",
        description="Camera device number (e.g., 2 for /dev/video2)",
    )

    # Config file path
    config_file = os.path.join(pkg_tractor_vision, "config", "realsense_config.yaml")

    # OpenCV camera node (alternative to RealSense SDK)
    opencv_camera_node = Node(
        package="tractor_vision",
        executable="opencv_camera_node",
        name="opencv_camera_node",
        output="screen",
        parameters=[
            {
                "camera_device": camera_device,
                "camera_name": "realsense",
                "frame_id": "camera_link",
                "width": 640,
                "height": 480,
                "fps": 30,
                "publish_rate": 30.0,
                "use_sim_time": use_sim_time,
            }
        ],
        remappings=[
            ("/realsense/color/image_raw", "/realsense/color/image_raw"),
            ("/realsense/color/camera_info", "/realsense/color/camera_info"),
        ],
    )

    # Obstacle detector node (can work with OpenCV camera)
    obstacle_detector_node = Node(
        package="tractor_vision",
        executable="obstacle_detector",
        name="obstacle_detector",
        output="screen",
        parameters=[config_file, {"use_sim_time": use_sim_time}],
    )

    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_camera_device_cmd)

    # Add camera node
    ld.add_action(opencv_camera_node)
    ld.add_action(obstacle_detector_node)

    return ld