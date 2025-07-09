#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Package directories
    pkg_tractor_vision = get_package_share_directory("tractor_vision")
    pkg_tractor_bringup = get_package_share_directory("tractor_bringup")

    # Launch arguments
    use_sim_time = LaunchConfiguration("use_sim_time")
    camera_device = LaunchConfiguration("camera_device")
    depth_device = LaunchConfiguration("depth_device")

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation (Gazebo) clock if true",
    )
    
    declare_camera_device_cmd = DeclareLaunchArgument(
        "camera_device",
        default_value="2",
        description="Color camera device number (e.g., 2 for /dev/video2)",
    )
    
    declare_depth_device_cmd = DeclareLaunchArgument(
        "depth_device",
        default_value="0",
        description="Depth camera device number (e.g., 0 for /dev/video0)",
    )

    # Config file path
    config_file = os.path.join(pkg_tractor_vision, "config", "realsense_config.yaml")

    # RealSense Nav2 camera node
    realsense_nav2_node = Node(
        package="tractor_vision",
        executable="realsense_nav2_node",
        name="realsense_nav2_node",
        output="screen",
        parameters=[
            {
                "camera_device": camera_device,
                "depth_device": depth_device,
                "camera_name": "realsense",
                "base_frame": "base_link",
                "camera_frame": "camera_link",
                "depth_frame": "camera_depth_frame",
                "width": 640,
                "height": 480,
                "fps": 30,
                "publish_rate": 30.0,
                "depth_scale": 0.001,  # mm to meters
                "max_depth": 10.0,
                "min_depth": 0.1,
                "use_sim_time": use_sim_time,
            }
        ],
    )

    # Point cloud to laser scan converter for Nav2
    pointcloud_to_laserscan_node = Node(
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        name="pointcloud_to_laserscan",
        output="screen",
        parameters=[
            {
                "target_frame": "base_link",
                "transform_tolerance": 0.01,
                "min_height": -0.5,
                "max_height": 2.0,
                "angle_min": -1.57,  # -90 degrees
                "angle_max": 1.57,   # 90 degrees
                "angle_increment": 0.0087,  # 0.5 degrees
                "scan_time": 0.1,
                "range_min": 0.1,
                "range_max": 10.0,
                "use_inf": True,
                "inf_epsilon": 1.0,
                "use_sim_time": use_sim_time,
            }
        ],
        remappings=[
            ("cloud_in", "/realsense/depth/points"),
            ("scan", "/realsense/scan"),
        ],
    )

    # Obstacle detector node (enhanced for Nav2)
    obstacle_detector_node = Node(
        package="tractor_vision",
        executable="obstacle_detector",
        name="obstacle_detector",
        output="screen",
        parameters=[config_file, {"use_sim_time": use_sim_time}],
    )

    # TF2 relay for camera transforms (if needed)
    tf2_relay_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_base_link_tf",
        arguments=[
            "0.2", "0.0", "0.15",  # x, y, z
            "0.0", "0.0", "0.0",   # roll, pitch, yaw
            "base_link", "camera_link"
        ],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_camera_device_cmd)
    ld.add_action(declare_depth_device_cmd)

    # Add nodes
    ld.add_action(realsense_nav2_node)
    ld.add_action(pointcloud_to_laserscan_node)
    ld.add_action(obstacle_detector_node)
    ld.add_action(tf2_relay_node)

    return ld