#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Package directories
    pkg_tractor_bringup = get_package_share_directory("tractor_bringup")
    pkg_nav2_bringup = get_package_share_directory("nav2_bringup")

    # Launch arguments
    use_sim_time = LaunchConfiguration("use_sim_time")
    map_save_period = LaunchConfiguration("map_save_period")
    mapping_duration = LaunchConfiguration("mapping_duration")
    max_speed = LaunchConfiguration("max_speed")
    safety_distance = LaunchConfiguration("safety_distance")

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation (Gazebo) clock if true",
    )
    
    declare_map_save_period_cmd = DeclareLaunchArgument(
        "map_save_period",
        default_value="30.0",
        description="Period to save map during exploration (seconds)",
    )
    
    declare_mapping_duration_cmd = DeclareLaunchArgument(
        "mapping_duration",
        default_value="600",
        description="Duration for autonomous mapping (seconds)",
    )
    
    declare_max_speed_cmd = DeclareLaunchArgument(
        "max_speed",
        default_value="0.3",
        description="Maximum speed during exploration (m/s)",
    )
    
    declare_safety_distance_cmd = DeclareLaunchArgument(
        "safety_distance",
        default_value="0.8",
        description="Minimum distance to obstacles (meters)",
    )

    # Configuration files
    nav2_params_file = os.path.join(pkg_tractor_bringup, "config", "nav2_params.yaml")
    slam_params_file = os.path.join(pkg_tractor_bringup, "config", "slam_toolbox_params.yaml")

    # 1. Robot Description and State Publisher
    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tractor_bringup, "launch", "robot_description.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    # 2. Control System (with feedback for better odometry)
    control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tractor_bringup, "launch", "control_with_feedback.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    # 3. Sensor Systems
    sensors_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tractor_bringup, "launch", "sensors.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    # 4. Robot Localization (sensor fusion)
    robot_localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tractor_bringup, "launch", "robot_localization.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    # 5. RealSense D435i with optimized configuration including IMU
    realsense_config_file = os.path.join(pkg_tractor_bringup, "config", "realsense_config.yaml")
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("realsense2_camera"), "launch", "rs_launch.py")
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "config_file": realsense_config_file,
            "camera_name": "camera",
            "camera_namespace": "camera",
            "enable_pointcloud": "true",
            "enable_sync": "true",
            "align_depth.enable": "true",
            "enable_gyro": "true",
            "enable_accel": "true",
            "gyro_fps": "200",
            "accel_fps": "250",
            "unite_imu_method": "linear_interpolation",
            "enable_infra1": "false",
            "enable_infra2": "false",
            "depth_module.depth_profile": "424x240x15",
            "rgb_camera.color_profile": "424x240x15",
            "depth_module.emitter_enabled": "0", # Disabled for outdoor use
            "clip_distance": "8.0",
            "linear_accel_cov": "0.01",
            "angular_velocity_cov": "0.01",
            "hold_back_imu_for_frames": "true",
        }.items(),
    )

    # 5a. Optimized point cloud to laser scan converter
    pointcloud_to_laserscan_node = Node(
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        name="pointcloud_to_laserscan",
        output="screen",
        parameters=[
            {
                "target_frame": "base_link",
                "transform_tolerance": 0.01,
                "min_height": -0.3,
                "max_height": 1.5,
                "angle_min": -1.57,  # -90 degrees
                "angle_max": 1.57,   # +90 degrees
                "angle_increment": 0.0087,  # 0.5 degrees for higher resolution
                "scan_time": 0.06666666666,  # 15 Hz (1.0/15.0)
                "range_min": 0.3,
                "range_max": 8.0,
                "use_inf": True,
                "inf_epsilon": 1.0,
                "concurrency_level": 1,
                "use_sim_time": use_sim_time,
            }
        ],
        remappings=[
            ("cloud_in", "/camera/camera/depth/color/points"),
            ("scan", "/scan"),
        ],
    )

    # 6. SLAM Mapping
    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tractor_bringup, "launch", "slam_mapping.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    # 7. Nav2 Navigation Stack
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_nav2_bringup, "launch", "navigation_launch.py")
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "params_file": nav2_params_file,
        }.items(),
    )

    # 8. Autonomous Mapping Node (delayed start to let everything initialize)
    autonomous_mapper_node = Node(
        package="tractor_bringup",
        executable="autonomous_mapping.py",
        name="autonomous_mapper",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "mapping_duration": mapping_duration,
                "max_speed": max_speed,
                "safety_distance": safety_distance,
                "exploration_radius": 5.0,
                "waypoint_timeout": 45.0,
            }
        ],
    )

    # 9. Map Saver (periodically save the map)
    map_saver_node = Node(
        package="nav2_map_server",
        executable="map_saver_cli",
        name="map_saver",
        output="screen",
        arguments=["-f", "/home/ubuntu/ros2-rover/autonomous_map"],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    # 10. Safety Monitor Node
    safety_monitor_node = Node(
        package="tractor_bringup",
        executable="safety_monitor.py",
        name="safety_monitor",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "max_speed_limit": max_speed,
                "emergency_stop_distance": 0.3,
                "warning_distance": safety_distance,
            }
        ],
    )

    # 11. Foxglove Bridge for visualization
    foxglove_bridge_node = Node(
        package="foxglove_bridge",
        executable="foxglove_bridge",
        name="foxglove_bridge",
        output="screen",
        parameters=[
            {
                "port": 8765,
                "address": "0.0.0.0",
                "tls": False,
                "certfile": "",
                "keyfile": "",
                "topic_whitelist": [
                    "/map",
                    "/tf",
                    "/tf_static", 
                    "/odom",
                    "/odometry/filtered",
                    "/scan",
                    "/camera/camera/depth/color/points",
                    "/camera/camera/color/image_raw",
                    "/camera/camera/color/camera_info",
                    "/camera/camera/depth/image_rect_raw",
                    "/camera/camera/depth/camera_info",
                    "/camera/camera/aligned_depth_to_color/camera_info",
                    "/camera/camera/imu",
                    "/camera/camera/accel/imu_info",
                    "/camera/camera/gyro/imu_info",
                    "/cmd_vel",
                    "/navigate_to_pose/_action/goal",
                    "/navigate_to_pose/_action/feedback",
                    "/mapping_status",
                    "/safety_status",
                    "/emergency_stop",
                    "/robot_description",
                    "/global_costmap/costmap",
                    "/local_costmap/costmap",
                    "/plan",
                    "/local_plan"
                ],
                "use_sim_time": use_sim_time,
            }
        ],
    )

    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_map_save_period_cmd)
    ld.add_action(declare_mapping_duration_cmd)
    ld.add_action(declare_max_speed_cmd)
    ld.add_action(declare_safety_distance_cmd)

    # Add launch files in sequence
    ld.add_action(robot_description_launch)
    ld.add_action(control_launch)
    ld.add_action(sensors_launch)
    ld.add_action(robot_localization_launch)
    
    # Delay RealSense to let other systems initialize
    ld.add_action(TimerAction(period=3.0, actions=[realsense_launch]))
    ld.add_action(TimerAction(period=5.0, actions=[pointcloud_to_laserscan_node]))
    
    # Delay SLAM and Nav2 to let sensors initialize
    ld.add_action(TimerAction(period=5.0, actions=[slam_launch]))
    ld.add_action(TimerAction(period=8.0, actions=[nav2_launch]))
    
    # Add safety monitor early
    ld.add_action(TimerAction(period=2.0, actions=[safety_monitor_node]))
    
    # Start autonomous mapping after everything is ready
    ld.add_action(TimerAction(period=15.0, actions=[autonomous_mapper_node]))
    
    # Start map saver
    ld.add_action(TimerAction(period=20.0, actions=[map_saver_node]))
    
    # Start Foxglove Bridge early for monitoring
    ld.add_action(TimerAction(period=1.0, actions=[foxglove_bridge_node]))

    return ld