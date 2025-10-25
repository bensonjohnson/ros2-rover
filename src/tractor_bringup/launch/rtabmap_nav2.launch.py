#!/usr/bin/env python3
"""
RTAB-Map + Nav2 launch for outdoor RGB-D mapping without GPS.
Uses EKF (/odometry/filtered) as odom prior and publishes map->odom TF.
Keeps Nav2 obstacle layers on RealSense pointcloud.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_tractor_bringup = get_package_share_directory("tractor_bringup")

    # Launch args
    use_sim_time = LaunchConfiguration("use_sim_time")
    max_speed = LaunchConfiguration("max_speed")
    mapping_duration = LaunchConfiguration("mapping_duration")
    nav2_params_file = LaunchConfiguration("nav2_params_file")
    with_teleop = LaunchConfiguration("with_teleop")
    with_autonomous_mapper = LaunchConfiguration("with_autonomous_mapper")
    with_motor = LaunchConfiguration("with_motor")
    with_rtabmap = LaunchConfiguration("with_rtabmap")
    with_safety = LaunchConfiguration("with_safety")

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        "use_sim_time", default_value="false", description="Use sim time"
    )
    declare_max_speed_cmd = DeclareLaunchArgument(
        "max_speed", default_value="0.25", description="Max linear speed"
    )
    declare_mapping_duration_cmd = DeclareLaunchArgument(
        "mapping_duration", default_value="900", description="Seconds to map"
    )
    declare_nav2_params_cmd = DeclareLaunchArgument(
        "nav2_params_file",
        default_value=os.path.join(pkg_tractor_bringup, "config", "nav2_params.yaml"),
        description="Nav2 params (mapping mode: odom global frame)",
    )
    declare_with_teleop_cmd = DeclareLaunchArgument(
        "with_teleop", default_value="false", description="Start Xbox teleop"
    )
    declare_with_autonomous_mapper_cmd = DeclareLaunchArgument(
        "with_autonomous_mapper", default_value="true", description="Start autonomous mapper"
    )
    declare_with_motor_cmd = DeclareLaunchArgument(
        "with_motor", default_value="true", description="Start motor driver"
    )
    declare_with_rtabmap_cmd = DeclareLaunchArgument(
        "with_rtabmap", default_value="true", description="Start RTAB-Map nodes"
    )
    declare_with_safety_cmd = DeclareLaunchArgument(
        "with_safety", default_value="false", description="Start safety monitor"
    )

    # 1) Robot description
    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tractor_bringup, "launch", "robot_description.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    # 2) Motor driver
    hiwonder_motor_node = Node(
        package="tractor_control",
        executable="hiwonder_motor_driver",
        name="hiwonder_motor_driver",
        output="screen",
        parameters=[
            os.path.join(pkg_tractor_bringup, "config", "hiwonder_motor_params.yaml"),
            {"use_sim_time": use_sim_time},
        ],
        remappings=[("cmd_vel", "cmd_vel_safe")],
        condition=IfCondition(with_motor),
    )

    # 3) Sensors
    lc29h_gps_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("tractor_sensors"), "launch", "lc29h_rtk.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    lsm9ds1_imu_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("tractor_sensors"), "launch", "lsm9ds1_imu.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    # 4) EKF localization (GPS disabled here, but node present for odom fusion)
    robot_localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tractor_bringup, "launch", "robot_localization.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time, "use_gps": "false"}.items(),
    )

    # 5) RealSense
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("realsense2_camera"), "launch", "rs_launch.py")
        ),
        launch_arguments={
            # Keep name+namespace as "camera" to match existing topics (/camera/camera/...)
            "use_sim_time": use_sim_time,
            "camera_name": "camera",
            "camera_namespace": "camera",
            # Core enablements - enable pointcloud but disable sync
            "enable_pointcloud": "true",
            "align_depth": "true",
            "device_type": "435i",
            "enable_color": "true",
            "enable_depth": "true",
            "enable_sync": "true",
            # Profiles (reduced bandwidth)
            "depth_module.depth_profile": "320x240x6",
            "rgb_camera.color_profile": "320x240x6",
            # Disable camera IMU to save bandwidth (we use LSM9DS1 IMU)
            "enable_imu": "false",
            "enable_gyro": "false",
            "enable_accel": "false",
            # Filters
            "decimation_filter.enable": "true",
            "decimation_filter.filter_magnitude": "3",
            "spatial_filter.enable": "true",
            "temporal_filter.enable": "true",
            "hole_filling_filter.enable": "true",
            # Timeout protection
            "wait_for_device_timeout": "10.0",
            "reconnect_timeout": "5.0",
        }.items(),
    )

    # 6) Pointcloud to laserscan converter (since pointcloud is disabled)
    pointcloud_to_laserscan_node = Node(
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        name="pointcloud_to_laserscan",
        output="screen",
        parameters=[
            os.path.join(pkg_tractor_bringup, "config", "pointcloud_to_laserscan.yaml"),
            {"use_sim_time": use_sim_time},
        ],
        remappings=[
            ("cloud_in", "/camera/camera/depth/color/points"),
            ("scan", "/scan"),
        ],
    )

    # 7) RTAB-Map RGBD sync
    rgbd_sync_node = Node(
        package="rtabmap_sync",
        executable="rgbd_sync",
        name="rgbd_sync",
        output="screen",
        parameters=[{"approx_sync": True, "queue_size": 50, "sync_queue_size": 50, "use_sim_time": use_sim_time}],
        remappings=[
            ("rgb/image", "/camera/camera/color/image_raw"),
            ("depth/image", "/camera/camera/aligned_depth_to_color/image_raw"),
            ("rgb/camera_info", "/camera/camera/color/camera_info"),
        ],
        condition=IfCondition(with_rtabmap),
    )

    # 7) RTAB-Map core
    rtabmap_node = Node(
        package="rtabmap_slam",
        executable="rtabmap",
        name="rtabmap",
        output="screen",
        parameters=[
            os.path.join(pkg_tractor_bringup, "config", "rtabmap_params.yaml"),
            {"use_sim_time": use_sim_time},
        ],
        remappings=[
            ("rgbd_image", "rgbd_image"),
            ("imu", "/imu/data"),
        ],
        condition=IfCondition(with_rtabmap),
    )

    # 8) Nav2 stack (mapping mode: odom global frame)
    nav2_controller_node = Node(
        package="nav2_controller",
        executable="controller_server",
        name="controller_server",
        output="screen",
        parameters=[nav2_params_file, {"use_sim_time": use_sim_time}],
        remappings=[("cmd_vel", "cmd_vel_nav")],
    )

    nav2_planner_node = Node(
        package="nav2_planner",
        executable="planner_server",
        name="planner_server",
        output="screen",
        parameters=[nav2_params_file, {"use_sim_time": use_sim_time}],
    )

    nav2_behavior_server_node = Node(
        package="nav2_behaviors",
        executable="behavior_server",
        name="behavior_server",
        output="screen",
        parameters=[nav2_params_file, {"use_sim_time": use_sim_time}],
    )

    nav2_bt_navigator_node = Node(
        package="nav2_bt_navigator",
        executable="bt_navigator",
        name="bt_navigator",
        output="screen",
        parameters=[nav2_params_file, {"use_sim_time": use_sim_time}],
    )

    velocity_smoother_node = Node(
        package="nav2_velocity_smoother",
        executable="velocity_smoother",
        name="velocity_smoother",
        output="screen",
        parameters=[nav2_params_file, {"use_sim_time": use_sim_time}],
        remappings=[("cmd_vel", "cmd_vel_nav"), ("cmd_vel_smoothed", "cmd_vel_smoothed")],
    )

    collision_monitor_node = Node(
        package="nav2_collision_monitor",
        executable="collision_monitor",
        name="collision_monitor",
        output="screen",
        parameters=[nav2_params_file, {"use_sim_time": use_sim_time}],
        remappings=[("cmd_vel_in", "cmd_vel_smoothed"), ("cmd_vel_out", "cmd_vel_safe")],
    )

    nav2_lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_navigation",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"autostart": True},
            {
                "node_names": [
                    "controller_server",
                    "planner_server",
                    "behavior_server",
                    "bt_navigator",
                    "velocity_smoother",
                    "collision_monitor",
                ]
            },
            {"bond_timeout": 60.0},
        ],
    )

    # 9) Simple autonomous mapper and safety monitor (reuse your existing nodes)
    autonomous_mapper_node = Node(
        package="tractor_bringup",
        executable="autonomous_mapping.py",
        name="simple_autonomous_mapper",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "max_speed": max_speed,
                "mapping_duration": mapping_duration,
                "exploration_distance": 2.5,
                "obstacle_avoidance_distance": 0.6,
            }
        ],
        condition=IfCondition(with_autonomous_mapper),
    )

    safety_monitor_node = Node(
        package="tractor_bringup",
        executable="simple_safety_monitor.py",
        name="safety_monitor",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "max_speed_limit": max_speed,
                "emergency_stop_distance": 0.25,
                "hard_stop_distance": 0.08,
                "warning_distance": 0.20,
                "pointcloud_topic": "/camera/camera/depth/color/points",
                "input_cmd_topic": "cmd_vel_nav",
                "output_cmd_topic": "cmd_vel_safe",
            }
        ],
        condition=IfCondition(with_safety),
    )

    foxglove_bridge_node = Node(
        package="foxglove_bridge",
        executable="foxglove_bridge",
        name="foxglove_bridge",
        output="screen",
        parameters=[{"port": 8765, "address": "0.0.0.0", "tls": False, "topic_whitelist": [".*"], "use_sim_time": use_sim_time}],
    )

    ld = LaunchDescription()

    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_max_speed_cmd)
    ld.add_action(declare_mapping_duration_cmd)
    ld.add_action(declare_nav2_params_cmd)
    ld.add_action(declare_with_teleop_cmd)
    ld.add_action(declare_with_autonomous_mapper_cmd)
    ld.add_action(declare_with_motor_cmd)
    ld.add_action(declare_with_rtabmap_cmd)
    ld.add_action(declare_with_safety_cmd)

    # Core
    ld.add_action(robot_description_launch)
    ld.add_action(hiwonder_motor_node)

    # Sensors and EKF
    ld.add_action(TimerAction(period=1.0, actions=[lsm9ds1_imu_launch]))
    ld.add_action(TimerAction(period=2.0, actions=[robot_localization_launch]))
    # GPS launcher is present but not essential; start late if available
    ld.add_action(TimerAction(period=20.0, actions=[lc29h_gps_launch]))

    # Camera
    ld.add_action(TimerAction(period=3.0, actions=[realsense_launch]))
    ld.add_action(TimerAction(period=5.0, actions=[pointcloud_to_laserscan_node]))

    # RTAB-Map
    ld.add_action(TimerAction(period=7.0, actions=[rgbd_sync_node]))
    ld.add_action(TimerAction(period=8.0, actions=[rtabmap_node]))

    # Nav2
    for t, node in enumerate([
        nav2_controller_node,
        nav2_planner_node,
        nav2_behavior_server_node,
        nav2_bt_navigator_node,
        velocity_smoother_node,
        collision_monitor_node,
    ]):
        ld.add_action(TimerAction(period=10.0 + 0.1 * t, actions=[node]))
    ld.add_action(TimerAction(period=12.0, actions=[nav2_lifecycle_manager]))

    # Safety + exploration
    ld.add_action(TimerAction(period=15.0, actions=[safety_monitor_node]))
    ld.add_action(TimerAction(period=18.0, actions=[autonomous_mapper_node]))

    # Optional teleop
    xbox_teleop_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tractor_bringup, "launch", "xbox_teleop.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
        condition=IfCondition(with_teleop),
    )
    ld.add_action(TimerAction(period=10.0, actions=[xbox_teleop_launch]))

    # Viz
    ld.add_action(TimerAction(period=4.0, actions=[foxglove_bridge_node]))

    return ld
