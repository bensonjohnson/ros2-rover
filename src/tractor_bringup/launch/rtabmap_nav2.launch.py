#!/usr/bin/env python3
"""
RTAB-Map + Nav2 launch for indoor/outdoor RGB-D mapping.

Sensor Fusion:
  - STL-19p LiDAR: 2D laser scans for obstacle detection
  - RF2O: Laser scan matching for drift-free odometry (position + yaw)
  - Wheel encoders: Velocity feedback
  - LSM9DS1 IMU: Angular velocity
  - EKF: Fuses all sensors into /odometry/filtered

RTAB-Map uses RGB-D from RealSense D435i for visual SLAM.
Nav2 uses LiDAR scans for obstacle avoidance and local planning.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition, UnlessCondition
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
    # DISABLE TF publishing to avoid conflict with EKF
    hiwonder_motor_node = Node(
        package="tractor_control",
        executable="hiwonder_motor_driver",
        name="hiwonder_motor_driver",
        output="screen",
        parameters=[
            os.path.join(pkg_tractor_bringup, "config", "hiwonder_motor_params.yaml"),
            {"use_sim_time": use_sim_time, "publish_tf": False},  # Let EKF handle odom -> base_link
        ],
        remappings=[("cmd_vel", "cmd_vel_safe")],
        condition=IfCondition(with_motor),
    )

    # 3) Sensors - GPS removed for indoor mapping
    lsm9ds1_imu_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("tractor_sensors"), "launch", "lsm9ds1_imu.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    # 3b) LiDAR (STL-19p) - provides accurate 2D laser scans
    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("tractor_sensors"), "launch", "stl19p_lidar.launch.py")
        ),
        launch_arguments={
            "port_name": "/dev/ttyUSB0",
            "frame_id": "laser_link",
        }.items(),
    )

    # 3c) LiDAR Odometry (RF2O) - scan matching for drift-free odometry
    rf2o_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("tractor_sensors"), "launch", "lidar_odometry.launch.py")
        ),
        launch_arguments={
            "publish_tf": "false",  # Let EKF handle odom -> base_link
        }.items(),
    )

    # 4) EKF localization (GPS disabled here, but node present for odom fusion)
    robot_localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tractor_bringup, "launch", "robot_localization.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time, "use_gps": "false"}.items(),
    )

    # 5) RealSense - optimized for RTAB-Map performance
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("realsense2_camera"), "launch", "rs_launch.py")
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "camera_name": "camera",
            "camera_namespace": "camera",
            "device_type": "435i",
            "depth_module.depth_profile": "424x240x30",  # Lower resolution for performance
            "rgb_camera.color_profile": "424x240x30",     # Lower resolution for performance
            "enable_color": "true",
            "enable_depth": "true",
            "enable_sync": "true",
            "align_depth.enable": "true",  # Keep alignment for RTAB-Map RGBD sync
            "pointcloud.enable": "false",   # Disable pointcloud (not needed, using LiDAR)
            "initial_reset": "true",
            "enable_gyro": "false",
            "enable_accel": "false",
            "enable_imu": "false",
        }.items(),
    )

    # 6) RTAB-Map RGBD sync
    rgbd_sync_node = Node(
        package="rtabmap_sync",
        executable="rgbd_sync",
        name="rgbd_sync",
        output="screen",
        parameters=[{"approx_sync": True, "approx_sync_max_interval": 0.05, "queue_size": 50, "sync_queue_size": 50, "use_sim_time": use_sim_time}],
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
        condition=UnlessCondition(with_safety),
    )

    nav2_lifecycle_manager_collision = Node(
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
        condition=UnlessCondition(with_safety),
    )

    nav2_lifecycle_manager_no_collision = Node(
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
                ]
            },
            {"bond_timeout": 60.0},
        ],
        condition=IfCondition(with_safety),
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

    # Velocity feedback controller for improved speed accuracy
    vfc_node = Node(
        package="tractor_control",
        executable="velocity_feedback_controller",
        name="velocity_feedback_controller",
        output="screen",
        parameters=[{"control_frequency": 50.0, "use_sim_time": use_sim_time}],
    )

    # LiDAR-based safety monitor (more reliable than depth-based)
    safety_monitor_node = Node(
        package="tractor_bringup",
        executable="lidar_safety_monitor.py",
        name="lidar_safety_monitor",
        output="screen",
        parameters=[
            {
                "scan_topic": "/scan",
                "input_cmd_topic": "cmd_vel_smoothed",
                "output_cmd_topic": "cmd_vel_safe",
                "emergency_stop_distance": 0.25,
                "hard_stop_distance": 0.15,
                "min_valid_range": 0.05,
                "max_eval_distance": 5.0,
                "use_sim_time": use_sim_time,
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

    # Sensors - staggered startup for stability
    ld.add_action(TimerAction(period=2.0, actions=[lidar_launch]))
    ld.add_action(TimerAction(period=3.0, actions=[lsm9ds1_imu_launch]))
    ld.add_action(TimerAction(period=4.0, actions=[rf2o_launch]))

    # EKF - start after sensor data is available
    ld.add_action(TimerAction(period=5.0, actions=[robot_localization_launch]))

    # Camera - start after core sensors
    ld.add_action(TimerAction(period=6.0, actions=[realsense_launch]))

    # RTAB-Map - ensure camera is fully initialized
    ld.add_action(TimerAction(period=10.0, actions=[rgbd_sync_node]))
    ld.add_action(TimerAction(period=12.0, actions=[rtabmap_node]))

    # Nav2 - start after mapping is ready
    nav2_start_time = 15.0
    for t, node in enumerate([
        nav2_controller_node,
        nav2_planner_node,
        nav2_behavior_server_node,
        nav2_bt_navigator_node,
        velocity_smoother_node,
        collision_monitor_node,
    ]):
        ld.add_action(TimerAction(period=nav2_start_time + 0.5 * t, actions=[node]))
    ld.add_action(TimerAction(period=nav2_start_time + 4.0, actions=[nav2_lifecycle_manager_collision]))
    ld.add_action(TimerAction(period=nav2_start_time + 4.0, actions=[nav2_lifecycle_manager_no_collision]))

    # Control and safety - start after Nav2 is ready
    ld.add_action(TimerAction(period=20.0, actions=[vfc_node]))
    ld.add_action(TimerAction(period=21.0, actions=[safety_monitor_node]))
    ld.add_action(TimerAction(period=23.0, actions=[autonomous_mapper_node]))

    # Optional teleop
    xbox_teleop_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tractor_bringup, "launch", "xbox_teleop.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
        condition=IfCondition(with_teleop),
    )
    ld.add_action(TimerAction(period=22.0, actions=[xbox_teleop_launch]))

    # Viz - start early for debugging
    ld.add_action(TimerAction(period=3.0, actions=[foxglove_bridge_node]))

    return ld
