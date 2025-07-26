#!/usr/bin/env python3
"""
Minimal Autonomous Mapping Launch File with SLAM support
Clean architecture: Essential nodes only, no duplicates, optimized for performance
"""

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

    # Launch arguments
    use_sim_time = LaunchConfiguration("use_sim_time")
    max_speed = LaunchConfiguration("max_speed")
    safety_distance = LaunchConfiguration("safety_distance")
    mapping_duration = LaunchConfiguration("mapping_duration")

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation (Gazebo) clock if true",
    )

    declare_max_speed_cmd = DeclareLaunchArgument(
        "max_speed",
        default_value="0.3",
        description="Maximum speed during exploration (m/s)",
    )

    declare_safety_distance_cmd = DeclareLaunchArgument(
        "safety_distance",
        default_value="0.5",
        description="Minimum distance to obstacles (meters)",
    )

    declare_mapping_duration_cmd = DeclareLaunchArgument(
        "mapping_duration",
        default_value="600",
        description="Mapping duration in seconds",
    )

    # 1. Robot Description (URDF/TF)
    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tractor_bringup, "launch", "robot_description.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    # 2. Motor Control ONLY (no duplicates)
    hiwonder_motor_node = Node(
        package="tractor_control",
        executable="hiwonder_motor_driver",
        name="hiwonder_motor_driver",
        output="screen",
        parameters=[
            os.path.join(pkg_tractor_bringup, "config", "hiwonder_motor_params.yaml"),
            {"use_sim_time": use_sim_time}
        ],
        remappings=[
            ("cmd_vel", "cmd_vel_safe")
        ]
    )

    # 3. GPS/Compass DISABLED - causes timing issues and noise
    # gps_compass_node = Node(
    #     package="tractor_sensors",
    #     executable="hglrc_m100_5883",
    #     name="hglrc_m100_5883",
    #     output="screen",
    #     parameters=[
    #         {
    #             "use_sim_time": use_sim_time,
    #             "gps_port": "/dev/ttyS6",
    #             "gps_baudrate": 115200,
    #             "i2c_bus": 5,
    #             "qmc5883_address": 0x0D,
    #             "compass_update_rate": 50.0,
    #         }
    #     ],
    # )

    # 4. Odometry will be published by motor driver - no static transform needed

    # 5. RealSense Camera (minimal but functional config)
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("realsense2_camera"), "launch", "rs_launch.py")
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "camera_name": "camera",
            "camera_namespace": "camera",
            "config_file": os.path.join(pkg_tractor_bringup, "config", "realsense_config.yaml"),
            "enable_pointcloud": "true",  # Ensure point cloud is enabled
            "align_depth": "true",  # Align depth for better accuracy
        }.items(),
    )

    # 6. PointCloud to LaserScan for SLAM
    pointcloud_to_laserscan_node = Node(
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        name="pointcloud_to_laserscan",
        output="screen",
        parameters=[
            os.path.join(pkg_tractor_bringup, "config", "pointcloud_to_laserscan.yaml"),
            {"use_sim_time": use_sim_time}
        ],
        remappings=[
            ("cloud_in", "/camera/camera/depth/color/points"),
            ("scan", "/scan"),  # For SLAM only
        ],
    )

    # 7. SLAM Toolbox for mapping and localization
    slam_toolbox_node = Node(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
        parameters=[
            os.path.join(pkg_tractor_bringup, "config", "slam_toolbox_params.yaml"),
            {"use_sim_time": use_sim_time}
        ],
        remappings=[
            ("scan", "/scan"),
        ],
    )

    # 8. Nav2 Lifecycle Manager (essential for activating nodes)
    nav2_lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_navigation",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"autostart": True},
            {"node_names": ["controller_server", "planner_server", "behavior_server", "bt_navigator", "velocity_smoother"]}
        ],
    )

    # 9. Basic Nav2 Navigation (minimal components)
    nav2_controller_node = Node(
        package="nav2_controller",
        executable="controller_server",
        name="controller_server",
        output="screen",
        parameters=[
            os.path.join(pkg_tractor_bringup, "config", "nav2_params.yaml"),
            {"use_sim_time": use_sim_time}
        ],
        remappings=[
            ("cmd_vel", "cmd_vel_nav")
        ]
    )

    nav2_planner_node = Node(
        package="nav2_planner",
        executable="planner_server",
        name="planner_server",
        output="screen",
        parameters=[
            os.path.join(pkg_tractor_bringup, "config", "nav2_params.yaml"),
            {"use_sim_time": use_sim_time}
        ],
    )

    nav2_behavior_server_node = Node(
        package="nav2_behaviors",
        executable="behavior_server",
        name="behavior_server",
        output="screen",
        parameters=[
            os.path.join(pkg_tractor_bringup, "config", "nav2_params.yaml"),
            {"use_sim_time": use_sim_time}
        ],
    )

    nav2_bt_navigator_node = Node(
        package="nav2_bt_navigator",
        executable="bt_navigator",
        name="bt_navigator",
        output="screen",
        parameters=[
            os.path.join(pkg_tractor_bringup, "config", "nav2_params.yaml"),
            {"use_sim_time": use_sim_time}
        ],
    )

    # 10. Velocity Smoother (optional but helpful)
    velocity_smoother_node = Node(
        package="nav2_velocity_smoother",
        executable="velocity_smoother",
        name="velocity_smoother",
        output="screen",
        parameters=[
            os.path.join(pkg_tractor_bringup, "config", "nav2_params.yaml"),
            {"use_sim_time": use_sim_time}
        ],
        remappings=[
            ("cmd_vel", "cmd_vel_nav"),
            ("cmd_vel_smoothed", "cmd_vel_smoothed")
        ]
    )

    collision_monitor_node = Node(
        package="nav2_collision_monitor",
        executable="collision_monitor",
        name="collision_monitor",
        output="screen",
        parameters=[
            os.path.join(pkg_tractor_bringup, "config", "nav2_params.yaml"),
            {"use_sim_time": use_sim_time}
        ],
        remappings=[
            ("cmd_vel_in", "cmd_vel_smoothed"),
            ("cmd_vel_out", "cmd_vel_safe")
        ]
    )

    # Costmaps are handled internally by Nav2 navigation stack

    # 11. Simple Autonomous Mapper (forward exploration with Nav2)
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
                "exploration_distance": 2.5,  # Reduced for better obstacle avoidance
                "obstacle_avoidance_distance": 0.6,  # Safety margin for obstacle detection
            }
        ],
    )

    # 12. Safety Monitor (essential)
    safety_monitor_node = Node(
        package="tractor_bringup",
        executable="safety_monitor.py",
        name="safety_monitor",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "max_speed_limit": max_speed,
                "emergency_stop_distance": 0.025,  # 1 inch - tank can zero turn
                "warning_distance": 0.15,  # 6 inches - close warning zone
            }
        ],
    )

    # 13. Map Saver (save maps periodically)
    map_saver_node = Node(
        package="nav2_map_server",
        executable="map_saver_server",
        name="map_saver_server",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time}
        ],
    )

    # 14. Foxglove Bridge (visualization)
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
                "topic_whitelist": [".*"],
                "use_sim_time": use_sim_time,
            }
        ],
    )

    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_max_speed_cmd)
    ld.add_action(declare_safety_distance_cmd)
    ld.add_action(declare_mapping_duration_cmd)

    # Core system (immediate start)
    ld.add_action(robot_description_launch)
    ld.add_action(hiwonder_motor_node)
    # ld.add_action(gps_compass_node)  # DISABLED

    # Camera system (delayed for stability)
    ld.add_action(TimerAction(period=3.0, actions=[realsense_launch]))

    # SLAM system (after camera is ready)
    ld.add_action(TimerAction(period=6.0, actions=[pointcloud_to_laserscan_node]))
    ld.add_action(TimerAction(period=7.0, actions=[slam_toolbox_node]))

    # Navigation only - Start lifecycle manager first, then nodes
    ld.add_action(TimerAction(period=9.0, actions=[nav2_lifecycle_manager]))
    ld.add_action(TimerAction(period=9.5, actions=[nav2_controller_node]))
    ld.add_action(TimerAction(period=10.0, actions=[nav2_planner_node]))
    ld.add_action(TimerAction(period=10.2, actions=[nav2_behavior_server_node]))
    ld.add_action(TimerAction(period=10.5, actions=[nav2_bt_navigator_node]))
    ld.add_action(TimerAction(period=11.0, actions=[velocity_smoother_node]))
    ld.add_action(TimerAction(period=11.2, actions=[collision_monitor_node]))

    # Add Nav2 activation commands
    ld.add_action(TimerAction(period=12.0, actions=[
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="nav2_activation",
            output="screen",
            parameters=[
                {"use_sim_time": use_sim_time},
                {"node_names": ["controller_server", "planner_server", "behavior_server", "bt_navigator", "velocity_smoother"]}
            ],
            arguments=["activate"]
        )
    ]))

    # Safety and exploration (after Nav2 is ready)
    ld.add_action(TimerAction(period=13.0, actions=[safety_monitor_node]))
    ld.add_action(TimerAction(period=12.0, actions=[autonomous_mapper_node]))

    # Utilities
    ld.add_action(TimerAction(period=8.0, actions=[map_saver_node]))
    ld.add_action(TimerAction(period=2.0, actions=[foxglove_bridge_node]))

    return ld
