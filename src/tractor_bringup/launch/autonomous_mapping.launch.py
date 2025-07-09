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

    # 5. RealSense Camera with Nav2 integration
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tractor_bringup, "launch", "realsense_nav2.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
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
    
    # Delay SLAM and Nav2 to let sensors initialize
    ld.add_action(TimerAction(period=5.0, actions=[slam_launch]))
    ld.add_action(TimerAction(period=8.0, actions=[nav2_launch]))
    
    # Add safety monitor early
    ld.add_action(TimerAction(period=2.0, actions=[safety_monitor_node]))
    
    # Start autonomous mapping after everything is ready
    ld.add_action(TimerAction(period=15.0, actions=[autonomous_mapper_node]))
    
    # Start map saver
    ld.add_action(TimerAction(period=20.0, actions=[map_saver_node]))

    return ld