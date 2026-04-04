#!/usr/bin/env python3
"""
Classical Autonomous Launch (Tier 1)

Single source of truth for the classical Nav2 + SLAM Toolbox + RF2O + EKF stack
used for agricultural / outdoor autonomous mapping and exploration.

Key differences from autonomous_slam_mapping.launch.py (kept as fallback):
  - Loads config/nav2_params.yaml for every Nav2 node (no inline duplicates).
  - Uses the real STL-19p LiDAR instead of pointcloud_to_laserscan.
  - Fuses wheel odometry + RF2O laser odometry + LSM9DS1 IMU via robot_localization EKF.
  - Motor driver runs with publish_tf:=false; EKF owns the odom -> base_footprint transform.
  - SLAM Toolbox runs under the Nav2 lifecycle manager (consistent with the yaml).
  - Adds nav2_velocity_smoother in the cmd_vel chain.
  - Drops the duplicate safety_monitor.py; collision_monitor is the sole safety authority.

cmd_vel chain:
  controller_server -> cmd_vel_nav
    -> velocity_smoother -> cmd_vel_smoothed
    -> collision_monitor -> cmd_vel_safe
    -> hiwonder_motor_driver (remapped from cmd_vel)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_tractor_bringup = get_package_share_directory("tractor_bringup")
    pkg_tractor_sensors = get_package_share_directory("tractor_sensors")

    nav2_params_file = os.path.join(pkg_tractor_bringup, "config", "nav2_params.yaml")
    slam_params_file = os.path.join(pkg_tractor_bringup, "config", "slam_toolbox_params.yaml")
    motor_params_file = os.path.join(pkg_tractor_bringup, "config", "hiwonder_motor_params.yaml")

    # ------------------------------------------------------------------
    # Launch arguments
    # ------------------------------------------------------------------
    use_sim_time = LaunchConfiguration("use_sim_time")
    exploration_radius = LaunchConfiguration("exploration_radius")
    with_frontier_explorer = LaunchConfiguration("with_frontier_explorer")
    with_realsense = LaunchConfiguration("with_realsense")

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation clock if true",
    )
    declare_exploration_radius_cmd = DeclareLaunchArgument(
        "exploration_radius",
        default_value="10.0",
        description="Maximum exploration radius from start (meters)",
    )
    declare_with_frontier_explorer_cmd = DeclareLaunchArgument(
        "with_frontier_explorer",
        default_value="false",
        description="Enable the frontier explorer (autonomous goal selection)",
    )
    declare_with_realsense_cmd = DeclareLaunchArgument(
        "with_realsense",
        default_value="false",
        description="Also launch RealSense D435i (off by default for Tier 1)",
    )

    # ------------------------------------------------------------------
    # 1. Robot description (URDF -> TF static)
    # ------------------------------------------------------------------
    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tractor_bringup, "launch", "robot_description.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    # ------------------------------------------------------------------
    # 2. Hiwonder motor driver (publishes /odom topic, TF disabled)
    # ------------------------------------------------------------------
    hiwonder_motor_node = Node(
        package="tractor_control",
        executable="hiwonder_motor_driver",
        name="hiwonder_motor_driver",
        output="screen",
        parameters=[
            motor_params_file,
            {
                "use_sim_time": use_sim_time,
                # CRITICAL: EKF owns odom -> base_footprint. Motor driver must NOT publish TF.
                "publish_tf": False,
            },
        ],
        remappings=[
            # Motor driver subscribes to cmd_vel (i.e. /cmd_vel_safe after collision_monitor).
            ("cmd_vel", "cmd_vel_safe"),
        ],
    )

    # ------------------------------------------------------------------
    # 3. Sensors: STL-19p LiDAR, LSM9DS1 IMU, (optional) RealSense
    # ------------------------------------------------------------------
    stl19p_lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tractor_sensors, "launch", "stl19p_lidar.launch.py")
        ),
        launch_arguments={
            "port_name": "/dev/ttyUSB0",
            "frame_id": "laser_link",
        }.items(),
    )

    lsm9ds1_imu_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tractor_sensors, "launch", "lsm9ds1_imu.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    # Optional RealSense - off by default. For Tier 1 the LiDAR is authoritative.
    # Kept available so the depth cloud can be added as a secondary obstacle source later.
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("realsense2_camera"),
                "launch",
                "rs_launch.py",
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "camera_name": "camera",
            "camera_namespace": "camera",
            "config_file": os.path.join(pkg_tractor_bringup, "config", "realsense_config.yaml"),
            "enable_pointcloud": "true",
            "align_depth": "true",
        }.items(),
        condition=IfCondition(with_realsense),
    )

    # ------------------------------------------------------------------
    # 4. RF2O laser odometry (no TF publish - EKF owns it)
    # ------------------------------------------------------------------
    rf2o_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tractor_sensors, "launch", "lidar_odometry.launch.py")
        ),
        launch_arguments={
            "publish_tf": "false",
        }.items(),
    )

    # ------------------------------------------------------------------
    # 5. robot_localization EKF (fuses /odom + /odom_rf2o + /imu/data,
    #    publishes odom -> base_footprint to match URDF static joint).
    # ------------------------------------------------------------------
    robot_localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tractor_bringup, "launch", "robot_localization.launch.py")
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "use_gps": "false",
        }.items(),
    )

    # ------------------------------------------------------------------
    # 6. SLAM Toolbox (lifecycle-managed, loads yaml)
    # ------------------------------------------------------------------
    slam_toolbox_node = Node(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
        parameters=[
            slam_params_file,
            {"use_sim_time": use_sim_time},
        ],
    )

    # ------------------------------------------------------------------
    # 7. Nav2 stack (all nodes load nav2_params.yaml - no inline params)
    # ------------------------------------------------------------------
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

    # velocity_smoother: cmd_vel_nav -> cmd_vel_smoothed
    velocity_smoother_node = Node(
        package="nav2_velocity_smoother",
        executable="velocity_smoother",
        name="velocity_smoother",
        output="screen",
        parameters=[nav2_params_file, {"use_sim_time": use_sim_time}],
        remappings=[
            ("cmd_vel", "cmd_vel_nav"),
            ("cmd_vel_smoothed", "cmd_vel_smoothed"),
        ],
    )

    # collision_monitor: cmd_vel_smoothed -> cmd_vel_safe (sole safety authority)
    collision_monitor_node = Node(
        package="nav2_collision_monitor",
        executable="collision_monitor",
        name="collision_monitor",
        output="screen",
        parameters=[nav2_params_file, {"use_sim_time": use_sim_time}],
        remappings=[
            ("cmd_vel_in", "cmd_vel_smoothed"),
            ("cmd_vel_out", "cmd_vel_safe"),
        ],
    )

    # Global + local costmaps run as part of controller_server / planner_server in Jazzy;
    # they're configured via nav2_params.yaml. No separate nav2_costmap_2d nodes needed.

    # ------------------------------------------------------------------
    # 8. Nav2 lifecycle manager (includes slam_toolbox + velocity_smoother)
    # ------------------------------------------------------------------
    nav2_lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_navigation",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "autostart": True,
                "node_names": [
                    "slam_toolbox",
                    "controller_server",
                    "planner_server",
                    "behavior_server",
                    "bt_navigator",
                    "velocity_smoother",
                    "collision_monitor",
                ],
                "bond_timeout": 60.0,
                "attempt_respawn_reconnection": True,
                "bond_respawn_max_duration": 10.0,
            }
        ],
    )

    # ------------------------------------------------------------------
    # 9. Frontier explorer (optional; disabled by default)
    # ------------------------------------------------------------------
    frontier_explorer_node = Node(
        package="tractor_bringup",
        executable="frontier_explorer.py",
        name="frontier_explorer",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "max_exploration_radius": exploration_radius,
                "frontier_search_radius": 2.0,
                "min_frontier_size": 10,
                "exploration_frequency": 1.0,
                "goal_timeout": 30.0,
            }
        ],
        condition=IfCondition(with_frontier_explorer),
    )

    # ------------------------------------------------------------------
    # 10. Map saver
    # ------------------------------------------------------------------
    map_saver_node = Node(
        package="nav2_map_server",
        executable="map_saver_server",
        name="map_saver_server",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "save_map_timeout": 5.0,
                "free_thresh_default": 0.25,
                "occupied_thresh_default": 0.65,
                "map_subscribe_transient_local": True,
            }
        ],
    )

    # ------------------------------------------------------------------
    # Assemble launch description with a staggered start-up so TF and
    # sensors are ready before Nav2 looks them up.
    # ------------------------------------------------------------------
    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_exploration_radius_cmd)
    ld.add_action(declare_with_frontier_explorer_cmd)
    ld.add_action(declare_with_realsense_cmd)

    # t=0: URDF + motor driver
    ld.add_action(robot_description_launch)
    ld.add_action(hiwonder_motor_node)

    # t=1: sensors (LiDAR, IMU, optional RealSense)
    ld.add_action(TimerAction(period=1.0, actions=[stl19p_lidar_launch]))
    ld.add_action(TimerAction(period=1.0, actions=[lsm9ds1_imu_launch]))
    ld.add_action(TimerAction(period=2.0, actions=[realsense_launch]))

    # t=3: RF2O (needs LiDAR running first)
    ld.add_action(TimerAction(period=3.0, actions=[rf2o_launch]))

    # t=4: EKF (needs /odom, /odom_rf2o, /imu/data)
    ld.add_action(TimerAction(period=4.0, actions=[robot_localization_launch]))

    # t=6: SLAM Toolbox (needs odom->base TF from EKF)
    ld.add_action(TimerAction(period=6.0, actions=[slam_toolbox_node]))

    # t=8: Nav2 servers
    ld.add_action(TimerAction(period=8.0, actions=[nav2_controller_node]))
    ld.add_action(TimerAction(period=8.0, actions=[nav2_planner_node]))
    ld.add_action(TimerAction(period=8.0, actions=[nav2_behavior_server_node]))
    ld.add_action(TimerAction(period=8.0, actions=[nav2_bt_navigator_node]))
    ld.add_action(TimerAction(period=8.0, actions=[velocity_smoother_node]))
    ld.add_action(TimerAction(period=8.0, actions=[collision_monitor_node]))

    # t=12: lifecycle manager (activates SLAM + Nav2 together)
    ld.add_action(TimerAction(period=12.0, actions=[nav2_lifecycle_manager]))

    # t=15: map saver + optional frontier explorer
    ld.add_action(TimerAction(period=15.0, actions=[map_saver_node]))
    ld.add_action(TimerAction(period=20.0, actions=[frontier_explorer_node]))

    return ld
