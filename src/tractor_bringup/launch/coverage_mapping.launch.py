#!/usr/bin/env python3
"""Lidar-driven SLAM mapping + Nav2 + frontier exploration for coverage.

This is the Phase A bring-up for the classical coverage pipeline. It differs
from ``autonomous_slam_mapping.launch.py`` in the key sensor wiring:

  * The **RPLidar/LD19** (360 deg) drives ``/scan`` -> slam_toolbox and rf2o.
    A 360 deg scan closes a full-room map far better than the depth camera's
    ~87 deg front arc (the old launch derived /scan from the depth cloud).
  * The **RealSense D435i** point cloud feeds *only* the Nav2 costmap obstacle
    layers + collision monitor, so obstacles above/below the lidar plane are
    still avoided.

Odometry/TF chain: wheel odom (hiwonder) + laser odom (rf2o) + IMU (lsm9ds1)
are fused by robot_localization's EKF, which owns ``odom -> base_footprint``.
slam_toolbox owns ``map -> odom``. robot_state_publisher owns the rest from URDF.

Cmd_vel safety chain:
  Nav2 controller -> cmd_vel_nav -> collision_monitor (depth) -> cmd_vel_safe
  -> lidar_safety_monitor (lidar) -> cmd_vel_raw -> hiwonder motor driver.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_bringup = get_package_share_directory("tractor_bringup")
    pkg_sensors = get_package_share_directory("tractor_sensors")

    use_sim_time = LaunchConfiguration("use_sim_time")
    max_speed = LaunchConfiguration("max_speed")
    exploration_radius = LaunchConfiguration("exploration_radius")

    declare_use_sim_time = DeclareLaunchArgument("use_sim_time", default_value="false")
    declare_max_speed = DeclareLaunchArgument(
        "max_speed", default_value="0.2", description="Max exploration speed (m/s)"
    )
    declare_exploration_radius = DeclareLaunchArgument(
        "exploration_radius",
        default_value="15.0",
        description="Max exploration radius from start (m). Make >= room size.",
    )

    def inc(pkg_share, rel, **launch_args):
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(pkg_share, "launch", rel)),
            launch_arguments=launch_args.items(),
        )

    # --- 1. Robot description / TF (URDF owns all static sensor extrinsics) ---
    robot_description = inc(
        pkg_bringup, "robot_description.launch.py", use_sim_time=use_sim_time
    )

    # --- 2. Wheel odometry + motor control ---
    hiwonder = Node(
        package="tractor_control",
        executable="hiwonder_motor_driver",
        name="hiwonder_motor_driver",
        output="screen",
        parameters=[
            os.path.join(pkg_bringup, "config", "hiwonder_motor_params.yaml"),
            {"use_sim_time": use_sim_time},
        ],
        remappings=[("cmd_vel", "cmd_vel_raw")],
    )

    # --- 3. 360 deg LiDAR -> /scan (drives SLAM + rf2o) ---
    lidar = inc(pkg_sensors, "stl19p_lidar.launch.py")

    # --- 4. IMU -> /imu/data ---
    imu = inc(pkg_sensors, "lsm9ds1_imu.launch.py")

    # --- 5. Laser odometry (rf2o) -> /odom_rf2o (EKF position source) ---
    rf2o = inc(pkg_sensors, "lidar_odometry.launch.py", publish_tf="false")

    # --- 6. EKF: fuse wheel + laser + IMU -> owns odom->base_footprint TF ---
    ekf = inc(pkg_bringup, "robot_localization.launch.py", use_gps="false")

    # --- 7. RealSense D435i: depth cloud for costmaps / collision (NOT SLAM) ---
    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("realsense2_camera"),
                "launch",
                "rs_launch.py",
            )
        ),
        launch_arguments={
            "camera_name": "camera",
            "camera_namespace": "camera",
            "config_file": os.path.join(
                pkg_bringup, "config", "realsense_config.yaml"
            ),
            "enable_pointcloud": "true",
            "align_depth": "true",
        }.items(),
    )

    # --- 8. SLAM Toolbox (mapping) fed by the 360 deg lidar /scan ---
    slam = Node(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
        parameters=[
            os.path.join(pkg_bringup, "config", "slam_toolbox_params.yaml"),
            {
                "use_sim_time": use_sim_time,
                "mode": "mapping",
                "scan_topic": "/scan",
                "map_frame": "map",
                "odom_frame": "odom",
                "base_frame": "base_link",
            },
        ],
    )

    # --- 9. Nav2 controller (RPP), planner, behaviors, BT navigator ---
    controller = Node(
        package="nav2_controller",
        executable="controller_server",
        name="controller_server",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "controller_frequency": 20.0,
                "min_x_velocity_threshold": 0.001,
                "min_y_velocity_threshold": 0.001,
                "min_theta_velocity_threshold": 0.001,
                "failure_tolerance": 0.3,
                "progress_checker_plugins": ["progress_checker"],
                "goal_checker_plugins": ["general_goal_checker"],
                "controller_plugins": ["FollowPath"],
                "progress_checker": {
                    "plugin": "nav2_controller::SimpleProgressChecker",
                    "required_movement_radius": 0.5,
                    "movement_time_allowance": 10.0,
                },
                "general_goal_checker": {
                    "stateful": True,
                    "plugin": "nav2_controller::SimpleGoalChecker",
                    "xy_goal_tolerance": 0.25,
                    "yaw_goal_tolerance": 0.5,
                },
                "FollowPath": {
                    "plugin": "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController",
                    "desired_linear_vel": 0.2,
                    "lookahead_dist": 0.6,
                    "min_lookahead_dist": 0.3,
                    "max_lookahead_dist": 0.9,
                    "lookahead_time": 1.5,
                    "rotate_to_heading_angular_vel": 1.8,
                    "transform_tolerance": 0.1,
                    "use_velocity_scaled_lookahead_dist": False,
                    "min_approach_linear_velocity": 0.05,
                    "approach_velocity_scaling_dist": 0.6,
                    "use_collision_detection": True,
                    "max_allowed_time_to_collision_up_to_carrot": 1.0,
                    "use_regulated_linear_velocity_scaling": True,
                    "use_cost_regulated_linear_velocity_scaling": False,
                    "regulated_linear_scaling_min_radius": 0.9,
                    "regulated_linear_scaling_min_speed": 0.05,
                    "use_rotate_to_heading": True,
                    "allow_reversing": False,
                    "rotate_to_heading_min_angle": 0.785,
                    "max_angular_accel": 3.2,
                    "max_robot_pose_search_dist": 10.0,
                },
            }
        ],
        remappings=[("cmd_vel", "cmd_vel_nav")],
    )

    planner = Node(
        package="nav2_planner",
        executable="planner_server",
        name="planner_server",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "expected_planner_frequency": 20.0,
                "planner_plugins": ["GridBased"],
                "GridBased": {
                    "plugin": "nav2_navfn_planner::NavfnPlanner",
                    "tolerance": 0.5,
                    "use_astar": False,
                    "allow_unknown": True,
                },
            }
        ],
    )

    behaviors = Node(
        package="nav2_behaviors",
        executable="behavior_server",
        name="behavior_server",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "costmap_topic": "local_costmap/costmap_raw",
                "footprint_topic": "local_costmap/published_footprint",
                "cycle_frequency": 10.0,
                "behavior_plugins": ["spin", "backup", "drive_on_heading", "wait"],
                "spin": {"plugin": "nav2_behaviors::Spin"},
                "backup": {"plugin": "nav2_behaviors::BackUp"},
                "drive_on_heading": {"plugin": "nav2_behaviors::DriveOnHeading"},
                "wait": {"plugin": "nav2_behaviors::Wait"},
                "global_frame": "odom",
                "robot_base_frame": "base_link",
                "transform_tolerance": 0.1,
                "simulate_ahead_time": 2.0,
                "max_rotational_vel": 1.0,
                "min_rotational_vel": 0.4,
                "rotational_acc_lim": 3.2,
            }
        ],
    )

    bt_navigator = Node(
        package="nav2_bt_navigator",
        executable="bt_navigator",
        name="bt_navigator",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "global_frame": "map",
                "robot_base_frame": "base_link",
                "odom_topic": "/odometry/filtered",
                "bt_loop_duration": 10,
                "default_server_timeout": 20,
                "navigators": ["navigate_to_pose", "navigate_through_poses"],
                "navigate_to_pose": {
                    "plugin": "nav2_bt_navigator::NavigateToPoseNavigator"
                },
                "navigate_through_poses": {
                    "plugin": "nav2_bt_navigator::NavigateThroughPosesNavigator"
                },
            }
        ],
    )

    # --- 10. Costmaps: static layer from /map, obstacle layers from depth cloud ---
    local_costmap = Node(
        package="nav2_costmap_2d",
        executable="nav2_costmap_2d",
        name="local_costmap",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "update_frequency": 5.0,
                "publish_frequency": 2.0,
                "global_frame": "odom",
                "robot_base_frame": "base_link",
                "rolling_window": True,
                "width": 3,
                "height": 3,
                "resolution": 0.05,
                "robot_radius": 0.18,
                "plugins": ["voxel_layer", "inflation_layer"],
                "voxel_layer": {
                    "plugin": "nav2_costmap_2d::VoxelLayer",
                    "enabled": True,
                    "publish_voxel_map": True,
                    "origin_z": 0.0,
                    "z_resolution": 0.05,
                    "z_voxels": 16,
                    "max_obstacle_height": 2.0,
                    "min_obstacle_height": 0.05,
                    "unknown_threshold": 15,
                    "mark_threshold": 0,
                    "observation_sources": "pointcloud",
                    "pointcloud": {
                        "data_type": "PointCloud2",
                        "topic": "/camera/depth/color/points",
                        "marking": True,
                        "clearing": True,
                        "obstacle_range": 2.5,
                        "raytrace_max_range": 3.0,
                        "max_obstacle_height": 2.0,
                        "min_obstacle_height": 0.05,
                        "inf_is_valid": False,
                    },
                },
                "inflation_layer": {
                    "plugin": "nav2_costmap_2d::InflationLayer",
                    "cost_scaling_factor": 3.0,
                    "inflation_radius": 0.45,
                },
            }
        ],
    )

    global_costmap = Node(
        package="nav2_costmap_2d",
        executable="nav2_costmap_2d",
        name="global_costmap",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "update_frequency": 1.0,
                "publish_frequency": 1.0,
                "global_frame": "map",
                "robot_base_frame": "base_link",
                "width": 50,
                "height": 50,
                "resolution": 0.05,
                "robot_radius": 0.18,
                "track_unknown_space": True,
                "plugins": ["static_layer", "obstacle_layer", "inflation_layer"],
                "static_layer": {
                    "plugin": "nav2_costmap_2d::StaticLayer",
                    "map_subscribe_transient_local": True,
                },
                "obstacle_layer": {
                    "plugin": "nav2_costmap_2d::ObstacleLayer",
                    "enabled": True,
                    "observation_sources": "pointcloud",
                    "pointcloud": {
                        "data_type": "PointCloud2",
                        "topic": "/camera/depth/color/points",
                        "marking": True,
                        "clearing": True,
                        "obstacle_range": 2.5,
                        "raytrace_max_range": 3.0,
                        "max_obstacle_height": 2.0,
                        "min_obstacle_height": 0.05,
                        "inf_is_valid": False,
                    },
                },
                "inflation_layer": {
                    "plugin": "nav2_costmap_2d::InflationLayer",
                    "cost_scaling_factor": 3.0,
                    "inflation_radius": 0.45,
                },
            }
        ],
    )

    lifecycle_manager = Node(
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
                    "collision_monitor",
                ],
                "bond_timeout": 30.0,
                "attempt_respawn_reconnection": True,
                "bond_respawn_max_duration": 10.0,
            }
        ],
    )

    # --- 11. Safety: depth collision monitor then lidar safety monitor ---
    collision_monitor = Node(
        package="nav2_collision_monitor",
        executable="collision_monitor",
        name="collision_monitor",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "base_frame_id": "base_link",
                "odom_frame_id": "odom",
                "cmd_vel_in_topic": "cmd_vel_nav",
                "cmd_vel_out_topic": "cmd_vel_safe",
                "transform_tolerance": 0.2,
                "source_timeout": 1.0,
                "stop_pub_timeout": 2.0,
                "observation_sources": ["pointcloud"],
                "pointcloud": {
                    "type": "pointcloud",
                    "topic": "/camera/depth/color/points",
                    "min_height": 0.05,
                    "max_height": 2.0,
                    "enabled": True,
                },
                "polygons": ["emergency_stop", "slowdown_zone"],
                "emergency_stop": {
                    "type": "circle",
                    "radius": 0.3,
                    "action_type": "stop",
                    "min_points": 4,
                    "visualize": True,
                    "polygon_pub_topic": "emergency_stop_polygon",
                },
                "slowdown_zone": {
                    "type": "circle",
                    "radius": 0.8,
                    "action_type": "slowdown",
                    "slowdown_ratio": 0.3,
                    "min_points": 3,
                    "visualize": True,
                    "polygon_pub_topic": "slowdown_polygon",
                },
            }
        ],
        remappings=[
            ("cmd_vel_in", "cmd_vel_nav"),
            ("cmd_vel_out", "cmd_vel_safe"),
        ],
    )

    lidar_safety = Node(
        package="tractor_bringup",
        executable="lidar_safety_monitor.py",
        name="lidar_safety_monitor",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "scan_topic": "/scan",
                "input_cmd_topic": "cmd_vel_safe",
                "output_cmd_topic": "cmd_vel_raw",
                "stop_distance": 0.20,
                "slow_distance": 0.40,
            }
        ],
    )

    # --- 12. Frontier exploration: drives to frontiers until map is closed ---
    frontier_explorer = Node(
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
    )

    ld = LaunchDescription()
    for a in (declare_use_sim_time, declare_max_speed, declare_exploration_radius):
        ld.add_action(a)

    # Core + sensors first.
    ld.add_action(robot_description)
    ld.add_action(hiwonder)
    ld.add_action(lidar)
    ld.add_action(imu)
    ld.add_action(TimerAction(period=2.0, actions=[realsense]))
    ld.add_action(TimerAction(period=3.0, actions=[rf2o, ekf]))

    # SLAM after odom/TF is flowing.
    ld.add_action(TimerAction(period=8.0, actions=[slam]))

    # Nav2 after SLAM.
    nav2_nodes = [
        controller,
        planner,
        behaviors,
        bt_navigator,
        local_costmap,
        global_costmap,
    ]
    ld.add_action(TimerAction(period=12.0, actions=nav2_nodes))
    ld.add_action(TimerAction(period=15.0, actions=[lifecycle_manager]))

    # Safety + exploration last.
    ld.add_action(TimerAction(period=18.0, actions=[collision_monitor, lidar_safety]))
    ld.add_action(TimerAction(period=25.0, actions=[frontier_explorer]))

    return ld
