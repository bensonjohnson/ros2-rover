#!/usr/bin/env python3
"""
RTAB-Map + Frontier-Based Exploration Launch File

This launch file combines the best of both worlds:
- RTAB-Map for RGB-D SLAM mapping (better visual quality than SLAM Toolbox)
- Frontier-based exploration for reliable autonomous navigation
- Nav2 for path planning and obstacle avoidance
- No PPO/RL - just proven classical robotics algorithms

Core components:
- RTAB-Map for visual SLAM and occupancy grid generation
- RealSense D435i for RGB-D sensing + wheel encoders
- Frontier explorer for autonomous area coverage
- Nav2 stack for navigation
- Safety monitors for collision avoidance
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
    exploration_radius = LaunchConfiguration("exploration_radius")

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation clock if true",
    )

    declare_max_speed_cmd = DeclareLaunchArgument(
        "max_speed",
        default_value="0.18",
        description="Maximum exploration speed (m/s)",
    )

    declare_safety_distance_cmd = DeclareLaunchArgument(
        "safety_distance",
        default_value="0.25",
        description="Emergency stop distance (m)",
    )

    declare_exploration_radius_cmd = DeclareLaunchArgument(
        "exploration_radius",
        default_value="10.0",
        description="Maximum exploration radius from start (meters)",
    )

    # 1. Robot Description and TF
    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tractor_bringup, "launch", "robot_description.launch.py")
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    # 2. Motor Control (provides wheel odometry)
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
            ("cmd_vel", "cmd_vel_raw")  # Raw motor commands
        ]
    )

    # 3. RealSense D435i Camera (RGB-D)
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("realsense2_camera"), "launch", "rs_launch.py")
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "pointcloud.enable": "false",
            "align_depth.enable": "true",
            "enable_color": "true",
            "enable_depth": "true",
            "enable_sync": "true",
            "device_type": "435i",
            "depth_module.depth_profile": "424x240x30",
            "rgb_camera.color_profile": "424x240x30",
            "enable_gyro": "false",
            "enable_accel": "false",
            "enable_imu": "false",
        }.items(),
    )

    # 4. IMU (for RTAB-Map)
    lsm9ds1_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("tractor_sensors"), "launch", "lsm9ds1_imu.launch.py")
        )
    )

    # 5. RTAB-Map (RGB-D SLAM + occupancy grid generation)
    rtabmap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("rtabmap_launch"), "launch", "rtabmap.launch.py")
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "frame_id": "base_link",
            "odom_topic": "/odom",
            "subscribe_depth": "true",
            "subscribe_rgb": "true",
            "approx_sync": "true",
            "queue_size": "30",
            "rgb_topic": "/camera/camera/color/image_raw",
            "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
            "camera_info_topic": "/camera/camera/color/camera_info",
            "imu_topic": "/lsm9ds1_imu_publisher/imu/data",
            "rtabmap": "true",
            "rtabmapviz": "false",
            "rviz": "false",
            "visual_odometry": "true",
            "odom_guess_frame_id": "/odom",
            "odom_guess_min_translation": "0.01",
            "odom_guess_min_rotation": "0.05",
            # RTAB-Map args: occupancy grid creation enabled, outdoor optimized
            "args": "--delete_db_on_start --Mem/IncrementalMemory true --subscribe_scan false --subscribe_imu true --RGBD/CreateOccupancyGrid true --Grid/Sensor 1 --Grid/RangeMax 5.0 --Grid/CellSize 0.05 --Vis/MinInliers 8 --Vis/CorType 0 --Odom/Strategy 0 --Odom/GuessMotion true --OdomF2M/MaxSize 1000 --Grid/3D false --Grid/GroundIsObstacle false",
        }.items()
    )

    # 6. Nav2 Lifecycle Manager
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
                    "controller_server",
                    "planner_server",
                    "behavior_server",
                    "bt_navigator",
                ],
                "bond_timeout": 30.0,
                "attempt_respawn_reconnection": True,
                "bond_respawn_max_duration": 10.0,
            }
        ],
    )

    # 7. Nav2 Controller Server
    nav2_controller_node = Node(
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
                    "yaw_goal_tolerance": 0.25,
                },

                "FollowPath": {
                    "plugin": "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController",
                    "desired_linear_vel": max_speed,
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

    # 8. Nav2 Planner Server
    nav2_planner_node = Node(
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

    # 9. Nav2 Behavior Server
    nav2_behavior_server_node = Node(
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
                "global_frame": "map",
                "robot_base_frame": "base_link",
                "transform_tolerance": 0.1,
                "simulate_ahead_time": 2.0,
                "max_rotational_vel": 1.0,
                "min_rotational_vel": 0.4,
                "rotational_acc_lim": 3.2,
            }
        ],
    )

    # 10. Nav2 BT Navigator
    nav2_bt_navigator_node = Node(
        package="nav2_bt_navigator",
        executable="bt_navigator",
        name="bt_navigator",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "global_frame": "map",
                "robot_base_frame": "base_link",
                "odom_topic": "/odom",
                "bt_loop_duration": 10,
                "default_server_timeout": 20,
                "enable_groot_monitoring": True,
                "groot_zmq_publisher_port": 1666,
                "groot_zmq_server_port": 1667,
                "default_nav_to_pose_bt_xml": "/opt/ros/jazzy/share/nav2_bt_navigator/behavior_trees/navigate_to_pose_w_replanning_and_recovery.xml",
                "navigators": ["navigate_to_pose", "navigate_through_poses"],
                "navigate_to_pose": {"plugin": "nav2_bt_navigator::NavigateToPoseNavigator"},
                "navigate_through_poses": {"plugin": "nav2_bt_navigator::NavigateThroughPosesNavigator"},
            }
        ],
    )

    # 11. Costmaps - Local (using depth data directly)
    local_costmap_node = Node(
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
                "robot_radius": 0.15,
                "plugins": ["obstacle_layer", "inflation_layer"],

                "obstacle_layer": {
                    "plugin": "nav2_costmap_2d::ObstacleLayer",
                    "enabled": True,
                    "observation_sources": "depth_camera",
                    "depth_camera": {
                        "data_type": "LaserScan",
                        "topic": "/rtabmap/scan",  # RTAB-Map provides a scan from depth
                        "marking": True,
                        "clearing": True,
                        "obstacle_range": 2.5,
                        "raytrace_max_range": 3.0,
                        "expected_update_rate": 0.0,
                        "observation_persistence": 0.0,
                    },
                },

                "inflation_layer": {
                    "plugin": "nav2_costmap_2d::InflationLayer",
                    "cost_scaling_factor": 3.0,
                    "inflation_radius": 0.55,
                },
            }
        ],
    )

    # 12. Costmaps - Global (using RTAB-Map occupancy grid)
    global_costmap_node = Node(
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
                "robot_radius": 0.15,
                "plugins": ["static_layer", "inflation_layer"],

                "static_layer": {
                    "plugin": "nav2_costmap_2d::StaticLayer",
                    "map_subscribe_transient_local": True,
                    "map_topic": "/rtabmap/grid_map",  # Use RTAB-Map's global occupancy grid
                },

                "inflation_layer": {
                    "plugin": "nav2_costmap_2d::InflationLayer",
                    "cost_scaling_factor": 3.0,
                    "inflation_radius": 0.55,
                },
            }
        ],
    )

    # 13. Frontier-based Explorer (uses Nav2 for navigation)
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
                "exploration_frequency": 1.0,  # Check for new frontiers every second
                "goal_timeout": 30.0,  # Timeout goals after 30 seconds
            }
        ],
        remappings=[
            ("map", "/rtabmap/grid_map"),  # Use RTAB-Map's occupancy grid
        ]
    )

    # 14. Safety Monitor (final velocity limiter)
    safety_monitor_node = Node(
        package="tractor_bringup",
        executable="simple_safety_monitor.py",
        name="safety_monitor",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "max_speed_limit": max_speed,
                "emergency_stop_distance": safety_distance,
                "warning_distance": 0.5,
            }
        ],
        remappings=[
            ("cmd_vel_in", "cmd_vel_nav"),
            ("cmd_vel_out", "cmd_vel_raw"),
        ],
    )

    # 15. Map Saver for saving RTAB-Map's occupancy grid
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
        remappings=[
            ("map", "/rtabmap/grid_map"),  # Save RTAB-Map's occupancy grid
        ]
    )

    # Build launch description with timed startup sequence
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_max_speed_cmd)
    ld.add_action(declare_safety_distance_cmd)
    ld.add_action(declare_exploration_radius_cmd)

    # Core system - immediate start
    ld.add_action(robot_description_launch)
    ld.add_action(hiwonder_motor_node)

    # Sensors - start with small delay
    ld.add_action(TimerAction(period=2.0, actions=[lsm9ds1_launch]))
    ld.add_action(TimerAction(period=5.0, actions=[realsense_launch]))

    # RTAB-Map - start after sensors are ready
    ld.add_action(TimerAction(period=8.0, actions=[rtabmap_launch]))

    # Navigation stack - start after RTAB-Map is running
    ld.add_action(TimerAction(period=12.0, actions=[nav2_controller_node]))
    ld.add_action(TimerAction(period=12.0, actions=[nav2_planner_node]))
    ld.add_action(TimerAction(period=12.0, actions=[nav2_behavior_server_node]))
    ld.add_action(TimerAction(period=12.0, actions=[nav2_bt_navigator_node]))
    ld.add_action(TimerAction(period=12.0, actions=[local_costmap_node]))
    ld.add_action(TimerAction(period=12.0, actions=[global_costmap_node]))

    # Lifecycle manager - activate all nav2 nodes
    ld.add_action(TimerAction(period=15.0, actions=[nav2_lifecycle_manager]))

    # Safety and exploration - start after navigation is ready
    ld.add_action(TimerAction(period=18.0, actions=[safety_monitor_node]))
    ld.add_action(TimerAction(period=20.0, actions=[frontier_explorer_node]))

    # Utilities
    ld.add_action(TimerAction(period=10.0, actions=[map_saver_node]))

    return ld
