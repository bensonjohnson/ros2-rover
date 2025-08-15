#!/usr/bin/env python3
"""
Simplified Autonomous SLAM Mapping Launch File for ROS2 Jazzy
Uses RealSense D435i (RGB-D + IMU) + wheel encoders for complete area mapping

Core components:
- SLAM Toolbox for mapping and localization
- RealSense D435i for visual odometry and obstacle detection
- Hiwonder motor controllers for wheel odometry
- Frontier-based exploration for complete area coverage
- Dynamic obstacle avoidance for people/moving objects
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
    exploration_radius = LaunchConfiguration("exploration_radius")

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation (Gazebo) clock if true",
    )

    declare_max_speed_cmd = DeclareLaunchArgument(
        "max_speed",
        default_value="0.2",
        description="Maximum exploration speed (m/s)",
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

    # 3. RealSense D435i Camera (RGB-D) - Using working config approach
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("realsense2_camera"), "launch", "rs_launch.py")
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "camera_name": "camera",
            "camera_namespace": "camera",
            "config_file": os.path.join(pkg_tractor_bringup, "config", "realsense_config.yaml"),
            "enable_pointcloud": "true",
            "align_depth": "true",
        }.items(),
    )

    # 4. Convert PointCloud to LaserScan for SLAM
    pointcloud_to_laserscan_node = Node(
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        name="pointcloud_to_laserscan",
        output="screen",
        parameters=[
            {
                "target_frame": "base_link",
                "transform_tolerance": 0.01,
                "min_height": 0.1,  # Ignore ground/grass
                "max_height": 2.0,  # Detect fences/walls
                "angle_min": -1.5708,  # -90 degrees
                "angle_max": 1.5708,   # +90 degrees
                "angle_increment": 0.0087,  # ~0.5 degrees
                "scan_time": 0.1,
                "range_min": 0.2,
                "range_max": 10.0,
                "use_inf": True,
                "inf_epsilon": 1.0,
                "use_sim_time": use_sim_time
            }
        ],
        remappings=[
            ("cloud_in", "/camera/depth/color/points"),
            ("scan", "/scan"),
        ],
    )

    # 5. SLAM Toolbox (mapping and localization)
    slam_toolbox_node = Node(
        package="slam_toolbox", 
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                # Frame configuration
                "odom_frame": "odom",
                "map_frame": "map", 
                "base_frame": "base_link",
                "scan_topic": "/scan",
                "mode": "mapping",
                
                # Basic SLAM parameters
                "resolution": 0.05,
                "max_laser_range": 8.0,
                "minimum_time_interval": 0.1,
                "transform_publish_period": 0.02,
                "map_update_interval": 1.0,
                
                # Outdoor mapping optimizations
                "minimum_travel_distance": 0.1,
                "minimum_travel_heading": 0.1,
                "scan_buffer_size": 10,
                "scan_buffer_maximum_scan_distance": 20.0,
                "link_match_minimum_response_fine": 0.1,
                "link_scan_maximum_distance": 1.5,
                
                # Loop closure for large areas
                "do_loop_closing": True,
                "loop_search_maximum_distance": 3.0,
                "loop_match_minimum_chain_size": 10,
                "loop_match_maximum_variance_coarse": 3.0,
                "loop_match_minimum_response_coarse": 0.35,
                "loop_match_minimum_response_fine": 0.45,
                
                # Performance
                "number_of_threads": 4,
                "stack_size_to_use": 40000000,
                "tf_buffer_duration": 30.0,
                
                # Map saving
                "map_file_name": "/home/ubuntu/ros2-rover/maps/autonomous_slam_map",
                "map_start_at_dock": True,
            }
        ],
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
                    "slam_toolbox",
                    "controller_server", 
                    "planner_server",
                    "behavior_server",
                    "bt_navigator",
                    "collision_monitor"
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

    # 11. Costmaps - Local
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
                "robot_radius": 0.1,
                "plugins": ["voxel_layer", "inflation_layer"],
                
                "voxel_layer": {
                    "plugin": "nav2_costmap_2d::VoxelLayer",
                    "enabled": True,
                    "publish_voxel_map": True,
                    "origin_z": 0.0,
                    "z_resolution": 0.05,
                    "z_voxels": 16,
                    "max_obstacle_height": 2.0,
                    "min_obstacle_height": 0.1,
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
                        "min_obstacle_height": 0.1,
                        "expected_update_rate": 0.0,
                        "observation_persistence": 0.0,
                        "inf_is_valid": False,
                        "clearing_use_maximum_range": False,
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

    # 12. Costmaps - Global
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
                "robot_radius": 0.1,
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
                        "min_obstacle_height": 0.1,
                        "expected_update_rate": 0.0,
                        "observation_persistence": 0.0,
                        "inf_is_valid": False,
                        "clearing_use_maximum_range": False,
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

    # 13. Collision Monitor for dynamic obstacles (people)
    collision_monitor_node = Node(
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
                "state_topic": "collision_monitor_state",
                "transform_tolerance": 0.2,
                "source_timeout": 1.0,
                "stop_pub_timeout": 2.0,
                "observation_sources": ["pointcloud"],
                
                "pointcloud": {
                    "type": "pointcloud",
                    "topic": "/camera/depth/color/points",
                    "min_height": 0.1,
                    "max_height": 2.0,
                    "enabled": True,
                },
                
                "polygons": ["emergency_stop", "slowdown_zone"],
                
                "emergency_stop": {
                    "type": "circle",
                    "radius": 0.3,  # 30cm emergency stop
                    "action_type": "stop",
                    "min_points": 4,
                    "visualize": True,
                    "polygon_pub_topic": "emergency_stop_polygon",
                },
                
                "slowdown_zone": {
                    "type": "circle", 
                    "radius": 0.8,  # 80cm slowdown zone
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

    # 14. Safety Monitor (additional layer)
    safety_monitor_node = Node(
        package="tractor_bringup",
        executable="safety_monitor.py", 
        name="safety_monitor",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "max_speed_limit": max_speed,
                "emergency_stop_distance": 0.15,  # 15cm emergency stop
                "warning_distance": 0.5,  # 50cm warning
            }
        ],
        remappings=[
            ("cmd_vel_in", "cmd_vel_safe"),
            ("cmd_vel_out", "cmd_vel_raw"),
        ],
    )

    # 15. Frontier-based Explorer
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
    )

    # 16. Map Saver
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

    # Build launch description
    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_max_speed_cmd)
    ld.add_action(declare_exploration_radius_cmd)

    # Core system - immediate start
    ld.add_action(robot_description_launch)
    ld.add_action(hiwonder_motor_node)

    # Sensors - start with small delay
    ld.add_action(TimerAction(period=2.0, actions=[realsense_launch]))
    ld.add_action(TimerAction(period=5.0, actions=[pointcloud_to_laserscan_node]))

    # SLAM - start after sensors are ready
    ld.add_action(TimerAction(period=8.0, actions=[slam_toolbox_node]))

    # Navigation stack - start after SLAM
    ld.add_action(TimerAction(period=12.0, actions=[nav2_controller_node]))
    ld.add_action(TimerAction(period=12.0, actions=[nav2_planner_node]))
    ld.add_action(TimerAction(period=12.0, actions=[nav2_behavior_server_node]))
    ld.add_action(TimerAction(period=12.0, actions=[nav2_bt_navigator_node]))
    ld.add_action(TimerAction(period=12.0, actions=[local_costmap_node]))
    ld.add_action(TimerAction(period=12.0, actions=[global_costmap_node]))
    
    # Lifecycle manager - activate all nav2 nodes
    ld.add_action(TimerAction(period=15.0, actions=[nav2_lifecycle_manager]))

    # Safety and exploration - start after navigation is ready
    ld.add_action(TimerAction(period=18.0, actions=[collision_monitor_node]))
    ld.add_action(TimerAction(period=20.0, actions=[safety_monitor_node]))
    ld.add_action(TimerAction(period=25.0, actions=[frontier_explorer_node]))

    # Utilities
    ld.add_action(TimerAction(period=10.0, actions=[map_saver_node]))

    return ld