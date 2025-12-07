#!/usr/bin/env python3
"""
LiDAR-Based Frontier Exploration Launch File

Uses:
- LD19 LiDAR for mapping and odometry (rf2o)
- RealSense D435i for 3D obstacle detection
- LSM9DS1 IMU for sensor fusion
- robot_localization EKF (rf2o + IMU)
- SLAM Toolbox for mapping
- Nav2 for navigation
- Wavefront Frontier Exploration for autonomous exploration
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
    pkg_tractor_bringup = get_package_share_directory('tractor_bringup')
    pkg_tractor_sensors = get_package_share_directory('tractor_sensors')

    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    max_speed = LaunchConfiguration('max_speed')
    exploration_radius = LaunchConfiguration('exploration_radius')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock if true'
    )

    declare_max_speed_cmd = DeclareLaunchArgument(
        'max_speed',
        default_value='0.25',
        description='Maximum speed during exploration (m/s)'
    )

    declare_exploration_radius_cmd = DeclareLaunchArgument(
        'exploration_radius',
        default_value='3.0',
        description='Exploration radius in meters'
    )

    # 1. Robot Description (URDF/TF)
    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tractor_bringup, 'launch', 'robot_description.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    # 2. LD19 LiDAR
    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tractor_sensors, 'launch', 'stl19p_lidar.launch.py')
        )
    )

    # 3. RF2O Laser Odometry (replaces wheel encoders)
    lidar_odometry_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tractor_sensors, 'launch', 'lidar_odometry.launch.py')
        ),
        launch_arguments={'publish_tf': 'false'}.items()  # Let robot_localization publish TF
    )

    # 4. LSM9DS1 IMU
    imu_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tractor_sensors, 'launch', 'lsm9ds1_imu.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    # 5. Robot Localization EKF (fuses rf2o + IMU)
    robot_localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_tractor_bringup, 'launch', 'robot_localization.launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'use_gps': 'false'  # Disable GPS for indoor exploration
        }.items()
    )

    # 6. RealSense D435i Camera (3D obstacle detection)
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'camera_name': 'camera',
            'camera_namespace': 'camera',
            'config_file': os.path.join(pkg_tractor_bringup, 'config', 'realsense_config.yaml'),
            'enable_pointcloud': 'true',
            'align_depth': 'true',
            'enable_color': 'true',
            'enable_depth': 'true'
        }.items()
    )

    # 7. PointCloud to LaserScan (merge with LiDAR for better obstacle detection)
    pointcloud_to_laserscan_node = Node(
        package='pointcloud_to_laserscan',
        executable='pointcloud_to_laserscan_node',
        name='pointcloud_to_laserscan',
        output='screen',
        parameters=[
            os.path.join(pkg_tractor_bringup, 'config', 'pointcloud_to_laserscan.yaml'),
            {'use_sim_time': use_sim_time}
        ],
        remappings=[
            ('cloud_in', '/camera/depth/color/points'),
            ('scan', '/scan_depth'),  # Separate topic for depth scan
        ]
    )

    # 8. SLAM Toolbox for mapping
    slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[
            os.path.join(pkg_tractor_bringup, 'config', 'slam_toolbox_params.yaml'),
            {'use_sim_time': use_sim_time}
        ],
        remappings=[
            ('scan', '/scan'),  # Use LiDAR scan for SLAM
        ]
    )

    # 9. Nav2 Controller
    nav2_controller_node = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[
            os.path.join(pkg_tractor_bringup, 'config', 'nav2_params.yaml'),
            {'use_sim_time': use_sim_time}
        ],
        remappings=[
            ('cmd_vel', 'cmd_vel_nav')
        ]
    )

    # 10. Nav2 Planner
    nav2_planner_node = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[
            os.path.join(pkg_tractor_bringup, 'config', 'nav2_params.yaml'),
            {'use_sim_time': use_sim_time}
        ]
    )

    # 11. Nav2 Behavior Server
    nav2_behavior_server_node = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        output='screen',
        parameters=[
            os.path.join(pkg_tractor_bringup, 'config', 'nav2_params.yaml'),
            {'use_sim_time': use_sim_time}
        ]
    )

    # 12. Nav2 BT Navigator
    nav2_bt_navigator_node = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[
            os.path.join(pkg_tractor_bringup, 'config', 'nav2_params.yaml'),
            {'use_sim_time': use_sim_time}
        ]
    )

    # 13. Velocity Smoother
    velocity_smoother_node = Node(
        package='nav2_velocity_smoother',
        executable='velocity_smoother',
        name='velocity_smoother',
        output='screen',
        parameters=[
            os.path.join(pkg_tractor_bringup, 'config', 'nav2_params.yaml'),
            {'use_sim_time': use_sim_time}
        ],
        remappings=[
            ('cmd_vel', 'cmd_vel_nav'),
            ('cmd_vel_smoothed', 'cmd_vel_smoothed')
        ]
    )

    # 14. Collision Monitor
    collision_monitor_node = Node(
        package='nav2_collision_monitor',
        executable='collision_monitor',
        name='collision_monitor',
        output='screen',
        parameters=[
            os.path.join(pkg_tractor_bringup, 'config', 'nav2_params.yaml'),
            {'use_sim_time': use_sim_time}
        ],
        remappings=[
            ('cmd_vel_in', 'cmd_vel_smoothed'),
            ('cmd_vel_out', 'cmd_vel_safe')
        ]
    )

    # 15. Motor Controller
    hiwonder_motor_node = Node(
        package='tractor_control',
        executable='hiwonder_motor_driver',
        name='hiwonder_motor_driver',
        output='screen',
        parameters=[
            os.path.join(pkg_tractor_bringup, 'config', 'hiwonder_motor_params.yaml'),
            {'use_sim_time': use_sim_time}
        ],
        remappings=[
            ('cmd_vel', 'cmd_vel_safe')
        ]
    )

    # 16. Nav2 Lifecycle Manager
    nav2_lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'autostart': True},
            {'node_names': [
                'slam_toolbox',
                'controller_server',
                'planner_server',
                'behavior_server',
                'bt_navigator',
                'velocity_smoother',
                'collision_monitor'
            ]},
            {'bond_timeout': 60.0},
            {'attempt_respawn_reconnection': True}
        ]
    )

    # 17. Wavefront Frontier Explorer
    frontier_explorer_node = Node(
        package='nav2_wfd',
        executable='explore',
        name='explore',
        output='screen',
        parameters=[
            {
                'use_sim_time': use_sim_time,
                'explore_clear_radius': exploration_radius,
                'frequency': 1.0,  # Hz
                'robot_radius': 0.15,  # meters (half of rover width)
                'min_frontier_size': 10  # minimum frontier points
            }
        ]
    )

    # 18. Foxglove Bridge (visualization)
    foxglove_bridge_node = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        output='screen',
        parameters=[
            {
                'port': 8765,
                'address': '0.0.0.0',
                'tls': False,
                'topic_whitelist': ['.*'],
                'use_sim_time': use_sim_time
            }
        ]
    )

    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_max_speed_cmd)
    ld.add_action(declare_exploration_radius_cmd)

    # Launch sequence with delays for stability
    ld.add_action(robot_description_launch)

    # Sensors (staggered start for stability)
    ld.add_action(TimerAction(period=1.0, actions=[lidar_launch]))
    ld.add_action(TimerAction(period=2.0, actions=[lidar_odometry_launch]))
    ld.add_action(TimerAction(period=2.5, actions=[imu_launch]))
    ld.add_action(TimerAction(period=3.0, actions=[robot_localization_launch]))
    ld.add_action(TimerAction(period=4.0, actions=[realsense_launch]))
    ld.add_action(TimerAction(period=8.0, actions=[pointcloud_to_laserscan_node]))

    # SLAM
    ld.add_action(TimerAction(period=10.0, actions=[slam_toolbox_node]))

    # Nav2 stack
    ld.add_action(TimerAction(period=12.0, actions=[nav2_controller_node]))
    ld.add_action(TimerAction(period=12.0, actions=[nav2_planner_node]))
    ld.add_action(TimerAction(period=12.0, actions=[nav2_behavior_server_node]))
    ld.add_action(TimerAction(period=12.0, actions=[nav2_bt_navigator_node]))
    ld.add_action(TimerAction(period=12.0, actions=[velocity_smoother_node]))
    ld.add_action(TimerAction(period=12.0, actions=[collision_monitor_node]))
    ld.add_action(TimerAction(period=14.0, actions=[nav2_lifecycle_manager]))

    # Motor control and exploration
    ld.add_action(hiwonder_motor_node)
    ld.add_action(TimerAction(period=18.0, actions=[frontier_explorer_node]))

    # Visualization
    ld.add_action(TimerAction(period=2.0, actions=[foxglove_bridge_node]))

    return ld
