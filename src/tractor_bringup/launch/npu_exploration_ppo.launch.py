#!/usr/bin/env python3
"""
PPO Live-Training Exploration Launch File
 - Reuses BEV processor, safety monitor, velocity controller
 - NPU node runs inference (RKNN); PPO manager runs bounded updates in background
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

    # Args
    declare_max_speed_cmd = DeclareLaunchArgument("max_speed", default_value="0.15")
    declare_safety_distance_cmd = DeclareLaunchArgument("safety_distance", default_value="0.2")
    declare_use_rtab_cmd = DeclareLaunchArgument("use_rtab_observation", default_value="false")

    # Robot description
    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_tractor_bringup, 'launch', 'robot_description.launch.py'))
    )

    # Hiwonder motor control
    hiwonder_motor_node = Node(
        package="tractor_control",
        executable="hiwonder_motor_driver",
        name="hiwonder_motor_driver",
        output="screen",
        parameters=[os.path.join(pkg_tractor_bringup, "config", "hiwonder_motor_params.yaml")]
    )

    use_rtab = LaunchConfiguration("use_rtab_observation")

    # RealSense
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("realsense2_camera"), "launch", "rs_launch.py")
        ),
        launch_arguments={
            "enable_pointcloud": "true",
            "align_depth": "true",
            "enable_color": "false",
            "enable_depth": "true",
            "enable_sync": "false",
            "device_type": "435i",
            "depth_module.depth_profile": "424x240x30",
            "rgb_camera.color_profile": "424x240x30",
            "enable_imu": "false",
            "enable_gyro": "false",
            "enable_accel": "false",
            "decimation_filter.enable": "true",
            "decimation_filter.filter_magnitude": "2",
            "config_file": os.path.join(get_package_share_directory("tractor_bringup"), "config", "realsense_config.yaml"),
        }.items()
    )

    # RTAB-Map bringup (optional)
    rtabmap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("rtabmap_ros"), "launch", "rtabmap.launch.py")
        ),
        condition=IfCondition(use_rtab),
        launch_arguments={
            'use_sim_time': 'false',
            'frame_id': 'base_link',
            'odom_topic': '/odom',
            'subscribe_depth': 'true',
            'subscribe_rgb': 'true',
            'approx_sync': 'true',
            'queue_size': '10',
        }.items()
    )

    # BEV processor (shared BEV)
    bev_processor_node = Node(
        package="tractor_bringup",
        executable="bev_processor_node.py",
        name="bev_processor",
        output="screen",
        parameters=[{
            "pointcloud_topic": "/camera/camera/depth/color/points",
            "bev_image_topic": "/bev/image",
            "publish_rate_hz": 10.0,
            "bev_size": [200, 200],
            "bev_range": [6.0, 4.0],
            "bev_height_channels": [0.2, 1.0],
            "enable_ground_removal": True,
            "bev_ground_update_interval": 20,
            "ground_ransac_iterations": 40,
            "ground_ransac_threshold": 0.06,
            "bev_enable_opencl": True,
            # IMU-assisted ground removal
            "imu_topic": "/lsm9ds1_imu_publisher/imu/data",
            "sensor_height_m": 0.17,
            "imu_ransac_interval_s": 4.0,
            "imu_roll_pitch_threshold_deg": 3.0,
            "min_obstacle_height_m": 0.25,
            "grass_height_tolerance_m": 0.15,
        }],
        condition=UnlessCondition(use_rtab)
    )

    # RTAB observation builder
    rtab_observation_node = Node(
        package="tractor_bringup",
        executable="rtab_observation_node.py",
        name="rtab_observation",
        output="screen",
        parameters=[{
            "depth_topic": "/camera/aligned_depth_to_color/image_raw",
            "occupancy_topic": "/rtabmap/local_grid_map",
            "frontier_topic": "/rtabmap/frontiers",
            "odom_topic": "/odom",
            "imu_topic": "/lsm9ds1_imu_publisher/imu/data",
            "publish_rate_hz": 10.0,
            "occupancy_window_m": 12.0,
            "output_resolution": [128, 128],
            "depth_clip_m": 6.0,
        }],
        condition=IfCondition(use_rtab)
    )

    # Safety monitor variants
    safety_monitor_node = Node(
        package="tractor_bringup",
        executable="simple_safety_monitor_bev.py",
        name="simple_safety_monitor",
        output="screen",
        parameters=[{
            "emergency_stop_distance": LaunchConfiguration("safety_distance"),
            "pointcloud_topic": "/camera/camera/depth/color/points",
            "input_cmd_topic": "cmd_vel_ai",
            "output_cmd_topic": "cmd_vel_raw",
            "use_shared_bev": True,
            "bev_image_topic": "/bev/image",
            "bev_freshness_timeout": 0.5,
            "bev_x_range": 6.0,
            "forward_min_distance": 0.05,
        }],
        condition=UnlessCondition(use_rtab)
    )

    safety_monitor_rtab = Node(
        package="tractor_bringup",
        executable="simple_safety_monitor_rtab.py",
        name="simple_safety_monitor_rtab",
        output="screen",
        parameters=[{
            "occupancy_topic": "/rtabmap/local_grid_map",
            "input_cmd_topic": "cmd_vel_ai",
            "output_cmd_topic": "cmd_vel_raw",
            "emergency_stop_distance": LaunchConfiguration("safety_distance"),
            "hard_stop_distance": 0.12,
            "forward_width_m": 1.2,
        }],
        condition=IfCondition(use_rtab)
    )

    # Velocity feedback controller
    vfc_node = Node(
        package="tractor_control",
        executable="velocity_feedback_controller",
        name="velocity_feedback_controller",
        output="screen",
        parameters=[{"control_frequency": 50.0}]
    )

    # NPU exploration (inference-only)
    npu_node = Node(
        package="tractor_bringup",
        executable="npu_exploration_bev.py",
        name="npu_exploration",
        output="screen",
        parameters=[{
            "max_speed": LaunchConfiguration("max_speed"),
            "safety_distance": LaunchConfiguration("safety_distance"),
            "operation_mode": "hybrid",
            "npu_inference_rate": 10.0,
            "enable_training_optimization": False,
            "enable_reward_optimization": False,
            "enable_multi_metric_evaluation": False,
            "enable_optimization_monitoring": False,
            "enable_reward_based_distill": False,
            "encoder_freeze_step": 0,
            # IMU proprio feed for runtime actor
            "enable_lsm_imu_proprio": True,
            "lsm_imu_topic": "/lsm9ds1_imu_publisher/imu/data",
            "enable_ground_removal": True,
            "bev_size": [200, 200],
            "bev_range": [6.0, 4.0],
            "bev_height_channels": [0.2, 1.0],
            "ground_ransac_iterations": 40,
            "ground_ransac_threshold": 0.06,
        }],
        remappings=[
            ("point_cloud", "/camera/camera/depth/color/points"),
            ("odom", "/odom"),
        ],
        condition=UnlessCondition(use_rtab)
    )

    npu_rtab_node = Node(
        package="tractor_bringup",
        executable="npu_exploration_rtab.py",
        name="npu_exploration_rtab",
        output="screen",
        parameters=[{
            "max_speed": LaunchConfiguration("max_speed"),
            "safety_distance": LaunchConfiguration("safety_distance"),
            "observation_topic": "/exploration/observation",
        }],
        condition=IfCondition(use_rtab)
    )

    # PPO Managers
    ppo_node = Node(
        package="tractor_bringup",
        executable="ppo_manager_node.py",
        name="ppo_manager",
        output="screen",
        parameters=[{
            "bev_image_topic": "/bev/image",
            "update_interval_sec": 25.0,
            "min_export_interval_sec": 120.0,
            "rollout_capacity": 2048,
            "minibatch_size": 64,
            "update_epochs": 2,
            "ppo_clip": 0.2,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "bev_channels": 4,
            "bev_size": [200, 200],
            "proprio_dim": 21,
            "imu_topic": "/lsm9ds1_imu_publisher/imu/data",
            "reward_forward_scale": 5.0,
            "reward_block_penalty": -1.0,
            "reward_emergency_penalty": -5.0,
            "encoder_freeze_step": 5,
            "validation_margin": 0.05,
            "low_activity_linear_threshold": 0.03,
            "low_activity_angular_threshold": 0.1,
            "export_wait_timeout_sec": 15.0,
            "rknn_drift_tolerance": 0.15,
            "min_effective_linear": 0.03,
            "min_effective_angular": 0.05,
            "small_action_penalty": -0.3,
            "small_action_patience": 3,
        }],
        condition=UnlessCondition(use_rtab)
    )

    ppo_rtab_node = Node(
        package="tractor_bringup",
        executable="ppo_manager_rtab.py",
        name="ppo_manager_rtab",
        output="screen",
        parameters=[{
            "observation_topic": "/exploration/observation",
            "cmd_topic": "cmd_vel_ai",
            "update_interval_sec": 20.0,
            "rollout_capacity": 4096,
            "minibatch_size": 128,
            "update_epochs": 3,
            "reward_forward_scale": 4.0,
            "reward_emergency_penalty": -5.0,
            "reward_block_penalty": -1.0,
            "reward_coverage_scale": 2.0,
            "max_speed": LaunchConfiguration("max_speed"),
            "min_forward_threshold": LaunchConfiguration("safety_distance"),
        }],
        condition=IfCondition(use_rtab)
    )

    # IMU (LSM9DS1)
    lsm9ds1_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("tractor_sensors"), "launch", "lsm9ds1_imu.launch.py")
        )
    )

    ld = LaunchDescription()
    ld.add_action(declare_max_speed_cmd)
    ld.add_action(declare_safety_distance_cmd)
    ld.add_action(declare_use_rtab_cmd)

    # Bringup order
    ld.add_action(robot_description_launch)
    ld.add_action(hiwonder_motor_node)
    ld.add_action(TimerAction(period=2.0, actions=[realsense_launch]))
    ld.add_action(TimerAction(period=3.5, actions=[rtabmap_launch]))
    ld.add_action(TimerAction(period=4.0, actions=[bev_processor_node]))
    ld.add_action(TimerAction(period=4.0, actions=[rtab_observation_node]))
    ld.add_action(TimerAction(period=5.0, actions=[lsm9ds1_launch]))
    ld.add_action(TimerAction(period=6.0, actions=[safety_monitor_node]))
    ld.add_action(TimerAction(period=6.0, actions=[safety_monitor_rtab]))
    ld.add_action(TimerAction(period=7.0, actions=[vfc_node]))
    ld.add_action(TimerAction(period=8.0, actions=[npu_node]))
    ld.add_action(TimerAction(period=8.0, actions=[npu_rtab_node]))
    ld.add_action(TimerAction(period=9.0, actions=[ppo_node]))
    ld.add_action(TimerAction(period=9.0, actions=[ppo_rtab_node]))

    return ld
