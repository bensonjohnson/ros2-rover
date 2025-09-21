#!/usr/bin/env python3
"""Launch RTAB observation builder and PPO manager for offline bag training."""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    declare_obs_topic = DeclareLaunchArgument('observation_topic', default_value='/exploration/observation')
    declare_depth_topic = DeclareLaunchArgument('depth_topic', default_value='/camera/aligned_depth_to_color/image_raw')
    declare_occ_topic = DeclareLaunchArgument('occupancy_topic', default_value='/rtabmap/local_grid_map')
    declare_frontier_topic = DeclareLaunchArgument('frontier_topic', default_value='/rtabmap/frontiers')
    declare_odom_topic = DeclareLaunchArgument('odom_topic', default_value='/odom')
    declare_imu_topic = DeclareLaunchArgument('imu_topic', default_value='/lsm9ds1_imu_publisher/imu/data')
    declare_joint_topic = DeclareLaunchArgument('joint_topic', default_value='joint_states')

    obs_node = Node(
        package='tractor_bringup',
        executable='rtab_observation_node.py',
        name='rtab_observation',
        output='screen',
        parameters=[{
            'depth_topic': LaunchConfiguration('depth_topic'),
            'occupancy_topic': LaunchConfiguration('occupancy_topic'),
            'frontier_topic': LaunchConfiguration('frontier_topic'),
            'odom_topic': LaunchConfiguration('odom_topic'),
            'imu_topic': LaunchConfiguration('imu_topic'),
            'publish_rate_hz': 10.0,
            'enable_sample_logging': False,
        }]
    )

    ppo_node = Node(
        package='tractor_bringup',
        executable='ppo_manager_rtab.py',
        name='ppo_manager_rtab',
        output='screen',
        parameters=[{
            'observation_topic': LaunchConfiguration('observation_topic'),
            'cmd_topic': LaunchConfiguration('cmd_topic'),
            'odom_topic': LaunchConfiguration('odom_topic'),
            'imu_topic': LaunchConfiguration('imu_topic'),
            'joint_topic': LaunchConfiguration('joint_topic'),
            'update_interval_sec': 15.0,
            'rollout_capacity': 4096,
            'minibatch_size': 128,
            'update_epochs': 3,
            'reward_forward_scale': 3.5,
            'reward_emergency_penalty': -5.0,
            'reward_block_penalty': -1.0,
            'reward_coverage_scale': 2.0,
            'max_speed': 0.18,
            'min_forward_threshold': 0.25,
        }]
    )

    ld = LaunchDescription()
    ld.add_action(declare_obs_topic)
    ld.add_action(declare_depth_topic)
    ld.add_action(declare_occ_topic)
    ld.add_action(declare_frontier_topic)
    ld.add_action(declare_odom_topic)
    ld.add_action(declare_imu_topic)
    ld.add_action(declare_joint_topic)
    ld.add_action(obs_node)
    ld.add_action(ppo_node)
    return ld
