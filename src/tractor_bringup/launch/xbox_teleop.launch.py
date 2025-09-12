#!/usr/bin/env python3
"""
Launch Xbox controller teleop over Bluetooth (joy + teleop_twist_joy).
Publishes Twist to cmd_vel_nav so it flows through smoother+collision monitor.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('tractor_bringup')

    use_sim_time = LaunchConfiguration('use_sim_time')
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='false', description='Use sim time'
    )

    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen',
        parameters=[
            os.path.join(pkg_share, 'config', 'xbox_teleop.yaml'),
            {'use_sim_time': use_sim_time},
        ],
    )

    teleop_node = Node(
        package='teleop_twist_joy',
        executable='teleop_node',
        name='teleop_twist_joy',
        output='screen',
        parameters=[
            os.path.join(pkg_share, 'config', 'xbox_teleop.yaml'),
            {'use_sim_time': use_sim_time},
        ],
        remappings=[('cmd_vel', 'cmd_vel_nav')],
    )

    return LaunchDescription([
        declare_use_sim_time_cmd,
        joy_node,
        teleop_node,
    ])

