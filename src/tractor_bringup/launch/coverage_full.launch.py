#!/usr/bin/env python3
"""Full classical coverage stack: map the room, then cover it.

Brings up Phase A (lidar SLAM + Nav2 + frontier exploration) and the coverage
orchestrator together. By default the orchestrator auto-starts boustrophedon
coverage as soon as frontier exploration reports the map is closed
(``auto_start_on_complete``); set it false to require a manual
``/start_coverage`` trigger (e.g. to inspect the planned path in RViz first).

Run coverage on a previously-built map by launching only this file's
orchestrator portion against a map_server-published ``/map`` and calling
``ros2 service call /start_coverage std_srvs/srv/Trigger``.
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

    use_sim_time = LaunchConfiguration("use_sim_time")
    exploration_radius = LaunchConfiguration("exploration_radius")
    auto_start = LaunchConfiguration("auto_start_on_complete")
    tool_width = LaunchConfiguration("tool_width")

    declares = [
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("exploration_radius", default_value="15.0"),
        DeclareLaunchArgument(
            "auto_start_on_complete",
            default_value="true",
            description="Start coverage automatically when mapping completes",
        ),
        DeclareLaunchArgument(
            "tool_width",
            default_value="0.30",
            description="Coverage swath width (m), ~ robot width",
        ),
    ]

    mapping = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_bringup, "launch", "coverage_mapping.launch.py")
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "exploration_radius": exploration_radius,
        }.items(),
    )

    orchestrator = Node(
        package="tractor_coverage",
        executable="coverage_orchestrator",
        name="coverage_orchestrator",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "tool_width": tool_width,
                "overlap": 0.10,
                "border_offset": 0.25,
                "turn_radius": 0.30,
                "auto_start_on_complete": auto_start,
                "optimize_path": True,
            }
        ],
    )

    ld = LaunchDescription()
    for d in declares:
        ld.add_action(d)
    ld.add_action(mapping)
    # Start the orchestrator after Nav2 is up so navigate_through_poses exists.
    ld.add_action(TimerAction(period=20.0, actions=[orchestrator]))
    return ld
