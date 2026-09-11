#!/usr/bin/env python3
"""Spatial explorer bringup — local spatial memory + frontier seeking.

    lidar -> /scan ─┬─> rf2o laser odometry ─┐
                    │                        ├─> EKF -> /odometry/filtered
    wheel odom, IMU ┴────────────────────────┘            │
                                                          v
                    /scan + pose -> spatial_explorer_runner (the brain)
                                          │
                                   /track_cmd_ai
                                          │  lidar_safety_monitor gates it
                                          v
                                    /track_cmd -> hiwonder motor driver

Unlike pc_active_inference.launch.py, this NEEDS odometry: the memory is a
metric occupancy grid in the odom frame, so the EKF chain (wheel + IMU + rf2o)
is load-bearing rather than optional. Drift is tolerated by design — the target
is always a nearby frontier reached in seconds, and the memory resets on an
odom jump.

slam_toolbox is optional (``slam:=true``) and purely for a human-viewable map;
the explorer does not consume it. Nav2 is deliberately absent — the explorer
does its own planning on its own local grid.

``dry_run:=true`` runs everything but routes the command to a dead topic, so
the stack can be verified on hardware without the rover moving.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    pkg_bringup = get_package_share_directory("tractor_bringup")
    pkg_sensors = get_package_share_directory("tractor_sensors")

    args = [
        DeclareLaunchArgument(
            "driver", default_value="frontier",
            description="'frontier' = local occupancy grid + BFS to the "
                        "nearest frontier (validated: ~1.40 rooms/house vs "
                        "1.00 and zero doorway crossings for reactive). "
                        "'pnn' = the distilled map-policy research path; no "
                        "checkpoint passes check_policy yet."),
        DeclareLaunchArgument(
            "policy_path",
            default_value=os.path.expanduser("~/.ros/spatial_policy.pt"),
            description="driver=pnn only. Must PASS "
                        "`python3 -m pnn_sim.spatial.check_policy <ckpt>` — a "
                        "policy that ignores the map still 'explores' well "
                        "enough to fool a coverage score."),
        DeclareLaunchArgument(
            "action_scale", default_value="1.0",
            description="Scales track output. 1.0 matches the envelope the "
                        "policy was distilled under; lowering it slows the "
                        "rover below what the pursuit law expects, and speed "
                        "is already the binding constraint on coverage. The "
                        "safety monitor — not this — is what bounds risk."),
        DeclareLaunchArgument("control_rate_hz", default_value="15.0"),
        DeclareLaunchArgument(
            "lidar_port",
            default_value="/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_"
                          "UART_Bridge_Controller_0001-if00-port0",
            description="The LD19, addressed by its STABLE by-id path. Do not "
                        "use /dev/ttyUSB0: the number depends on USB "
                        "enumeration order and shifts when other devices (the "
                        "GPS) are plugged or unplugged. /dev/ldlidar is no "
                        "better — ldlidar.rules matches ANY CP210x, and this "
                        "rover has two, so the symlink lands on whichever "
                        "enumerated first. Verified: this port streams 47-byte "
                        "LD19 frames (0x54 0x2C)."),
        DeclareLaunchArgument(
            "imu_type", default_value="bno085",
            description="'bno085' (on-chip fusion, better EKF heading) or "
                        "'lsm9ds1'"),
        DeclareLaunchArgument(
            "slam", default_value="false",
            description="Also run slam_toolbox for a human-viewable map"),
        DeclareLaunchArgument(
            "dry_run", default_value="false",
            description="Publish the brain's command to a DEAD topic instead "
                        "of /track_cmd_ai, so the whole stack (lidar, EKF, "
                        "spatial memory, planner) runs and can be inspected on "
                        "/pnn/spatial_diag while the motors stay idle."),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
    ]
    use_sim_time = LaunchConfiguration("use_sim_time")

    def inc(pkg_share, rel, condition=None, **kw):
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_share, "launch", rel)),
            launch_arguments=kw.items(),
            condition=condition)

    # --- robot description / TF ------------------------------------------
    robot_description = inc(pkg_bringup, "robot_description.launch.py",
                            use_sim_time=use_sim_time)

    # --- motors + wheel odometry -----------------------------------------
    # publish_tf False: the EKF owns odom->base_footprint.
    hiwonder = Node(
        package="tractor_control",
        executable="hiwonder_motor_driver",
        name="hiwonder_motor_driver",
        output="screen",
        parameters=[
            os.path.join(pkg_bringup, "config", "hiwonder_motor_params.yaml"),
            {"publish_tf": False, "use_sim_time": use_sim_time},
        ])

    # --- lidar -> /scan ---------------------------------------------------
    lidar = inc(pkg_sensors, "stl19p_lidar.launch.py",
                port_name=LaunchConfiguration("lidar_port"),
                frame_id="laser_link")

    # --- IMU -> /imu/data -------------------------------------------------
    imu_bno = inc(pkg_sensors, "bno085_imu.launch.py",
                  condition=IfCondition(PythonExpression(
                      ["'", LaunchConfiguration("imu_type"), "' == 'bno085'"])))
    imu_lsm = inc(pkg_sensors, "lsm9ds1_imu.launch.py",
                  condition=IfCondition(PythonExpression(
                      ["'", LaunchConfiguration("imu_type"), "' == 'lsm9ds1'"])))

    # --- laser odometry + EKF -> /odometry/filtered -----------------------
    rf2o = inc(pkg_sensors, "lidar_odometry.launch.py", publish_tf="false")
    # The EKF node is declared HERE rather than via robot_localization.launch.py
    # on purpose: routed through that include, ekf_node came up with an empty
    # parameter set — no odom0/odom1/imu0, base_link_frame silently defaulting
    # to base_link — and published nothing on /odometry/filtered, while still
    # appearing healthy in `ros2 node list`. An absolute path passed straight to
    # the node removes that failure mode. (The YAML itself also had to be fixed:
    # it was three '---'-separated documents, and ROS 2's parameter loader reads
    # only the first.)
    ekf = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[os.path.join(pkg_bringup, "config",
                                 "robot_localization.yaml")],
        remappings=[("odometry/filtered", "odometry/filtered")])

    # --- optional human-viewable SLAM map ---------------------------------
    slam = Node(
        package="slam_toolbox",
        executable="async_slam_toolbox_node",
        name="slam_toolbox",
        output="screen",
        parameters=[
            os.path.join(pkg_bringup, "config", "slam_toolbox_params.yaml"),
            {"use_sim_time": use_sim_time},
        ],
        condition=IfCondition(LaunchConfiguration("slam")))

    # --- safety gate: /track_cmd_ai -> /track_cmd -------------------------
    # The twist path is parked on an unused topic: the motor driver listens to
    # BOTH /cmd_vel and /track_cmd, so letting both carry traffic double-drives
    # the motors.
    safety = Node(
        package="tractor_bringup",
        executable="lidar_safety_monitor.py",
        name="lidar_safety_monitor",
        output="screen",
        parameters=[{
            "scan_topic": "/scan",
            "input_cmd_topic": "/cmd_vel_unused",
            "output_cmd_topic": "/cmd_vel",
            "input_track_topic": "/track_cmd_ai",
            "output_track_topic": "/track_cmd",
            "stop_distance": 0.15,
            "stop_distance_rear": 0.30,
            "use_sim_time": use_sim_time,
        }])

    # --- the brain --------------------------------------------------------
    brain = Node(
        package="tractor_bringup",
        executable="spatial_explorer_runner",
        name="spatial_explorer_runner",
        output="screen",
        parameters=[{
            "driver": LaunchConfiguration("driver"),
            "policy_path": LaunchConfiguration("policy_path"),
            "action_scale": LaunchConfiguration("action_scale"),
            "control_rate_hz": LaunchConfiguration("control_rate_hz"),
            "scan_topic": "/scan",
            "odom_topic": "/odometry/filtered",
            "track_cmd_topic": PythonExpression(
                ["'/track_cmd_dryrun' if '",
                 LaunchConfiguration("dry_run"), "'.lower() == 'true' "
                 "else '/track_cmd_ai'"]),
            "use_sim_time": use_sim_time,
        }])

    ld = LaunchDescription(args)
    for a in (robot_description, hiwonder, lidar, imu_bno, imu_lsm, safety):
        ld.add_action(a)
    # Odometry needs the lidar and IMU streaming first; the brain needs odom.
    ld.add_action(TimerAction(period=3.0, actions=[rf2o, ekf, slam]))
    ld.add_action(TimerAction(period=6.0, actions=[brain]))
    return ld
