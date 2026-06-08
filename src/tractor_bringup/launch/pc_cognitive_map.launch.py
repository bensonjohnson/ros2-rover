"""Cognitive-map active-inference brain — bringup.

Same minimal stack as pc_active_inference, plus rf2o lidar odometry for the
allocentric pose the growing latent map needs:

    stl19p lidar  -> /scan
    rf2o          -> /odom_rf2o   (lidar odometry, no TF)
    safety        -> gates /track_cmd_ai -> /track_cmd
    motor driver  -> tracks
    cognitive-map brain (scan + odom -> growing latent map -> frontier seeking)

No RealSense, IMU, joystick, or remote server. Online PC learning, no
pretraining, single house.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    tractor_bringup_dir = get_package_share_directory("tractor_bringup")
    tractor_sensors_dir = get_package_share_directory("tractor_sensors")

    action_scale_arg = DeclareLaunchArgument("action_scale", default_value="0.6")
    control_rate_arg = DeclareLaunchArgument("control_rate_hz", default_value="15.0")
    learn_arg = DeclareLaunchArgument("learn", default_value="true")
    lidar_port_arg = DeclareLaunchArgument("lidar_port", default_value="/dev/ttyUSB0")
    cell_size_arg = DeclareLaunchArgument("cell_size", default_value="0.5")

    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tractor_bringup_dir, "launch", "robot_description.launch.py")))

    hiwonder_motor_node = Node(
        package="tractor_control", executable="hiwonder_motor_driver",
        name="hiwonder_motor_driver", output="screen",
        parameters=[
            os.path.join(tractor_bringup_dir, "config", "hiwonder_motor_params.yaml"),
            {"publish_tf": False}])

    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tractor_sensors_dir, "launch", "stl19p_lidar.launch.py")),
        launch_arguments={"port_name": LaunchConfiguration("lidar_port"),
                          "frame_id": "laser_link"}.items())

    rf2o_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tractor_sensors_dir, "launch", "lidar_odometry.launch.py")),
        launch_arguments={"publish_tf": "false"}.items())

    safety_monitor_node = Node(
        package="tractor_bringup", executable="lidar_safety_monitor.py",
        name="lidar_safety_monitor", output="screen",
        parameters=[{
            "scan_topic": "/scan",
            "input_cmd_topic": "/cmd_vel_teleop", "output_cmd_topic": "/cmd_vel",
            "input_track_topic": "/track_cmd_ai", "output_track_topic": "/track_cmd",
            "stop_distance": 0.15, "slow_distance": 0.15, "hysteresis": 0.10,
            "min_valid_range": 0.05, "max_eval_distance": 5.0,
            "robot_front_offset": 0.06, "robot_half_width": 0.12,
            "stale_timeout": 0.2, "min_block_duration": 0.3}])

    brain_node = Node(
        package="tractor_bringup", executable="pc_cognitive_map_runner",
        name="pc_cognitive_map_runner", output="screen",
        parameters=[{
            "scan_topic": "/scan",
            "odom_topic": "/odom_rf2o",
            "track_cmd_topic": "/track_cmd_ai",
            "control_rate_hz": LaunchConfiguration("control_rate_hz"),
            "action_scale": LaunchConfiguration("action_scale"),
            "learn": LaunchConfiguration("learn"),
            "cell_size": LaunchConfiguration("cell_size"),
            "num_bins": 72, "max_range": 5.0, "latent_dim": 24,
            "torch_threads": 4, "dashboard_port": 8083}])

    return LaunchDescription([
        action_scale_arg, control_rate_arg, learn_arg, lidar_port_arg, cell_size_arg,
        robot_description_launch,
        hiwonder_motor_node,
        TimerAction(period=2.0, actions=[lidar_launch]),
        TimerAction(period=4.0, actions=[rf2o_launch]),
        TimerAction(period=5.0, actions=[safety_monitor_node]),
        TimerAction(period=7.0, actions=[brain_node]),
    ])
