"""Active-inference rover brain — minimal bringup.

Reuses ONLY the pieces this experiment needs, nothing else:
    - stl19p lidar bringup            -> /scan
    - lsm9ds1 IMU                     -> /imu/data (proprio yaw-rate channel)
    - lidar_safety_monitor (track)    -> gates /track_cmd_ai -> /track_cmd
    - hiwonder_motor_driver           -> tracks + /joint_states (wheel velocity)
    - pc_active_inference_runner      -> the predictive-coding brain

The brain learns online on the rover CPU from lidar + proprioception (wheel
velocity + IMU yaw rate — the rover sensing its own motion), and the safety
monitor hard-stops the tracks near obstacles so early erratic behavior stays
bounded. No RealSense, odometry EKF, joystick, or remote training server.
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

    action_scale_arg = DeclareLaunchArgument(
        "action_scale", default_value="0.6",
        description="Scales track output [-1,1] (gentler while the brain is young)")
    control_rate_arg = DeclareLaunchArgument(
        "control_rate_hz", default_value="15.0",
        description="Brain inference/control rate (match the lidar scan rate so "
                    "each fresh scan drives exactly one infer/learn step)")
    learn_arg = DeclareLaunchArgument(
        "learn", default_value="true",
        description="Set false to freeze the brain and evaluate")
    lidar_port_arg = DeclareLaunchArgument(
        "lidar_port", default_value="/dev/ttyUSB0")
    forward_bias_arg = DeclareLaunchArgument(
        "forward_bias", default_value="0.3",
        description="0 = pure epistemic, 1 = pure forward translation (anti-spin)")
    action_persist_arg = DeclareLaunchArgument(
        "action_persist", default_value="5",
        description="Hold a chosen action this many ticks (anti-twitch)")
    imu_yaw_axis_arg = DeclareLaunchArgument(
        "imu_yaw_axis", default_value="z",
        description="Chip axis that is VERTICAL on the (strangely mounted) IMU "
                    "= the yaw axis. VERIFY with a spin + gravity check.")
    imu_yaw_sign_arg = DeclareLaunchArgument(
        "imu_yaw_sign", default_value="1.0",
        description="Flip to -1.0 if a left turn reads as negative yaw rate")
    max_yaw_rate_arg = DeclareLaunchArgument(
        "max_yaw_rate", default_value="2.5",
        description="rad/s that maps yaw to the proprio range edge (set ~1.3x the "
                    "rover's peak powered pivot rate so spins fill the range "
                    "without clipping)")
    max_wheel_vel_arg = DeclareLaunchArgument(
        "max_wheel_vel", default_value="8.0",
        description="rad/s that maps wheel velocity to the proprio range edge")
    dashboard_port_arg = DeclareLaunchArgument(
        "dashboard_port", default_value="8082",
        description="Web dashboard port (0 disables)")

    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tractor_bringup_dir, "launch", "robot_description.launch.py")))

    hiwonder_motor_node = Node(
        package="tractor_control",
        executable="hiwonder_motor_driver",
        name="hiwonder_motor_driver",
        output="screen",
        parameters=[
            os.path.join(tractor_bringup_dir, "config", "hiwonder_motor_params.yaml"),
            {"publish_tf": False},
        ])

    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tractor_sensors_dir, "launch", "stl19p_lidar.launch.py")),
        launch_arguments={
            "port_name": LaunchConfiguration("lidar_port"),
            "frame_id": "laser_link",
        }.items())

    # IMU (LSM9DS1) for the proprio yaw-rate channel -> /imu/data
    imu_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tractor_sensors_dir, "launch", "lsm9ds1_imu.launch.py")))

    # Track-space safety gate: /track_cmd_ai -> (clamp near obstacles) -> /track_cmd
    safety_monitor_node = Node(
        package="tractor_bringup",
        executable="lidar_safety_monitor.py",
        name="lidar_safety_monitor",
        output="screen",
        parameters=[{
            "scan_topic": "/scan",
            "input_cmd_topic": "/cmd_vel_teleop",   # unused here, kept for the node's API
            "output_cmd_topic": "/cmd_vel",
            "input_track_topic": "/track_cmd_ai",
            "output_track_topic": "/track_cmd",
            "stop_distance": 0.15,
            "slow_distance": 0.15,
            "hysteresis": 0.10,
            "min_valid_range": 0.05,
            "max_eval_distance": 5.0,
            "robot_front_offset": 0.06,
            "robot_half_width": 0.12,
            "stale_timeout": 0.2,
            "min_block_duration": 0.3,
        }])

    brain_node = Node(
        package="tractor_bringup",
        executable="pc_active_inference_runner",
        name="pc_active_inference_runner",
        output="screen",
        parameters=[{
            "scan_topic": "/scan",
            "track_cmd_topic": "/track_cmd_ai",
            "control_rate_hz": LaunchConfiguration("control_rate_hz"),
            "action_scale": LaunchConfiguration("action_scale"),
            "forward_bias": LaunchConfiguration("forward_bias"),
            "action_persist": LaunchConfiguration("action_persist"),
            "learn": LaunchConfiguration("learn"),
            "use_proprio": True,
            "imu_yaw_axis": LaunchConfiguration("imu_yaw_axis"),
            "imu_yaw_sign": LaunchConfiguration("imu_yaw_sign"),
            "max_yaw_rate": LaunchConfiguration("max_yaw_rate"),
            "max_wheel_vel": LaunchConfiguration("max_wheel_vel"),
            "num_bins": 72,
            "max_range": 5.0,
            "latent_dim": 64,
            "ensemble_size": 5,
            "torch_threads": 4,
            "dashboard_port": LaunchConfiguration("dashboard_port"),
        }])

    return LaunchDescription([
        action_scale_arg,
        control_rate_arg,
        learn_arg,
        lidar_port_arg,
        forward_bias_arg,
        action_persist_arg,
        imu_yaw_axis_arg,
        imu_yaw_sign_arg,
        max_yaw_rate_arg,
        max_wheel_vel_arg,
        dashboard_port_arg,

        robot_description_launch,
        hiwonder_motor_node,

        TimerAction(period=2.0, actions=[lidar_launch]),
        TimerAction(period=4.0, actions=[imu_launch]),
        TimerAction(period=5.0, actions=[safety_monitor_node]),
        TimerAction(period=7.0, actions=[brain_node]),
    ])
