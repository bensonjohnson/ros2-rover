import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    tractor_bringup_dir = get_package_share_directory('tractor_bringup')
    realsense_dir = get_package_share_directory('realsense2_camera')
    tractor_sensors_dir = get_package_share_directory('tractor_sensors')

    max_speed_arg = DeclareLaunchArgument(
        'max_speed', default_value='0.18',
        description='Maximum linear speed (m/s)')

    invert_vel_arg = DeclareLaunchArgument(
        'invert_linear_vel', default_value='true',
        description='Invert linear velocity from odometry')

    checkpoint_dir_arg = DeclareLaunchArgument(
        'checkpoint_dir', default_value='./checkpoints_ppo',
        description='Directory for PPO checkpoints')

    log_dir_arg = DeclareLaunchArgument(
        'log_dir', default_value='./logs_ppo',
        description='Directory for TensorBoard logs')

    rollout_steps_arg = DeclareLaunchArgument(
        'rollout_steps', default_value='2048',
        description='Steps per PPO rollout before training')

    # 1. Robot Description
    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tractor_bringup_dir, 'launch', 'robot_description.launch.py')))

    # 2. Motor Driver
    hiwonder_motor_node = Node(
        package="tractor_control",
        executable="hiwonder_motor_driver",
        name="hiwonder_motor_driver",
        output="screen",
        parameters=[
            os.path.join(tractor_bringup_dir, "config", "hiwonder_motor_params.yaml"),
            {"publish_tf": False}
        ])

    # 3. RealSense Camera (Depth-only)
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(realsense_dir, "launch", "rs_launch.py")),
        launch_arguments={
            "pointcloud.enable": "false",
            "align_depth.enable": "false",
            "enable_color": "false",
            "enable_depth": "true",
            "enable_sync": "false",
            "device_type": "435i",
            "depth_module.depth_profile": "848x100x100",
            "enable_gyro": "false",
            "enable_accel": "false",
            "enable_imu": "false",
        }.items())

    # 4. IMU
    lsm9ds1_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tractor_sensors_dir, "launch", "lsm9ds1_imu.launch.py")))

    # 4b. LiDAR
    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tractor_sensors_dir, "launch", "stl19p_lidar.launch.py")),
        launch_arguments={
            'port_name': '/dev/ttyUSB0',
            'frame_id': 'laser_link'
        }.items())

    # 4c. LiDAR Odometry (RF2O)
    rf2o_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tractor_sensors_dir, "launch", "lidar_odometry.launch.py")),
        launch_arguments={
            'publish_tf': 'false'
        }.items())

    # 5. Safety Monitor (LiDAR-based)
    safety_monitor_node = Node(
        package="tractor_bringup",
        executable="lidar_safety_monitor.py",
        name="lidar_safety_monitor",
        output="screen",
        parameters=[{
            "scan_topic": "/scan",
            "input_cmd_topic": "/cmd_vel_teleop",
            "output_cmd_topic": "/cmd_vel",
            "input_track_topic": "/track_cmd_ai",
            "output_track_topic": "/track_cmd",
            "stop_distance": 0.15,
            "slow_distance": 0.20,
            "hysteresis": 0.10,
            "min_valid_range": 0.05,
            "max_eval_distance": 5.0,
            "robot_front_offset": 0.06,
            "robot_half_width": 0.12,
            "stale_timeout": 0.2,
            "min_block_duration": 0.3,
        }])

    # 6. Robot Localization (EKF)
    robot_localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tractor_bringup_dir, 'launch', 'robot_localization.launch.py')),
        launch_arguments={
            'use_gps': 'false'
        }.items())

    # 7. PPO Local Runner
    ppo_runner_node = Node(
        package='tractor_bringup',
        executable='ppo_local_runner',
        name='ppo_local_runner',
        output='screen',
        parameters=[{
            'max_linear_speed': LaunchConfiguration('max_speed'),
            'max_angular_speed': 1.0,
            'inference_rate_hz': 30.0,
            'rollout_steps': LaunchConfiguration('rollout_steps'),
            'checkpoint_dir': LaunchConfiguration('checkpoint_dir'),
            'log_dir': LaunchConfiguration('log_dir'),
            'invert_linear_vel': LaunchConfiguration('invert_linear_vel'),
        }])

    return LaunchDescription([
        max_speed_arg,
        invert_vel_arg,
        checkpoint_dir_arg,
        log_dir_arg,
        rollout_steps_arg,

        # Core
        robot_description_launch,
        hiwonder_motor_node,

        # Sensors (delayed)
        TimerAction(period=2.0, actions=[lidar_launch]),
        TimerAction(period=4.0, actions=[rf2o_launch]),
        TimerAction(period=5.0, actions=[realsense_launch]),
        TimerAction(period=6.0, actions=[lsm9ds1_launch]),

        # State Estimation
        TimerAction(period=7.0, actions=[robot_localization_launch]),

        # Safety
        TimerAction(period=8.0, actions=[safety_monitor_node]),

        # PPO (last)
        TimerAction(period=10.0, actions=[ppo_runner_node]),
    ])
