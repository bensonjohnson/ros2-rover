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

    server_addr_arg = DeclareLaunchArgument(
        'server_addr', default_value='192.168.1.100',
        description='Training server IP address')

    max_speed_arg = DeclareLaunchArgument(
        'max_speed', default_value='0.18',
        description='Maximum linear speed (m/s)')

    invert_vel_arg = DeclareLaunchArgument(
        'invert_linear_vel', default_value='true',
        description='Invert linear velocity from odometry')

    chunk_len_arg = DeclareLaunchArgument(
        'chunk_len', default_value='64',
        description='Steps per Dreamer chunk before shipping to server')

    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tractor_bringup_dir, 'launch', 'robot_description.launch.py')))

    hiwonder_motor_node = Node(
        package="tractor_control",
        executable="hiwonder_motor_driver",
        name="hiwonder_motor_driver",
        output="screen",
        parameters=[
            os.path.join(tractor_bringup_dir, "config", "hiwonder_motor_params.yaml"),
            {"publish_tf": False}
        ])

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(realsense_dir, "launch", "rs_launch.py")),
        launch_arguments={
            "pointcloud.enable": "false",
            "align_depth.enable": "false",
            "enable_color": "true",
            "rgb_camera.color_profile": "424x240x30",
            "enable_depth": "true",
            "enable_sync": "true",
            "device_type": "435i",
            "depth_module.depth_profile": "848x100x100",
            "enable_gyro": "false",
            "enable_accel": "false",
            "enable_imu": "false",
        }.items())

    lsm9ds1_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tractor_sensors_dir, "launch", "lsm9ds1_imu.launch.py")))

    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tractor_sensors_dir, "launch", "stl19p_lidar.launch.py")),
        launch_arguments={
            'port_name': '/dev/ttyUSB0',
            'frame_id': 'laser_link'
        }.items())

    rf2o_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tractor_sensors_dir, "launch", "lidar_odometry.launch.py")),
        launch_arguments={
            'publish_tf': 'false'
        }.items())

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

    robot_localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tractor_bringup_dir, 'launch', 'robot_localization.launch.py')),
        launch_arguments={
            'use_gps': 'false'
        }.items())

    dreamer_runner_node = Node(
        package='tractor_bringup',
        executable='dreamer_remote_runner',
        name='dreamer_remote_runner',
        output='screen',
        parameters=[{
            'server_addr': LaunchConfiguration('server_addr'),
            'max_linear_speed': LaunchConfiguration('max_speed'),
            'max_angular_speed': 1.0,
            'inference_rate_hz': 30.0,
            'chunk_len': LaunchConfiguration('chunk_len'),
            'invert_linear_vel': LaunchConfiguration('invert_linear_vel'),
        }])

    return LaunchDescription([
        server_addr_arg,
        max_speed_arg,
        invert_vel_arg,
        chunk_len_arg,

        robot_description_launch,
        hiwonder_motor_node,

        TimerAction(period=2.0, actions=[lidar_launch]),
        TimerAction(period=4.0, actions=[rf2o_launch]),
        TimerAction(period=5.0, actions=[realsense_launch]),
        TimerAction(period=6.0, actions=[lsm9ds1_launch]),

        TimerAction(period=7.0, actions=[robot_localization_launch]),
        TimerAction(period=8.0, actions=[safety_monitor_node]),

        TimerAction(period=10.0, actions=[dreamer_runner_node]),
    ])
