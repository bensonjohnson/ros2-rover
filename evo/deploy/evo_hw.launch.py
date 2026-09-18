"""Hardware-only bringup for evo genome tests (no PC brain).

    ros2 launch evo_hw.launch.py

Same nodes and parameters as tractor_bringup/pc_active_inference.launch.py
minus the brain, joystick and camera: robot description, hiwonder motor
driver, STL-19P lidar on /scan, BNO085 IMU on /imu/data (the LSM9DS1 is not
fitted on this rover — it fails to initialise), and the track-space
lidar_safety_monitor (/track_cmd_ai -> /track_cmd) with the parameters the
evo sim gate mirrors. evo_runner.py supplies /track_cmd_ai.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    bringup = get_package_share_directory("tractor_bringup")
    sensors = get_package_share_directory("tractor_sensors")
    return LaunchDescription([
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(bringup, "launch", "robot_description.launch.py"))),
        Node(package="tractor_control", executable="hiwonder_motor_driver",
             name="hiwonder_motor_driver", output="screen",
             parameters=[os.path.join(bringup, "config",
                                      "hiwonder_motor_params.yaml"),
                         {"publish_tf": False}]),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(sensors, "launch", "stl19p_lidar.launch.py")),
            launch_arguments={"port_name": "/dev/ttyUSB0",
                              "frame_id": "laser_link"}.items()),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(sensors, "launch", "bno085_imu.launch.py"))),
        Node(package="tractor_bringup", executable="lidar_safety_monitor.py",
             name="lidar_safety_monitor", output="screen",
             parameters=[{
                 "scan_topic": "/scan",
                 "input_cmd_topic": "/cmd_vel_unused",
                 "output_cmd_topic": "/cmd_vel",
                 "input_track_topic": "/track_cmd_ai",
                 "output_track_topic": "/track_cmd",
                 "stop_distance": 0.15,
                 "stop_distance_rear": 0.30,
                 "slow_distance": 0.15,
                 "hysteresis": 0.10,
                 "min_block_points": 3,
                 "block_scans": 2,
                 "min_valid_range": 0.05,
                 "max_eval_distance": 5.0,
                 "robot_front_offset": 0.06,
                 "robot_half_width": 0.12,
                 "stale_timeout": 0.2,
                 "min_block_duration": 0.3,
             }]),
    ])
