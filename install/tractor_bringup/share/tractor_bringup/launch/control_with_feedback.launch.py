#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration("use_sim_time")

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation (Gazebo) clock if true",
    )

    # Package Share for tractor_bringup
    pkg_tractor_bringup_share = FindPackageShare("tractor_bringup")

    # Parameters file for Hiwonder motor
    motor_params_file = PathJoinSubstitution(
        [pkg_tractor_bringup_share, "config", "hiwonder_motor_params.yaml"]
    )

    # Hiwonder motor driver node (includes battery publishing and odometry)
    hiwonder_motor_node = Node(
        package="tractor_control",
        executable="hiwonder_motor_driver",
        name="hiwonder_motor_driver",
        output="screen",
        parameters=[
            motor_params_file,
            {"use_sim_time": use_sim_time}
            # Add any overrides specific to this launch file if needed, e.g.:
            # {"max_motor_speed": 25} # Example override
        ],
    )

    # Velocity feedback controller for encoder-based correction
    velocity_controller_node = Node(
        package="tractor_control",
        executable="velocity_feedback_controller",
        name="velocity_feedback_controller",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "wheel_separation": 0.5,
                "wheel_radius": 0.15,
                "kp_linear": 0.8,  # Proportional gain for linear velocity
                "kp_angular": 0.6,  # Proportional gain for angular velocity
                "ki_linear": 0.1,  # Integral gain for linear velocity
                "ki_angular": 0.05,  # Integral gain for angular velocity
                "max_integral": 0.5,  # Maximum integral windup
                "control_frequency": 50.0,  # Control loop frequency
                "deadband": 0.001,  # Minimum velocity command
                # Gain for encoder drift correction in velocity_feedback_controller
                "drift_correction_gain": 0.001,
            }
        ],
        remappings=[
            # Controller subscribes to desired velocity, publishes corrected
            # velocity
            ("cmd_vel_desired", "cmd_vel_raw"),  # Input from joystick/teleop
            ("cmd_vel", "cmd_vel"),  # Output to motor driver
            ("odometry/filtered", "odometry/filtered"),  # Use robot_localization output
        ],
    )

    # Xbox controller teleop node
    xbox_controller_node = Node(
        package="tractor_control",
        executable="xbox_controller_teleop",
        name="xbox_controller_teleop",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "max_linear_speed": 0.5,  # ~65% of motor max (0.76 m/s) for good control margin
                "max_angular_speed": 1.0,  # Based on wheel separation and max speed
                "deadzone": 0.05,  # Reduced from 0.15 to allow smaller inputs
                "tank_drive_mode": True,
                "controller_index": 0,
                "use_feedback_control": False,
            }
        ],
        remappings=[
            ("cmd_vel", "cmd_vel_raw"),  # Output to velocity feedback controller
        ],
    )

    # GPS and compass node for heading-based drift correction
    gps_compass_node = Node(
        package="tractor_sensors",
        executable="hglrc_m100_5883",
        name="hglrc_m100_5883",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "gps_port": "/dev/ttyS6",
                "gps_baudrate": 115200,
                "i2c_bus": 5,
                "qmc5883_address": 0x0D,
                "magnetic_declination": 0.0,
                "gps_frame_id": "gps_link",
                "compass_frame_id": "compass_link",
                "compass_update_rate": 100.0,
            }
        ],
    )

    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)

    # Include robot localization launch file
    robot_localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("tractor_bringup"),
                "launch",
                "robot_localization.launch.py"
            ])
        ]),
        launch_arguments={
            "use_sim_time": use_sim_time
        }.items()
    )

    # Add control nodes
    ld.add_action(hiwonder_motor_node)
    ld.add_action(velocity_controller_node)
    ld.add_action(xbox_controller_node)
    ld.add_action(gps_compass_node)
    ld.add_action(robot_localization_launch)

    return ld
