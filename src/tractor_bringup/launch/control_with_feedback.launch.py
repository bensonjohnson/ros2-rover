#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


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
                "deadband": 0.01,  # Minimum velocity command
                # Gain for encoder drift correction in velocity_feedback_controller
                "drift_correction_gain": 0.0001,
            }
        ],
        remappings=[
            # Controller subscribes to desired velocity, publishes corrected
            # velocity
            ("cmd_vel_desired", "cmd_vel_raw"),  # Input from joystick/teleop
            ("cmd_vel", "cmd_vel"),  # Output to motor driver
            ("odom", "odom"),  # Feedback from motor driver
        ],
    )

    ld = LaunchDescription()

    # Add launch arguments
    ld.add_action(declare_use_sim_time_cmd)

    # Add control nodes
    ld.add_action(hiwonder_motor_node)
    ld.add_action(velocity_controller_node)

    return ld
