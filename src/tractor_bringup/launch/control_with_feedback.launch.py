#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration("use_sim_time")

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation (Gazebo) clock if true",
    )

    # Hiwonder motor driver node (includes battery publishing and odometry)
    hiwonder_motor_node = Node(
        package="tractor_control",
        executable="hiwonder_motor_driver",
        name="hiwonder_motor_driver",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "i2c_bus": 5,
                "motor_controller_address": 0x34,
                "wheel_separation": 0.5,
                "wheel_radius": 0.15,
                "max_motor_speed": 25,
                "deadband": 0.05,
                "encoder_ppr": 1980,  # JGB3865-520R45-12: 44 pulses × 45:1 ratio
                "publish_rate": 100.0,  # High rate for good control
                "use_pwm_control": True,  # PWM mode for JGB3865
                "motor_type": 3,
                "min_samples_for_estimation": 10,
                "max_history_minutes": 60,
            }
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
