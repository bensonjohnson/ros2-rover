#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition


def generate_launch_description():

    # Launch arguments
    i2c_bus_arg = DeclareLaunchArgument(
        "i2c_bus", default_value="5", description="I2C bus number"
    )

    motor_address_arg = DeclareLaunchArgument(
        "motor_address",
        default_value="0x34",
        description="Motor controller I2C address (corrected)",
    )

    max_motor_speed_arg = DeclareLaunchArgument(
        "max_motor_speed",
        default_value="25",
        description="Maximum motor speed (reduced for JGB3865 testing)",
    )

    tank_drive_mode_arg = DeclareLaunchArgument(
        "tank_drive_mode",
        default_value="true",
        description="Enable tank drive mode for Xbox controller",
    )

    use_xbox_controller_arg = DeclareLaunchArgument(
        "use_xbox_controller",
        default_value="true",
        description="Enable Xbox controller teleop",
    )

    # Hiwonder motor driver node
    hiwonder_motor_driver = Node(
        package="tractor_control",
        executable="hiwonder_motor_driver",
        name="hiwonder_motor_driver",
        parameters=[
            {
                "i2c_bus": LaunchConfiguration("i2c_bus"),
                "motor_controller_address": 0x34,  # Corrected address from ESP32 testing
                "wheel_separation": 0.5,
                "wheel_radius": 0.15,
                "max_motor_speed": LaunchConfiguration("max_motor_speed"),
                "deadband": 0.05,
                "encoder_ppr": 1440,
                "publish_rate": 5.0,
                "use_pwm_control": False,  # Use speed control for JGB3865 encoded motors
                "motor_type": 3,  # 3=JGB37 series (for JGB3865 motors)
            }
        ],
        output="screen",
    )

    # Xbox controller teleop node
    xbox_controller_teleop = Node(
        package="tractor_control",
        executable="xbox_controller_teleop",
        name="xbox_controller_teleop",
        parameters=[
            {
                "max_linear_speed": 1.0,
                "max_angular_speed": 2.0,
                "deadzone": 0.15,
                "tank_drive_mode": LaunchConfiguration("tank_drive_mode"),
                "controller_index": 0,
            }
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_xbox_controller")),
    )

    return LaunchDescription(
        [
            i2c_bus_arg,
            motor_address_arg,
            max_motor_speed_arg,
            tank_drive_mode_arg,
            use_xbox_controller_arg,
            hiwonder_motor_driver,
            xbox_controller_teleop,
        ]
    )
