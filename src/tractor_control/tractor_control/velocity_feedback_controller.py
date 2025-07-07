#!/usr/bin/env python3
"""
ROS 2 Node for implementing a PI velocity feedback controller.

This node subscribes to raw velocity commands (typically from a teleoperation
node or a planner) and current odometry (for actual velocity feedback).
It then calculates corrected velocity commands using a Proportional-Integral (PI)
controller to make the robot more accurately achieve the desired velocities.
The corrected commands are published on a new /cmd_vel topic.
It also includes an emergency stop feature.
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
import time
from threading import Lock


class VelocityFeedbackController(Node):
    def __init__(self):
        super().__init__("velocity_feedback_controller")

        # Parameters
        self.declare_parameter("wheel_separation", 0.5)  # meters
        self.declare_parameter("wheel_radius", 0.15)  # meters
        # Proportional gain for linear velocity
        self.declare_parameter("kp_linear", 0.8)
        # Proportional gain for angular velocity
        self.declare_parameter("kp_angular", 0.6)
        # Integral gain for linear velocity
        self.declare_parameter("ki_linear", 0.1)
        # Integral gain for angular velocity
        self.declare_parameter("ki_angular", 0.05)
        self.declare_parameter("max_integral", 0.5)  # Maximum integral windup
        self.declare_parameter("control_frequency", 50.0)  # Hz
        self.declare_parameter("deadband", 0.01)  # Minimum velocity command

        self.wheel_separation = self.get_parameter("wheel_separation").value
        self.wheel_radius = self.get_parameter("wheel_radius").value
        self.kp_linear = self.get_parameter("kp_linear").value
        self.kp_angular = self.get_parameter("kp_angular").value
        self.ki_linear = self.get_parameter("ki_linear").value
        self.ki_angular = self.get_parameter("ki_angular").value
        self.max_integral = self.get_parameter("max_integral").value
        self.control_frequency = self.get_parameter("control_frequency").value
        self.deadband = self.get_parameter("deadband").value

        # Control state
        self.desired_linear = 0.0
        self.desired_angular = 0.0
        self.current_linear = 0.0
        self.current_angular = 0.0
        self.emergency_stop = False

        # PID state
        self.linear_integral = 0.0
        self.angular_integral = 0.0
        self.last_time = time.time()

        # Thread safety
        self.control_lock = Lock()

        # Subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            "cmd_vel_raw",  # Input: raw velocity commands from teleop
            self.cmd_vel_callback,
            10,
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            "odom",  # Feedback: actual velocity from encoders
            self.odom_callback,
            10,
        )

        self.emergency_stop_sub = self.create_subscription(
            Bool, "emergency_stop", self.emergency_stop_callback, 10
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist, "cmd_vel", 10  # Output: corrected velocity command
        )

        # Control timer
        self.control_timer = self.create_timer(
            1.0 / self.control_frequency, self.control_loop
        )

        self.get_logger().info(
            f"Velocity Feedback Controller initialized at {
                self.control_frequency} Hz"
        )
        self.get_logger().info(
            f"PID gains - Linear: Kp={self.kp_linear}, Ki={self.ki_linear}"
        )
        self.get_logger().info(
            f"PID gains - Angular: Kp={self.kp_angular}, Ki={self.ki_angular}"
        )

    def cmd_vel_callback(self, msg):
        """Receive desired velocity commands"""
        with self.control_lock:
            self.desired_linear = msg.linear.x
            self.desired_angular = msg.angular.z

    def odom_callback(self, msg):
        """Receive actual velocity feedback from encoders"""
        with self.control_lock:
            self.current_linear = msg.twist.twist.linear.x
            self.current_angular = msg.twist.twist.angular.z

    def emergency_stop_callback(self, msg):
        """Handle emergency stop"""
        self.emergency_stop = msg.data
        if self.emergency_stop:
            with self.control_lock:
                self.desired_linear = 0.0
                self.desired_angular = 0.0
                self.linear_integral = 0.0
                self.angular_integral = 0.0
            self.get_logger().warn(
                "Emergency stop activated - clearing velocity commands"
            )

    def control_loop(self):
        """Main PID control loop"""
        if self.emergency_stop:
            # Send stop command during emergency stop
            self.publish_velocity_command(0.0, 0.0)
            return

        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        if dt <= 0.001:  # Avoid division by very small dt
            return

        with self.control_lock:
            # Calculate errors
            linear_error = self.desired_linear - self.current_linear
            angular_error = self.desired_angular - self.current_angular

            # Apply deadband to desired velocities
            if abs(self.desired_linear) < self.deadband:
                self.desired_linear = 0.0
                linear_error = -self.current_linear  # Drive actual to zero

            if abs(self.desired_angular) < self.deadband:
                self.desired_angular = 0.0
                angular_error = -self.current_angular  # Drive actual to zero

            # Update integral terms (with windup protection)
            # Only integrate if there's a non-negligible desired velocity, to
            # prevent windup when stopped.
            if abs(self.desired_linear) > 0.01:
                self.linear_integral += linear_error * dt
                self.linear_integral = max(
                    -self.max_integral, min(self.max_integral, self.linear_integral)
                )
            else:
                self.linear_integral = 0.0  # Reset when stopped

            if abs(self.desired_angular) > 0.01:  # Only integrate when turning
                self.angular_integral += angular_error * dt
                self.angular_integral = max(
                    -self.max_integral, min(self.max_integral, self.angular_integral)
                )
            else:
                self.angular_integral = 0.0  # Reset when not turning

            # PI control law: Output = Kp * error + Ki * integral(error)
            # The desired velocity is added to this output to form the new command.
            # This means the PI controller is trying to correct the *error* on
            # top of the desired command.
            linear_output = (
                self.kp_linear * linear_error + self.ki_linear * self.linear_integral
            )

            angular_output = (
                self.kp_angular * angular_error
                + self.ki_angular * self.angular_integral
            )

            # Calculate corrected velocity commands
            corrected_linear = self.desired_linear + linear_output
            corrected_angular = self.desired_angular + angular_output

            # Publish corrected commands
            self.publish_velocity_command(corrected_linear, corrected_angular)

            # Debug logging (reduced rate)
            if abs(linear_error) > 0.05 or abs(angular_error) > 0.05:
                self.get_logger().debug(
                    f"PID: Des[{
                        self.desired_linear:.2f}, {
                        self.desired_angular:.2f}] "
                    f"Act[{
                        self.current_linear:.2f}, {
                        self.current_angular:.2f}] "
                    f"Err[{
                        linear_error:.2f}, {
                            angular_error:.2f}] "
                    f"Out[{
                                corrected_linear:.2f}, {
                                    corrected_angular:.2f}]"
                )

    def publish_velocity_command(self, linear, angular):
        """Publish corrected velocity command"""
        cmd_msg = Twist()
        cmd_msg.linear.x = linear
        cmd_msg.linear.y = 0.0
        cmd_msg.linear.z = 0.0
        cmd_msg.angular.x = 0.0
        cmd_msg.angular.y = 0.0
        cmd_msg.angular.z = angular

        self.cmd_vel_pub.publish(cmd_msg)

    def destroy_node(self):
        """Clean shutdown"""
        # Send stop command
        self.publish_velocity_command(0.0, 0.0)
        time.sleep(0.1)
        self.get_logger().info("Velocity feedback controller shutdown complete")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    controller = VelocityFeedbackController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
