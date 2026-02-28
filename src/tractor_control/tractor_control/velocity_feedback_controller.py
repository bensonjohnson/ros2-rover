import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
import time
import math
from threading import Lock


class VelocityFeedbackController(Node):
    def __init__(self):
        super().__init__("velocity_feedback_controller")

        # Parameters
        self.declare_parameter("control_frequency", 30.0)  # Hz (match SAC rate)
        self.declare_parameter("deadband", 0.001)  # Minimum velocity command
        self.declare_parameter("cmd_timeout_sec", 0.2)  # Zero commands if no input for 200ms
        # Heading correction
        self.declare_parameter("heading_correction_gain", 0.5)  # P gain for heading error
        self.declare_parameter("max_heading_correction", 0.3)  # Max angular correction (rad/s)
        self.declare_parameter("straight_threshold", 0.05)  # Below this angular = "going straight"
        self.declare_parameter("min_linear_for_correction", 0.03)  # Min linear speed to apply correction
        self.declare_parameter("heading_error_deadzone", 0.02)  # ~1 degree, ignore smaller errors
        # Logging controls
        self.declare_parameter("log_cmd_vel", False)
        self.declare_parameter("log_throttle_sec", 1.0)

        self.control_frequency = self.get_parameter("control_frequency").value
        self.deadband = self.get_parameter("deadband").value
        self.cmd_timeout_sec = self.get_parameter("cmd_timeout_sec").value
        self.heading_correction_gain = self.get_parameter("heading_correction_gain").value
        self.max_heading_correction = self.get_parameter("max_heading_correction").value
        self.straight_threshold = self.get_parameter("straight_threshold").value
        self.min_linear_for_correction = self.get_parameter("min_linear_for_correction").value
        self.heading_error_deadzone = self.get_parameter("heading_error_deadzone").value
        self.log_cmd_vel = bool(self.get_parameter("log_cmd_vel").value)
        self.log_throttle_sec = float(self.get_parameter("log_throttle_sec").value)
        self._last_log_time = 0.0

        # Control state
        self.desired_linear = 0.0
        self.desired_angular = 0.0
        self.current_yaw = 0.0  # Current heading from EKF
        self.emergency_stop = False
        self.last_cmd_time = time.time()  # For command timeout

        # Heading tracking
        self.target_heading = None
        self.heading_initialized = False

        # Thread safety
        self.control_lock = Lock()

        # Subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            "cmd_vel_raw",  # Input: from safety monitor
            self.cmd_vel_callback,
            10,
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            "/odometry/filtered",  # EKF filtered odometry
            self.odom_callback,
            10,
        )

        self.emergency_stop_sub = self.create_subscription(
            Bool, "emergency_stop", self.emergency_stop_callback, 10
        )

        # Publisher
        self.cmd_vel_pub = self.create_publisher(
            Twist, "cmd_vel", 10  # Output: corrected velocity to motor driver
        )

        # Control timer
        self.control_timer = self.create_timer(
            1.0 / self.control_frequency, self.control_loop
        )

        self.get_logger().info(
            f"VFC initialized at {self.control_frequency} Hz, "
            f"heading_gain={self.heading_correction_gain}, "
            f"max_correction={self.max_heading_correction} rad/s"
        )

    def cmd_vel_callback(self, msg):
        """Receive desired velocity commands from safety monitor."""
        with self.control_lock:
            self.desired_linear = msg.linear.x
            self.desired_angular = msg.angular.z
            self.last_cmd_time = time.time()

    def odom_callback(self, msg):
        """Receive filtered odometry from robot_localization EKF."""
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

        with self.control_lock:
            self.current_yaw = yaw
            if not self.heading_initialized:
                self.heading_initialized = True
                self.get_logger().info(
                    f"EKF heading initialized: {math.degrees(yaw):.1f} deg"
                )

    def emergency_stop_callback(self, msg):
        """Handle emergency stop."""
        self.emergency_stop = msg.data
        if self.emergency_stop:
            with self.control_lock:
                self.desired_linear = 0.0
                self.desired_angular = 0.0
                self.target_heading = None
            self.get_logger().warn(
                "Emergency stop activated - clearing velocity commands"
            )

    def control_loop(self):
        """Apply heading drift correction and publish corrected cmd_vel."""
        if self.emergency_stop:
            self.publish_cmd(0.0, 0.0)
            return

        with self.control_lock:
            # Command timeout — zero everything if no recent input
            if time.time() - self.last_cmd_time > self.cmd_timeout_sec:
                self.target_heading = None
                self.publish_cmd(0.0, 0.0)
                return

            desired_linear = self.desired_linear
            desired_angular = self.desired_angular

        # Apply deadband
        if abs(desired_linear) < self.deadband:
            desired_linear = 0.0
        if abs(desired_angular) < self.deadband:
            desired_angular = 0.0

        # If stopped or very slow, reset heading target and pass through
        if abs(desired_linear) < self.min_linear_for_correction:
            self.target_heading = None
            self.publish_cmd(desired_linear, desired_angular)
            return

        # If intentional turn (any non-trivial angular command), reset heading and pass through
        if abs(desired_angular) >= self.straight_threshold:
            self.target_heading = None
            self.publish_cmd(desired_linear, desired_angular)
            return

        # Going straight (linear > threshold, angular < threshold) — apply heading correction
        with self.control_lock:
            current_yaw = self.current_yaw

        # Set target heading when starting to go straight
        if self.target_heading is None:
            self.target_heading = current_yaw
            self.get_logger().debug(
                f"Target heading set: {math.degrees(current_yaw):.1f} deg"
            )

        # Calculate heading error
        heading_error = current_yaw - self.target_heading

        # Normalize to [-pi, pi]
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

        # If EKF jumped wildly (>90 deg error), adapt target instead of fighting
        if abs(heading_error) > 1.57:
            self.target_heading = current_yaw
            self.get_logger().warn(
                f"EKF jump detected ({math.degrees(heading_error):.0f} deg), resetting target heading"
            )
            self.publish_cmd(desired_linear, desired_angular)
            return

        # Apply proportional correction within deadzone
        corrected_angular = desired_angular
        if abs(heading_error) > self.heading_error_deadzone:
            correction = -heading_error * self.heading_correction_gain
            correction = max(-self.max_heading_correction,
                             min(self.max_heading_correction, correction))
            corrected_angular += correction

            self.get_logger().debug(
                f"Heading correction: error={math.degrees(heading_error):.1f} deg, "
                f"correction={correction:.3f} rad/s"
            )

        self.publish_cmd(desired_linear, corrected_angular)

    def publish_cmd(self, linear, angular):
        """Publish velocity command."""
        cmd_msg = Twist()
        cmd_msg.linear.x = linear
        cmd_msg.angular.z = angular

        if self.log_cmd_vel and (abs(linear) > 0.001 or abs(angular) > 0.001):
            now = time.time()
            if now - self._last_log_time >= self.log_throttle_sec:
                self._last_log_time = now
                self.get_logger().info(
                    f"cmd_vel: linear={linear:.3f}, angular={angular:.3f}"
                )

        self.cmd_vel_pub.publish(cmd_msg)

    def destroy_node(self):
        """Clean shutdown."""
        self.publish_cmd(0.0, 0.0)
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
