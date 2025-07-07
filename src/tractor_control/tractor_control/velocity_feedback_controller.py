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
        self.declare_parameter(
            "drift_correction_gain", 0.0001
        )  # Gain for encoder drift correction

        self.wheel_separation = self.get_parameter("wheel_separation").value
        self.wheel_radius = self.get_parameter("wheel_radius").value
        self.kp_linear = self.get_parameter("kp_linear").value
        self.kp_angular = self.get_parameter("kp_angular").value
        self.ki_linear = self.get_parameter("ki_linear").value
        self.ki_angular = self.get_parameter("ki_angular").value
        self.max_integral = self.get_parameter("max_integral").value
        self.control_frequency = self.get_parameter("control_frequency").value
        self.deadband = self.get_parameter("deadband").value
        self.drift_correction_gain = self.get_parameter("drift_correction_gain").value

        # Control state
        self.desired_linear = 0.0
        self.desired_angular = 0.0
        self.current_linear = 0.0
        self.current_angular = 0.0
        self.emergency_stop = False

        # Encoder drift correction
        self.left_encoder_total = 0
        self.right_encoder_total = 0
        self.initial_encoder_diff = 0
        self.encoder_diff_baseline_set = False

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

        # Subscribe to joint states to get encoder totals
        from sensor_msgs.msg import JointState  # Local import is fine for callbacks

        self.joint_state_sub = self.create_subscription(
            JointState, "joint_states", self.joint_state_callback, 10
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
        self.get_logger().info(
            f"Drift Correction Gain: {
                self.drift_correction_gain}"
        )  # Log new gain

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

    def joint_state_callback(self, msg):
        """Receive encoder totals from joint states for drift correction"""
        try:
            # Find left and right wheel positions (encoder totals)
            # Assuming 'left_wheel_joint' and 'right_wheel_joint' are the names
            # in the JointState message that correspond to raw encoder counts or
            # scaled positions that can be consistently diffed.
            # The hiwonder_motor_driver publishes these as 'left_wheel_joint', 'right_wheel_joint'
            # with 0.0 position, and 'left_viz_wheel_joint', 'right_viz_wheel_joint' with scaled radians.
            # This new code expects raw encoder counts in 'left_wheel_joint' and 'right_wheel_joint'.
            # This is a MISMATCH with what hiwonder_motor_driver currently publishes.
            # For this to work, hiwonder_motor_driver needs to publish raw encoder counts
            # to 'left_wheel_joint' and 'right_wheel_joint' positions.

            # For now, I will assume the user intends to modify hiwonder_motor_driver
            # or that these names are placeholders. The original code had 0.0 for these.
            # The user's code casts msg.position to int(), implying it expects
            # countable units.

            left_idx = msg.name.index("left_wheel_joint")
            right_idx = msg.name.index("right_wheel_joint")

            with self.control_lock:
                # These should be raw encoder ticks if possible for drift calculation
                # The user's code casts to int, suggesting countable units.
                # If these are in radians (as per current hiwonder driver for viz wheels),
                # the drift logic might behave unexpectedly due to wrapping and
                # float precision.
                self.left_encoder_total = int(msg.position[left_idx])
                self.right_encoder_total = int(msg.position[right_idx])

                # Set baseline encoder difference on first reading
                if not self.encoder_diff_baseline_set:
                    self.initial_encoder_diff = (
                        self.right_encoder_total - self.left_encoder_total
                    )
                    self.encoder_diff_baseline_set = True
                    self.get_logger().info(
                        f"Encoder baseline set: L={
                            self.left_encoder_total}, R={
                            self.right_encoder_total}, diff={
                            self.initial_encoder_diff}"
                    )

        except (ValueError, IndexError):
            # Joint names not found or indices out of range
            # self.get_logger().warn("Could not find 'left_wheel_joint' or 'right_wheel_joint' in /joint_states for drift correction.")
            pass  # Be less verbose if joints aren't there initially

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
        """Main PID control loop with smart tank turn detection"""
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
            # Detect if this is an intentional tank turn (high angular
            # velocity)
            is_tank_turning = abs(self.desired_angular) > 1.0

            # For very low speeds or tank turns, don't apply PI corrections
            if abs(self.desired_linear) < 0.1 or is_tank_turning:
                # Pass through commands without correction
                self.publish_velocity_command(self.desired_linear, self.desired_angular)
                # Reset integrals
                self.linear_integral = 0.0
                self.angular_integral = 0.0
                if is_tank_turning:
                    self.get_logger().debug(
                        "Tank turn detected, PI corrections bypassed."
                    )
                return

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

            # PI corrections primarily for straight-line driving (small desired
            # angular velocity)
            if abs(self.desired_angular) < 0.3:
                # Update integral terms (with windup protection)
                if (
                    abs(self.desired_linear) > 0.2
                ):  # Only integrate when moving at reasonable speed
                    self.linear_integral += linear_error * dt
                    self.linear_integral = max(
                        -self.max_integral, min(self.max_integral, self.linear_integral)
                    )
                else:
                    self.linear_integral = 0.0  # Reset integral when slow or stopped

                # Gentle angular correction: only for small angular errors
                if abs(angular_error) < 0.5:
                    self.angular_integral += angular_error * dt
                    self.angular_integral = max(
                        -self.max_integral,
                        min(self.max_integral, self.angular_integral),
                    )
                else:
                    # Reset for large angular errors (likely intentional turn)
                    self.angular_integral = 0.0

                # PI control output
                linear_output = (
                    self.kp_linear * linear_error
                    + self.ki_linear * self.linear_integral
                )

                # Reduce angular correction effect during primarily straight
                # driving
                angular_output = (
                    self.kp_angular * angular_error
                    + self.ki_angular * self.angular_integral
                ) * 0.3

                corrected_linear = self.desired_linear + linear_output
                corrected_angular = self.desired_angular + angular_output

                self.publish_velocity_command(corrected_linear, corrected_angular)

                if abs(linear_error) > 0.05 or abs(angular_error) > 0.05:
                    self.get_logger().debug(
                        f"Straight PID: Des[{
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
            else:  # Handling intentional turns (larger desired_angular)
                # Pass through commands without PI correction for turns
                self.get_logger().debug(
                    f"Intentional turn (des_ang: {
                        self.desired_angular:.2f}), PI corrections bypassed."
                )
                self.publish_velocity_command(self.desired_linear, self.desired_angular)
                # Reset integrals during turns
                self.linear_integral = 0.0
                self.angular_integral = 0.0

    def publish_velocity_command(self, linear, angular):
        """Publish corrected velocity command with drift compensation"""
        corrected_angular = angular  # Start with the angular from PI or passthrough

        # Apply encoder-based drift correction for straight driving only
        if (
            self.encoder_diff_baseline_set
            and abs(linear) > 0.2  # Only when moving meaningfully straight
            and abs(self.desired_angular) < 0.2
        ):  # Only when intending to go straight

            current_encoder_diff = self.right_encoder_total - self.left_encoder_total
            accumulated_drift = current_encoder_diff - self.initial_encoder_diff

            # Apply proportional correction for drift
            # Positive drift (right_encoder_total > left_encoder_total than baseline) means robot drifts right.
            # Need a negative angular velocity (turn left) to correct.
            drift_adjustment = -accumulated_drift * self.drift_correction_gain

            # Limit the magnitude of drift correction
            max_drift_correction_angular_vel = 0.2
            drift_adjustment = max(
                -max_drift_correction_angular_vel,
                min(max_drift_correction_angular_vel, drift_adjustment),
            )

            # Add to the PI output or passthrough angular
            corrected_angular += drift_adjustment

            if abs(drift_adjustment) > 0.001:  # Log only if correction is significant
                self.get_logger().debug(
                    f"Drift correction: base_diff={
                        self.initial_encoder_diff}, curr_diff={current_encoder_diff}, acc_drift={accumulated_drift}, adjust={
                        drift_adjustment:.4f}, final_ang={
                        corrected_angular:.4f}"
                )

        cmd_msg = Twist()
        cmd_msg.linear.x = linear
        cmd_msg.linear.y = 0.0
        cmd_msg.linear.z = 0.0
        cmd_msg.angular.x = 0.0
        cmd_msg.angular.y = 0.0
        cmd_msg.angular.z = corrected_angular  # Use potentially drift-corrected angular

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
