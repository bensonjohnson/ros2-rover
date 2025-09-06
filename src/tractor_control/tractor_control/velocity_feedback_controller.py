import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, QuaternionStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
import time
import math
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
        self.declare_parameter("deadband", 0.001)  # Minimum velocity command
        self.declare_parameter(
            "drift_correction_gain", 0.0001
        )  # Gain for encoder drift correction
        # Logging controls
        self.declare_parameter("log_cmd_vel", False)  # If true, log published cmd_vel at INFO
        self.declare_parameter("log_received_cmd", False)  # If true, log received cmd_vel at INFO
        self.declare_parameter("log_throttle_sec", 1.0)  # Minimum seconds between repeated logs

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
        # Logging controls
        self.log_cmd_vel = bool(self.get_parameter("log_cmd_vel").value)
        self.log_received_cmd = bool(self.get_parameter("log_received_cmd").value)
        self.log_throttle_sec = float(self.get_parameter("log_throttle_sec").value)
        self._last_pub_log_time = 0.0
        self._last_recv_log_time = 0.0

        # Control state
        self.desired_linear = 0.0
        self.desired_angular = 0.0
        self.current_linear = 0.0
        self.current_angular = 0.0
        self.current_yaw = 0.0  # Current heading from filtered odometry
        self.emergency_stop = False
        
        # Compass-based heading tracking
        self.current_heading = 0.0  # Current compass heading in radians
        self.target_heading = None  # Target heading when going straight
        self.heading_set = False

        # Encoder drift correction
        self.left_encoder_total = 0
        self.right_encoder_total = 0
        self.initial_encoder_diff = 0
        self.encoder_diff_baseline_set = False

        # PID state
        self.linear_integral = 0.0
        self.angular_integral = 0.0
        self.last_time = time.time()

        # Velocity tracking (initialize to zero)
        self.left_velocity = 0.0
        self.right_velocity = 0.0

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
            "odometry/filtered",  # Use robot_localization's filtered odometry
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

        # Subscribe to filtered odometry for heading (from robot_localization EKF)
        self.compass_sub = self.create_subscription(
            Odometry, "/odometry/filtered", self.odometry_callback, 10
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
            
        # Log received commands for debugging
        if self.log_received_cmd and (abs(msg.linear.x) > 0.001 or abs(msg.angular.z) > 0.001):
            now = time.time()
            if now - self._last_recv_log_time >= self.log_throttle_sec:
                self._last_recv_log_time = now
                self.get_logger().info(f"Received cmd_vel: linear={msg.linear.x:.3f}, angular={msg.angular.z:.3f}")
        else:
            # Quiet by default
            self.get_logger().debug(f"Received cmd_vel: linear={msg.linear.x:.3f}, angular={msg.angular.z:.3f}")

    def odom_callback(self, msg):
        """Receive filtered odometry from robot_localization EKF"""
        with self.control_lock:
            self.current_linear = msg.twist.twist.linear.x
            self.current_angular = msg.twist.twist.angular.z
            
            # Extract yaw from quaternion in pose
            q = msg.pose.pose.orientation
            self.current_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 
                                        1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            
            # Calculate individual wheel speeds from odometry
            # This is the reverse of the kinematics in the motor driver
            wheel_separation = self.wheel_separation
            self.left_velocity = (self.current_linear - self.current_angular * wheel_separation / 2.0) / self.wheel_radius
            self.right_velocity = (self.current_linear + self.current_angular * wheel_separation / 2.0) / self.wheel_radius

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

    def odometry_callback(self, msg):
        """Receive filtered odometry data for heading"""
        # Extract quaternion from odometry pose
        q = msg.pose.pose.orientation
        # Extract yaw from quaternion
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 
                        1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        
        with self.control_lock:
            self.current_heading = yaw
            if not self.heading_set:
                self.heading_set = True
                self.get_logger().info(f"EKF heading initialized: {math.degrees(yaw):.1f}°")

    def control_loop(self):
        """Simple drift correction - slow down faster track"""
        if self.emergency_stop:
            # Send stop command during emergency stop
            self.publish_velocity_command(0.0, 0.0)
            return

        with self.control_lock:
            # Apply deadband to desired velocities
            desired_linear = self.desired_linear
            desired_angular = self.desired_angular
            
            if abs(desired_linear) < self.deadband:
                desired_linear = 0.0
            if abs(desired_angular) < self.deadband:
                desired_angular = 0.0

            # For very low speeds or intentional turns, pass through without correction
            if abs(desired_linear) < 0.05 or abs(desired_angular) > 0.3:
                self.publish_velocity_command(desired_linear, desired_angular)
                if abs(desired_angular) > 0.3:
                    self.get_logger().debug("Intentional turn detected, no drift correction")
                    # Reset target heading when turning
                    self.target_heading = None
                return

            # EKF-based drift correction with stability monitoring (ready for D435i integration)
            corrected_angular = desired_angular
            
            if (abs(desired_linear) > 0.02 and abs(desired_angular) < 0.1):
                
                # Set target heading when starting to go straight (use EKF filtered heading)
                if self.target_heading is None:
                    self.target_heading = self.current_yaw  # Use EKF heading
                    self.get_logger().debug(f"Target heading set from EKF: {math.degrees(self.current_yaw):.1f}°")
                
                # Calculate heading error using EKF filtered odometry
                if self.target_heading is not None:
                    heading_error = self.current_yaw - self.target_heading
                    
                    # Normalize heading error to [-pi, pi]
                    while heading_error > math.pi:
                        heading_error -= 2 * math.pi
                    while heading_error < -math.pi:
                        heading_error += 2 * math.pi
                    
                    # EKF stability check - if huge errors, gradually adapt target instead of fighting
                    if abs(heading_error) > 1.57:  # 90 degrees - EKF likely unstable
                        # Gradually move target toward current heading (adaptive target)
                        target_adjustment = heading_error * 0.1  # Move target 10% toward current
                        self.target_heading += target_adjustment
                        heading_error *= 0.9  # Reduce error for this cycle
                        self.get_logger().warn(f"EKF instability detected, adapting target by {math.degrees(target_adjustment):.1f}°")
                    
                    # Apply correction - stronger for stable EKF, gentler for unstable
                    if abs(heading_error) > 0.02:  # ~1 degree threshold
                        if abs(heading_error) < 0.52:  # Less than 30° - stable correction
                            heading_correction = -heading_error * 0.6  # Strong gain for small errors
                            heading_correction = max(-0.5, min(0.5, heading_correction))
                            correction_type = "stable"
                        else:  # Large error - gentler correction
                            heading_correction = -heading_error * 0.3  # Gentler for large errors
                            heading_correction = max(-0.4, min(0.4, heading_correction))
                            correction_type = "adaptive"
                        
                        corrected_angular += heading_correction
                        
                        self.get_logger().debug(
                            f"EKF correction ({correction_type}): heading_error={math.degrees(heading_error):.1f}°, "
                            f"correction={heading_correction:.3f}"
                        )
            
            self.publish_velocity_command(desired_linear, corrected_angular)

    def publish_velocity_command(self, linear, angular):
        """Publish velocity command"""
        cmd_msg = Twist()
        cmd_msg.linear.x = linear
        cmd_msg.linear.y = 0.0
        cmd_msg.linear.z = 0.0
        cmd_msg.angular.x = 0.0
        cmd_msg.angular.y = 0.0
        cmd_msg.angular.z = angular

        # Log published commands for debugging
        if self.log_cmd_vel and (abs(linear) > 0.001 or abs(angular) > 0.001):
            now = time.time()
            if now - self._last_pub_log_time >= self.log_throttle_sec:
                self._last_pub_log_time = now
                self.get_logger().info(f"Publishing cmd_vel: linear={linear:.3f}, angular={angular:.3f}")
        else:
            self.get_logger().debug(f"Publishing cmd_vel: linear={linear:.3f}, angular={angular:.3f}")

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
