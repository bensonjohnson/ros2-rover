#!/usr/bin/env python3
"""
ROS 2 Node for controlling Hiwonder I2C DC motor driver board.

This node interfaces with the Hiwonder motor controller to drive two motors
(typically for a differential drive robot), read encoder values for odometry,
and monitor battery voltage. It subscribes to /cmd_vel for velocity commands
and publishes odometry, joint states, and battery information.
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from std_msgs.msg import Float32MultiArray, Float32
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
import tf2_ros
import smbus
import time
import struct
import math
from threading import Lock


class HiwonderMotorDriver(Node):
    def __init__(self):
        super().__init__("hiwonder_motor_driver")

        # Parameters (using corrected addresses from ESP32 testing)
        self.declare_parameter("i2c_bus", 5)
        # Default I2C address for Hiwonder controller. Verified via ESP32
        # testing.
        self.declare_parameter("motor_controller_address", 0x34)
        # meters, distance between the centers of the two wheels.
        self.declare_parameter("wheel_separation", 0.5)
        # meters, radius of the drive wheels.
        self.declare_parameter("wheel_radius", 0.15)
        # Max speed value sent to controller. Matches Arduino MAX_MOTOR_SPEED
        # from Hiwonder examples.
        self.declare_parameter("max_motor_speed", 50)
        # Joystick deadband for velocity commands.
        self.declare_parameter("deadband", 0.05)
        # Pulses per revolution for motor encoders. JGB3865-520R45: 44 base
        # pulses * 45:1 gear ratio = 1980.
        self.declare_parameter("encoder_ppr", 1980)
        # DEPRECATED. Sensor data is published at sensor_timer rate (100Hz).
        # This param is not actively used.
        self.declare_parameter("publish_rate", 5.0)
        # True for open-loop PWM control (e.g., for motors without encoders
        # like JGB3865). False for closed-loop speed control.
        self.declare_parameter("use_pwm_control", True)
        # Controller's motor type setting. 0 for generic/no-encoder, 3 for
        # JGB37-520. See controller docs.
        self.declare_parameter("motor_type", 0)
        # Min battery voltage samples needed to estimate runtime.
        self.declare_parameter("min_samples_for_estimation", 10)
        # Duration of voltage history for runtime estimation (minutes).
        self.declare_parameter("max_history_minutes", 60)
        # Seconds to sleep after an I2C write operation.
        self.declare_parameter("i2c_write_delay_secs", 0.005)
        # Seconds to sleep between reading consecutive bytes in a multi-byte
        # read.
        self.declare_parameter("i2c_read_inter_byte_delay_secs", 0.001)
        # Minimum interval between sending motor commands.
        self.declare_parameter("motor_command_rate_limit_secs", 0.1)
        # Delay after setting motor type during initialization.
        self.declare_parameter("init_motor_type_delay_secs", 0.5)
        # Delay after setting encoder polarity during initialization.
        self.declare_parameter("init_encoder_polarity_delay_secs", 0.1)
        # Use hardware encoder clear/reset on startup (I2C register 0x52)?
        # If false, will establish software baseline for encoder counts.
        self.declare_parameter("hardware_clear_encoders", True)
        # Tolerance in encoder counts for verifying successful hardware clear.
        self.declare_parameter("encoder_clear_verify_tolerance", 50)  # counts within which hardware clear considered successful
        # NEW: Timeout for receiving /cmd_vel before auto stop (seconds)
        self.declare_parameter("cmd_vel_timeout_secs", 0.5)
        # NEW: Watchdog check frequency (Hz)
        self.declare_parameter("watchdog_check_hz", 20.0)

        self.i2c_bus = self.get_parameter("i2c_bus").value
        self.motor_address = self.get_parameter("motor_controller_address").value
        self.wheel_separation = self.get_parameter("wheel_separation").value
        self.wheel_radius = self.get_parameter("wheel_radius").value
        self.max_motor_speed = self.get_parameter("max_motor_speed").value
        self.deadband = self.get_parameter("deadband").value
        self.encoder_ppr = self.get_parameter("encoder_ppr").value
        self.publish_rate = self.get_parameter("publish_rate").value
        self.use_pwm_control = self.get_parameter("use_pwm_control").value
        self.motor_type = self.get_parameter("motor_type").value
        self.min_samples_for_estimation = self.get_parameter(
            "min_samples_for_estimation"
        ).value
        self.max_history_minutes = self.get_parameter("max_history_minutes").value
        self.i2c_write_delay = self.get_parameter("i2c_write_delay_secs").value
        self.i2c_read_inter_byte_delay = self.get_parameter(
            "i2c_read_inter_byte_delay_secs"
        ).value
        self.motor_command_rate_limit = self.get_parameter(
            "motor_command_rate_limit_secs"
        ).value
        self.init_motor_type_delay = self.get_parameter(
            "init_motor_type_delay_secs"
        ).value
        self.init_encoder_polarity_delay = self.get_parameter(
            "init_encoder_polarity_delay_secs"
        ).value
        self.hardware_clear_encoders = self.get_parameter("hardware_clear_encoders").value
        self.encoder_clear_verify_tolerance = self.get_parameter("encoder_clear_verify_tolerance").value
        # NEW: Retrieve watchdog parameters
        self.cmd_vel_timeout_secs = self.get_parameter("cmd_vel_timeout_secs").value
        self.watchdog_check_hz = self.get_parameter("watchdog_check_hz").value

        # Battery monitoring history
        # List of (timestamp, voltage, percentage) tuples
        self.voltage_history = []
        self.start_time = time.time()

        # I2C Register addresses (corrected from ESP32 testing)
        self.ADC_BAT_ADDR = 0x00
        self.MOTOR_TYPE_ADDR = 0x14
        self.MOTOR_ENCODER_POLARITY_ADDR = 0x15
        self.MOTOR_FIXED_PWM_ADDR = 0x1F  # PWM control for non-encoder motors
        self.MOTOR_FIXED_SPEED_ADDR = 0x33  # Speed control for encoder motors
        self.MOTOR_ENCODER_TOTAL_ADDR = 0x3C
        self.MOTOR_ENCODER_CLEAR_ADDR = 0x52  # Clear/reset encoder counts

        # Motor types
        # For 3865-520 motors (if no encoder)
        self.MOTOR_TYPE_WITHOUT_ENCODER = 0
        self.MOTOR_TYPE_TT = 1
        self.MOTOR_TYPE_N20 = 2
        # 90:1 gear ratio (44 pulses per rev)
        self.MOTOR_TYPE_JGB37_520_12V = 3
        # Note: JGB3865-520R45-12 has 45:1 gear ratio (44 pulses × 45:1 = 1980 PPR)
        # Use motor_type=3 with encoder_ppr=1980 for JGB3865-520R45-12

        # Initialize I2C
        try:
            self.bus = smbus.SMBus(self.i2c_bus)
            self.get_logger().info(
                f"I2C bus {self.i2c_bus} initialized successfully"
            )
            self.init_motor_driver()
        except Exception as e:
            self.get_logger().error(f"Failed to initialize I2C bus: {e}")
            self.bus = None

        # State tracking
        self.encoder_lock = Lock()
        self.last_encoder_time = time.time()
        self.prev_left_encoder = 0
        self.prev_right_encoder = 0
        self.left_velocity = 0.0
        self.right_velocity = 0.0
        self.read_count = 0

        # Motor command rate limiting
        self.last_motor_command_time = 0.0
        # Track last sent speeds for immediate stop logic
        self.last_sent_left = 0
        self.last_sent_right = 0

        # Odometry state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Watchdog tracking
        self.last_cmd_vel_msg_time = time.time()
        self.watchdog_last_active = True  # motors considered active until proven idle

        # Subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist, "cmd_vel", self.cmd_vel_callback, 10
        )

        # Publishers
        self.motor_speeds_pub = self.create_publisher(
            Float32MultiArray, "motor_speeds", 10
        )

        self.battery_voltage_pub = self.create_publisher(Float32, "battery_voltage", 10)

        self.battery_percentage_pub = self.create_publisher(
            Float32, "battery_percentage", 10
        )

        self.battery_runtime_pub = self.create_publisher(Float32, "battery_runtime", 10)

        self.joint_state_pub = self.create_publisher(JointState, "joint_states", 10)

        self.odom_pub = self.create_publisher(Odometry, "odom", 10)

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Timer for sensor readings - 100Hz for excellent odometry feedback
        # (faster I2C after speed fix)
        self.sensor_timer = self.create_timer(
            1.0 / 100.0, self.sensor_callback
        )  # 100 Hz (10ms intervals)
        self.battery_timer = self.create_timer(
            5.0, self.battery_callback
        )  # Every 5 seconds
        # NEW: Watchdog timer for cmd_vel timeout
        if self.watchdog_check_hz > 0:
            self.watchdog_timer = self.create_timer(1.0 / self.watchdog_check_hz, self.watchdog_check)
        else:
            self.watchdog_timer = None

        self.get_logger().info("Hiwonder Motor Driver initialized")

    def init_motor_driver(self):
        """Initialize motor driver with correct settings from ESP32 testing"""
        self.get_logger().info("Initializing motor driver with corrected addresses...")

        try:
            # Check if I2C device is responding
            # Try to read a byte from the device
            try:
                self.bus.read_byte(self.motor_address)
                self.get_logger().info(f"I2C device found at address 0x{self.motor_address:02X}")
            except Exception as e:
                self.get_logger().error(f"I2C device not responding at address 0x{self.motor_address:02X}: {e}")
                return

            # Set motor type based on parameter - using official documentation
            # method
            self.bus.write_byte_data(
                self.motor_address, self.MOTOR_TYPE_ADDR, self.motor_type
            )
            motor_names = {0: "No Encoder", 1: "TT", 2: "N20", 3: "JGB37"}
            motor_name = motor_names.get(self.motor_type, f"Unknown({self.motor_type})")
            self.get_logger().info(
                f"Motor type set to {motor_name} (value {
                    self.motor_type}) at address 0x{
                    self.MOTOR_TYPE_ADDR:02X}"
            )
            if self.init_motor_type_delay > 0:
                # Delay based on official documentation or hardware needs.
                time.sleep(self.init_motor_type_delay)

            # Set encoder polarity - using official documentation method
            # Default: 0. Try 1 if motors behave unexpectedly or don't respond
            # with encoders.
            encoder_polarity = 0
            self.bus.write_byte_data(
                self.motor_address, self.MOTOR_ENCODER_POLARITY_ADDR, encoder_polarity
            )
            self.get_logger().info(
                f"Encoder polarity set to {encoder_polarity} at address 0x{
                    self.MOTOR_ENCODER_POLARITY_ADDR:02X}"
            )
            if self.init_encoder_polarity_delay > 0:
                # Delay based on official documentation or hardware needs.
                time.sleep(self.init_encoder_polarity_delay)

            # Reset/clear encoder counts to start fresh
            self.reset_encoders()

            # Log control method being used
            control_method = (
                "Speed control (with encoders)"
                if not self.use_pwm_control
                else "PWM control (no encoders)"
            )
            self.get_logger().info(f"Using {control_method} for JGB3865 motors")

            self.get_logger().info("Motor driver initialized successfully!")

        except Exception as e:
            self.get_logger().error(f"Failed to initialize motor driver: {e}")

    def reset_encoders(self):
        """Reset encoder counts to zero via I2C or establish software baseline."""
        try:
            hw_attempted = False
            hw_success = False
            raw_left = None
            raw_right = None
            if self.bus is not None and self.hardware_clear_encoders:
                try:
                    self.bus.write_byte_data(self.motor_address, self.MOTOR_ENCODER_CLEAR_ADDR, 1)
                    hw_attempted = True
                    time.sleep(0.1)
                    # Read back counts to verify
                    encoder_data = self.read_data_array(self.MOTOR_ENCODER_TOTAL_ADDR, 8)
                    if encoder_data and len(encoder_data) == 8:
                        encoders = struct.unpack("<2i", bytes(encoder_data))
                        raw_right = encoders[0]
                        raw_left = encoders[1]
                        if (
                            abs(raw_left) <= self.encoder_clear_verify_tolerance and
                            abs(raw_right) <= self.encoder_clear_verify_tolerance
                        ):
                            hw_success = True
                except Exception as e:
                    self.get_logger().warn(f"Hardware encoder clear failed: {e}")
            if hw_success:
                with self.encoder_lock:
                    self.encoder_baseline_left = 0
                    self.encoder_baseline_right = 0
                    self.prev_left_encoder = 0
                    self.prev_right_encoder = 0
                    self.left_velocity = 0.0
                    self.right_velocity = 0.0
                self.x = 0.0; self.y = 0.0; self.theta = 0.0
                self.get_logger().info("Encoder hardware clear succeeded (counts near zero)")
            else:
                # Establish software baseline from first read of current (if not already provided)
                if raw_left is None or raw_right is None:
                    encoder_data = self.read_data_array(self.MOTOR_ENCODER_TOTAL_ADDR, 8)
                    if encoder_data and len(encoder_data) == 8:
                        encoders = struct.unpack("<2i", bytes(encoder_data))
                        raw_right = encoders[0]
                        raw_left = encoders[1]
                if raw_left is not None and raw_right is not None:
                    self.encoder_baseline_left = raw_left
                    self.encoder_baseline_right = raw_right
                    with self.encoder_lock:
                        self.prev_left_encoder = 0
                        self.prev_right_encoder = 0
                        self.left_velocity = 0.0
                        self.right_velocity = 0.0
                    self.x = 0.0; self.y = 0.0; self.theta = 0.0
                    source = "hardware clear failed" if hw_attempted else "hardware clear skipped"
                    self.get_logger().warn(
                        f"Using software encoder baseline (left={raw_left}, right={raw_right}) because {source}. Effective counts will start at zero.")
                else:
                    self.get_logger().error("Unable to read encoders to set software baseline; proceeding with zeros")
                    self.encoder_baseline_left = 0
                    self.encoder_baseline_right = 0
            self.get_logger().debug(f"Encoder baselines: L={self.encoder_baseline_left}, R={self.encoder_baseline_right}")
        except Exception as e:
            self.get_logger().error(f"Failed to reset/initialize encoders: {e}")

    def write_byte(self, val):
        """Write a single byte to I2C device"""
        try:
            self.bus.write_byte(self.motor_address, val)
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to write byte: {e}")
            return False

    def write_data_array(self, reg, data):
        """
        Write a list of bytes to a specified I2C register.

        This method attempts to replicate the Hiwonder Arduino library's approach,
        which typically involves writing the register address then the data bytes
        in a single I2C transaction.
        A small delay, as found in manufacturer's code, is added after the write.
        """
        try:
            if len(data) == 1:
                # For a single byte, use write_byte_data
                self.bus.write_byte_data(self.motor_address, reg, data[0])
            else:
                # For multiple bytes, use write_i2c_block_data
                # This sends the register address followed by the data bytes.
                self.bus.write_i2c_block_data(self.motor_address, reg, data)

            # Delay found in some manufacturer example code, potentially for command processing.
            # Evaluate if this is necessary or can be tuned.
            if self.i2c_write_delay > 0:
                time.sleep(self.i2c_write_delay)
            return True
        except Exception as e:
            self.get_logger().error(
                f"Failed to write data array to reg 0x{
                    reg:02X}: {e}"
            )
            return False

    def read_data_array(self, reg, length):
        """
        Read a specified number of bytes from a starting I2C register.

        This method reads bytes individually by addressing `reg + i` for each byte.
        This approach was found to be reliable ("working shell method") during development.
        Includes small inter-byte delays, also found helpful during testing.
        """
        try:
            data = []
            for i in range(length):
                byte_val = self.bus.read_byte_data(self.motor_address, reg + i)
                data.append(byte_val)
                # Small delay between reads, except for the last byte.
                if i < length - 1:
                    # This delay was found to improve read stability with this specific controller.
                    # Evaluate if this is always necessary or can be
                    # tuned/removed.
                    if self.i2c_read_inter_byte_delay > 0:
                        time.sleep(self.i2c_read_inter_byte_delay)

            return data
        except Exception as e:
            self.get_logger().error(
                f"Failed to read data array from reg 0x{
                    reg:02X}: {e}"
            )
            return None

    def cmd_vel_callback(self, msg):
        """Convert twist command to tank steering motor speeds"""
        linear_vel = msg.linear.x
        angular_vel = msg.angular.z

        # Apply deadband
        if abs(linear_vel) < self.deadband:
            linear_vel = 0.0
        if abs(angular_vel) < self.deadband:
            angular_vel = 0.0

        # Tank steering kinematics
        left_speed = linear_vel - (angular_vel * self.wheel_separation / 2.0)
        right_speed = linear_vel + (angular_vel * self.wheel_separation / 2.0)

        # Convert to motor speeds (-max_motor_speed to max_motor_speed)
        left_motor = int(left_speed * self.max_motor_speed)
        right_motor = int(right_speed * self.max_motor_speed)

        # Clamp motor speeds
        left_motor = max(-self.max_motor_speed, min(self.max_motor_speed, left_motor))
        right_motor = max(-self.max_motor_speed, min(self.max_motor_speed, right_motor))

        # Send to motor controller
        self.send_motor_speeds(left_motor, right_motor)

        # Publish motor speeds for feedback
        motor_msg = Float32MultiArray()
        motor_msg.data = [float(left_motor), float(right_motor)]
        self.motor_speeds_pub.publish(motor_msg)

        if left_motor != 0 or right_motor != 0:
            self.get_logger().debug(f"Motor speeds: L={left_motor}, R={right_motor}")

        # Update watchdog timestamp
        self.last_cmd_vel_msg_time = time.time()

    def send_motor_speeds(self, left_speed, right_speed):
        """Send motor speeds via I2C using corrected protocol with rate limiting

        Immediate stop (both speeds zero) bypasses rate limiting if previous command was non-zero.
        """
        if self.bus is None:
            return

        current_time = time.time()
        # Determine if this is an emergency / immediate stop
        is_stop_command = (left_speed == 0 and right_speed == 0)
        was_moving = (self.last_sent_left != 0 or self.last_sent_right != 0)
        bypass_rate_limit = is_stop_command and was_moving

        # Rate limiting to prevent I2C bus overload (unless bypass)
        if not bypass_rate_limit and (current_time - self.last_motor_command_time < self.motor_command_rate_limit):
            self.get_logger().debug("Motor command rate limited")
            return

        try:
            speeds = [right_speed, left_speed, 0, 0]
            speeds_bytes = [max(-127, min(127, int(s))) for s in speeds]

            if left_speed != 0 or right_speed != 0:
                control_type = "PWM" if self.use_pwm_control else "Speed"
                self.get_logger().debug(
                    f"Sending {control_type} motor command: L={left_speed}, R={right_speed}"
                )
            elif bypass_rate_limit:
                self.get_logger().debug("Immediate STOP command sent (bypassed rate limit)")

            control_addr = (
                self.MOTOR_FIXED_PWM_ADDR
                if self.use_pwm_control
                else self.MOTOR_FIXED_SPEED_ADDR
            )
            self.bus.write_i2c_block_data(
                self.motor_address, control_addr, speeds_bytes
            )
            if left_speed != 0 or right_speed != 0:
                self.get_logger().debug("Motor command sent successfully")
            self.last_motor_command_time = current_time
            self.last_sent_left = left_speed
            self.last_sent_right = right_speed
        except Exception as e:
            self.get_logger().error(f"Motor command error: {e}")

    def sensor_callback(self):
        """Read encoders and publish sensor data"""
        if self.bus is None:
            return
        try:
            encoder_data = self.read_data_array(self.MOTOR_ENCODER_TOTAL_ADDR, 8)
            if encoder_data and len(encoder_data) == 8:
                encoders = struct.unpack("<2i", bytes(encoder_data))
                raw_right = encoders[0]
                raw_left = encoders[1]
                # Apply software baseline if set
                if self.encoder_baseline_left is None or self.encoder_baseline_right is None:
                    # First time sensor callback before reset_encoders established baseline
                    self.encoder_baseline_left = raw_left
                    self.encoder_baseline_right = raw_right
                    self.get_logger().info(f"Encoder baseline auto-set in sensor loop L={raw_left} R={raw_right}")
                effective_left = raw_left - self.encoder_baseline_left
                effective_right = raw_right - self.encoder_baseline_right
                self.read_count += 1
                current_time = time.time()
                dt = current_time - self.last_encoder_time
                if dt > 0:
                    with self.encoder_lock:
                        left_delta = effective_left - self.prev_left_encoder
                        right_delta = effective_right - self.prev_right_encoder
                        if dt > 0.001:
                            self.left_velocity = (left_delta / self.encoder_ppr) * 2 * math.pi / dt
                            self.right_velocity = (right_delta / self.encoder_ppr) * 2 * math.pi / dt
                        self.prev_left_encoder = effective_left
                        self.prev_right_encoder = effective_right
                self.last_encoder_time = current_time
                # Publish using effective counts so downstream sees zeroed start
                self.publish_joint_states(effective_left, effective_right)
                self.publish_odometry()
                # Log at reduced rate to avoid spam (every 2.5 seconds = 250
                # reads at 100Hz)
                if self.read_count % 250 == 0:
                    self.get_logger().info(
                        f"Encoders @100Hz raw L={raw_left} R={raw_right} eff L={effective_left} R={effective_right} Vel L={self.left_velocity:.2f} R={self.right_velocity:.2f}")
                else:
                    self.get_logger().debug(
                        f"Encoders: L={raw_left}, R={raw_right}, "
                        f"Velocities: L={
                            self.left_velocity:.2f}, R={
                            self.right_velocity:.2f}"
                    )
            else:
                self.get_logger().warning("Invalid encoder data received")
                self.publish_joint_states(0, 0)
                self.publish_odometry()
        except Exception as e:
            self.get_logger().error(f"Encoder reading error: {e}")
            self.publish_joint_states(0, 0)
            self.publish_odometry()

    def publish_joint_states(self, left_encoder, right_encoder):
        """Publish joint states from encoder data"""
        # Tank tracks are fixed joints - don't publish rotating positions!
        # The encoder data is used for odometry calculation, not joint
        # visualization

        # Convert encoder counts to wheel radians for visualization wheels
        left_wheel_pos = (left_encoder / self.encoder_ppr) * 2 * math.pi
        right_wheel_pos = (right_encoder / self.encoder_ppr) * 2 * math.pi

        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = [
            "left_wheel_joint",
            "right_wheel_joint",
            "left_viz_wheel_joint",
            "right_viz_wheel_joint",
        ]
        # Publish raw encoder counts for left_wheel_joint and right_wheel_joint
        # and scaled radians for visualization wheels.
        joint_msg.position = [
            float(left_encoder),
            float(right_encoder),
            left_wheel_pos,
            right_wheel_pos,
        ]
        # Fixed tracks + wheel velocities
        joint_msg.velocity = [0.0, 0.0, self.left_velocity, self.right_velocity]

        self.joint_state_pub.publish(joint_msg)

    def publish_odometry(self):
        """Calculate and publish wheel odometry"""
        # Tank steering kinematics (fixed coordinate frame orientation)
        linear_vel = (
            (self.left_velocity + self.right_velocity) * self.wheel_radius / 2.0
        )
        angular_vel = (
            -(self.left_velocity - self.right_velocity)
            * self.wheel_radius
            / self.wheel_separation
        )

        # Update pose (integrate velocities)
        dt = 1.0 / 100.0  # 100Hz sensor callback frequency (10ms)
        self.x += linear_vel * math.cos(self.theta) * dt
        self.y += linear_vel * math.sin(self.theta) * dt
        self.theta += angular_vel * dt

        # Normalize theta
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        # Create odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_footprint"

        # Position
        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.position.z = 0.0

        # Orientation (quaternion from yaw)
        odom_msg.pose.pose.orientation.x = 0.0
        odom_msg.pose.pose.orientation.y = 0.0
        odom_msg.pose.pose.orientation.z = math.sin(self.theta / 2.0)
        odom_msg.pose.pose.orientation.w = math.cos(self.theta / 2.0)

        # Velocity
        odom_msg.twist.twist.linear.x = linear_vel
        odom_msg.twist.twist.linear.y = 0.0
        odom_msg.twist.twist.angular.z = angular_vel

        # Covariance (simplified)
        odom_msg.pose.covariance[0] = 0.1  # x
        odom_msg.pose.covariance[7] = 0.1  # y
        odom_msg.pose.covariance[35] = 0.1  # theta
        odom_msg.twist.covariance[0] = 0.1  # vx
        odom_msg.twist.covariance[35] = 0.1  # vtheta

        self.odom_pub.publish(odom_msg)

        # Publish TF transform (odom -> base_footprint, base_footprint ->
        # base_link is in URDF)
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = "odom"
        tf_msg.child_frame_id = "base_footprint"
        tf_msg.transform.translation.x = self.x
        tf_msg.transform.translation.y = self.y
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation = odom_msg.pose.pose.orientation

        self.tf_broadcaster.sendTransform(tf_msg)

    def battery_callback(self):
        """Read and publish battery voltage from motor controller"""
        if self.bus is None:
            return

        try:
            # Read 2 bytes from battery voltage register (like ESP32 code)
            voltage_data = self.read_data_array(self.ADC_BAT_ADDR, 2)
            if voltage_data and len(voltage_data) == 2:
                # Combine bytes: data[0] + (data[1] << 8) like ESP32
                voltage_raw = voltage_data[0] + (voltage_data[1] << 8)

                # Convert to volts (assuming millivolts from controller)
                voltage_volts = voltage_raw / 1000.0

                # Publish the real voltage
                voltage_msg = Float32()
                voltage_msg.data = voltage_volts
                self.battery_voltage_pub.publish(voltage_msg)

                # Calculate and publish battery percentage (3S LiPo: 9.9V=0%,
                # 12.6V=100%)
                battery_percentage = self.calculate_battery_percentage(voltage_volts)
                percentage_msg = Float32()
                percentage_msg.data = battery_percentage
                self.battery_percentage_pub.publish(percentage_msg)

                # Update voltage history for drain rate calculation
                current_time = time.time()
                self.update_voltage_history(
                    current_time, voltage_volts, battery_percentage
                )

                # Calculate and publish estimated runtime remaining based on
                # actual drain rate
                runtime_hours = self.calculate_dynamic_runtime()
                runtime_msg = Float32()
                runtime_msg.data = runtime_hours
                self.battery_runtime_pub.publish(runtime_msg)

                # Enhanced logging with runtime status
                if runtime_hours < 0:
                    runtime_status = "calculating..."
                else:
                    runtime_status = f"{runtime_hours:.1f}h remaining"

                self.get_logger().debug(
                    f"Battery: {voltage_raw}mV ({
                        voltage_volts:.1f}V, {
                        battery_percentage:.1f}%, {runtime_status}) [samples: {
                        len(
                            self.voltage_history)}]"
                )
            else:
                # Fallback to dummy value if read fails
                voltage_msg = Float32()
                voltage_msg.data = 12.0
                self.battery_voltage_pub.publish(voltage_msg)

                # Publish dummy percentage and runtime too
                dummy_percentage = self.calculate_battery_percentage(12.0)
                percentage_msg = Float32()
                percentage_msg.data = dummy_percentage
                self.battery_percentage_pub.publish(percentage_msg)

                runtime_msg = Float32()
                runtime_msg.data = -1.0  # Indicate unknown runtime
                self.battery_runtime_pub.publish(runtime_msg)

                self.get_logger().debug("Battery read failed - using dummy 12V")

        except Exception as e:
            # Fallback to dummy value on error
            voltage_msg = Float32()
            voltage_msg.data = 12.0
            self.battery_voltage_pub.publish(voltage_msg)

            # Publish dummy percentage and runtime too
            dummy_percentage = self.calculate_battery_percentage(12.0)
            percentage_msg = Float32()
            percentage_msg.data = dummy_percentage
            self.battery_percentage_pub.publish(percentage_msg)

            runtime_msg = Float32()
            runtime_msg.data = -1.0  # Indicate unknown runtime
            self.battery_runtime_pub.publish(runtime_msg)

            self.get_logger().debug(f"Battery reading error: {e} - using dummy 12V")

    def calculate_battery_percentage(self, voltage):
        """Calculate battery percentage for 3S LiPo (9.9V = 0%, 12.6V = 100%)"""
        min_voltage = 9.9  # 0% charge (3.3V per cell)
        max_voltage = 12.6  # 100% charge (4.2V per cell)

        # Clamp voltage to valid range
        voltage = max(min_voltage, min(max_voltage, voltage))

        # Calculate percentage
        percentage = ((voltage - min_voltage) / (max_voltage - min_voltage)) * 100.0

        return max(0.0, min(100.0, percentage))

    def update_voltage_history(self, timestamp, voltage, percentage):
        """Update voltage history and clean old entries"""
        # Add new reading
        self.voltage_history.append((timestamp, voltage, percentage))

        # Remove entries older than max_history_minutes
        cutoff_time = timestamp - (self.max_history_minutes * 60)
        self.voltage_history = [
            entry for entry in self.voltage_history if entry[0] > cutoff_time
        ]

    def calculate_dynamic_runtime(self):
        """Calculate runtime based on actual battery drain rate"""
        if len(self.voltage_history) < self.min_samples_for_estimation:
            return -1.0  # Not enough data yet

        # Get current and oldest readings
        current_time, current_voltage, current_percentage = self.voltage_history[-1]
        oldest_time, oldest_voltage, oldest_percentage = self.voltage_history[0]

        # Calculate time elapsed and voltage/percentage change
        time_elapsed_hours = (current_time - oldest_time) / 3600.0
        # voltage_drop = oldest_voltage - current_voltage # Unused variable
        percentage_drop = oldest_percentage - current_percentage

        # If no significant change or time, can't estimate
        if time_elapsed_hours < 0.01 or percentage_drop <= 0:
            return -1.0  # No meaningful drain detected

        # Calculate drain rate (percentage per hour)
        drain_rate_percent_per_hour = percentage_drop / time_elapsed_hours

        # Estimate remaining time based on current percentage and drain rate
        if drain_rate_percent_per_hour <= 0:
            return -1.0  # Invalid drain rate

        # Calculate hours remaining: current_percentage / drain_rate
        runtime_hours = current_percentage / drain_rate_percent_per_hour

        # Cap unrealistic estimates
        max_reasonable_hours = 24.0
        runtime_hours = min(runtime_hours, max_reasonable_hours)

        return max(0.0, runtime_hours)

    def watchdog_check(self):
        """Stop motors if /cmd_vel not received recently."""
        if self.cmd_vel_timeout_secs <= 0:
            return
        elapsed = time.time() - self.last_cmd_vel_msg_time
        if elapsed > self.cmd_vel_timeout_secs:
            # Timeout expired; ensure motors stopped
            if self.last_sent_left != 0 or self.last_sent_right != 0:
                self.get_logger().warn(
                    f"/cmd_vel timeout ({elapsed:.2f}s > {self.cmd_vel_timeout_secs}s). Stopping motors." 
                )
                self.send_motor_speeds(0, 0)
        # (Optional future: publish diagnostic status)

    def destroy_node(self):
        """Clean shutdown"""
        if self.bus is not None:
            try:
                # Stop motors immediately (bypass rate limit) before closing
                self.send_motor_speeds(0, 0)
                time.sleep(0.05)
                self.bus.close()
                self.get_logger().info("Motor driver shutdown complete")
            except Exception as e:
                self.get_logger().error(f"Error during cleanup: {e}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    motor_driver = HiwonderMotorDriver()

    try:
        rclpy.spin(motor_driver)
    except KeyboardInterrupt:
        pass
    finally:
        motor_driver.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
