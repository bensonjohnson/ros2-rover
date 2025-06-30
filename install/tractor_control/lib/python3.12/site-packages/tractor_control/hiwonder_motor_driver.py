#!/usr/bin/env python3

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
        super().__init__('hiwonder_motor_driver')
        
        # Parameters (using corrected addresses from ESP32 testing)
        self.declare_parameter('i2c_bus', 5)
        self.declare_parameter('motor_controller_address', 0x34)  # Corrected from ESP32 testing
        self.declare_parameter('wheel_separation', 0.5)  # meters
        self.declare_parameter('wheel_radius', 0.15)  # meters
        self.declare_parameter('max_motor_speed', 50)  # Matches Arduino MAX_MOTOR_SPEED
        self.declare_parameter('deadband', 0.05)
        self.declare_parameter('encoder_ppr', 1980)  # JGB3865-520R45: 44 pulses * 45:1 ratio = 1980
        self.declare_parameter('publish_rate', 5.0)  # Hz - Reduced to prevent I2C overload
        self.declare_parameter('use_pwm_control', True)  # Use PWM for JGB3865 (open-loop)
        self.declare_parameter('motor_type', 0)  # 0=no encoder (PWM mode for JGB3865)
        
        self.i2c_bus = self.get_parameter('i2c_bus').value
        self.motor_address = self.get_parameter('motor_controller_address').value
        self.wheel_separation = self.get_parameter('wheel_separation').value
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.max_motor_speed = self.get_parameter('max_motor_speed').value
        self.deadband = self.get_parameter('deadband').value
        self.encoder_ppr = self.get_parameter('encoder_ppr').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.use_pwm_control = self.get_parameter('use_pwm_control').value
        self.motor_type = self.get_parameter('motor_type').value
        
        # I2C Register addresses (corrected from ESP32 testing)
        self.ADC_BAT_ADDR = 0x00
        self.MOTOR_TYPE_ADDR = 0x14
        self.MOTOR_ENCODER_POLARITY_ADDR = 0x15
        self.MOTOR_FIXED_PWM_ADDR = 0x1F      # PWM control for non-encoder motors
        self.MOTOR_FIXED_SPEED_ADDR = 0x33    # Speed control for encoder motors
        self.MOTOR_ENCODER_TOTAL_ADDR = 0x3C
        
        # Motor types
        self.MOTOR_TYPE_WITHOUT_ENCODER = 0   # For 3865-520 motors (if no encoder)
        self.MOTOR_TYPE_TT = 1
        self.MOTOR_TYPE_N20 = 2  
        self.MOTOR_TYPE_JGB37_520_12V = 3     # 90:1 gear ratio (44 pulses per rev)
        # Note: JGB3865-520R45-12 has 45:1 gear ratio - different encoder count
        
        # Initialize I2C
        try:
            self.bus = smbus.SMBus(self.i2c_bus)
            self.get_logger().info(f'I2C bus {self.i2c_bus} initialized successfully')
            self.init_motor_driver()
        except Exception as e:
            self.get_logger().error(f'Failed to initialize I2C bus: {e}')
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
        self.motor_command_rate_limit = 0.1  # Minimum 100ms between commands
        
        # Odometry state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        
        # Subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        # Publishers
        self.motor_speeds_pub = self.create_publisher(
            Float32MultiArray,
            'motor_speeds',
            10
        )
        
        self.battery_voltage_pub = self.create_publisher(
            Float32,
            'battery_voltage',
            10
        )
        
        self.joint_state_pub = self.create_publisher(
            JointState,
            'joint_states',
            10
        )
        
        self.odom_pub = self.create_publisher(
            Odometry,
            'odom',
            10
        )
        
        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Timer for sensor readings - 100Hz for excellent odometry feedback (faster I2C after speed fix)
        self.sensor_timer = self.create_timer(1.0/100.0, self.sensor_callback)  # 100 Hz (10ms intervals)
        self.battery_timer = self.create_timer(5.0, self.battery_callback)  # Every 5 seconds
        
        self.get_logger().info('Hiwonder Motor Driver initialized')
    
    def init_motor_driver(self):
        """Initialize motor driver with correct settings from ESP32 testing"""
        self.get_logger().info("Initializing motor driver with corrected addresses...")
        
        try:
            # Set motor type based on parameter - using official documentation method
            self.bus.write_byte_data(self.motor_address, self.MOTOR_TYPE_ADDR, self.motor_type)
            motor_names = {0: "No Encoder", 1: "TT", 2: "N20", 3: "JGB37"}
            motor_name = motor_names.get(self.motor_type, f"Unknown({self.motor_type})")
            self.get_logger().info(f"Motor type set to {motor_name} (value {self.motor_type}) at address 0x{self.MOTOR_TYPE_ADDR:02X}")
            time.sleep(0.5)  # Official documentation uses 0.5 second delay
            
            # Set encoder polarity - using official documentation method
            encoder_polarity = 0  # Try 1 if motors don't respond
            self.bus.write_byte_data(self.motor_address, self.MOTOR_ENCODER_POLARITY_ADDR, encoder_polarity)
            self.get_logger().info(f"Encoder polarity set to {encoder_polarity} at address 0x{self.MOTOR_ENCODER_POLARITY_ADDR:02X}")
            time.sleep(0.1)
            
            # Log control method being used
            control_method = "Speed control (with encoders)" if not self.use_pwm_control else "PWM control (no encoders)"
            self.get_logger().info(f"Using {control_method} for JGB3865 motors")
            
            self.get_logger().info("Motor driver initialized successfully!")
            
        except Exception as e:
            self.get_logger().error(f"Failed to initialize motor driver: {e}")
    
    def write_byte(self, val):
        """Write a single byte to I2C device"""
        try:
            self.bus.write_byte(self.motor_address, val)
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to write byte: {e}")
            return False
    
    def write_data_array(self, reg, data):
        """Write data array to I2C register - using Arduino-compatible single transaction method"""
        try:
            # Arduino method: Write register + all data in single I2C transaction
            # This matches: Wire.beginTransmission() -> Wire.write(reg) -> Wire.write(data[i]) -> Wire.endTransmission()
            
            if len(data) == 1:
                # Single byte write using standard method
                self.bus.write_byte_data(self.motor_address, reg, data[0])
            else:
                # Multi-byte write: Use I2C block write (standard SMBus method)
                self.bus.write_i2c_block_data(self.motor_address, reg, data)
            
            time.sleep(0.005)  # 5ms delay like manufacturer code
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to write data array to reg 0x{reg:02X}: {e}")
            return False
    
    def read_data_array(self, reg, length):
        """Read data array from I2C register - using working shell method"""
        try:
            # Method: Read individual bytes with register addressing
            # This matches the working shell command approach
            data = []
            for i in range(length):
                byte_val = self.bus.read_byte_data(self.motor_address, reg + i)
                data.append(byte_val)
                if i < length - 1:  # Small delay between reads except for last byte
                    time.sleep(0.001)
            
            return data
        except Exception as e:
            self.get_logger().error(f"Failed to read data array from reg 0x{reg:02X}: {e}")
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
            self.get_logger().debug(f'Motor speeds: L={left_motor}, R={right_motor}')
    
    def send_motor_speeds(self, left_speed, right_speed):
        """Send motor speeds via I2C using corrected protocol with rate limiting"""
        if self.bus is None:
            return
        
        # Rate limiting to prevent I2C bus overload
        current_time = time.time()
        if current_time - self.last_motor_command_time < self.motor_command_rate_limit:
            self.get_logger().debug(f"Motor command rate limited")
            return
        
        try:
            # Pack speeds as signed integers (M1=right, M2=left, M3=0, M4=0) 
            # Official documentation uses signed values directly: [-50,-50,-50,-50]
            speeds = [right_speed, left_speed, 0, 0]
            # Clamp to reasonable range for JGB37 motors
            speeds_bytes = [max(-127, min(127, int(s))) for s in speeds]
            
            # Add logging to see if commands are being sent
            if left_speed != 0 or right_speed != 0:
                control_type = "PWM" if self.use_pwm_control else "Speed"
                self.get_logger().info(f"Sending {control_type} motor command: L={left_speed}, R={right_speed}")
            
            # Choose control register based on motor type
            control_addr = self.MOTOR_FIXED_PWM_ADDR if self.use_pwm_control else self.MOTOR_FIXED_SPEED_ADDR
            
            # Send to motor controller using official documentation method
            self.bus.write_i2c_block_data(self.motor_address, control_addr, speeds_bytes)
            if left_speed != 0 or right_speed != 0:
                self.get_logger().info("✅ Motor command sent successfully")
            self.last_motor_command_time = current_time
                
        except Exception as e:
            self.get_logger().error(f'Motor command error: {e}')
    
    def sensor_callback(self):
        """Read encoders and publish sensor data"""
        if self.bus is None:
            return
            
        try:
            # Read encoder data from I2C motor controller (only need first 8 bytes for M1 and M2)
            encoder_data = self.read_data_array(self.MOTOR_ENCODER_TOTAL_ADDR, 8)
            if encoder_data and len(encoder_data) == 8:
                # Unpack 2 signed 32-bit integers (M1=right, M2=left encoder counts)
                encoders = struct.unpack('<2i', bytes(encoder_data))
                right_encoder = encoders[0]  # M1 (right motor)
                left_encoder = encoders[1]   # M2 (left motor)
                self.read_count += 1
                
                # Calculate velocities
                current_time = time.time()
                dt = current_time - self.last_encoder_time
                
                if dt > 0:
                    with self.encoder_lock:
                        left_delta = left_encoder - self.prev_left_encoder
                        right_delta = right_encoder - self.prev_right_encoder
                        
                        # Convert encoder deltas to wheel velocities (rad/s)
                        self.left_velocity = (left_delta / self.encoder_ppr) * 2 * math.pi / dt
                        self.right_velocity = (right_delta / self.encoder_ppr) * 2 * math.pi / dt
                        
                        self.prev_left_encoder = left_encoder
                        self.prev_right_encoder = right_encoder
                
                self.last_encoder_time = current_time
                
                # Publish joint states and odometry with real encoder data
                self.publish_joint_states(left_encoder, right_encoder)
                self.publish_odometry()
                
                # Log at reduced rate to avoid spam (every 2.5 seconds = 250 reads at 100Hz)
                if self.read_count % 250 == 0:
                    self.get_logger().info(f"Encoders @100Hz: L={left_encoder}, R={right_encoder}, "
                                         f"Velocities: L={self.left_velocity:.2f}, R={self.right_velocity:.2f}")
                else:
                    self.get_logger().debug(f"Encoders: L={left_encoder}, R={right_encoder}, "
                                          f"Velocities: L={self.left_velocity:.2f}, R={self.right_velocity:.2f}")
            else:
                self.get_logger().warning("Invalid encoder data received")
                # Publish zero values if read fails
                self.publish_joint_states(0, 0)
                self.publish_odometry()
                
        except Exception as e:
            self.get_logger().error(f'Encoder reading error: {e}')
            # Publish zero values on error
            self.publish_joint_states(0, 0)
            self.publish_odometry()
    
    def publish_joint_states(self, left_encoder, right_encoder):
        """Publish joint states from encoder data"""
        # Convert encoder counts to wheel positions
        left_pos = (left_encoder / self.encoder_ppr) * 2 * math.pi
        right_pos = (right_encoder / self.encoder_ppr) * 2 * math.pi
        
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = ['left_wheel_joint', 'right_wheel_joint']
        joint_msg.position = [left_pos, right_pos]
        joint_msg.velocity = [self.left_velocity, self.right_velocity]
        
        self.joint_state_pub.publish(joint_msg)
    
    def publish_odometry(self):
        """Calculate and publish wheel odometry"""
        # Tank steering kinematics
        linear_vel = (self.left_velocity + self.right_velocity) * self.wheel_radius / 2.0
        angular_vel = (self.right_velocity - self.left_velocity) * self.wheel_radius / self.wheel_separation
        
        # Update pose (integrate velocities)
        dt = 1.0/100.0  # 100Hz sensor callback frequency (10ms)
        self.x += linear_vel * math.cos(self.theta) * dt
        self.y += linear_vel * math.sin(self.theta) * dt
        self.theta += angular_vel * dt
        
        # Normalize theta
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
        
        # Create odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'
        
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
        odom_msg.pose.covariance[0] = 0.1   # x
        odom_msg.pose.covariance[7] = 0.1   # y
        odom_msg.pose.covariance[35] = 0.1  # theta
        odom_msg.twist.covariance[0] = 0.1  # vx
        odom_msg.twist.covariance[35] = 0.1 # vtheta
        
        self.odom_pub.publish(odom_msg)
        
        # Publish TF transform
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = 'odom'
        tf_msg.child_frame_id = 'base_link'
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
                
                self.get_logger().debug(f"Battery voltage: {voltage_raw}mV ({voltage_volts:.1f}V)")
            else:
                # Fallback to dummy value if read fails
                voltage_msg = Float32()
                voltage_msg.data = 12.0
                self.battery_voltage_pub.publish(voltage_msg)
                self.get_logger().debug("Battery read failed - using dummy 12V")
                
        except Exception as e:
            # Fallback to dummy value on error
            voltage_msg = Float32()
            voltage_msg.data = 12.0
            self.battery_voltage_pub.publish(voltage_msg)
            self.get_logger().debug(f'Battery reading error: {e} - using dummy 12V')
    
    def destroy_node(self):
        """Clean shutdown"""
        if self.bus is not None:
            try:
                # Stop motors
                self.send_motor_speeds(0, 0)
                time.sleep(0.1)
                self.bus.close()
                self.get_logger().info("Motor driver shutdown complete")
            except Exception as e:
                self.get_logger().error(f'Error during cleanup: {e}')
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


if __name__ == '__main__':
    main()