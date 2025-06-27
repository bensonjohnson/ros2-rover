#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import SetBool
from geometry_msgs.msg import Polygon, Point32
from action_msgs.msg import GoalStatus
import smbus2
import RPi.GPIO as GPIO
import time
import math
from threading import Lock


class MowerController(Node):
    def __init__(self):
        super().__init__('mower_controller')
        
        # Parameters
        self.declare_parameter('mower_enable_pin', 22)
        self.declare_parameter('mower_pwm_pin', 23)
        self.declare_parameter('blade_rpm_sensor_pin', 24)
        self.declare_parameter('safety_stop_pin', 25)
        self.declare_parameter('max_blade_rpm', 3000)
        self.declare_parameter('min_blade_rpm', 1500)
        self.declare_parameter('mower_height_i2c_addr', 0x40)
        self.declare_parameter('default_cut_height', 25)  # mm
        self.declare_parameter('safety_timeout', 5.0)  # seconds
        
        self.mower_enable_pin = self.get_parameter('mower_enable_pin').value
        self.mower_pwm_pin = self.get_parameter('mower_pwm_pin').value
        self.blade_rpm_sensor_pin = self.get_parameter('blade_rpm_sensor_pin').value
        self.safety_stop_pin = self.get_parameter('safety_stop_pin').value
        self.max_blade_rpm = self.get_parameter('max_blade_rpm').value
        self.min_blade_rpm = self.get_parameter('min_blade_rpm').value
        self.mower_height_i2c_addr = self.get_parameter('mower_height_i2c_addr').value
        self.default_cut_height = self.get_parameter('default_cut_height').value
        self.safety_timeout = self.get_parameter('safety_timeout').value
        
        # Initialize GPIO
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.mower_enable_pin, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self.mower_pwm_pin, GPIO.OUT)
            GPIO.setup(self.blade_rpm_sensor_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.setup(self.safety_stop_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            
            # PWM for blade speed control
            self.pwm = GPIO.PWM(self.mower_pwm_pin, 1000)  # 1kHz frequency
            self.pwm.start(0)
            
            self.get_logger().info('GPIO initialized for mower control')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize GPIO: {e}')
        
        # Initialize I2C for height adjustment
        try:
            self.i2c_bus = smbus2.SMBus(1)
            self.get_logger().info('I2C initialized for height control')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize I2C: {e}')
            self.i2c_bus = None
        
        # State variables
        self.mower_enabled = False
        self.current_cut_height = self.default_cut_height
        self.blade_rpm = 0.0
        self.safety_stop_active = False
        self.last_safety_check = time.time()
        self.rpm_lock = Lock()
        
        # RPM measurement
        self.rpm_pulse_count = 0
        self.rpm_last_time = time.time()
        GPIO.add_event_detect(self.blade_rpm_sensor_pin, GPIO.FALLING, 
                            callback=self.rpm_callback, bouncetime=10)
        
        # Safety monitoring
        GPIO.add_event_detect(self.safety_stop_pin, GPIO.BOTH,
                            callback=self.safety_callback, bouncetime=50)
        
        # Publishers
        self.mower_status_pub = self.create_publisher(String, 'mower/status', 10)
        self.blade_rpm_pub = self.create_publisher(Float32, 'mower/blade_rpm', 10)
        self.cut_height_pub = self.create_publisher(Float32, 'mower/cut_height', 10)
        self.safety_status_pub = self.create_publisher(Bool, 'mower/safety_status', 10)
        
        # Subscribers
        self.mower_enable_sub = self.create_subscription(
            Bool, 'mower/enable', self.enable_callback, 10)
        self.cut_height_sub = self.create_subscription(
            Float32, 'mower/set_cut_height', self.cut_height_callback, 10)
        
        # Services
        self.emergency_stop_srv = self.create_service(
            SetBool, 'mower/emergency_stop', self.emergency_stop_callback)
        self.calibrate_height_srv = self.create_service(
            SetBool, 'mower/calibrate_height', self.calibrate_height_callback)
        
        # Timers
        self.status_timer = self.create_timer(0.5, self.publish_status)  # 2 Hz
        self.safety_timer = self.create_timer(0.1, self.safety_check)    # 10 Hz
        
        self.get_logger().info('Mower Controller initialized')
    
    def rpm_callback(self, channel):
        """GPIO interrupt callback for RPM measurement"""
        with self.rpm_lock:
            self.rpm_pulse_count += 1
    
    def safety_callback(self, channel):
        """GPIO interrupt callback for safety stop"""
        safety_state = GPIO.input(self.safety_stop_pin)
        self.safety_stop_active = (safety_state == GPIO.LOW)
        
        if self.safety_stop_active:
            self.get_logger().warn('Safety stop activated!')
            self.emergency_stop()
    
    def enable_callback(self, msg):
        """Enable/disable mower"""
        if msg.data and not self.safety_stop_active:
            self.start_mower()
        else:
            self.stop_mower()
    
    def cut_height_callback(self, msg):
        """Set cutting height"""
        height = max(10, min(100, msg.data))  # Clamp between 10-100mm
        self.set_cut_height(height)
    
    def start_mower(self):
        """Start the mower"""
        if self.safety_stop_active:
            self.get_logger().warn('Cannot start mower - safety stop active')
            return
        
        try:
            # Enable mower power
            GPIO.output(self.mower_enable_pin, GPIO.HIGH)
            
            # Gradually ramp up blade speed
            for duty_cycle in range(0, 80, 5):  # Ramp to 80% max speed
                self.pwm.ChangeDutyCycle(duty_cycle)
                time.sleep(0.1)
            
            self.mower_enabled = True
            self.get_logger().info('Mower started successfully')
            
        except Exception as e:
            self.get_logger().error(f'Failed to start mower: {e}')
            self.emergency_stop()
    
    def stop_mower(self):
        """Stop the mower"""
        try:
            # Gradually ramp down blade speed
            current_duty = 80
            for duty_cycle in range(current_duty, -1, -5):
                self.pwm.ChangeDutyCycle(duty_cycle)
                time.sleep(0.1)
            
            # Disable mower power
            GPIO.output(self.mower_enable_pin, GPIO.LOW)
            
            self.mower_enabled = False
            self.get_logger().info('Mower stopped')
            
        except Exception as e:
            self.get_logger().error(f'Error stopping mower: {e}')
    
    def emergency_stop(self):
        """Emergency stop - immediate shutdown"""
        try:
            self.pwm.ChangeDutyCycle(0)
            GPIO.output(self.mower_enable_pin, GPIO.LOW)
            self.mower_enabled = False
            self.get_logger().warn('Emergency stop executed')
        except Exception as e:
            self.get_logger().error(f'Emergency stop error: {e}')
    
    def set_cut_height(self, height_mm):
        """Set cutting height via I2C"""
        if self.i2c_bus is None:
            self.get_logger().warn('I2C not available for height adjustment')
            return
        
        try:
            # Convert height to servo position (example mapping)
            servo_position = int((height_mm - 10) * 180 / 90)  # 10-100mm -> 0-180 degrees
            servo_position = max(0, min(180, servo_position))
            
            # Send command to height adjustment servo controller
            self.i2c_bus.write_byte_data(self.mower_height_i2c_addr, 0x00, servo_position)
            
            self.current_cut_height = height_mm
            self.get_logger().info(f'Cut height set to {height_mm}mm')
            
        except Exception as e:
            self.get_logger().error(f'Failed to set cut height: {e}')
    
    def calculate_rpm(self):
        """Calculate blade RPM from pulse count"""
        current_time = time.time()
        time_diff = current_time - self.rpm_last_time
        
        if time_diff >= 1.0:  # Update every second
            with self.rpm_lock:
                # Assuming 2 pulses per revolution (magnetic sensor setup)
                rpm = (self.rpm_pulse_count / 2.0) * (60.0 / time_diff)
                self.blade_rpm = rpm
                self.rpm_pulse_count = 0
                self.rpm_last_time = current_time
    
    def safety_check(self):
        """Periodic safety checks"""
        current_time = time.time()
        
        # Check safety stop pin
        safety_pin_state = GPIO.input(self.safety_stop_pin)
        self.safety_stop_active = (safety_pin_state == GPIO.LOW)
        
        # Check blade RPM if mower is enabled
        if self.mower_enabled:
            if self.blade_rpm < self.min_blade_rpm:
                self.get_logger().warn(f'Blade RPM too low: {self.blade_rpm}')
                # Could implement automatic restart or alert
            elif self.blade_rpm > self.max_blade_rpm:
                self.get_logger().error(f'Blade RPM too high: {self.blade_rpm}')
                self.emergency_stop()
        
        # Check communication timeout
        if current_time - self.last_safety_check > self.safety_timeout:
            if self.mower_enabled:
                self.get_logger().warn('Safety timeout - stopping mower')
                self.stop_mower()
        
        self.last_safety_check = current_time
    
    def publish_status(self):
        """Publish mower status information"""
        # Calculate current RPM
        self.calculate_rpm()
        
        # Publish status
        status_msg = String()
        if self.safety_stop_active:
            status_msg.data = "SAFETY_STOP"
        elif self.mower_enabled:
            status_msg.data = "RUNNING"
        else:
            status_msg.data = "STOPPED"
        self.mower_status_pub.publish(status_msg)
        
        # Publish RPM
        rpm_msg = Float32()
        rpm_msg.data = self.blade_rpm
        self.blade_rpm_pub.publish(rpm_msg)
        
        # Publish cut height
        height_msg = Float32()
        height_msg.data = self.current_cut_height
        self.cut_height_pub.publish(height_msg)
        
        # Publish safety status
        safety_msg = Bool()
        safety_msg.data = not self.safety_stop_active
        self.safety_status_pub.publish(safety_msg)
    
    def emergency_stop_callback(self, request, response):
        """Service callback for emergency stop"""
        if request.data:
            self.emergency_stop()
            response.success = True
            response.message = "Emergency stop executed"
        else:
            response.success = False
            response.message = "Invalid request"
        return response
    
    def calibrate_height_callback(self, request, response):
        """Service callback for height calibration"""
        if request.data:
            try:
                # Perform height calibration sequence
                self.get_logger().info('Starting height calibration...')
                
                # Move to minimum height
                self.set_cut_height(10)
                time.sleep(2)
                
                # Move to maximum height
                self.set_cut_height(100)
                time.sleep(2)
                
                # Return to default height
                self.set_cut_height(self.default_cut_height)
                
                response.success = True
                response.message = "Height calibration completed"
            except Exception as e:
                response.success = False
                response.message = f"Calibration failed: {e}"
        else:
            response.success = False
            response.message = "Invalid request"
        return response
    
    def destroy_node(self):
        """Clean shutdown"""
        try:
            self.emergency_stop()
            if hasattr(self, 'pwm'):
                self.pwm.stop()
            GPIO.cleanup()
            if self.i2c_bus:
                self.i2c_bus.close()
        except Exception as e:
            self.get_logger().error(f'Error during cleanup: {e}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    controller = MowerController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()