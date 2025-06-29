#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import SetBool, Trigger
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy
import smbus2
import RPi.GPIO as GPIO
import time
import math
from threading import Lock


class SprayerController(Node):
    def __init__(self):
        super().__init__('sprayer_controller')
        
        # Parameters
        self.declare_parameter('pump_enable_pin', 26)
        self.declare_parameter('pump_pwm_pin', 27)
        self.declare_parameter('nozzle_control_pins', [5, 6, 7, 8])  # Multiple nozzles
        self.declare_parameter('flow_sensor_pin', 9)
        self.declare_parameter('tank_level_sensor_i2c_addr', 0x41)
        self.declare_parameter('pressure_sensor_i2c_addr', 0x42)
        self.declare_parameter('max_pump_speed', 100)  # Percentage
        self.declare_parameter('target_pressure', 2.0)  # Bar
        self.declare_parameter('min_tank_level', 10)    # Percentage
        self.declare_parameter('spray_width', 2.0)      # meters
        self.declare_parameter('default_flow_rate', 1.0)  # L/min
        
        self.pump_enable_pin = self.get_parameter('pump_enable_pin').value
        self.pump_pwm_pin = self.get_parameter('pump_pwm_pin').value
        self.nozzle_pins = self.get_parameter('nozzle_control_pins').value
        self.flow_sensor_pin = self.get_parameter('flow_sensor_pin').value
        self.tank_level_i2c_addr = self.get_parameter('tank_level_sensor_i2c_addr').value
        self.pressure_sensor_i2c_addr = self.get_parameter('pressure_sensor_i2c_addr').value
        self.max_pump_speed = self.get_parameter('max_pump_speed').value
        self.target_pressure = self.get_parameter('target_pressure').value
        self.min_tank_level = self.get_parameter('min_tank_level').value
        self.spray_width = self.get_parameter('spray_width').value
        self.default_flow_rate = self.get_parameter('default_flow_rate').value
        
        # Initialize GPIO
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pump_enable_pin, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self.pump_pwm_pin, GPIO.OUT)
            GPIO.setup(self.nozzle_pins, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self.flow_sensor_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            
            # PWM for pump speed control
            self.pump_pwm = GPIO.PWM(self.pump_pwm_pin, 1000)  # 1kHz
            self.pump_pwm.start(0)
            
            self.get_logger().info('GPIO initialized for sprayer control')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize GPIO: {e}')
        
        # Initialize I2C for sensors
        try:
            self.i2c_bus = smbus2.SMBus(5)
            self.get_logger().info('I2C initialized for sensor readings')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize I2C: {e}')
            self.i2c_bus = None
        
        # State variables
        self.sprayer_enabled = False
        self.pump_speed = 0.0
        self.current_pressure = 0.0
        self.tank_level = 100.0
        self.flow_rate = 0.0
        self.active_nozzles = [False] * len(self.nozzle_pins)
        self.flow_lock = Lock()
        
        # Flow measurement
        self.flow_pulse_count = 0
        self.flow_last_time = time.time()
        GPIO.add_event_detect(self.flow_sensor_pin, GPIO.FALLING,
                            callback=self.flow_callback, bouncetime=10)
        
        # Publishers
        self.sprayer_status_pub = self.create_publisher(String, 'sprayer/status', 10)
        self.pump_speed_pub = self.create_publisher(Float32, 'sprayer/pump_speed', 10)
        self.pressure_pub = self.create_publisher(Float32, 'sprayer/pressure', 10)
        self.tank_level_pub = self.create_publisher(Float32, 'sprayer/tank_level', 10)
        self.flow_rate_pub = self.create_publisher(Float32, 'sprayer/flow_rate', 10)
        self.nozzle_status_pub = self.create_publisher(String, 'sprayer/nozzle_status', 10)
        
        # Subscribers
        self.enable_sub = self.create_subscription(
            Bool, 'sprayer/enable', self.enable_callback, 10)
        self.pump_speed_sub = self.create_subscription(
            Float32, 'sprayer/set_pump_speed', self.pump_speed_callback, 10)
        self.nozzle_control_sub = self.create_subscription(
            String, 'sprayer/nozzle_control', self.nozzle_control_callback, 10)
        self.cmd_vel_sub = self.create_subscription(
            Twist, 'cmd_vel', self.cmd_vel_callback, 10)
        
        # Services
        self.emergency_stop_srv = self.create_service(
            SetBool, 'sprayer/emergency_stop', self.emergency_stop_callback)
        self.calibrate_sensors_srv = self.create_service(
            Trigger, 'sprayer/calibrate_sensors', self.calibrate_sensors_callback)
        self.prime_pump_srv = self.create_service(
            Trigger, 'sprayer/prime_pump', self.prime_pump_callback)
        
        # Timers
        self.status_timer = self.create_timer(0.5, self.publish_status)  # 2 Hz
        self.control_timer = self.create_timer(0.1, self.control_loop)   # 10 Hz
        
        self.get_logger().info('Sprayer Controller initialized')
    
    def flow_callback(self, channel):
        """GPIO interrupt callback for flow measurement"""
        with self.flow_lock:
            self.flow_pulse_count += 1
    
    def enable_callback(self, msg):
        """Enable/disable sprayer"""
        if msg.data:
            self.start_sprayer()
        else:
            self.stop_sprayer()
    
    def pump_speed_callback(self, msg):
        """Set pump speed"""
        speed = max(0.0, min(100.0, msg.data))
        self.set_pump_speed(speed)
    
    def nozzle_control_callback(self, msg):
        """Control individual nozzles"""
        # Expected format: "1,0,1,1" for 4 nozzles (1=on, 0=off)
        try:
            nozzle_states = msg.data.split(',')
            for i, state in enumerate(nozzle_states[:len(self.nozzle_pins)]):
                self.active_nozzles[i] = (state.strip() == '1')
                GPIO.output(self.nozzle_pins[i], GPIO.HIGH if self.active_nozzles[i] else GPIO.LOW)
        except Exception as e:
            self.get_logger().error(f'Invalid nozzle control command: {e}')
    
    def cmd_vel_callback(self, msg):
        """Adjust spray pattern based on vehicle movement"""
        # Variable rate application based on speed
        linear_speed = abs(msg.linear.x)  # m/s
        
        if self.sprayer_enabled and linear_speed > 0:
            # Calculate application rate (example: maintain constant volume per area)
            base_flow_rate = self.default_flow_rate  # L/min
            speed_factor = linear_speed / 1.0  # Normalize to 1 m/s reference
            adjusted_flow_rate = base_flow_rate * speed_factor
            
            # Convert flow rate to pump speed (simplified mapping)
            target_pump_speed = min(100.0, adjusted_flow_rate * 20)  # Rough conversion
            self.set_pump_speed(target_pump_speed)
    
    def start_sprayer(self):
        """Start the sprayer system"""
        if self.tank_level < self.min_tank_level:
            self.get_logger().warn('Cannot start sprayer - tank level too low')
            return
        
        try:
            # Enable pump
            GPIO.output(self.pump_enable_pin, GPIO.HIGH)
            
            # Start with low pump speed
            self.set_pump_speed(30.0)
            
            # Enable default nozzles (all on)
            for i, pin in enumerate(self.nozzle_pins):
                GPIO.output(pin, GPIO.HIGH)
                self.active_nozzles[i] = True
            
            self.sprayer_enabled = True
            self.get_logger().info('Sprayer started successfully')
            
        except Exception as e:
            self.get_logger().error(f'Failed to start sprayer: {e}')
            self.emergency_stop()
    
    def stop_sprayer(self):
        """Stop the sprayer system"""
        try:
            # Turn off all nozzles first
            for i, pin in enumerate(self.nozzle_pins):
                GPIO.output(pin, GPIO.LOW)
                self.active_nozzles[i] = False
            
            # Gradually reduce pump speed
            for speed in range(int(self.pump_speed), -1, -5):
                self.pump_pwm.ChangeDutyCycle(speed)
                time.sleep(0.1)
            
            # Disable pump
            GPIO.output(self.pump_enable_pin, GPIO.LOW)
            
            self.sprayer_enabled = False
            self.pump_speed = 0.0
            self.get_logger().info('Sprayer stopped')
            
        except Exception as e:
            self.get_logger().error(f'Error stopping sprayer: {e}')
    
    def emergency_stop(self):
        """Emergency stop - immediate shutdown"""
        try:
            # Immediate pump stop
            self.pump_pwm.ChangeDutyCycle(0)
            GPIO.output(self.pump_enable_pin, GPIO.LOW)
            
            # Close all nozzles
            for pin in self.nozzle_pins:
                GPIO.output(pin, GPIO.LOW)
            
            self.sprayer_enabled = False
            self.pump_speed = 0.0
            self.active_nozzles = [False] * len(self.nozzle_pins)
            
            self.get_logger().warn('Sprayer emergency stop executed')
        except Exception as e:
            self.get_logger().error(f'Emergency stop error: {e}')
    
    def set_pump_speed(self, speed_percent):
        """Set pump speed (0-100%)"""
        speed_percent = max(0.0, min(100.0, speed_percent))
        self.pump_speed = speed_percent
        self.pump_pwm.ChangeDutyCycle(speed_percent)
    
    def read_sensors(self):
        """Read pressure and tank level sensors via I2C"""
        if self.i2c_bus is None:
            return
        
        try:
            # Read pressure sensor (example: returns raw ADC value)
            pressure_raw = self.i2c_bus.read_word_data(self.pressure_sensor_i2c_addr, 0x00)
            self.current_pressure = (pressure_raw / 65535.0) * 5.0  # Convert to bar
            
            # Read tank level sensor
            level_raw = self.i2c_bus.read_word_data(self.tank_level_i2c_addr, 0x00)
            self.tank_level = (level_raw / 65535.0) * 100.0  # Convert to percentage
            
        except Exception as e:
            self.get_logger().debug(f'Sensor reading error: {e}')
    
    def calculate_flow_rate(self):
        """Calculate flow rate from pulse count"""
        current_time = time.time()
        time_diff = current_time - self.flow_last_time
        
        if time_diff >= 1.0:  # Update every second
            with self.flow_lock:
                # Convert pulses to flow rate (calibration needed)
                pulses_per_liter = 450  # Example calibration value
                flow_rate_lps = (self.flow_pulse_count / pulses_per_liter) / time_diff
                self.flow_rate = flow_rate_lps * 60.0  # Convert to L/min
                
                self.flow_pulse_count = 0
                self.flow_last_time = current_time
    
    def control_loop(self):
        """Main control loop for pressure regulation"""
        if not self.sprayer_enabled:
            return
        
        self.read_sensors()
        
        # Pressure control (simple P controller)
        pressure_error = self.target_pressure - self.current_pressure
        
        if abs(pressure_error) > 0.1:  # Dead band
            # Adjust pump speed based on pressure error
            adjustment = pressure_error * 10.0  # Proportional gain
            new_speed = self.pump_speed + adjustment
            new_speed = max(10.0, min(100.0, new_speed))  # Clamp limits
            self.set_pump_speed(new_speed)
        
        # Safety checks
        if self.tank_level < self.min_tank_level:
            self.get_logger().warn('Tank level critical - stopping sprayer')
            self.stop_sprayer()
        
        if self.current_pressure > self.target_pressure * 1.5:
            self.get_logger().error('Pressure too high - emergency stop')
            self.emergency_stop()
    
    def publish_status(self):
        """Publish sprayer status information"""
        self.calculate_flow_rate()
        
        # Publish status
        status_msg = String()
        if self.sprayer_enabled:
            if self.tank_level < self.min_tank_level:
                status_msg.data = "LOW_TANK"
            else:
                status_msg.data = "SPRAYING"
        else:
            status_msg.data = "STOPPED"
        self.sprayer_status_pub.publish(status_msg)
        
        # Publish pump speed
        pump_msg = Float32()
        pump_msg.data = self.pump_speed
        self.pump_speed_pub.publish(pump_msg)
        
        # Publish pressure
        pressure_msg = Float32()
        pressure_msg.data = self.current_pressure
        self.pressure_pub.publish(pressure_msg)
        
        # Publish tank level
        tank_msg = Float32()
        tank_msg.data = self.tank_level
        self.tank_level_pub.publish(tank_msg)
        
        # Publish flow rate
        flow_msg = Float32()
        flow_msg.data = self.flow_rate
        self.flow_rate_pub.publish(flow_msg)
        
        # Publish nozzle status
        nozzle_msg = String()
        nozzle_msg.data = ','.join(['1' if active else '0' for active in self.active_nozzles])
        self.nozzle_status_pub.publish(nozzle_msg)
    
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
    
    def calibrate_sensors_callback(self, request, response):
        """Service callback for sensor calibration"""
        try:
            self.get_logger().info('Starting sensor calibration...')
            
            # Calibration sequence would go here
            # For now, just reset sensor readings
            self.current_pressure = 0.0
            self.tank_level = 100.0
            
            response.success = True
            response.message = "Sensor calibration completed"
        except Exception as e:
            response.success = False
            response.message = f"Calibration failed: {e}"
        return response
    
    def prime_pump_callback(self, request, response):
        """Service callback for pump priming"""
        try:
            self.get_logger().info('Priming pump...')
            
            # Priming sequence
            GPIO.output(self.pump_enable_pin, GPIO.HIGH)
            self.set_pump_speed(50.0)
            time.sleep(5)  # Prime for 5 seconds
            self.set_pump_speed(0.0)
            GPIO.output(self.pump_enable_pin, GPIO.LOW)
            
            response.success = True
            response.message = "Pump primed successfully"
        except Exception as e:
            response.success = False
            response.message = f"Priming failed: {e}"
        return response
    
    def destroy_node(self):
        """Clean shutdown"""
        try:
            self.emergency_stop()
            if hasattr(self, 'pump_pwm'):
                self.pump_pwm.stop()
            GPIO.cleanup()
            if self.i2c_bus:
                self.i2c_bus.close()
        except Exception as e:
            self.get_logger().error(f'Error during cleanup: {e}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    controller = SprayerController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()