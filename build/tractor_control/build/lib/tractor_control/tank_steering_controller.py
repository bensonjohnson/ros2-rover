#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
import smbus2
import time
import math


class TankSteeringController(Node):
    def __init__(self):
        super().__init__('tank_steering_controller')
        
        # Parameters
        self.declare_parameter('i2c_bus', 1)
        self.declare_parameter('motor_controller_address', 0x60)
        self.declare_parameter('wheel_separation', 0.5)  # meters
        self.declare_parameter('max_motor_speed', 255)
        self.declare_parameter('deadband', 0.05)
        
        self.i2c_bus = self.get_parameter('i2c_bus').value
        self.motor_address = self.get_parameter('motor_controller_address').value
        self.wheel_separation = self.get_parameter('wheel_separation').value
        self.max_motor_speed = self.get_parameter('max_motor_speed').value
        self.deadband = self.get_parameter('deadband').value
        
        # Initialize I2C
        try:
            self.bus = smbus2.SMBus(self.i2c_bus)
            self.get_logger().info(f'I2C bus {self.i2c_bus} initialized successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize I2C bus: {e}')
            self.bus = None
        
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
        
        self.get_logger().info('Tank Steering Controller initialized')
    
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
        # v_left = linear_vel - (angular_vel * wheel_separation / 2)
        # v_right = linear_vel + (angular_vel * wheel_separation / 2)
        left_speed = linear_vel - (angular_vel * self.wheel_separation / 2.0)
        right_speed = linear_vel + (angular_vel * self.wheel_separation / 2.0)
        
        # Convert to motor speeds (-255 to 255)
        left_motor = int(left_speed * self.max_motor_speed)
        right_motor = int(right_speed * self.max_motor_speed)
        
        # Clamp motor speeds
        left_motor = max(-self.max_motor_speed, min(self.max_motor_speed, left_motor))
        right_motor = max(-self.max_motor_speed, min(self.max_motor_speed, right_motor))
        
        # Send to I2C motor controller
        self.send_motor_commands(left_motor, right_motor)
        
        # Publish motor speeds for feedback
        motor_msg = Float32MultiArray()
        motor_msg.data = [float(left_motor), float(right_motor)]
        self.motor_speeds_pub.publish(motor_msg)
        
        self.get_logger().debug(f'Motor speeds: L={left_motor}, R={right_motor}')
    
    def send_motor_commands(self, left_speed, right_speed):
        """Send motor commands via I2C"""
        if self.bus is None:
            return
        
        try:
            # Convert speeds to bytes (assuming signed 16-bit values)
            left_bytes = left_speed.to_bytes(2, byteorder='big', signed=True)
            right_bytes = right_speed.to_bytes(2, byteorder='big', signed=True)
            
            # Send left motor command (register 0x00)
            self.bus.write_i2c_block_data(self.motor_address, 0x00, list(left_bytes))
            # Send right motor command (register 0x02)
            self.bus.write_i2c_block_data(self.motor_address, 0x02, list(right_bytes))
            
        except Exception as e:
            self.get_logger().error(f'I2C communication error: {e}')
    
    def destroy_node(self):
        """Clean shutdown"""
        if self.bus is not None:
            try:
                # Stop motors
                self.send_motor_commands(0, 0)
                self.bus.close()
            except Exception as e:
                self.get_logger().error(f'Error during cleanup: {e}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    controller = TankSteeringController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()