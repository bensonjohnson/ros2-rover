#!/usr/bin/env python3
"""
Simple script to start the Hiwonder motor controller and read battery voltage and percentage
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import time

class BatteryMonitor(Node):
    def __init__(self):
        super().__init__('battery_monitor')
        
        # Store latest voltage and percentage values
        self.latest_voltage = None
        self.latest_percentage = None
        
        # Subscribe to battery voltage topic
        self.battery_voltage_sub = self.create_subscription(
            Float32,
            '/battery_voltage',
            self.battery_voltage_callback,
            10
        )
        
        # Subscribe to battery percentage topic
        self.battery_percentage_sub = self.create_subscription(
            Float32,
            '/battery_percentage',
            self.battery_percentage_callback,
            10
        )
        
        self.get_logger().info('Battery monitor started. Waiting for voltage and percentage readings...')
        
    def battery_voltage_callback(self, msg):
        self.latest_voltage = msg.data
        self.print_battery_info()
        
    def battery_percentage_callback(self, msg):
        self.latest_percentage = msg.data
        self.print_battery_info()
        
    def print_battery_info(self):
        # Only print when we have both voltage and percentage values
        if self.latest_voltage is not None and self.latest_percentage is not None:
            self.get_logger().info(f'Current battery: {self.latest_voltage:.2f}V ({self.latest_percentage:.1f}%)')

def main():
    # Initialize ROS 2
    rclpy.init()
    
    # Create and spin the battery monitor node
    battery_monitor = BatteryMonitor()
    
    try:
        rclpy.spin(battery_monitor)
    except KeyboardInterrupt:
        pass
    finally:
        battery_monitor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
