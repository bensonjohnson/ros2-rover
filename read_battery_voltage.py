#!/usr/bin/env python3
"""
Simple script to start the Hiwonder motor controller and read battery voltage
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import time

class BatteryMonitor(Node):
    def __init__(self):
        super().__init__('battery_monitor')
        
        # Subscribe to battery voltage topic
        self.battery_sub = self.create_subscription(
            Float32,
            '/battery_voltage',
            self.battery_callback,
            10
        )
        
        self.get_logger().info('Battery monitor started. Waiting for voltage readings...')
        
    def battery_callback(self, msg):
        voltage = msg.data
        self.get_logger().info(f'Current battery voltage: {voltage:.2f}V')

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