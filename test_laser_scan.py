#!/usr/bin/env python3
"""
Simple laser scan publisher for testing SLAM
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math
import random

class TestLaserScanPublisher(Node):
    def __init__(self):
        super().__init__('test_laser_scan_publisher')
        self.publisher = self.create_publisher(LaserScan, '/scan', 10)
        self.timer = self.create_timer(0.1, self.publish_scan)  # 10 Hz
        self.get_logger().info('Test laser scan publisher started')
    
    def publish_scan(self):
        """Publish a simple laser scan with some simulated obstacles"""
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        
        msg.angle_min = -math.pi/2  # -90 degrees
        msg.angle_max = math.pi/2   # 90 degrees
        msg.angle_increment = 0.0174  # 1 degree
        msg.time_increment = 0.0
        msg.scan_time = 0.1
        msg.range_min = 0.1
        msg.range_max = 10.0
        
        # Create ranges array
        num_readings = int((msg.angle_max - msg.angle_min) / msg.angle_increment)
        msg.ranges = []
        
        for i in range(num_readings):
            angle = msg.angle_min + i * msg.angle_increment
            # Simulate some obstacles and open space
            if abs(angle) < 0.5:  # Front area
                distance = 5.0 + random.uniform(-0.5, 0.5)  # Open space ahead
            else:  # Side areas
                distance = 2.0 + random.uniform(-0.5, 0.5)  # Some obstacles
            
            msg.ranges.append(distance)
        
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = TestLaserScanPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()