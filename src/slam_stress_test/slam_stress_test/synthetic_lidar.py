#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math
import time
import random


class SyntheticLidarNode(Node):
    def __init__(self):
        super().__init__("synthetic_lidar")

        # Parameters for stress testing
        self.declare_parameter("scan_frequency", 20.0)  # Start with 20Hz
        # Dense scan (0.5 degree)
        self.declare_parameter("num_ranges", 720)
        self.declare_parameter("complexity", "high")  # high, medium, low

        freq = self.get_parameter("scan_frequency").value
        self.num_ranges = self.get_parameter("num_ranges").value
        complexity = self.get_parameter("complexity").value

        # Publisher
        self.scan_pub = self.create_publisher(LaserScan, "scan", 10)

        # Timer
        timer_period = 1.0 / freq
        self.timer = self.create_timer(timer_period, self.publish_scan)

        # Simulation parameters
        self.scan_time = time.time()
        self.environment_complexity = complexity

        self.get_logger().info(
            f"Synthetic LIDAR: {freq}Hz, {
                self.num_ranges} ranges, {complexity} complexity"
        )

    def generate_complex_environment(self, scan_count):
        """Generate a complex synthetic environment"""
        ranges = []

        for i in range(self.num_ranges):
            angle = (i * 2 * math.pi) / self.num_ranges

            # Create complex environment with moving obstacles
            base_distance = 3.0 + 2.0 * math.sin(angle * 3)

            # Add time-varying obstacles (simulates moving through environment)
            time_factor = scan_count * 0.01
            moving_obstacle = 1.0 + 0.5 * math.sin(angle * 5 + time_factor)

            # Add noise and complexity based on setting
            if self.environment_complexity == "high":
                noise = random.uniform(-0.1, 0.1)
                complexity_factor = 0.5 * math.sin(angle * 10 + time_factor * 2)
                distance = base_distance + moving_obstacle + complexity_factor + noise
            elif self.environment_complexity == "medium":
                noise = random.uniform(-0.05, 0.05)
                distance = base_distance + moving_obstacle + noise
            else:  # low
                distance = base_distance + 0.2 * moving_obstacle

            # Clamp to realistic LIDAR range
            distance = max(0.1, min(distance, 10.0))
            ranges.append(distance)

        return ranges

    def publish_scan(self):
        scan = LaserScan()
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.header.frame_id = "laser"

        # LIDAR parameters (similar to RPLIDAR A2M12)
        scan.angle_min = 0.0
        scan.angle_max = 2 * math.pi
        scan.angle_increment = (2 * math.pi) / self.num_ranges
        scan.time_increment = 0.0
        scan.scan_time = 0.05  # 20Hz = 0.05 seconds per scan
        scan.range_min = 0.1
        scan.range_max = 10.0

        # Generate complex scan data
        current_time = time.time()
        scan_count = int(
            (current_time - self.scan_time) * 20
        )  # Approximate scan number
        scan.ranges = self.generate_complex_environment(scan_count)

        self.scan_pub.publish(scan)


def main(args=None):
    rclpy.init(args=args)
    node = SyntheticLidarNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
