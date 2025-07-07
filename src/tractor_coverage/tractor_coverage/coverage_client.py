#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point32, Polygon
from std_msgs.msg import String
import json
import math


class CoverageClient(Node):
    """
    Simple client for commanding coverage operations.
    Provides convenience methods for common coverage patterns.
    """

    def __init__(self):
        super().__init__("coverage_client")

        # Publishers for sending coverage commands
        self.boundary_pub = self.create_publisher(Polygon, "coverage_boundary", 10)
        self.command_pub = self.create_publisher(String, "coverage_command", 10)

        # Subscribers for status updates
        self.status_sub = self.create_subscription(
            String, "coverage_status", self.status_callback, 10
        )

        self.current_status = "IDLE"
        self.get_logger().info("Coverage Client initialized")

    def status_callback(self, msg):
        """Handle coverage status updates"""
        self.current_status = msg.data
        self.get_logger().info(f"Coverage status: {self.current_status}")

    def create_rectangular_boundary(
        self, center_x: float, center_y: float, width: float, height: float
    ) -> Polygon:
        """Create rectangular boundary polygon"""
        polygon = Polygon()

        # Create rectangle corners
        half_width = width / 2.0
        half_height = height / 2.0

        corners = [
            (center_x - half_width, center_y - half_height),
            (center_x + half_width, center_y - half_height),
            (center_x + half_width, center_y + half_height),
            (center_x - half_width, center_y + half_height),
            (center_x - half_width, center_y - half_height),  # Close the polygon
        ]

        for x, y in corners:
            point = Point32()
            point.x = x
            point.y = y
            point.z = 0.0
            polygon.points.append(point)

        return polygon

    def create_circular_boundary(
        self, center_x: float, center_y: float, radius: float, num_points: int = 20
    ) -> Polygon:
        """Create circular boundary polygon"""
        polygon = Polygon()

        for i in range(num_points + 1):  # +1 to close the polygon
            angle = 2 * math.pi * i / num_points
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)

            point = Point32()
            point.x = x
            point.y = y
            point.z = 0.0
            polygon.points.append(point)

        return polygon

    def start_mowing_operation(
        self,
        boundary: Polygon,
        tool_width: float = 1.0,
        overlap: float = 0.1,
        include_perimeter: bool = True,
    ):
        """Start mowing operation with specified parameters"""

        # Publish boundary
        self.boundary_pub.publish(boundary)

        # Create command
        command = {
            "operation_type": "mowing",
            "tool_width": tool_width,
            "overlap_percentage": overlap,
            "work_speed": 0.5,
            "include_perimeter": include_perimeter,
            "optimize_path": True,
        }

        command_msg = String()
        command_msg.data = json.dumps(command)
        self.command_pub.publish(command_msg)

        self.get_logger().info(
            f"Started mowing operation with {len(boundary.points)} boundary points"
        )

    def start_spraying_operation(
        self,
        boundary: Polygon,
        tool_width: float = 2.0,
        overlap: float = 0.05,
        include_perimeter: bool = False,
    ):
        """Start spraying operation with specified parameters"""

        # Publish boundary
        self.boundary_pub.publish(boundary)

        # Create command
        command = {
            "operation_type": "spraying",
            "tool_width": tool_width,
            "overlap_percentage": overlap,
            "work_speed": 0.8,
            "include_perimeter": include_perimeter,
            "optimize_path": True,
        }

        command_msg = String()
        command_msg.data = json.dumps(command)
        self.command_pub.publish(command_msg)

        self.get_logger().info(
            f"Started spraying operation with {len(boundary.points)} boundary points"
        )

    def stop_operation(self):
        """Stop current coverage operation"""
        command = {"operation_type": "stop"}

        command_msg = String()
        command_msg.data = json.dumps(command)
        self.command_pub.publish(command_msg)

        self.get_logger().info("Sent stop command")

    def get_status(self) -> str:
        """Get current operation status"""
        return self.current_status


def main(args=None):
    rclpy.init(args=args)

    client = CoverageClient()

    # Example usage - create and execute a mowing pattern
    if args and len(args) > 1 and args[1] == "demo":
        # Demo rectangular mowing area
        boundary = client.create_rectangular_boundary(
            center_x=0.0, center_y=0.0, width=20.0, height=15.0
        )

        # Start mowing operation
        client.start_mowing_operation(
            boundary=boundary, tool_width=1.2, overlap=0.1, include_perimeter=True
        )

        # Let it run for demo
        try:
            rclpy.spin(client)
        except KeyboardInterrupt:
            client.stop_operation()
    else:
        # Interactive mode
        print("\nCoverage Client Commands:")
        print("1. Demo rectangular mowing")
        print("2. Demo circular spraying")
        print("3. Stop operation")
        print("4. Check status")
        print("Ctrl+C to exit")

        try:
            while rclpy.ok():
                try:
                    choice = input("\nEnter command (1-4): ").strip()

                    if choice == "1":
                        boundary = client.create_rectangular_boundary(0, 0, 10, 8)
                        client.start_mowing_operation(boundary, tool_width=1.0)

                    elif choice == "2":
                        boundary = client.create_circular_boundary(0, 0, 6)
                        client.start_spraying_operation(boundary, tool_width=1.5)

                    elif choice == "3":
                        client.stop_operation()

                    elif choice == "4":
                        print(f"Status: {client.get_status()}")

                    else:
                        print("Invalid choice")

                    # Process callbacks
                    rclpy.spin_once(client, timeout_sec=0.1)

                except EOFError:
                    break

        except KeyboardInterrupt:
            pass

    client.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    import sys

    main(sys.argv)
