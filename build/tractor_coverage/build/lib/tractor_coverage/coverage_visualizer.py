#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Polygon, Point32
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from std_msgs.msg import String, ColorRGBA
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import List, Optional
import threading


class CoverageVisualizer(Node):
    """
    Visualization tool for coverage planning and execution.
    Provides both RViz markers and matplotlib plotting.
    """

    def __init__(self):
        super().__init__("coverage_visualizer")

        # Publishers
        self.boundary_markers_pub = self.create_publisher(
            MarkerArray, "boundary_markers", 10
        )
        self.path_markers_pub = self.create_publisher(
            MarkerArray, "path_visualization", 10
        )

        # Subscribers
        self.boundary_sub = self.create_subscription(
            Polygon, "coverage_boundary", self.boundary_callback, 10
        )
        self.path_sub = self.create_subscription(
            Path, "coverage_path", self.path_callback, 10
        )
        self.status_sub = self.create_subscription(
            String, "coverage_status", self.status_callback, 10
        )

        # Data storage
        self.current_boundary = None
        self.current_path = None
        self.current_status = "IDLE"
        self.path_history = []

        # Matplotlib setup
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title("Tractor Coverage Planning Visualization")
        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")

        # Animation
        self.animation = None
        self.show_animation = False

        self.get_logger().info("Coverage Visualizer initialized")

    def boundary_callback(self, msg):
        """Handle boundary updates"""
        self.current_boundary = msg
        self.publish_boundary_markers(msg)
        self.update_plot()
        self.get_logger().info(f"Updated boundary with {len(msg.points)} points")

    def path_callback(self, msg):
        """Handle path updates"""
        self.current_path = msg
        self.publish_path_markers(msg)
        self.update_plot()
        self.get_logger().info(f"Updated path with {len(msg.poses)} waypoints")

    def status_callback(self, msg):
        """Handle status updates"""
        self.current_status = msg.data
        self.update_plot()

    def publish_boundary_markers(self, boundary: Polygon):
        """Publish RViz markers for boundary visualization"""
        marker_array = MarkerArray()

        # Boundary outline
        boundary_marker = Marker()
        boundary_marker.header.stamp = self.get_clock().now().to_msg()
        boundary_marker.header.frame_id = "map"
        boundary_marker.ns = "boundary"
        boundary_marker.id = 0
        boundary_marker.type = Marker.LINE_STRIP
        boundary_marker.action = Marker.ADD
        boundary_marker.scale.x = 0.2
        boundary_marker.color.r = 1.0
        boundary_marker.color.g = 0.0
        boundary_marker.color.b = 0.0
        boundary_marker.color.a = 1.0

        for point in boundary.points:
            p = Point()
            p.x = point.x
            p.y = point.y
            p.z = 0.1
            boundary_marker.points.append(p)

        marker_array.markers.append(boundary_marker)

        # Boundary area fill
        area_marker = Marker()
        area_marker.header = boundary_marker.header
        area_marker.ns = "boundary_fill"
        area_marker.id = 1
        area_marker.type = Marker.TRIANGLE_LIST
        area_marker.action = Marker.ADD
        area_marker.scale.x = 1.0
        area_marker.scale.y = 1.0
        area_marker.scale.z = 1.0
        area_marker.color.r = 1.0
        area_marker.color.g = 0.0
        area_marker.color.b = 0.0
        area_marker.color.a = 0.2

        # Simple triangulation for polygon fill
        if len(boundary.points) >= 3:
            center = self.calculate_centroid(boundary.points)
            for i in range(len(boundary.points) - 1):
                # Triangle: center -> point[i] -> point[i+1]
                area_marker.points.append(center)

                p1 = Point()
                p1.x = boundary.points[i].x
                p1.y = boundary.points[i].y
                p1.z = 0.05
                area_marker.points.append(p1)

                p2 = Point()
                p2.x = boundary.points[i + 1].x
                p2.y = boundary.points[i + 1].y
                p2.z = 0.05
                area_marker.points.append(p2)

        marker_array.markers.append(area_marker)
        self.boundary_markers_pub.publish(marker_array)

    def publish_path_markers(self, path: Path):
        """Publish RViz markers for path visualization"""
        marker_array = MarkerArray()

        if not path.poses:
            return

        # Path line
        path_marker = Marker()
        path_marker.header.stamp = self.get_clock().now().to_msg()
        path_marker.header.frame_id = "map"
        path_marker.ns = "coverage_path"
        path_marker.id = 0
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD
        path_marker.scale.x = 0.15
        path_marker.color.r = 0.0
        path_marker.color.g = 1.0
        path_marker.color.b = 0.0
        path_marker.color.a = 0.8

        for pose in path.poses:
            path_marker.points.append(pose.pose.position)

        marker_array.markers.append(path_marker)

        # Waypoint markers
        for i, pose in enumerate(path.poses[::5]):  # Show every 5th waypoint
            waypoint_marker = Marker()
            waypoint_marker.header = path_marker.header
            waypoint_marker.ns = "waypoints"
            waypoint_marker.id = i
            waypoint_marker.type = Marker.SPHERE
            waypoint_marker.action = Marker.ADD
            waypoint_marker.pose = pose.pose
            waypoint_marker.scale.x = 0.3
            waypoint_marker.scale.y = 0.3
            waypoint_marker.scale.z = 0.3
            waypoint_marker.color.r = 0.0
            waypoint_marker.color.g = 0.0
            waypoint_marker.color.b = 1.0
            waypoint_marker.color.a = 0.6

            marker_array.markers.append(waypoint_marker)

        # Start and end markers
        start_marker = Marker()
        start_marker.header = path_marker.header
        start_marker.ns = "start_end"
        start_marker.id = 100
        start_marker.type = Marker.ARROW
        start_marker.action = Marker.ADD
        start_marker.pose = path.poses[0].pose
        start_marker.scale.x = 1.0
        start_marker.scale.y = 0.3
        start_marker.scale.z = 0.3
        start_marker.color.r = 0.0
        start_marker.color.g = 1.0
        start_marker.color.b = 0.0
        start_marker.color.a = 1.0

        end_marker = Marker()
        end_marker.header = path_marker.header
        end_marker.ns = "start_end"
        end_marker.id = 101
        end_marker.type = Marker.CUBE
        end_marker.action = Marker.ADD
        end_marker.pose = path.poses[-1].pose
        end_marker.scale.x = 0.5
        end_marker.scale.y = 0.5
        end_marker.scale.z = 0.5
        end_marker.color.r = 1.0
        end_marker.color.g = 0.0
        end_marker.color.b = 0.0
        end_marker.color.a = 1.0

        marker_array.markers.extend([start_marker, end_marker])
        self.path_markers_pub.publish(marker_array)

    def calculate_centroid(self, points: List[Point32]) -> Point:
        """Calculate centroid of polygon"""
        if not points:
            return Point()

        sum_x = sum(p.x for p in points)
        sum_y = sum(p.y for p in points)

        centroid = Point()
        centroid.x = sum_x / len(points)
        centroid.y = sum_y / len(points)
        centroid.z = 0.05

        return centroid

    def update_plot(self):
        """Update matplotlib visualization"""
        self.ax.clear()
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title(f"Coverage Planning - Status: {self.current_status}")
        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")

        # Plot boundary
        if self.current_boundary and self.current_boundary.points:
            boundary_x = [p.x for p in self.current_boundary.points]
            boundary_y = [p.y for p in self.current_boundary.points]

            self.ax.plot(boundary_x, boundary_y, "r-", linewidth=3, label="Boundary")
            self.ax.fill(boundary_x, boundary_y, "red", alpha=0.2)

        # Plot path
        if self.current_path and self.current_path.poses:
            path_x = [pose.pose.position.x for pose in self.current_path.poses]
            path_y = [pose.pose.position.y for pose in self.current_path.poses]

            self.ax.plot(path_x, path_y, "g-", linewidth=2, label="Coverage Path")

            # Mark start and end
            if len(path_x) > 0:
                self.ax.plot(path_x[0], path_y[0], "go", markersize=10, label="Start")
                self.ax.plot(path_x[-1], path_y[-1], "ro", markersize=10, label="End")

                # Show direction arrows
                for i in range(0, len(path_x) - 1, max(1, len(path_x) // 20)):
                    dx = path_x[i + 1] - path_x[i]
                    dy = path_y[i + 1] - path_y[i]
                    if abs(dx) > 0.01 or abs(dy) > 0.01:  # Avoid zero-length arrows
                        self.ax.arrow(
                            path_x[i],
                            path_y[i],
                            dx * 0.3,
                            dy * 0.3,
                            head_width=0.2,
                            head_length=0.1,
                            fc="blue",
                            ec="blue",
                            alpha=0.6,
                        )

        # Add statistics
        if self.current_boundary and self.current_path:
            stats_text = self.calculate_statistics()
            self.ax.text(
                0.02,
                0.98,
                stats_text,
                transform=self.ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

        self.ax.legend()

        # Auto-scale with some padding
        if self.current_boundary or self.current_path:
            self.ax.relim()
            self.ax.autoscale_view(scalex=True, scaley=True)
            # Add 10% padding
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            x_padding = (xlim[1] - xlim[0]) * 0.1
            y_padding = (ylim[1] - ylim[0]) * 0.1
            self.ax.set_xlim(xlim[0] - x_padding, xlim[1] + x_padding)
            self.ax.set_ylim(ylim[0] - y_padding, ylim[1] + y_padding)

        plt.draw()

    def calculate_statistics(self) -> str:
        """Calculate and format coverage statistics"""
        stats = []

        if self.current_boundary:
            area = self.calculate_polygon_area(self.current_boundary.points)
            stats.append(f"Boundary Area: {area:.1f} m²")

        if self.current_path:
            path_length = self.calculate_path_length(self.current_path.poses)
            stats.append(f"Path Length: {path_length:.1f} m")
            stats.append(f"Waypoints: {len(self.current_path.poses)}")

            if self.current_boundary:
                # Estimate coverage efficiency
                boundary_perimeter = self.calculate_polygon_perimeter(
                    self.current_boundary.points
                )
                if boundary_perimeter > 0:
                    efficiency = (area / path_length) * 100 if path_length > 0 else 0
                    stats.append(f"Efficiency: {efficiency:.1f}%")

        stats.append(f"Status: {self.current_status}")
        return "\n".join(stats)

    def calculate_polygon_area(self, points: List[Point32]) -> float:
        """Calculate polygon area using shoelace formula"""
        if len(points) < 3:
            return 0.0

        area = 0.0
        for i in range(len(points)):
            j = (i + 1) % len(points)
            area += points[i].x * points[j].y
            area -= points[j].x * points[i].y

        return abs(area) / 2.0

    def calculate_polygon_perimeter(self, points: List[Point32]) -> float:
        """Calculate polygon perimeter"""
        if len(points) < 2:
            return 0.0

        perimeter = 0.0
        for i in range(len(points)):
            j = (i + 1) % len(points)
            dx = points[j].x - points[i].x
            dy = points[j].y - points[i].y
            perimeter += (dx * dx + dy * dy) ** 0.5

        return perimeter

    def calculate_path_length(self, poses) -> float:
        """Calculate total path length"""
        if len(poses) < 2:
            return 0.0

        length = 0.0
        for i in range(1, len(poses)):
            dx = poses[i].pose.position.x - poses[i - 1].pose.position.x
            dy = poses[i].pose.position.y - poses[i - 1].pose.position.y
            length += (dx * dx + dy * dy) ** 0.5

        return length

    def show_interactive_plot(self):
        """Show interactive matplotlib plot"""
        self.show_animation = True
        plt.ion()  # Turn on interactive mode
        plt.show()

    def save_plot(self, filename: str):
        """Save current plot to file"""
        self.fig.savefig(filename, dpi=300, bbox_inches="tight")
        self.get_logger().info(f"Plot saved to {filename}")


def main(args=None):
    rclpy.init(args=args)

    visualizer = CoverageVisualizer()

    # Start matplotlib in separate thread
    plot_thread = threading.Thread(target=visualizer.show_interactive_plot)
    plot_thread.daemon = True
    plot_thread.start()

    try:
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        pass
    finally:
        plt.close("all")
        visualizer.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
