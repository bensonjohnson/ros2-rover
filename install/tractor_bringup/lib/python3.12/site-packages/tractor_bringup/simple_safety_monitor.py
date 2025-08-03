#!/usr/bin/env python3
"""
Simple Safety Monitor for NPU Exploration
Emergency stop only - no complex navigation
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Twist
import numpy as np
import struct

class SimpleSafetyMonitor(Node):
    def __init__(self):
        super().__init__('simple_safety_monitor')
        
        # Parameters
        self.declare_parameter('emergency_stop_distance', 0.1)
        self.declare_parameter('max_speed_limit', 0.15)
        
        self.emergency_distance = self.get_parameter('emergency_stop_distance').value
        self.max_speed = self.get_parameter('max_speed_limit').value
        
        # State
        self.emergency_stop = False
        self.last_cmd = Twist()
        
        # Subscribers
        self.cmd_sub = self.create_subscription(
            Twist, 'cmd_vel_in',
            self.cmd_callback, 10
        )
        
        self.pc_sub = self.create_subscription(
            PointCloud2, 'point_cloud',
            self.pointcloud_callback, 10
        )
        
        # Publisher
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel_out', 10)
        
        self.get_logger().info(f"Simple Safety Monitor initialized")
        self.get_logger().info(f"  Emergency stop distance: {self.emergency_distance}m")
        
    def pointcloud_callback(self, msg):
        """Check for immediate obstacles"""
        try:
            points = self.ros_pc2_to_numpy(msg)
            
            if len(points) == 0:
                return
                
            # Check for very close obstacles in front
            forward_mask = (points[:, 0] > 0) & (points[:, 0] < self.emergency_distance * 2) & (np.abs(points[:, 1]) < 0.3)
            
            if np.any(forward_mask):
                forward_points = points[forward_mask]
                min_distance = np.min(np.linalg.norm(forward_points, axis=1))
                
                if min_distance < self.emergency_distance:
                    if not self.emergency_stop:
                        self.get_logger().warn(f"EMERGENCY STOP: Obstacle at {min_distance:.2f}m")
                        self.emergency_stop = True
                else:
                    if self.emergency_stop:
                        self.get_logger().info("Emergency stop cleared")
                        self.emergency_stop = False
            else:
                if self.emergency_stop:
                    self.get_logger().info("Emergency stop cleared - no forward obstacles")
                    self.emergency_stop = False
                    
        except Exception as e:
            self.get_logger().warn(f"Safety check failed: {e}")
            
    def cmd_callback(self, msg):
        """Process and potentially modify velocity commands"""
        safe_cmd = Twist()
        
        if self.emergency_stop:
            # Complete stop
            safe_cmd.linear.x = 0.0
            safe_cmd.angular.z = 0.0
        else:
            # Apply speed limits
            safe_cmd.linear.x = max(-self.max_speed, min(self.max_speed, msg.linear.x))
            safe_cmd.angular.z = max(-2.0, min(2.0, msg.angular.z))  # Limit angular velocity
            
        self.cmd_pub.publish(safe_cmd)
        self.last_cmd = safe_cmd
        
    def ros_pc2_to_numpy(self, pc_msg):
        """Convert ROS PointCloud2 to numpy array"""
        points = []
        point_step = pc_msg.point_step
        
        for i in range(0, len(pc_msg.data), point_step):
            if i + 12 <= len(pc_msg.data):
                try:
                    x, y, z = struct.unpack('fff', pc_msg.data[i:i+12])
                    if np.isfinite([x, y, z]).all():
                        points.append([x, y, z])
                except:
                    continue
                    
        return np.array(points) if points else np.zeros((0, 3))

def main(args=None):
    rclpy.init(args=args)
    node = SimpleSafetyMonitor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Safety monitor interrupted")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()