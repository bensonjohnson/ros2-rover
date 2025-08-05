#!/usr/bin/env python3
"""
Simple Safety Monitor for NPU Depth Exploration
Emergency stop only - no complex navigation
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import numpy as np
from cv_bridge import CvBridge
import cv2

class SimpleSafetyMonitorDepth(Node):
    def __init__(self):
        super().__init__('simple_safety_monitor_depth')
        
        # Parameters
        self.declare_parameter('emergency_stop_distance', 0.1)
        self.declare_parameter('max_speed_limit', 0.15)
        
        self.emergency_distance = self.get_parameter('emergency_stop_distance').value
        self.max_speed = self.get_parameter('max_speed_limit').value
        
        # State
        self.emergency_stop = False
        self.last_cmd = Twist()
        self.bridge = CvBridge()
        
        # Subscribers
        self.cmd_sub = self.create_subscription(
            Twist, 'cmd_vel_in',
            self.cmd_callback, 10
        )
        
        self.depth_sub = self.create_subscription(
            Image, 'depth_image',
            self.depth_callback, 10
        )
        
        # Publisher
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel_out', 10)
        
        self.get_logger().info(f"Simple Safety Monitor (Depth) initialized")
        self.get_logger().info(f"  Emergency stop distance: {self.emergency_distance}m")
        
    def depth_callback(self, msg):
        """Check for immediate obstacles using depth image"""
        try:
            # Convert ROS image to OpenCV format
            cv_depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            
            if cv_depth is None or cv_depth.size == 0:
                # Clear emergency stop if no depth data
                if self.emergency_stop:
                    self.get_logger().info("Emergency stop cleared - no depth data")
                    self.emergency_stop = False
                return
                
            # Convert to meters if needed (assuming depth is in millimeters)
            if cv_depth.dtype == np.uint16:
                depth_meters = cv_depth.astype(np.float32) / 1000.0
            else:
                depth_meters = cv_depth.astype(np.float32)
                
            # Check for very close obstacles in front (wider detection area)
            height, width = depth_meters.shape
            
            # Define region of interest (front center area)
            roi_height_start = int(height * 0.3)  # Top 30%
            roi_height_end = int(height * 0.7)    # Bottom 70%
            roi_width_start = int(width * 0.4)    # Left 40%
            roi_width_end = int(width * 0.6)      # Right 60%
            
            front_roi = depth_meters[roi_height_start:roi_height_end, roi_width_start:roi_width_end]
            
            # Filter out invalid depth values (0 or NaN)
            valid_depths = front_roi[(front_roi > 0.01) & (front_roi < 10.0)]  # 1cm to 10m range
            
            if len(valid_depths) > 0:
                min_distance = np.min(valid_depths)
                
                # Emergency stop if very close
                if min_distance < self.emergency_distance:
                    if not self.emergency_stop:
                        self.get_logger().warn(f"EMERGENCY STOP: Obstacle at {min_distance:.2f}m")
                        self.emergency_stop = True
                # Clear emergency stop if obstacle moves away
                elif min_distance > self.emergency_distance * 1.5:
                    if self.emergency_stop:
                        self.get_logger().info(f"Emergency stop cleared - closest obstacle at {min_distance:.2f}m")
                        self.emergency_stop = False
            else:
                # Clear emergency stop if no valid depth data in ROI
                if self.emergency_stop:
                    self.get_logger().info("Emergency stop cleared - no valid depth data in ROI")
                    self.emergency_stop = False
                    
        except Exception as e:
            self.get_logger().warn(f"Safety check failed: {e}")
            # Clear emergency stop on error to avoid getting stuck
            if self.emergency_stop:
                self.get_logger().info("Emergency stop cleared due to error")
                self.emergency_stop = False
            
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
        
def main(args=None):
    rclpy.init(args=args)
    node = SimpleSafetyMonitorDepth()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Safety monitor interrupted")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
