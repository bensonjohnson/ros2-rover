#!/usr/bin/env python3
"""
Safety Monitor Node for Autonomous Mapping
Provides additional safety checks and emergency stop functionality
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import Bool, String
from nav_msgs.msg import Odometry
import math
import time
from threading import Lock


class SafetyMonitor(Node):
    def __init__(self):
        super().__init__('safety_monitor')
        
        # Parameters
        self.declare_parameter('max_speed_limit', 0.5)
        self.declare_parameter('emergency_stop_distance', 0.3)
        self.declare_parameter('warning_distance', 0.8)
        self.declare_parameter('max_angular_velocity', 1.0)
        self.declare_parameter('heartbeat_timeout', 5.0)
        
        # Get parameters
        self.max_speed_limit = self.get_parameter('max_speed_limit').get_parameter_value().double_value
        self.emergency_distance = self.get_parameter('emergency_stop_distance').get_parameter_value().double_value
        self.warning_distance = self.get_parameter('warning_distance').get_parameter_value().double_value
        self.max_angular_vel = self.get_parameter('max_angular_velocity').get_parameter_value().double_value
        self.heartbeat_timeout = self.get_parameter('heartbeat_timeout').get_parameter_value().double_value
        
        # State variables
        self.emergency_stop_active = False
        self.last_cmd_vel = None
        self.last_heartbeat = time.time()
        self.min_obstacle_distance = float('inf')
        self.current_speed = 0.0
        self.lock = Lock()
        
        # Publishers
        self.safe_cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel_safe', 10)
        self.emergency_stop_pub = self.create_publisher(Bool, '/emergency_stop', 10)
        self.safety_status_pub = self.create_publisher(String, '/safety_status', 10)
        
        # Subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.realsense_scan_sub = self.create_subscription(
            LaserScan, '/realsense/scan', self.realsense_scan_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odometry/filtered', self.odom_callback, 10
        )
        self.heartbeat_sub = self.create_subscription(
            String, '/mapping_status', self.heartbeat_callback, 10
        )
        
        # Timers
        self.safety_timer = self.create_timer(0.05, self.safety_check)  # 20Hz
        self.status_timer = self.create_timer(2.0, self.publish_status)
        self.heartbeat_timer = self.create_timer(1.0, self.check_heartbeat)
        
        self.get_logger().info("Safety Monitor initialized")
        self.get_logger().info(f"Emergency stop distance: {self.emergency_distance}m")
        self.get_logger().info(f"Warning distance: {self.warning_distance}m")
        self.get_logger().info(f"Max speed limit: {self.max_speed_limit}m/s")
    
    def cmd_vel_callback(self, msg):
        """Monitor command velocity and apply safety limits"""
        with self.lock:
            self.last_cmd_vel = msg
    
    def odom_callback(self, msg):
        """Monitor current speed"""
        linear_vel = msg.twist.twist.linear
        self.current_speed = math.sqrt(linear_vel.x**2 + linear_vel.y**2)
    
    def scan_callback(self, msg):
        """Process laser scan for safety"""
        self.process_scan_safety(msg, "laser")
    
    def realsense_scan_callback(self, msg):
        """Process RealSense scan for safety"""
        self.process_scan_safety(msg, "realsense")
    
    def process_scan_safety(self, msg, source):
        """Process scan data for safety monitoring"""
        if not msg.ranges:
            return
        
        min_distance = float('inf')
        emergency_detected = False
        
        for i, distance in enumerate(msg.ranges):
            if msg.range_min <= distance <= msg.range_max:
                # Check front 180 degrees
                angle = msg.angle_min + i * msg.angle_increment
                if -math.pi/2 <= angle <= math.pi/2:
                    min_distance = min(min_distance, distance)
                    
                    # Check for emergency stop condition
                    if distance < self.emergency_distance:
                        emergency_detected = True
        
        # Update safety state
        with self.lock:
            self.min_obstacle_distance = min_distance
            if emergency_detected:
                self.emergency_stop_active = True
                self.get_logger().warn(f"EMERGENCY: Obstacle at {min_distance:.2f}m ({source})")
    
    def heartbeat_callback(self, msg):
        """Update heartbeat from mapping node"""
        self.last_heartbeat = time.time()
    
    def check_heartbeat(self):
        """Check if mapping node is still alive"""
        if time.time() - self.last_heartbeat > self.heartbeat_timeout:
            with self.lock:
                self.emergency_stop_active = True
            self.get_logger().error("Mapping node heartbeat lost - activating emergency stop")
    
    def apply_safety_limits(self, cmd_vel):
        """Apply safety limits to command velocity"""
        safe_cmd = Twist()
        
        # Copy original command
        safe_cmd.linear.x = cmd_vel.linear.x
        safe_cmd.linear.y = cmd_vel.linear.y
        safe_cmd.linear.z = cmd_vel.linear.z
        safe_cmd.angular.x = cmd_vel.angular.x
        safe_cmd.angular.y = cmd_vel.angular.y
        safe_cmd.angular.z = cmd_vel.angular.z
        
        # Apply speed limits
        linear_speed = math.sqrt(cmd_vel.linear.x**2 + cmd_vel.linear.y**2)
        if linear_speed > self.max_speed_limit:
            scale = self.max_speed_limit / linear_speed
            safe_cmd.linear.x *= scale
            safe_cmd.linear.y *= scale
        
        # Apply angular velocity limits
        if abs(cmd_vel.angular.z) > self.max_angular_vel:
            safe_cmd.angular.z = math.copysign(self.max_angular_vel, cmd_vel.angular.z)
        
        # Apply distance-based speed reduction
        if self.min_obstacle_distance < self.warning_distance:
            # Reduce speed based on distance
            speed_factor = max(0.1, (self.min_obstacle_distance - self.emergency_distance) / 
                             (self.warning_distance - self.emergency_distance))
            safe_cmd.linear.x *= speed_factor
            safe_cmd.linear.y *= speed_factor
        
        return safe_cmd
    
    def safety_check(self):
        """Main safety monitoring loop"""
        with self.lock:
            # Check emergency stop conditions
            if self.emergency_stop_active:
                # Send stop command
                stop_cmd = Twist()
                self.safe_cmd_vel_pub.publish(stop_cmd)
                
                # Publish emergency stop status
                emergency_msg = Bool()
                emergency_msg.data = True
                self.emergency_stop_pub.publish(emergency_msg)
                
                # Clear emergency after obstacle moves away
                if self.min_obstacle_distance > self.warning_distance:
                    self.emergency_stop_active = False
                    self.get_logger().info("Emergency stop cleared - obstacle moved away")
                
                return
            
            # Process normal command velocity with safety limits
            if self.last_cmd_vel is not None:
                safe_cmd = self.apply_safety_limits(self.last_cmd_vel)
                self.safe_cmd_vel_pub.publish(safe_cmd)
            
            # Publish non-emergency status
            emergency_msg = Bool()
            emergency_msg.data = False
            self.emergency_stop_pub.publish(emergency_msg)
    
    def publish_status(self):
        """Publish safety status"""
        status_msg = String()
        status_msg.data = f"Safety Status: Emergency={'YES' if self.emergency_stop_active else 'NO'}, " \
                         f"Min Distance={self.min_obstacle_distance:.2f}m, " \
                         f"Speed={self.current_speed:.2f}m/s, " \
                         f"Heartbeat={'OK' if time.time() - self.last_heartbeat < self.heartbeat_timeout else 'LOST'}"
        
        self.safety_status_pub.publish(status_msg)
        
        if self.emergency_stop_active:
            self.get_logger().warn(status_msg.data)


def main(args=None):
    rclpy.init(args=args)
    
    try:
        safety_monitor = SafetyMonitor()
        rclpy.spin(safety_monitor)
    except KeyboardInterrupt:
        print("Safety monitor interrupted")
    except Exception as e:
        print(f"Safety monitor error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()