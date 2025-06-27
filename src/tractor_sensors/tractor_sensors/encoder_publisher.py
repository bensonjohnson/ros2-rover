#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Twist
import tf2_ros
import math
import time
import RPi.GPIO as GPIO
from threading import Lock


class EncoderPublisher(Node):
    def __init__(self):
        super().__init__('encoder_publisher')
        
        # Parameters
        self.declare_parameter('left_encoder_pin_a', 18)
        self.declare_parameter('left_encoder_pin_b', 19)
        self.declare_parameter('right_encoder_pin_a', 20)
        self.declare_parameter('right_encoder_pin_b', 21)
        self.declare_parameter('encoder_ppr', 1440)  # pulses per revolution
        self.declare_parameter('wheel_radius', 0.15)  # meters
        self.declare_parameter('wheel_separation', 0.5)  # meters
        self.declare_parameter('publish_rate', 50.0)  # Hz
        
        self.left_pin_a = self.get_parameter('left_encoder_pin_a').value
        self.left_pin_b = self.get_parameter('left_encoder_pin_b').value
        self.right_pin_a = self.get_parameter('right_encoder_pin_a').value
        self.right_pin_b = self.get_parameter('right_encoder_pin_b').value
        self.encoder_ppr = self.get_parameter('encoder_ppr').value
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.wheel_separation = self.get_parameter('wheel_separation').value
        self.publish_rate = self.get_parameter('publish_rate').value
        
        # Initialize GPIO
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup([self.left_pin_a, self.left_pin_b, 
                       self.right_pin_a, self.right_pin_b], GPIO.IN, pull_up_down=GPIO.PUD_UP)
            
            # Encoder state
            self.left_encoder_count = 0
            self.right_encoder_count = 0
            self.encoder_lock = Lock()
            self.last_time = time.time()
            
            # Odometry state
            self.x = 0.0
            self.y = 0.0
            self.theta = 0.0
            
            # Setup interrupt callbacks
            GPIO.add_event_detect(self.left_pin_a, GPIO.BOTH, callback=self.left_encoder_callback)
            GPIO.add_event_detect(self.right_pin_a, GPIO.BOTH, callback=self.right_encoder_callback)
            
            self.get_logger().info('GPIO encoders initialized successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize GPIO: {e}')
        
        # Publishers
        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        
        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Timer for publishing
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_callback)
        
        self.get_logger().info('Encoder Publisher initialized')
    
    def left_encoder_callback(self, channel):
        """Interrupt callback for left encoder"""
        with self.encoder_lock:
            a_state = GPIO.input(self.left_pin_a)
            b_state = GPIO.input(self.left_pin_b)
            
            if a_state != b_state:
                self.left_encoder_count += 1
            else:
                self.left_encoder_count -= 1
    
    def right_encoder_callback(self, channel):
        """Interrupt callback for right encoder"""
        with self.encoder_lock:
            a_state = GPIO.input(self.right_pin_a)
            b_state = GPIO.input(self.right_pin_b)
            
            if a_state != b_state:
                self.right_encoder_count += 1
            else:
                self.right_encoder_count -= 1
    
    def publish_callback(self):
        """Main publishing callback"""
        current_time = time.time()
        dt = current_time - self.last_time
        
        with self.encoder_lock:
            left_count = self.left_encoder_count
            right_count = self.right_encoder_count
        
        # Convert encoder counts to wheel positions and velocities
        left_pos = (left_count / self.encoder_ppr) * 2 * math.pi
        right_pos = (right_count / self.encoder_ppr) * 2 * math.pi
        
        # Calculate wheel velocities (simple differentiation)
        if hasattr(self, 'prev_left_pos'):
            left_vel = (left_pos - self.prev_left_pos) / dt
            right_vel = (right_pos - self.prev_right_pos) / dt
        else:
            left_vel = 0.0
            right_vel = 0.0
        
        self.prev_left_pos = left_pos
        self.prev_right_pos = right_pos
        
        # Publish joint states
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = ['left_wheel_joint', 'right_wheel_joint']
        joint_msg.position = [left_pos, right_pos]
        joint_msg.velocity = [left_vel, right_vel]
        self.joint_state_pub.publish(joint_msg)
        
        # Calculate and publish odometry
        self.update_odometry(left_vel, right_vel, dt)
        
        self.last_time = current_time
    
    def update_odometry(self, left_vel, right_vel, dt):
        """Update and publish odometry"""
        # Tank steering kinematics
        linear_vel = (left_vel + right_vel) * self.wheel_radius / 2.0
        angular_vel = (right_vel - left_vel) * self.wheel_radius / self.wheel_separation
        
        # Update pose
        self.x += linear_vel * math.cos(self.theta) * dt
        self.y += linear_vel * math.sin(self.theta) * dt
        self.theta += angular_vel * dt
        
        # Normalize theta
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))
        
        # Create odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'
        
        # Position
        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.position.z = 0.0
        
        # Orientation (quaternion from yaw)
        odom_msg.pose.pose.orientation.x = 0.0
        odom_msg.pose.pose.orientation.y = 0.0
        odom_msg.pose.pose.orientation.z = math.sin(self.theta / 2.0)
        odom_msg.pose.pose.orientation.w = math.cos(self.theta / 2.0)
        
        # Velocity
        odom_msg.twist.twist.linear.x = linear_vel
        odom_msg.twist.twist.linear.y = 0.0
        odom_msg.twist.twist.angular.z = angular_vel
        
        # Covariance (simplified)
        odom_msg.pose.covariance[0] = 0.1  # x
        odom_msg.pose.covariance[7] = 0.1  # y
        odom_msg.pose.covariance[35] = 0.1  # theta
        odom_msg.twist.covariance[0] = 0.1  # vx
        odom_msg.twist.covariance[35] = 0.1  # vtheta
        
        self.odom_pub.publish(odom_msg)
        
        # Publish TF transform
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = 'odom'
        tf_msg.child_frame_id = 'base_link'
        tf_msg.transform.translation.x = self.x
        tf_msg.transform.translation.y = self.y
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation = odom_msg.pose.pose.orientation
        
        self.tf_broadcaster.sendTransform(tf_msg)
    
    def destroy_node(self):
        """Clean shutdown"""
        try:
            GPIO.cleanup()
        except Exception as e:
            self.get_logger().error(f'Error during GPIO cleanup: {e}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    encoder_pub = EncoderPublisher()
    
    try:
        rclpy.spin(encoder_pub)
    except KeyboardInterrupt:
        pass
    finally:
        encoder_pub.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()