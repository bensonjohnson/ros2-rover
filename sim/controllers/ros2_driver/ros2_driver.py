#!/usr/bin/env python3
"""
ROS 2 Driver for Webots TractorRover simulation.

Bridges Webots sensors/actuators to ROS 2 topics matching the real rover:
- Subscribes: /cmd_vel_ai (Twist)
- Publishes: /scan, /camera/camera/depth/image_rect_raw, /imu/data, /odometry/filtered, /joint_states

For extern controller: PYTHONPATH must prioritize Webots controller library.
"""

import sys
import os

# IMPORTANT: Import Webots controller FIRST before ROS 2 to avoid version conflicts
# The ROS 2 Jazzy image has its own 'controller' package that conflicts with Webots
webots_path = os.environ.get('WEBOTS_HOME', '/usr/local/webots')
webots_controller_path = os.path.join(webots_path, 'lib', 'controller', 'python')
if webots_controller_path not in sys.path:
    sys.path.insert(0, webots_controller_path)

from controller import Robot

# Now import ROS 2 modules
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import math

from geometry_msgs.msg import Twist, TransformStamped
from sensor_msgs.msg import Image, LaserScan, Imu, JointState
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32, Header
from tf2_ros import TransformBroadcaster


class ROS2Driver(Node):
    """Webots to ROS 2 bridge for TractorRover."""

    def __init__(self, robot: Robot):
        super().__init__('webots_ros2_driver')
        
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep())
        
        # Robot parameters (matching real rover)
        self.wheel_radius = 0.0485  # meters
        self.wheel_separation = 0.144  # meters
        self.max_speed = 0.18  # m/s
        
        # Get devices
        self._init_devices()
        
        # ROS 2 Publishers
        self._init_publishers()
        
        # ROS 2 Subscribers
        self.create_subscription(Twist, '/cmd_vel_ai', self._cmd_vel_callback, 10)
        
        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # State
        self._cmd_linear = 0.0
        self._cmd_angular = 0.0
        self._prev_left_pos = 0.0
        self._prev_right_pos = 0.0
        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0
        
        self.get_logger().info('✅ Webots ROS2 Driver initialized')

    def _init_devices(self):
        """Initialize Webots devices."""
        # Motors
        self.left_motor = self.robot.getDevice('left_motor')
        self.right_motor = self.robot.getDevice('right_motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        
        # Position sensors
        self.left_sensor = self.robot.getDevice('left_wheel_sensor')
        self.right_sensor = self.robot.getDevice('right_wheel_sensor')
        self.left_sensor.enable(self.timestep)
        self.right_sensor.enable(self.timestep)
        
        # Depth camera (RangeFinder)
        self.depth_camera = self.robot.getDevice('depth_camera')
        self.depth_camera.enable(self.timestep)
        
        # LiDAR
        self.lidar = self.robot.getDevice('lidar')
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()
        
        # IMU sensors
        self.imu = self.robot.getDevice('imu')
        self.gyro = self.robot.getDevice('gyro')
        self.accelerometer = self.robot.getDevice('accelerometer')
        self.imu.enable(self.timestep)
        self.gyro.enable(self.timestep)
        self.accelerometer.enable(self.timestep)
        
        # GPS (for ground truth)
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)
        
        # Compass
        self.compass = self.robot.getDevice('compass')
        self.compass.enable(self.timestep)

    def _init_publishers(self):
        """Initialize ROS 2 publishers matching real rover topics."""
        # Depth camera - matches /camera/camera/depth/image_rect_raw
        self.depth_pub = self.create_publisher(
            Image, '/camera/camera/depth/image_rect_raw', qos_profile_sensor_data)
        
        # LiDAR - matches /scan
        self.scan_pub = self.create_publisher(
            LaserScan, '/scan', qos_profile_sensor_data)
        
        # IMU - matches /imu/data
        self.imu_pub = self.create_publisher(
            Imu, '/imu/data', qos_profile_sensor_data)
        
        # Odometry - matches /odometry/filtered (EKF output on real rover)
        self.odom_pub = self.create_publisher(
            Odometry, '/odometry/filtered', 10)
        
        # Joint states - matches /joint_states
        self.joint_pub = self.create_publisher(
            JointState, '/joint_states', 10)
        
        # Safety (always safe in sim)
        self.safety_pub = self.create_publisher(
            Bool, '/safety_monitor_status', 10)
        
        # Velocity confidence (always 1.0 in sim)
        self.vel_conf_pub = self.create_publisher(
            Float32, '/velocity_confidence', 10)

    def _cmd_vel_callback(self, msg: Twist):
        """Handle velocity commands."""
        self._cmd_linear = msg.linear.x
        self._cmd_angular = msg.angular.z

    def step(self):
        """Execute one simulation step."""
        # Apply motor commands (differential drive)
        v = self._cmd_linear
        w = self._cmd_angular
        
        # Convert to wheel velocities
        v_left = (v - w * self.wheel_separation / 2.0) / self.wheel_radius
        v_right = (v + w * self.wheel_separation / 2.0) / self.wheel_radius
        
        # Clamp to motor limits
        max_motor_speed = self.max_speed / self.wheel_radius * 2.0
        v_left = np.clip(v_left, -max_motor_speed, max_motor_speed)
        v_right = np.clip(v_right, -max_motor_speed, max_motor_speed)
        
        self.left_motor.setVelocity(v_left)
        self.right_motor.setVelocity(v_right)
        
        # Publish sensor data
        now = self.get_clock().now().to_msg()
        
        self._publish_depth(now)
        self._publish_scan(now)
        self._publish_imu(now)
        self._publish_odometry(now)
        self._publish_joint_states(now)
        self._publish_safety(now)
        self._publish_tf(now)

    def _publish_depth(self, stamp):
        """Publish depth image matching D435i format."""
        # Get depth image (float meters)
        depth_data = self.depth_camera.getRangeImage()
        if depth_data is None:
            return
        
        width = self.depth_camera.getWidth()
        height = self.depth_camera.getHeight()
        
        # Convert to numpy and then to uint16 millimeters (like real D435i)
        depth_np = np.array(depth_data, dtype=np.float32).reshape((height, width))
        depth_mm = (depth_np * 1000.0).astype(np.uint16)
        
        # Create Image message
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = 'camera_link'
        msg.height = height
        msg.width = width
        msg.encoding = '16UC1'
        msg.is_bigendian = False
        msg.step = width * 2
        msg.data = depth_mm.tobytes()
        
        self.depth_pub.publish(msg)

    def _publish_scan(self, stamp):
        """Publish LaserScan matching LD19 format."""
        ranges = self.lidar.getRangeImage()
        if ranges is None:
            return
        
        msg = LaserScan()
        msg.header.stamp = stamp
        msg.header.frame_id = 'laser_link'
        msg.angle_min = -math.pi
        msg.angle_max = math.pi
        msg.angle_increment = 2.0 * math.pi / len(ranges)
        msg.time_increment = 0.0
        msg.scan_time = 0.1  # 10Hz
        msg.range_min = 0.15
        msg.range_max = 12.0
        msg.ranges = list(ranges)
        msg.intensities = []
        
        self.scan_pub.publish(msg)

    def _publish_imu(self, stamp):
        """Publish IMU data matching LSM9DS1 format."""
        # Get sensor values
        orientation = self.imu.getRollPitchYaw()  # [roll, pitch, yaw]
        gyro_vals = self.gyro.getValues()  # [x, y, z] rad/s
        accel_vals = self.accelerometer.getValues()  # [x, y, z] m/s^2
        
        # Convert Euler to quaternion
        roll, pitch, yaw = orientation
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        msg = Imu()
        msg.header.stamp = stamp
        msg.header.frame_id = 'imu_link'
        
        # Quaternion
        msg.orientation.w = cr * cp * cy + sr * sp * sy
        msg.orientation.x = sr * cp * cy - cr * sp * sy
        msg.orientation.y = cr * sp * cy + sr * cp * sy
        msg.orientation.z = cr * cp * sy - sr * sp * cy
        
        # Angular velocity
        msg.angular_velocity.x = gyro_vals[0]
        msg.angular_velocity.y = gyro_vals[1]
        msg.angular_velocity.z = gyro_vals[2]
        
        # Linear acceleration
        msg.linear_acceleration.x = accel_vals[0]
        msg.linear_acceleration.y = accel_vals[1]
        msg.linear_acceleration.z = accel_vals[2]
        
        self.imu_pub.publish(msg)

    def _publish_odometry(self, stamp):
        """Publish odometry using GPS ground truth."""
        # Use GPS for ground truth position
        gps_vals = self.gps.getValues()  # [x, y, z]
        compass_vals = self.compass.getValues()  # [x, y, z] magnetic field
        
        # Calculate heading from compass
        heading = math.atan2(compass_vals[0], compass_vals[1])
        
        # Convert to quaternion
        cy = math.cos(heading * 0.5)
        sy = math.sin(heading * 0.5)
        
        # Calculate velocity from wheel encoders
        left_pos = self.left_sensor.getValue()
        right_pos = self.right_sensor.getValue()
        
        dt = self.timestep / 1000.0  # Convert to seconds
        
        d_left = (left_pos - self._prev_left_pos) * self.wheel_radius
        d_right = (right_pos - self._prev_right_pos) * self.wheel_radius
        
        self._prev_left_pos = left_pos
        self._prev_right_pos = right_pos
        
        linear_vel = (d_left + d_right) / (2.0 * dt)
        angular_vel = (d_right - d_left) / (self.wheel_separation * dt)
        
        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_footprint'
        
        # Position (from GPS)
        msg.pose.pose.position.x = gps_vals[0]
        msg.pose.pose.position.y = gps_vals[1]
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.w = cy
        msg.pose.pose.orientation.z = sy
        
        # Velocity
        msg.twist.twist.linear.x = linear_vel
        msg.twist.twist.angular.z = angular_vel
        
        self.odom_pub.publish(msg)

    def _publish_joint_states(self, stamp):
        """Publish joint states for wheel velocities."""
        left_vel = self.left_motor.getVelocity() if hasattr(self.left_motor, 'getVelocity') else 0.0
        right_vel = self.right_motor.getVelocity() if hasattr(self.right_motor, 'getVelocity') else 0.0
        
        msg = JointState()
        msg.header.stamp = stamp
        msg.name = ['left_wheel_joint', 'right_wheel_joint', 'left_wheel', 'right_wheel']
        msg.position = [self.left_sensor.getValue(), self.right_sensor.getValue(), 0.0, 0.0]
        msg.velocity = [left_vel, right_vel, left_vel * self.wheel_radius, right_vel * self.wheel_radius]
        msg.effort = []
        
        self.joint_pub.publish(msg)

    def _publish_safety(self, stamp):
        """Publish safety status (always safe in sim)."""
        self.safety_pub.publish(Bool(data=False))
        self.vel_conf_pub.publish(Float32(data=1.0))

    def _publish_tf(self, stamp):
        """Publish TF transforms."""
        # odom -> base_footprint
        gps_vals = self.gps.getValues()
        compass_vals = self.compass.getValues()
        heading = math.atan2(compass_vals[0], compass_vals[1])
        
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_footprint'
        t.transform.translation.x = gps_vals[0]
        t.transform.translation.y = gps_vals[1]
        t.transform.translation.z = 0.0
        
        cy = math.cos(heading * 0.5)
        sy = math.sin(heading * 0.5)
        t.transform.rotation.w = cy
        t.transform.rotation.z = sy
        
        self.tf_broadcaster.sendTransform(t)


def main():
    # Initialize Webots robot
    robot = Robot()
    
    # Initialize ROS 2
    rclpy.init()
    
    # Create driver node
    driver = ROS2Driver(robot)
    
    # Main loop
    timestep = int(robot.getBasicTimeStep())
    
    while robot.step(timestep) != -1:
        rclpy.spin_once(driver, timeout_sec=0)
        driver.step()
    
    # Cleanup
    driver.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
