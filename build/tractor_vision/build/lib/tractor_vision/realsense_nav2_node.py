#!/usr/bin/env python3
"""
RealSense D435i Node for Nav2 Integration
Provides point cloud, IMU data, and coordinate frame transforms for navigation
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, Imu
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
import cv2
import numpy as np
import threading
import time
import struct
import math
from builtin_interfaces.msg import Time


class RealSenseNav2Node(Node):
    def __init__(self):
        super().__init__('realsense_nav2_node')
        
        # Parameters
        self.declare_parameter('camera_device', 2)  # Color camera
        self.declare_parameter('depth_device', 0)   # Depth camera
        self.declare_parameter('camera_name', 'realsense')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('depth_frame', 'camera_depth_frame')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('publish_rate', 30.0)
        self.declare_parameter('depth_scale', 0.001)  # mm to meters
        self.declare_parameter('max_depth', 10.0)     # max depth in meters
        self.declare_parameter('min_depth', 0.1)      # min depth in meters
        
        # Get parameters
        self.camera_device = self.get_parameter('camera_device').get_parameter_value().integer_value
        self.depth_device = self.get_parameter('depth_device').get_parameter_value().integer_value
        self.camera_name = self.get_parameter('camera_name').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        self.depth_frame = self.get_parameter('depth_frame').get_parameter_value().string_value
        self.width = self.get_parameter('width').get_parameter_value().integer_value
        self.height = self.get_parameter('height').get_parameter_value().integer_value
        self.fps = self.get_parameter('fps').get_parameter_value().integer_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.depth_scale = self.get_parameter('depth_scale').get_parameter_value().double_value
        self.max_depth = self.get_parameter('max_depth').get_parameter_value().double_value
        self.min_depth = self.get_parameter('min_depth').get_parameter_value().double_value
        
        # Initialize components
        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        
        # Camera setup
        self.color_cap = None
        self.depth_cap = None
        self.current_color_frame = None
        self.current_depth_frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.frame_thread = None
        
        # Camera intrinsics (approximate for D435i)
        self.fx = 615.0
        self.fy = 615.0
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        
        # Publishers
        self.color_pub = self.create_publisher(
            Image, f'/{self.camera_name}/color/image_raw', 10
        )
        self.depth_pub = self.create_publisher(
            Image, f'/{self.camera_name}/depth/image_raw', 10
        )
        self.camera_info_pub = self.create_publisher(
            CameraInfo, f'/{self.camera_name}/color/camera_info', 10
        )
        self.depth_info_pub = self.create_publisher(
            CameraInfo, f'/{self.camera_name}/depth/camera_info', 10
        )
        self.pointcloud_pub = self.create_publisher(
            PointCloud2, f'/{self.camera_name}/depth/points', 10
        )
        self.imu_pub = self.create_publisher(
            Imu, f'/{self.camera_name}/imu', 10
        )
        
        # Initialize camera
        self.init_cameras()
        
        # Publish static transforms
        self.publish_static_transforms()
        
        # Create timers
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_frames)
        self.imu_timer = self.create_timer(0.01, self.publish_imu)  # 100Hz for IMU
        
        self.get_logger().info(f"RealSense Nav2 Node initialized")
        self.get_logger().info(f"Color: /dev/video{self.camera_device}, Depth: /dev/video{self.depth_device}")
    
    def init_cameras(self):
        """Initialize color and depth cameras"""
        try:
            # Color camera
            self.color_cap = cv2.VideoCapture(self.camera_device)
            if not self.color_cap.isOpened():
                raise RuntimeError(f"Failed to open color camera {self.camera_device}")
            
            # Depth camera
            self.depth_cap = cv2.VideoCapture(self.depth_device)
            if not self.depth_cap.isOpened():
                raise RuntimeError(f"Failed to open depth camera {self.depth_device}")
            
            # Set camera properties
            for cap in [self.color_cap, self.depth_cap]:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Start capture thread
            self.running = True
            self.frame_thread = threading.Thread(target=self.capture_frames)
            self.frame_thread.daemon = True
            self.frame_thread.start()
            
            self.get_logger().info("Cameras initialized successfully")
            
        except Exception as e:
            self.get_logger().error(f"Failed to initialize cameras: {e}")
            raise
    
    def capture_frames(self):
        """Capture frames in separate thread"""
        while self.running:
            color_ret, color_frame = self.color_cap.read()
            depth_ret, depth_frame = self.depth_cap.read()
            
            if color_ret and depth_ret:
                with self.frame_lock:
                    self.current_color_frame = color_frame
                    self.current_depth_frame = depth_frame
            
            time.sleep(0.01)  # Small delay to prevent CPU overload
    
    def publish_static_transforms(self):
        """Publish static transforms for camera frames"""
        transforms = []
        
        # Transform from base_link to camera_link
        # Adjust these values based on your camera mounting position
        t1 = TransformStamped()
        t1.header.stamp = self.get_clock().now().to_msg()
        t1.header.frame_id = self.base_frame
        t1.child_frame_id = self.camera_frame
        t1.transform.translation.x = 0.2   # 20cm forward from base_link
        t1.transform.translation.y = 0.0   # centered
        t1.transform.translation.z = 0.15  # 15cm above base_link
        t1.transform.rotation.x = 0.0
        t1.transform.rotation.y = 0.0
        t1.transform.rotation.z = 0.0
        t1.transform.rotation.w = 1.0
        transforms.append(t1)
        
        # Transform from camera_link to camera_depth_frame
        t2 = TransformStamped()
        t2.header.stamp = self.get_clock().now().to_msg()
        t2.header.frame_id = self.camera_frame
        t2.child_frame_id = self.depth_frame
        t2.transform.translation.x = 0.0
        t2.transform.translation.y = 0.0
        t2.transform.translation.z = 0.0
        t2.transform.rotation.x = 0.0
        t2.transform.rotation.y = 0.0
        t2.transform.rotation.z = 0.0
        t2.transform.rotation.w = 1.0
        transforms.append(t2)
        
        self.static_tf_broadcaster.sendTransform(transforms)
    
    def create_camera_info(self, frame_id):
        """Create camera info message"""
        camera_info = CameraInfo()
        camera_info.header.frame_id = frame_id
        camera_info.width = self.width
        camera_info.height = self.height
        
        camera_info.k = [self.fx, 0.0, self.cx,
                        0.0, self.fy, self.cy,
                        0.0, 0.0, 1.0]
        
        camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        camera_info.r = [1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0]
        
        camera_info.p = [self.fx, 0.0, self.cx, 0.0,
                        0.0, self.fy, self.cy, 0.0,
                        0.0, 0.0, 1.0, 0.0]
        
        return camera_info
    
    def depth_to_pointcloud(self, depth_image):
        """Convert depth image to point cloud"""
        if depth_image is None:
            return None
        
        # Convert depth image to grayscale if needed
        if len(depth_image.shape) == 3:
            depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
        
        # Create point cloud
        points = []
        for v in range(0, self.height, 2):  # Decimate for performance
            for u in range(0, self.width, 2):
                # Get depth value (assuming depth in mm, convert to meters)
                z = depth_image[v, u] * self.depth_scale
                
                # Filter out invalid depths
                if z < self.min_depth or z > self.max_depth:
                    continue
                
                # Project to 3D
                x = (u - self.cx) * z / self.fx
                y = (v - self.cy) * z / self.fy
                
                # Add point (x, y, z) - pack as binary
                points.append(struct.pack('fff', x, y, z))
        
        if not points:
            return None
        
        # Create PointCloud2 message
        pc2 = PointCloud2()
        pc2.header.frame_id = self.depth_frame
        pc2.height = 1
        pc2.width = len(points)
        pc2.fields = [
            {'name': 'x', 'offset': 0, 'datatype': 7, 'count': 1},
            {'name': 'y', 'offset': 4, 'datatype': 7, 'count': 1},
            {'name': 'z', 'offset': 8, 'datatype': 7, 'count': 1}
        ]
        pc2.is_bigendian = False
        pc2.point_step = 12
        pc2.row_step = pc2.point_step * pc2.width
        pc2.data = b''.join(points)
        pc2.is_dense = True
        
        return pc2
    
    def publish_frames(self):
        """Publish camera frames and point cloud"""
        if self.current_color_frame is None or self.current_depth_frame is None:
            return
        
        try:
            with self.frame_lock:
                color_frame = self.current_color_frame.copy()
                depth_frame = self.current_depth_frame.copy()
            
            now = self.get_clock().now()
            
            # Publish color image
            color_msg = self.bridge.cv2_to_imgmsg(color_frame, encoding='bgr8')
            color_msg.header.stamp = now.to_msg()
            color_msg.header.frame_id = self.camera_frame
            self.color_pub.publish(color_msg)
            
            # Publish depth image
            depth_msg = self.bridge.cv2_to_imgmsg(depth_frame, encoding='16UC1')
            depth_msg.header.stamp = now.to_msg()
            depth_msg.header.frame_id = self.depth_frame
            self.depth_pub.publish(depth_msg)
            
            # Publish camera info
            color_info = self.create_camera_info(self.camera_frame)
            color_info.header.stamp = now.to_msg()
            self.camera_info_pub.publish(color_info)
            
            depth_info = self.create_camera_info(self.depth_frame)
            depth_info.header.stamp = now.to_msg()
            self.depth_info_pub.publish(depth_info)
            
            # Publish point cloud
            pc2 = self.depth_to_pointcloud(depth_frame)
            if pc2:
                pc2.header.stamp = now.to_msg()
                self.pointcloud_pub.publish(pc2)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing frames: {e}")
    
    def publish_imu(self):
        """Publish IMU data (simulated for now since we can't access actual IMU)"""
        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = self.camera_frame
        
        # Set covariance matrices (high uncertainty since this is simulated)
        imu_msg.orientation_covariance = [0.1, 0.0, 0.0,
                                         0.0, 0.1, 0.0,
                                         0.0, 0.0, 0.1]
        imu_msg.angular_velocity_covariance = [0.1, 0.0, 0.0,
                                              0.0, 0.1, 0.0,
                                              0.0, 0.0, 0.1]
        imu_msg.linear_acceleration_covariance = [0.1, 0.0, 0.0,
                                                 0.0, 0.1, 0.0,
                                                 0.0, 0.0, 0.1]
        
        # For now, just publish zero values - in a real implementation,
        # you would read from the actual IMU via HID interface
        imu_msg.orientation.w = 1.0
        imu_msg.linear_acceleration.z = 9.81  # Gravity
        
        self.imu_pub.publish(imu_msg)
    
    def destroy_node(self):
        """Cleanup"""
        self.running = False
        if self.frame_thread:
            self.frame_thread.join(timeout=1.0)
        if self.color_cap:
            self.color_cap.release()
        if self.depth_cap:
            self.depth_cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = RealSenseNav2Node()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()