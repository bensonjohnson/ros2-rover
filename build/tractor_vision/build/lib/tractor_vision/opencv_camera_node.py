#!/usr/bin/env python3
"""
OpenCV Camera Node for RealSense D435i
Uses OpenCV instead of RealSense SDK when SDK doesn't recognize the camera
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import threading
import time


class OpenCVCameraNode(Node):
    def __init__(self):
        super().__init__('opencv_camera_node')
        
        # Parameters
        self.declare_parameter('camera_device', 2)  # /dev/video2 by default
        self.declare_parameter('camera_name', 'realsense')
        self.declare_parameter('frame_id', 'camera_link')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('publish_rate', 30.0)
        
        # Get parameters
        self.camera_device = self.get_parameter('camera_device').get_parameter_value().integer_value
        self.camera_name = self.get_parameter('camera_name').get_parameter_value().string_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.width = self.get_parameter('width').get_parameter_value().integer_value
        self.height = self.get_parameter('height').get_parameter_value().integer_value
        self.fps = self.get_parameter('fps').get_parameter_value().integer_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        
        # Initialize camera
        self.bridge = CvBridge()
        self.cap = None
        self.camera_info = None
        self.running = False
        self.frame_thread = None
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Publishers
        self.image_pub = self.create_publisher(
            Image, f'/{self.camera_name}/color/image_raw', 10
        )
        self.camera_info_pub = self.create_publisher(
            CameraInfo, f'/{self.camera_name}/color/camera_info', 10
        )
        
        # Initialize camera
        self.init_camera()
        
        # Create timer for publishing
        self.timer = self.create_timer(
            1.0 / self.publish_rate, self.publish_frame
        )
        
        self.get_logger().info(f"OpenCV Camera Node initialized for {self.camera_name}")
        self.get_logger().info(f"Using device: /dev/video{self.camera_device}")
        self.get_logger().info(f"Resolution: {self.width}x{self.height} @ {self.fps} fps")
    
    def init_camera(self):
        """Initialize the camera"""
        try:
            # Try to open the camera
            self.cap = cv2.VideoCapture(self.camera_device)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera device {self.camera_device}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.get_logger().info(f"Actual camera settings: {actual_width}x{actual_height} @ {actual_fps} fps")
            
            # Create camera info
            self.camera_info = self.create_camera_info()
            
            # Start capture thread
            self.running = True
            self.frame_thread = threading.Thread(target=self.capture_frames)
            self.frame_thread.daemon = True
            self.frame_thread.start()
            
            self.get_logger().info("Camera initialized successfully")
            
        except Exception as e:
            self.get_logger().error(f"Failed to initialize camera: {e}")
            raise
    
    def capture_frames(self):
        """Capture frames in a separate thread"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.current_frame = frame
            else:
                self.get_logger().warn("Failed to capture frame")
                time.sleep(0.1)
    
    def create_camera_info(self):
        """Create camera info message"""
        camera_info = CameraInfo()
        camera_info.header.frame_id = self.frame_id
        camera_info.width = self.width
        camera_info.height = self.height
        
        # Basic camera matrix for RealSense D435i (approximate values)
        # These should be calibrated for your specific camera
        fx = 615.0  # focal length x
        fy = 615.0  # focal length y
        cx = self.width / 2.0   # optical center x
        cy = self.height / 2.0  # optical center y
        
        camera_info.k = [fx, 0.0, cx,
                        0.0, fy, cy,
                        0.0, 0.0, 1.0]
        
        camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # No distortion by default
        
        camera_info.r = [1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0]
        
        camera_info.p = [fx, 0.0, cx, 0.0,
                        0.0, fy, cy, 0.0,
                        0.0, 0.0, 1.0, 0.0]
        
        return camera_info
    
    def publish_frame(self):
        """Publish current frame"""
        if self.current_frame is None:
            return
            
        try:
            with self.frame_lock:
                frame = self.current_frame.copy()
            
            # Create timestamp
            now = self.get_clock().now()
            
            # Convert to ROS image
            image_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            image_msg.header.stamp = now.to_msg()
            image_msg.header.frame_id = self.frame_id
            
            # Update camera info timestamp
            self.camera_info.header.stamp = now.to_msg()
            
            # Publish
            self.image_pub.publish(image_msg)
            self.camera_info_pub.publish(self.camera_info)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing frame: {e}")
    
    def destroy_node(self):
        """Cleanup when node is destroyed"""
        self.running = False
        if self.frame_thread:
            self.frame_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = OpenCVCameraNode()
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