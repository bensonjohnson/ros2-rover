#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from geometry_msgs.msg import TransformStamped
import tf2_ros
import cv2
from cv_bridge import CvBridge
import numpy as np
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("pyrealsense2 not available. Will use official realsense2_camera node instead.")


class RealSenseProcessor(Node):
    def __init__(self):
        super().__init__('realsense_processor')
        
        if not REALSENSE_AVAILABLE:
            self.get_logger().info("pyrealsense2 library not available. Using ROS2 RealSense topics instead.")
            self.setup_ros_subscribers()
            return
        
        # Parameters
        self.declare_parameter('camera_name', 'realsense_435i')
        self.declare_parameter('frame_id', 'camera_link')
        self.declare_parameter('depth_frame_id', 'camera_depth_frame')
        self.declare_parameter('color_width', 640)
        self.declare_parameter('color_height', 480)
        self.declare_parameter('color_fps', 30)
        self.declare_parameter('depth_width', 640)
        self.declare_parameter('depth_height', 480)
        self.declare_parameter('depth_fps', 30)
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('min_depth', 0.1)
        self.declare_parameter('max_depth', 10.0)
        self.declare_parameter('enable_pointcloud', True)
        self.declare_parameter('pointcloud_decimation', 2)
        self.declare_parameter('enable_align', True)
        
        self.camera_name = self.get_parameter('camera_name').value
        self.frame_id = self.get_parameter('frame_id').value
        self.depth_frame_id = self.get_parameter('depth_frame_id').value
        self.color_width = self.get_parameter('color_width').value
        self.color_height = self.get_parameter('color_height').value
        self.color_fps = self.get_parameter('color_fps').value
        self.depth_width = self.get_parameter('depth_width').value
        self.depth_height = self.get_parameter('depth_height').value
        self.depth_fps = self.get_parameter('depth_fps').value
        self.depth_scale = self.get_parameter('depth_scale').value
        self.min_depth = self.get_parameter('min_depth').value
        self.max_depth = self.get_parameter('max_depth').value
        self.enable_pointcloud = self.get_parameter('enable_pointcloud').value
        self.pointcloud_decimation = self.get_parameter('pointcloud_decimation').value
        self.enable_align = self.get_parameter('enable_align').value
        
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure streams
        self.config.enable_stream(rs.stream.color, self.color_width, self.color_height, rs.format.bgr8, self.color_fps)
        self.config.enable_stream(rs.stream.depth, self.depth_width, self.depth_height, rs.format.z16, self.depth_fps)
        
        # Start streaming
        try:
            self.profile = self.pipeline.start(self.config)
            self.get_logger().info("RealSense camera started successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to start RealSense camera: {e}")
            return
        
        # Get device info
        device = self.profile.get_device()
        depth_sensor = device.first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        # Alignment
        if self.enable_align:
            self.align = rs.align(rs.stream.color)
        
        # Point cloud
        if self.enable_pointcloud:
            self.pc = rs.pointcloud()
            self.decimation = rs.decimation_filter()
            self.decimation.set_option(rs.option.filter_magnitude, self.pointcloud_decimation)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Publishers
        self.color_pub = self.create_publisher(Image, f'{self.camera_name}/color/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, f'{self.camera_name}/depth/image_rect_raw', 10)
        self.color_info_pub = self.create_publisher(CameraInfo, f'{self.camera_name}/color/camera_info', 10)
        self.depth_info_pub = self.create_publisher(CameraInfo, f'{self.camera_name}/depth/camera_info', 10)
        
        if self.enable_pointcloud:
            self.pointcloud_pub = self.create_publisher(PointCloud2, f'{self.camera_name}/depth/points', 10)
        
        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Timer for frame processing
        self.timer = self.create_timer(1.0 / 30.0, self.process_frames)  # 30 Hz
        
        self.get_logger().info('RealSense Processor initialized')
    
    def setup_ros_subscribers(self):
        """Setup ROS subscribers for RealSense topics when pyrealsense2 is not available"""
        # CV Bridge
        self.bridge = CvBridge()
        
        # Parameters
        self.declare_parameter('camera_name', 'realsense')
        self.camera_name = self.get_parameter('camera_name').value
        
        # Subscribers to official RealSense node topics
        self.color_sub = self.create_subscription(
            Image,
            f'/{self.camera_name}/color/image_raw',
            self.color_callback,
            10
        )
        
        self.depth_sub = self.create_subscription(
            Image,
            f'/{self.camera_name}/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10
        )
        
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            f'/{self.camera_name}/depth/color/points',
            self.pointcloud_callback,
            10
        )
        
        # Publishers for processed data
        self.processed_color_pub = self.create_publisher(
            Image, 
            f'{self.camera_name}/processed/color/image_raw', 
            10
        )
        
        self.processed_depth_pub = self.create_publisher(
            Image, 
            f'{self.camera_name}/processed/depth/image_raw', 
            10
        )
        
        self.get_logger().info('RealSense ROS subscribers initialized')
    
    def color_callback(self, msg):
        """Process color image from RealSense camera"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # Add any custom processing here
            
            # Republish processed image
            processed_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            processed_msg.header = msg.header
            self.processed_color_pub.publish(processed_msg)
            
        except Exception as e:
            self.get_logger().error(f'Color image processing error: {e}')
    
    def depth_callback(self, msg):
        """Process depth image from RealSense camera"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            # Add any custom depth processing here
            
            # Republish processed depth image
            processed_msg = self.bridge.cv2_to_imgmsg(cv_image, "16UC1")
            processed_msg.header = msg.header
            self.processed_depth_pub.publish(processed_msg)
            
        except Exception as e:
            self.get_logger().error(f'Depth image processing error: {e}')
    
    def pointcloud_callback(self, msg):
        """Process point cloud from RealSense camera"""
        try:
            # Add any custom point cloud processing here
            self.get_logger().debug('Received point cloud with {} points'.format(
                msg.width * msg.height))
            
        except Exception as e:
            self.get_logger().error(f'Point cloud processing error: {e}')
    
    def get_logger(self):
        return super().get_logger()
    
    def process_frames(self):
        """Process and publish camera frames"""
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames(timeout_ms=100)
            
            if self.enable_align:
                frames = self.align.process(frames)
            
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return
            
            # Get timestamps
            timestamp = self.get_clock().now()
            
            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Apply depth filtering
            depth_image_filtered = np.where(
                (depth_image * self.depth_scale >= self.min_depth) & 
                (depth_image * self.depth_scale <= self.max_depth),
                depth_image, 0
            )
            
            # Publish color image
            color_msg = self.bridge.cv2_to_imgmsg(color_image, encoding='bgr8')
            color_msg.header.stamp = timestamp.to_msg()
            color_msg.header.frame_id = self.frame_id
            self.color_pub.publish(color_msg)
            
            # Publish depth image
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image_filtered, encoding='16UC1')
            depth_msg.header.stamp = timestamp.to_msg()
            depth_msg.header.frame_id = self.depth_frame_id
            self.depth_pub.publish(depth_msg)
            
            # Publish camera info
            self.publish_camera_info(color_frame, depth_frame, timestamp)
            
            # Publish point cloud
            if self.enable_pointcloud:
                self.publish_pointcloud(frames, timestamp)
            
            # Publish TF
            self.publish_tf(timestamp)
            
        except Exception as e:
            self.get_logger().debug(f"Frame processing error: {e}")
    
    def publish_camera_info(self, color_frame, depth_frame, timestamp):
        """Publish camera info messages"""
        # Color camera info
        color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        color_info = CameraInfo()
        color_info.header.stamp = timestamp.to_msg()
        color_info.header.frame_id = self.frame_id
        color_info.width = color_intrinsics.width
        color_info.height = color_intrinsics.height
        color_info.k = [color_intrinsics.fx, 0.0, color_intrinsics.ppx,
                       0.0, color_intrinsics.fy, color_intrinsics.ppy,
                       0.0, 0.0, 1.0]
        color_info.d = [color_intrinsics.coeffs[0], color_intrinsics.coeffs[1],
                       color_intrinsics.coeffs[2], color_intrinsics.coeffs[3],
                       color_intrinsics.coeffs[4]]
        color_info.distortion_model = "plumb_bob"
        self.color_info_pub.publish(color_info)
        
        # Depth camera info
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        depth_info = CameraInfo()
        depth_info.header.stamp = timestamp.to_msg()
        depth_info.header.frame_id = self.depth_frame_id
        depth_info.width = depth_intrinsics.width
        depth_info.height = depth_intrinsics.height
        depth_info.k = [depth_intrinsics.fx, 0.0, depth_intrinsics.ppx,
                       0.0, depth_intrinsics.fy, depth_intrinsics.ppy,
                       0.0, 0.0, 1.0]
        depth_info.d = [depth_intrinsics.coeffs[0], depth_intrinsics.coeffs[1],
                       depth_intrinsics.coeffs[2], depth_intrinsics.coeffs[3],
                       depth_intrinsics.coeffs[4]]
        depth_info.distortion_model = "plumb_bob"
        self.depth_info_pub.publish(depth_info)
    
    def publish_pointcloud(self, frames, timestamp):
        """Publish point cloud data"""
        try:
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            # Apply decimation filter
            depth_frame = self.decimation.process(depth_frame)
            
            # Generate point cloud
            points = self.pc.calculate(depth_frame)
            self.pc.map_to(color_frame)
            
            # Convert to ROS PointCloud2 message
            # This is a simplified version - full implementation would convert
            # the pyrealsense2 points to sensor_msgs/PointCloud2
            pointcloud_msg = PointCloud2()
            pointcloud_msg.header.stamp = timestamp.to_msg()
            pointcloud_msg.header.frame_id = self.depth_frame_id
            
            # Note: Full point cloud conversion implementation would be needed here
            # This is a placeholder for the structure
            self.pointcloud_pub.publish(pointcloud_msg)
            
        except Exception as e:
            self.get_logger().debug(f"Point cloud processing error: {e}")
    
    def publish_tf(self, timestamp):
        """Publish camera TF transforms"""
        # Camera link to camera color frame
        tf_msg = TransformStamped()
        tf_msg.header.stamp = timestamp.to_msg()
        tf_msg.header.frame_id = self.frame_id
        tf_msg.child_frame_id = f"{self.frame_id}_color_optical_frame"
        tf_msg.transform.translation.x = 0.0
        tf_msg.transform.translation.y = 0.0
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation.x = -0.5
        tf_msg.transform.rotation.y = 0.5
        tf_msg.transform.rotation.z = -0.5
        tf_msg.transform.rotation.w = 0.5
        
        self.tf_broadcaster.sendTransform(tf_msg)
        
        # Camera link to camera depth frame
        tf_msg.child_frame_id = f"{self.frame_id}_depth_optical_frame"
        tf_msg.transform.translation.x = 0.0  # Small offset for D435i
        self.tf_broadcaster.sendTransform(tf_msg)
    
    def destroy_node(self):
        """Clean shutdown"""
        try:
            if hasattr(self, 'pipeline'):
                self.pipeline.stop()
        except Exception as e:
            self.get_logger().error(f'Error during camera cleanup: {e}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    if not REALSENSE_AVAILABLE:
        print("RealSense library not available. Cannot start node.")
        return
    
    processor = RealSenseProcessor()
    
    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        pass
    finally:
        processor.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()