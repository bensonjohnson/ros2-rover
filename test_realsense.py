#!/usr/bin/env python3
"""
Test script to verify RealSense camera functionality
"""
import cv2
import numpy as np
import sys

def test_realsense_via_opencv():
    """Test RealSense camera using OpenCV"""
    print("Testing RealSense D435i via OpenCV...")
    
    # Try to open the camera
    cap = cv2.VideoCapture(0)  # Try video0 first
    if not cap.isOpened():
        print("Failed to open camera on /dev/video0")
        # Try other video devices
        for i in range(1, 6):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Found camera on /dev/video{i}")
                break
        else:
            print("No camera found on any video device")
            return False
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Camera opened successfully!")
    print(f"Frame width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"Frame height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    
    # Capture a few frames
    for i in range(10):
        ret, frame = cap.read()
        if ret:
            print(f"Frame {i+1}: {frame.shape} - OK")
        else:
            print(f"Frame {i+1}: Failed to capture")
            break
    
    cap.release()
    return True

def test_realsense_via_pyrealsense():
    """Test RealSense camera using pyrealsense2"""
    try:
        import pyrealsense2 as rs
        
        print("Testing RealSense D435i via pyrealsense2...")
        
        # Create pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Configure streams
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start streaming
        pipeline.start(config)
        
        print("Pipeline started successfully!")
        
        # Get a few frames
        for i in range(10):
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if depth_frame and color_frame:
                print(f"Frame {i+1}: Color={color_frame.get_width()}x{color_frame.get_height()}, Depth={depth_frame.get_width()}x{depth_frame.get_height()} - OK")
            else:
                print(f"Frame {i+1}: Failed to get frames")
                break
        
        pipeline.stop()
        return True
        
    except ImportError:
        print("pyrealsense2 not available")
        return False
    except Exception as e:
        print(f"Error with pyrealsense2: {e}")
        return False

if __name__ == "__main__":
    print("RealSense D435i Camera Test")
    print("=" * 40)
    
    # Test via OpenCV
    opencv_result = test_realsense_via_opencv()
    
    print("\n" + "=" * 40)
    
    # Test via pyrealsense2
    pyrealsense_result = test_realsense_via_pyrealsense()
    
    print("\n" + "=" * 40)
    print("Test Results:")
    print(f"OpenCV: {'PASS' if opencv_result else 'FAIL'}")
    print(f"pyrealsense2: {'PASS' if pyrealsense_result else 'FAIL'}")
    
    if opencv_result:
        print("\nCamera is working via OpenCV! You can use it as a standard USB camera.")
    if pyrealsense_result:
        print("\nCamera is working via RealSense SDK! Full depth functionality available.")
    
    sys.exit(0 if (opencv_result or pyrealsense_result) else 1)