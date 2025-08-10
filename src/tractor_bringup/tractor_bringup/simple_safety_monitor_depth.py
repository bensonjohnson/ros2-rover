#!/usr/bin/env python3
"""
Simple Safety Monitor for NPU Depth Exploration
Emergency stop only - no complex navigation
Enhanced with hysteresis, smoothing, minimum valid pixel requirement, and hold time to prevent chatter.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import numpy as np
from cv_bridge import CvBridge
import cv2
from collections import deque

class SimpleSafetyMonitorDepth(Node):
    def __init__(self):
        super().__init__('simple_safety_monitor_depth')
        
        # Parameters
        self.declare_parameter('emergency_stop_distance', 0.1)
        # Separate clear distance (if <=0 use multiplier)
        self.declare_parameter('clear_distance', 0.0)
        self.declare_parameter('clear_distance_multiplier', 1.5)
        self.declare_parameter('max_speed_limit', 0.15)
        # Minimum number of valid depth pixels required in ROI for a reliable measurement
        self.declare_parameter('min_valid_pixels', 200)
        # Number of consecutive clear frames required to release stop
        self.declare_parameter('hysteresis_frames', 4)
        # Median smoothing window size for min distance
        self.declare_parameter('smoothing_window', 5)
        # Minimum time to hold emergency stop once engaged (seconds)
        self.declare_parameter('minimum_hold_time_secs', 0.4)
        # Publish zero command immediately on trigger
        self.declare_parameter('publish_zero_on_trigger', True)
        # ROI percentages (center crop)
        self.declare_parameter('roi_top_pct', 0.3)
        self.declare_parameter('roi_bottom_pct', 0.7)
        self.declare_parameter('roi_left_pct', 0.4)
        self.declare_parameter('roi_right_pct', 0.6)
        # Optional debug logging for every depth evaluation
        self.declare_parameter('debug_depth_eval', False)

        self.emergency_distance = float(self.get_parameter('emergency_stop_distance').value)
        self.clear_distance = float(self.get_parameter('clear_distance').value)
        if self.clear_distance <= 0.0:
            self.clear_distance = self.emergency_distance * float(self.get_parameter('clear_distance_multiplier').value)
        self.max_speed = float(self.get_parameter('max_speed_limit').value)
        self.min_valid_pixels = int(self.get_parameter('min_valid_pixels').value)
        self.hysteresis_frames = int(self.get_parameter('hysteresis_frames').value)
        self.smoothing_window = int(self.get_parameter('smoothing_window').value)
        self.minimum_hold_time = float(self.get_parameter('minimum_hold_time_secs').value)
        self.publish_zero_on_trigger = bool(self.get_parameter('publish_zero_on_trigger').value)
        self.roi_top_pct = float(self.get_parameter('roi_top_pct').value)
        self.roi_bottom_pct = float(self.get_parameter('roi_bottom_pct').value)
        self.roi_left_pct = float(self.get_parameter('roi_left_pct').value)
        self.roi_right_pct = float(self.get_parameter('roi_right_pct').value)
        self.debug_depth_eval = bool(self.get_parameter('debug_depth_eval').value)
        
        # State
        self.emergency_stop = False
        self.last_cmd = Twist()
        self.bridge = CvBridge()
        self.recent_mins: deque = deque(maxlen=self.smoothing_window)
        self.consecutive_clear = 0
        self.stop_engaged_time = 0.0
        self.last_smoothed_min = None
        
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
        self.get_logger().info(f"  Stop distance: {self.emergency_distance}m | Clear distance: {self.clear_distance}m")
        
    def _publish_zero_now(self):
        zero = Twist()
        self.cmd_pub.publish(zero)
        self.last_cmd = zero

    def depth_callback(self, msg):
        """Check for immediate obstacles using depth image with hysteresis and smoothing"""
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            if cv_depth is None or cv_depth.size == 0:
                # Insufficient data: keep stop if already engaged; otherwise ignore
                if self.emergency_stop and self.debug_depth_eval:
                    self.get_logger().debug("Depth empty; maintaining stop state")
                return
            # Convert to meters if needed
            if cv_depth.dtype == np.uint16:
                depth_meters = cv_depth.astype(np.float32) / 1000.0
            else:
                depth_meters = cv_depth.astype(np.float32)
            h, w = depth_meters.shape
            top = int(h * self.roi_top_pct)
            bottom = int(h * self.roi_bottom_pct)
            left = int(w * self.roi_left_pct)
            right = int(w * self.roi_right_pct)
            if bottom <= top or right <= left:
                # Fallback to center crop defaults if params invalid
                top = int(h * 0.3); bottom = int(h * 0.7); left = int(w * 0.4); right = int(w * 0.6)
            roi = depth_meters[top:bottom, left:right]
            # Filter invalids
            valid = roi[(roi > 0.01) & (roi < 10.0) & np.isfinite(roi)]
            valid_count = valid.size
            smoothed_min = self.last_smoothed_min
            if valid_count >= self.min_valid_pixels:
                current_min = float(np.min(valid))
                self.recent_mins.append(current_min)
                smoothed_min = float(np.median(self.recent_mins))
                self.last_smoothed_min = smoothed_min
            else:
                # Not enough valid pixels: if not stopped, do nothing; if stopped, retain last smoothed_min
                if self.emergency_stop and self.debug_depth_eval:
                    self.get_logger().debug(f"Insufficient valid pixels ({valid_count}) retaining stop")
                if not self.emergency_stop:
                    return
            now = self.get_clock().now().seconds_nanoseconds()[0] + self.get_clock().now().seconds_nanoseconds()[1]/1e9
            if smoothed_min is None:
                return
            if not self.emergency_stop:
                if smoothed_min < self.emergency_distance:
                    self.emergency_stop = True
                    self.stop_engaged_time = now
                    self.consecutive_clear = 0
                    self.get_logger().warn(f"EMERGENCY STOP: Obstacle median {smoothed_min:.2f}m (raw count {valid_count})")
                    if self.publish_zero_on_trigger:
                        self._publish_zero_now()
            else:
                # Already stopped; evaluate release
                time_held = now - self.stop_engaged_time
                if smoothed_min > self.clear_distance:
                    self.consecutive_clear += 1
                else:
                    self.consecutive_clear = 0
                if (smoothed_min > self.clear_distance and 
                    self.consecutive_clear >= self.hysteresis_frames and 
                    time_held >= self.minimum_hold_time):
                    self.emergency_stop = False
                    self.get_logger().info(f"Emergency stop cleared (median {smoothed_min:.2f}m after {time_held:.2f}s)")
                    self.consecutive_clear = 0
            if self.debug_depth_eval and smoothed_min is not None:
                self.get_logger().debug(
                    f"DepthEval min_med={smoothed_min:.3f} stop={self.emergency_stop} clear_frames={self.consecutive_clear} valid={valid_count}" )
        except Exception as e:
            self.get_logger().warn(f"Safety check failed: {e}")
            # Do not clear automatically on processing error; keep conservative state

    def cmd_callback(self, msg):
        """Process and potentially modify velocity commands"""
        safe_cmd = Twist()
        if self.emergency_stop:
            safe_cmd.linear.x = 0.0
            safe_cmd.angular.z = 0.0
        else:
            safe_cmd.linear.x = max(-self.max_speed, min(self.max_speed, msg.linear.x))
            safe_cmd.angular.z = max(-2.0, min(2.0, msg.angular.z))
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
