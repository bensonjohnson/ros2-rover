#!/usr/bin/env python3
"""
Simple Command Multiplexer

This node multiplexes multiple cmd_vel sources with priority handling:
1. Teleop (highest priority)
2. VLM control 
3. Autonomous navigation (lowest priority)

It ensures that higher priority commands override lower priority ones,
and handles timeouts for safety.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time
from typing import Dict, Optional


class SimpleCmdMux(Node):
    def __init__(self):
        super().__init__('simple_cmd_mux')
        
        # Parameters
        self.declare_parameter('timeout_teleop', 1.0)  # seconds
        self.declare_parameter('timeout_vlm', 5.0)     # seconds
        self.declare_parameter('timeout_autonomous', 2.0)  # seconds
        self.declare_parameter('priority_order', ['teleop', 'vlm', 'autonomous'])
        
        self.timeout_teleop = self.get_parameter('timeout_teleop').value
        self.timeout_vlm = self.get_parameter('timeout_vlm').value
        self.timeout_autonomous = self.get_parameter('timeout_autonomous').value
        self.priority_order = self.get_parameter('priority_order').value
        
        # State tracking
        self.last_commands: Dict[str, Twist] = {
            'teleop': Twist(),
            'vlm': Twist(),
            'autonomous': Twist()
        }
        self.last_times: Dict[str, float] = {
            'teleop': 0.0,
            'vlm': 0.0,
            'autonomous': 0.0
        }
        self.timeouts: Dict[str, float] = {
            'teleop': self.timeout_teleop,
            'vlm': self.timeout_vlm,
            'autonomous': self.timeout_autonomous
        }
        
        # Subscribers for each input source
        self.teleop_sub = self.create_subscription(
            Twist, 'cmd_vel_teleop', self.teleop_callback, 10
        )
        self.vlm_sub = self.create_subscription(
            Twist, 'cmd_vel_vlm', self.vlm_callback, 10
        )
        self.autonomous_sub = self.create_subscription(
            Twist, 'cmd_vel_autonomous', self.autonomous_callback, 10
        )
        
        # Output publisher
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel_out', 10)
        
        # Timer for periodic output generation
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz
        
        # Track current active source for logging
        self.current_source: Optional[str] = None
        
        self.get_logger().info("Command multiplexer initialized")
        self.get_logger().info(f"Priority order: {self.priority_order}")
        self.get_logger().info(f"Timeouts: teleop={self.timeout_teleop}s, vlm={self.timeout_vlm}s, autonomous={self.timeout_autonomous}s")
    
    def teleop_callback(self, msg: Twist):
        """Handle teleop commands (highest priority)"""
        self.last_commands['teleop'] = msg
        self.last_times['teleop'] = time.time()
        
        # Log if this is a non-zero command
        if self.is_non_zero_command(msg):
            if self.current_source != 'teleop':
                self.get_logger().info("Switching to TELEOP control")
                self.current_source = 'teleop'
    
    def vlm_callback(self, msg: Twist):
        """Handle VLM commands (medium priority)"""
        self.last_commands['vlm'] = msg
        self.last_times['vlm'] = time.time()
        
        # Log if this is a non-zero command and we're not in teleop
        if self.is_non_zero_command(msg) and not self.is_source_active('teleop'):
            if self.current_source != 'vlm':
                self.get_logger().info("Switching to VLM control")
                self.current_source = 'vlm'
    
    def autonomous_callback(self, msg: Twist):
        """Handle autonomous commands (lowest priority)"""
        self.last_commands['autonomous'] = msg
        self.last_times['autonomous'] = time.time()
        
        # Log if this is a non-zero command and no higher priority sources are active
        if (self.is_non_zero_command(msg) and 
            not self.is_source_active('teleop') and 
            not self.is_source_active('vlm')):
            if self.current_source != 'autonomous':
                self.get_logger().info("Switching to AUTONOMOUS control")
                self.current_source = 'autonomous'
    
    def is_non_zero_command(self, cmd: Twist) -> bool:
        """Check if command has non-zero velocities"""
        return (abs(cmd.linear.x) > 0.001 or 
                abs(cmd.linear.y) > 0.001 or 
                abs(cmd.angular.z) > 0.001)
    
    def is_source_active(self, source: str) -> bool:
        """Check if a command source is currently active (within timeout)"""
        current_time = time.time()
        time_since_last = current_time - self.last_times[source]
        return time_since_last <= self.timeouts[source]
    
    def timer_callback(self):
        """Select and publish the highest priority active command"""
        current_time = time.time()
        output_cmd = Twist()  # Default to stop
        active_source = None
        
        # Check each source in priority order
        for source in self.priority_order:
            if self.is_source_active(source):
                output_cmd = self.last_commands[source]
                active_source = source
                break
        
        # Update current source tracking
        if active_source != self.current_source:
            if active_source is None:
                self.get_logger().info("No active command sources - STOPPING")
            self.current_source = active_source
        
        # Publish the selected command
        self.cmd_vel_pub.publish(output_cmd)
        
        # Debug logging (reduced frequency)
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
        
        # Log status every 2 seconds (20 calls at 10Hz)
        if self._debug_counter % 20 == 0:
            active_sources = [source for source in self.priority_order if self.is_source_active(source)]
            if active_sources:
                self.get_logger().debug(f"Active sources: {active_sources}, using: {active_source}")
                if self.is_non_zero_command(output_cmd):
                    self.get_logger().debug(f"Output: linear={output_cmd.linear.x:.2f}, angular={output_cmd.angular.z:.2f}")
    
    def destroy_node(self):
        """Clean shutdown"""
        # Send stop command before shutting down
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    cmd_mux = SimpleCmdMux()
    
    try:
        rclpy.spin(cmd_mux)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cmd_mux.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()