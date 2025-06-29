#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import pygame
import math
import time


class XboxControllerTeleop(Node):
    def __init__(self):
        super().__init__('xbox_controller_teleop')
        
        # Parameters
        self.declare_parameter('max_linear_speed', 1.0)  # m/s
        self.declare_parameter('max_angular_speed', 2.0)  # rad/s
        self.declare_parameter('deadzone', 0.15)  # Joystick deadzone
        self.declare_parameter('tank_drive_mode', True)  # Tank drive vs arcade drive
        self.declare_parameter('controller_index', 0)  # Which controller to use
        
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value
        self.deadzone = self.get_parameter('deadzone').value
        self.tank_drive_mode = self.get_parameter('tank_drive_mode').value
        self.controller_index = self.get_parameter('controller_index').value
        
        # Initialize pygame for joystick support
        try:
            pygame.init()
            pygame.joystick.init()
            
            if pygame.joystick.get_count() == 0:
                self.get_logger().error("No joystick/controller detected")
                self.controller = None
            else:
                self.controller = pygame.joystick.Joystick(self.controller_index)
                self.controller.init()
                self.get_logger().info(f"Controller initialized: {self.controller.get_name()}")
                
        except Exception as e:
            self.get_logger().error(f"Failed to initialize controller: {e}")
            self.controller = None
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.emergency_stop_pub = self.create_publisher(Bool, 'emergency_stop', 10)
        
        # Control state
        self.emergency_stop_active = False
        self.last_button_y_state = False
        
        # Timer for controller polling
        self.timer = self.create_timer(0.05, self.controller_callback)  # 20 Hz
        
        self.get_logger().info(f'Xbox Controller Teleop initialized (Tank Drive: {self.tank_drive_mode})')
        self.print_controls()
    
    def print_controls(self):
        """Print control instructions"""
        self.get_logger().info("=== Xbox Controller Controls ===")
        if self.tank_drive_mode:
            self.get_logger().info("Tank Drive Mode:")
            self.get_logger().info("- Left stick Y: Left motor")
            self.get_logger().info("- Right stick Y: Right motor")
        else:
            self.get_logger().info("Arcade Drive Mode:")
            self.get_logger().info("- Left stick Y: Forward/Backward")
            self.get_logger().info("- Left stick X: Left/Right turn")
        self.get_logger().info("- Y button: Emergency stop toggle")
        self.get_logger().info("- A button: Resume from emergency stop")
        self.get_logger().info("================================")
    
    def apply_deadzone(self, value, deadzone):
        """Apply deadzone to joystick input"""
        if abs(value) < deadzone:
            return 0.0
        # Scale the remaining range to full range
        sign = 1 if value > 0 else -1
        return sign * (abs(value) - deadzone) / (1.0 - deadzone)
    
    def controller_callback(self):
        """Main controller polling callback"""
        if self.controller is None:
            return
        
        try:
            # Update pygame events
            pygame.event.pump()
            
            # Check if controller is still connected
            if not self.controller.get_init():
                self.get_logger().error("Controller disconnected")
                self.publish_stop_command()
                return
            
            # Read controller inputs
            left_stick_x = self.controller.get_axis(0)  # Left stick horizontal
            left_stick_y = -self.controller.get_axis(1)  # Left stick vertical (inverted)
            right_stick_x = self.controller.get_axis(2)  # Right stick horizontal  
            right_stick_y = -self.controller.get_axis(3)  # Right stick vertical (inverted)
            
            # Button states
            button_a = self.controller.get_button(0)  # A button
            button_y = self.controller.get_button(3)  # Y button
            
            # Handle emergency stop toggle
            if button_y and not self.last_button_y_state:
                self.emergency_stop_active = not self.emergency_stop_active
                stop_msg = Bool()
                stop_msg.data = self.emergency_stop_active
                self.emergency_stop_pub.publish(stop_msg)
                
                if self.emergency_stop_active:
                    self.get_logger().warn("EMERGENCY STOP ACTIVATED")
                else:
                    self.get_logger().info("Emergency stop deactivated")
            
            # Handle resume with A button
            if button_a and self.emergency_stop_active:
                self.emergency_stop_active = False
                stop_msg = Bool()
                stop_msg.data = False
                self.emergency_stop_pub.publish(stop_msg)
                self.get_logger().info("Emergency stop deactivated (A button)")
            
            self.last_button_y_state = button_y
            
            # If emergency stop is active, send stop command
            if self.emergency_stop_active:
                self.publish_stop_command()
                return
            
            # Apply deadzone to joystick inputs
            left_stick_x = self.apply_deadzone(left_stick_x, self.deadzone)
            left_stick_y = self.apply_deadzone(left_stick_y, self.deadzone)
            right_stick_x = self.apply_deadzone(right_stick_x, self.deadzone)
            right_stick_y = self.apply_deadzone(right_stick_y, self.deadzone)
            
            # Generate motion commands based on drive mode
            if self.tank_drive_mode:
                self.tank_drive_control(left_stick_y, right_stick_y)
            else:
                self.arcade_drive_control(left_stick_x, left_stick_y)
                
        except Exception as e:
            self.get_logger().error(f'Controller callback error: {e}')
            self.publish_stop_command()
    
    def tank_drive_control(self, left_y, right_y):
        """Tank drive control using both sticks"""
        # In tank drive, we need to convert stick inputs to twist
        # Left stick controls left wheel, right stick controls right wheel
        
        # Calculate equivalent linear and angular velocities
        # This is an approximation for tank drive using Twist messages
        left_wheel_speed = left_y * self.max_linear_speed
        right_wheel_speed = right_y * self.max_linear_speed
        
        # Convert to twist (approximation)
        linear_vel = (left_wheel_speed + right_wheel_speed) / 2.0
        angular_vel = (right_wheel_speed - left_wheel_speed) / 2.0  # Simplified
        
        # Limit angular velocity
        angular_vel = max(-self.max_angular_speed, min(self.max_angular_speed, angular_vel))
        
        self.publish_twist_command(linear_vel, angular_vel)
        
        # Debug output for significant movement
        if abs(left_y) > 0.1 or abs(right_y) > 0.1:
            self.get_logger().debug(f'Tank drive - Left: {left_y:.2f}, Right: {right_y:.2f} -> Lin: {linear_vel:.2f}, Ang: {angular_vel:.2f}')
    
    def arcade_drive_control(self, stick_x, stick_y):
        """Arcade drive control using single stick"""
        # Forward/backward from Y axis
        linear_vel = stick_y * self.max_linear_speed
        
        # Left/right from X axis
        angular_vel = stick_x * self.max_angular_speed
        
        self.publish_twist_command(linear_vel, angular_vel)
        
        # Debug output for significant movement
        if abs(stick_x) > 0.1 or abs(stick_y) > 0.1:
            self.get_logger().debug(f'Arcade drive - X: {stick_x:.2f}, Y: {stick_y:.2f} -> Lin: {linear_vel:.2f}, Ang: {angular_vel:.2f}')
    
    def publish_twist_command(self, linear_vel, angular_vel):
        """Publish twist command"""
        twist_msg = Twist()
        twist_msg.linear.x = linear_vel
        twist_msg.linear.y = 0.0
        twist_msg.linear.z = 0.0
        twist_msg.angular.x = 0.0
        twist_msg.angular.y = 0.0
        twist_msg.angular.z = angular_vel
        
        self.cmd_vel_pub.publish(twist_msg)
    
    def publish_stop_command(self):
        """Publish stop command"""
        self.publish_twist_command(0.0, 0.0)
    
    def destroy_node(self):
        """Clean shutdown"""
        try:
            # Send stop command
            self.publish_stop_command()
            time.sleep(0.1)
            
            # Cleanup pygame
            if self.controller:
                self.controller.quit()
            pygame.quit()
            
            self.get_logger().info("Xbox controller teleop shutdown complete")
        except Exception as e:
            self.get_logger().error(f'Error during cleanup: {e}')
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    xbox_teleop = XboxControllerTeleop()
    
    try:
        rclpy.spin(xbox_teleop)
    except KeyboardInterrupt:
        pass
    finally:
        xbox_teleop.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()