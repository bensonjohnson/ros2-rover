#!/usr/bin/env python3
"""
ROS2 Training Monitor Node
Provides real-time monitoring of training progress and overtraining detection
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String, Bool, Float64
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import json
import time
import numpy as np
from collections import deque

try:
    from .training_monitor import TrainingMonitor
    from .improved_reward_system import ImprovedRewardCalculator
    from .anti_overtraining_config import get_safe_config
except ImportError:
    # Handle direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from training_monitor import TrainingMonitor
    from improved_reward_system import ImprovedRewardCalculator
    from anti_overtraining_config import get_safe_config

class TrainingMonitorNode(Node):
    """ROS2 node for monitoring training progress and detecting overtraining"""
    
    def __init__(self):
        super().__init__('training_monitor')
        
        # Parameters
        self.declare_parameter('monitor_frequency', 1.0)
        self.declare_parameter('diversity_window', 20)
        self.declare_parameter('alert_threshold', 0.7)
        self.declare_parameter('enable_plotting', False)
        
        self.monitor_frequency = self.get_parameter('monitor_frequency').value
        self.diversity_window = self.get_parameter('diversity_window').value
        self.alert_threshold = self.get_parameter('alert_threshold').value
        self.enable_plotting = self.get_parameter('enable_plotting').value
        
        # Initialize monitoring system
        self.training_monitor = TrainingMonitor(window_size=1000)
        
        # Initialize reward calculator if available
        try:
            config = get_safe_config()
            self.reward_calculator = ImprovedRewardCalculator(**config)
            self.get_logger().info("✓ Anti-overtraining reward system initialized")
        except Exception as e:
            self.get_logger().warning(f"Could not initialize reward system: {e}")
            self.reward_calculator = None
        
        # State tracking
        self.episode_count = 0
        self.step_count = 0
        self.current_episode_reward = 0.0
        self.last_monitor_time = time.time()
        
        # Action and position tracking for reward calculation
        self.last_action = np.array([0.0, 0.0])
        self.last_position = np.array([0.0, 0.0])
        self.position_history = deque(maxlen=10)
        
        # Publishers
        self.training_health_pub = self.create_publisher(String, 'training_health', 10)
        self.overtraining_alert_pub = self.create_publisher(String, 'overtraining_alerts', 10)
        self.diversity_score_pub = self.create_publisher(Float64, 'behavioral_diversity', 10)
        self.early_stop_pub = self.create_publisher(Bool, 'early_stop_recommendation', 10)
        
        # Subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist, 'cmd_vel_raw', self.cmd_vel_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)
        
        # Optional: Subscribe to custom training topics if available
        self.reward_sub = self.create_subscription(
            Float64, 'current_reward', self.reward_callback, 10)
        self.episode_end_sub = self.create_subscription(
            String, 'episode_status', self.episode_callback, 10)
        
        # Timer for regular monitoring
        self.monitor_timer = self.create_timer(
            1.0 / self.monitor_frequency, self.monitor_callback)
        
        self.get_logger().info("🛡️ Training Monitor Node started")
        self.get_logger().info(f"Monitor frequency: {self.monitor_frequency} Hz")
        self.get_logger().info(f"Alert threshold: {self.alert_threshold}")
    
    def cmd_vel_callback(self, msg):
        """Track robot actions for behavioral diversity analysis"""
        self.last_action = np.array([msg.linear.x, msg.angular.z])
        self.step_count += 1
        
        # Estimate position change (simple integration)
        dt = 0.1  # Assume 10Hz
        dx = msg.linear.x * dt
        dy = 0.0  # Simplified 2D movement
        
        if len(self.position_history) > 0:
            last_pos = self.position_history[-1]
            new_pos = np.array([last_pos[0] + dx, last_pos[1] + dy])
        else:
            new_pos = np.array([dx, dy])
        
        self.position_history.append(new_pos)
        self.last_position = new_pos
    
    def depth_callback(self, msg):
        """Process depth images for reward calculation if needed"""
        # For now, just log that we're receiving depth data
        pass
    
    def reward_callback(self, msg):
        """Track rewards if published by training system"""
        self.current_episode_reward += msg.data
    
    def episode_callback(self, msg):
        """Handle episode end notifications"""
        try:
            episode_data = json.loads(msg.data)
            if episode_data.get('status') == 'completed':
                self.episode_count += 1
                episode_reward = episode_data.get('total_reward', self.current_episode_reward)
                
                if self.reward_calculator:
                    # Update monitoring with episode data
                    report = self.training_monitor.update(
                        self.reward_calculator, episode_reward, self.step_count)
                    
                    # Publish training health
                    self.publish_training_health(report)
                
                # Reset episode tracking
                self.current_episode_reward = 0.0
                
        except Exception as e:
            self.get_logger().warning(f"Error processing episode data: {e}")
    
    def monitor_callback(self):
        """Regular monitoring callback"""
        current_time = time.time()
        
        # Only do full monitoring if we have reward calculator and sufficient data
        if not self.reward_calculator:
            return
        
        # Calculate a synthetic reward based on current action
        if len(self.position_history) >= 2:
            # Simple reward calculation for monitoring
            action = self.last_action
            position = self.last_position
            collision = False  # Would need actual collision detection
            near_collision = False  # Would need actual sensor data
            
            # Calculate progress
            if len(self.position_history) >= 2:
                progress = np.linalg.norm(
                    self.position_history[-1] - self.position_history[-2])
            else:
                progress = 0.0
            
            # Get reward and update monitoring
            try:
                reward, breakdown = self.reward_calculator.calculate_comprehensive_reward(
                    action=action,
                    position=position,
                    collision=collision,
                    near_collision=near_collision,
                    progress=progress
                )
                
                # Update monitor every few seconds
                if current_time - self.last_monitor_time > 5.0:
                    report = self.training_monitor.update(
                        self.reward_calculator, reward, self.step_count)
                    
                    # Publish monitoring data
                    self.publish_training_health(report)
                    self.publish_diversity_score()
                    self.check_early_stopping()
                    
                    self.last_monitor_time = current_time
                    
            except Exception as e:
                self.get_logger().warning(f"Error in reward calculation: {e}")
    
    def publish_training_health(self, report):
        """Publish training health report"""
        try:
            health_msg = String()
            health_data = {
                'timestamp': time.time(),
                'step': self.step_count,
                'episode': self.episode_count,
                'health_indicators': report.get('health_indicators', {}),
                'current_stats': report.get('current_stats', {}),
                'recommendations': report.get('recommendations', [])
            }
            health_msg.data = json.dumps(health_data, indent=2)
            self.training_health_pub.publish(health_msg)
            
            # Log health status
            health_status = report.get('health_indicators', {}).get('overall_health', 'UNKNOWN')
            if health_status == 'POOR':
                self.get_logger().warning(f"⚠️ Training health: {health_status}")
            else:
                self.get_logger().info(f"✓ Training health: {health_status}")
            
            # Publish alerts if any
            alerts = report.get('recent_alerts', [])
            for alert in alerts:
                alert_msg = String()
                alert_msg.data = json.dumps(alert)
                self.overtraining_alert_pub.publish(alert_msg)
                
                if alert['severity'] == 'CRITICAL':
                    self.get_logger().error(f"🚨 CRITICAL: {alert['message']}")
                elif alert['severity'] == 'WARNING':
                    self.get_logger().warning(f"⚠️ WARNING: {alert['message']}")
                    
        except Exception as e:
            self.get_logger().error(f"Error publishing training health: {e}")
    
    def publish_diversity_score(self):
        """Publish behavioral diversity score"""
        try:
            if self.reward_calculator:
                stats = self.reward_calculator.get_reward_statistics()
                diversity = stats.get('behavioral_diversity', 0.0)
                
                diversity_msg = Float64()
                diversity_msg.data = float(diversity)
                self.diversity_score_pub.publish(diversity_msg)
                
        except Exception as e:
            self.get_logger().warning(f"Error publishing diversity score: {e}")
    
    def check_early_stopping(self):
        """Check if early stopping is recommended"""
        try:
            should_stop = self.training_monitor.get_early_stopping_recommendation()
            
            early_stop_msg = Bool()
            early_stop_msg.data = should_stop
            self.early_stop_pub.publish(early_stop_msg)
            
            if should_stop:
                self.get_logger().warning("🛑 Early stopping recommended due to overtraining risk!")
                
        except Exception as e:
            self.get_logger().warning(f"Error checking early stopping: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = TrainingMonitorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in training monitor node: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
