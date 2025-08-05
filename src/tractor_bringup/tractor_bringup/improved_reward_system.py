#!/usr/bin/env python3
"""
Improved Reward System for NPU Exploration
Addresses issues with movement incentives and reward balance
"""

import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
import time

class ImprovedRewardCalculator:
    """
    Enhanced reward calculator that encourages exploration and movement
    while maintaining safety constraints
    """
    
    def __init__(self):
        # Movement tracking
        self.position_history = deque(maxlen=100)
        self.action_history = deque(maxlen=50)
        self.reward_history = deque(maxlen=1000)
        
        # Exploration tracking
        self.visited_areas = set()
        self.grid_size = 0.5  # 50cm grid for exploration tracking
        
        # Time tracking
        self.step_count = 0
        self.last_movement_time = time.time()
        
        # Differential drive tracking
        self.wheel_velocity_history = deque(maxlen=20)  # Track L/R wheel velocities
        self.track_efficiency_history = deque(maxlen=50)  # Track turning efficiency
        
        # Reward scaling factors
        self.reward_config = {
            # Movement rewards
            'base_movement_reward': 15.0,  # Increased from 10.0
            'speed_bonus_threshold': 0.05,  # Minimum speed for bonus
            'optimal_speed_range': (0.08, 0.25),  # Sweet spot for speed
            'speed_bonus_multiplier': 3.0,
            
            # Straight-line movement rewards
            'straight_line_bonus': 8.0,  # Strong reward for going straight
            'forward_progress_multiplier': 25.0,  # Heavy emphasis on forward movement
            'angular_penalty_threshold': 0.3,  # rad/s threshold for penalizing turning
            'excessive_turning_penalty': -4.0,  # Penalty for too much turning
            
            # Wall following and frontier exploration
            'wall_following_bonus': 5.0,  # Reward for maintaining distance from walls
            'frontier_approach_bonus': 10.0,  # Reward for heading toward unexplored areas
            'optimal_wall_distance': 0.8,  # Ideal distance from walls (meters)
            'wall_distance_tolerance': 0.3,  # Tolerance around optimal distance
            
            # Exploration rewards
            'new_area_bonus': 12.0,  # Increased reward for visiting new areas
            'exploration_streak_bonus': 3.0,  # Bonus for continuous exploration
            'exploration_efficiency_bonus': 6.0,  # Reward for efficient exploration patterns
            
            # Safety penalties
            'collision_penalty': -25.0,  # Reduced from -50.0
            'near_collision_penalty': -5.0,  # Penalty for getting too close
            'stationary_penalty': -2.0,  # Penalty for not moving
            
            # Control smoothness
            'smooth_control_bonus': 2.0,  # Increased to encourage smooth movement
            'jerky_control_penalty': -3.0,  # Increased penalty for jerky movement
            
            # Time-based rewards
            'continuous_movement_bonus': 1.5,
            'stagnation_penalty': -3.0,
            
            # Differential drive tracking
            'track_efficiency_bonus': 4.0,      # Reward for efficient differential drive
            'wheel_slip_penalty': -2.0,         # Penalty for wheel slippage
            'coordinated_turning_bonus': 3.0,   # Reward for proper tank steering
        }
    
    def calculate_comprehensive_reward(self, 
                                     action: np.ndarray,
                                     position: np.ndarray,
                                     collision: bool,
                                     near_collision: bool,
                                     progress: float,
                                     depth_data: Optional[np.ndarray] = None,
                                     wheel_velocities: Optional[Tuple[float, float]] = None) -> Tuple[float, Dict[str, float]]:
        """
        Calculate a comprehensive reward that encourages exploration and movement
        
        Returns:
            tuple: (total_reward, reward_breakdown)
        """
        
        reward_breakdown = {}
        total_reward = 0.0
        
        # 1. Movement Rewards
        movement_reward = self._calculate_movement_reward(action, progress)
        reward_breakdown['movement'] = movement_reward
        total_reward += movement_reward
        
        # 2. Exploration Rewards
        exploration_reward = self._calculate_exploration_reward(position)
        reward_breakdown['exploration'] = exploration_reward
        total_reward += exploration_reward
        
        # 3. Safety Penalties
        safety_penalty = self._calculate_safety_penalty(collision, near_collision)
        reward_breakdown['safety'] = safety_penalty
        total_reward += safety_penalty
        
        # 4. Control Smoothness
        smoothness_reward = self._calculate_smoothness_reward(action)
        reward_breakdown['smoothness'] = smoothness_reward
        total_reward += smoothness_reward
        
        # 5. Time-based rewards
        time_reward = self._calculate_time_based_reward(action)
        reward_breakdown['time_based'] = time_reward
        total_reward += time_reward
        
        # 6. Straight-line and directional movement rewards
        straight_line_reward = self._calculate_straight_line_reward(action, position)
        reward_breakdown['straight_line'] = straight_line_reward
        total_reward += straight_line_reward
        
        # 7. Wall following and frontier exploration (if depth data available)
        if depth_data is not None:
            wall_frontier_reward = self._calculate_wall_frontier_reward(depth_data, action, position)
            reward_breakdown['wall_frontier'] = wall_frontier_reward
            total_reward += wall_frontier_reward
        
        # 8. Curiosity/Obstacle Interaction (if depth data available)
        if depth_data is not None:
            curiosity_reward = self._calculate_curiosity_reward(depth_data, action)
            reward_breakdown['curiosity'] = curiosity_reward
            total_reward += curiosity_reward
        
        # 9. Differential Drive Efficiency (if wheel velocities available)
        if wheel_velocities is not None:
            differential_reward = self._calculate_differential_drive_reward(action, wheel_velocities)
            reward_breakdown['differential_drive'] = differential_reward
            total_reward += differential_reward
        
        # Update internal state
        self._update_tracking(action, position, wheel_velocities)
        
        # Store reward for analysis
        self.reward_history.append(total_reward)
        
        return total_reward, reward_breakdown
    
    def _calculate_movement_reward(self, action: np.ndarray, progress: float) -> float:
        """Calculate rewards for movement"""
        reward = 0.0
        
        linear_speed = abs(action[0])
        
        # Base movement reward (encourage any movement)
        if linear_speed > 0.01:  # Moving at all
            reward += self.reward_config['base_movement_reward'] * linear_speed
        
        # Speed bonus for optimal range
        if self.reward_config['optimal_speed_range'][0] <= linear_speed <= self.reward_config['optimal_speed_range'][1]:
            speed_bonus = self.reward_config['speed_bonus_multiplier'] * linear_speed
            reward += speed_bonus
        
        # Progress-based reward (scaled up)
        if progress > 0:
            reward += progress * 50.0  # Increased from 10.0
        
        # Penalty for being completely stationary
        if linear_speed < 0.01:
            reward += self.reward_config['stationary_penalty']
        
        return reward
    
    def _calculate_exploration_reward(self, position: np.ndarray) -> float:
        """Calculate rewards for exploring new areas"""
        reward = 0.0
        
        # Discretize position into grid
        grid_x = int(position[0] / self.grid_size)
        grid_y = int(position[1] / self.grid_size)
        grid_cell = (grid_x, grid_y)
        
        # Reward for visiting new areas
        if grid_cell not in self.visited_areas:
            self.visited_areas.add(grid_cell)
            reward += self.reward_config['new_area_bonus']
            
            # Streak bonus for continuous exploration
            if len(self.position_history) > 5:
                recent_positions = list(self.position_history)[-5:]
                if all(np.linalg.norm(pos - position) > 0.3 for pos in recent_positions):
                    reward += self.reward_config['exploration_streak_bonus']
        
        return reward
    
    def _calculate_safety_penalty(self, collision: bool, near_collision: bool) -> float:
        """Calculate safety-related penalties"""
        penalty = 0.0
        
        if collision:
            penalty += self.reward_config['collision_penalty']
        elif near_collision:
            penalty += self.reward_config['near_collision_penalty']
        
        return penalty
    
    def _calculate_smoothness_reward(self, action: np.ndarray) -> float:
        """Reward smooth control actions"""
        reward = 0.0
        
        if len(self.action_history) > 0:
            last_action = self.action_history[-1]
            action_change = np.linalg.norm(action - last_action)
            
            # Reward smooth transitions
            if action_change < 0.1:
                reward += self.reward_config['smooth_control_bonus']
            elif action_change > 0.5:  # Jerky movement
                reward += self.reward_config['jerky_control_penalty']
        
        return reward
    
    def _calculate_time_based_reward(self, action: np.ndarray) -> float:
        """Calculate time-based rewards and penalties"""
        reward = 0.0
        current_time = time.time()
        
        # Continuous movement bonus
        if abs(action[0]) > 0.02:  # Robot is moving
            self.last_movement_time = current_time
            reward += self.reward_config['continuous_movement_bonus']
        else:
            # Penalty for being stationary too long
            stationary_time = current_time - self.last_movement_time
            if stationary_time > 3.0:  # Stationary for more than 3 seconds
                reward += self.reward_config['stagnation_penalty'] * (stationary_time / 3.0)
        
        return reward
    
    def _calculate_curiosity_reward(self, depth_data: np.ndarray, action: np.ndarray) -> float:
        """Reward curiosity-driven behavior based on depth information"""
        reward = 0.0
        
        try:
            # Encourage approaching (but not colliding with) interesting objects
            valid_depths = depth_data[(depth_data > 0.1) & (depth_data < 5.0)]
            
            if len(valid_depths) > 0:
                min_distance = np.min(valid_depths)
                
                # Sweet spot: approach objects but maintain safe distance
                if 0.5 <= min_distance <= 1.5:  # 50cm to 1.5m
                    # Reward moving toward interesting objects
                    if action[0] > 0.02:  # Moving forward
                        reward += 2.0
                
                # Small reward for turning when close to obstacles (exploration)
                elif min_distance < 0.5 and abs(action[1]) > 0.1:  # Turning when close
                    reward += 1.0
        
        except Exception:
            pass  # Don't let depth processing errors affect rewards
        
        return reward
    
    def _calculate_straight_line_reward(self, action: np.ndarray, position: np.ndarray) -> float:
        """Reward straight-line movement and penalize excessive turning"""
        reward = 0.0
        
        linear_speed = abs(action[0])
        angular_speed = abs(action[1])
        
        # Strong reward for forward movement with minimal turning
        if linear_speed > 0.05:  # Moving forward
            if angular_speed < self.reward_config['angular_penalty_threshold']:
                # Reward straight-line movement
                straight_bonus = self.reward_config['straight_line_bonus'] * linear_speed
                reward += straight_bonus
                
                # Extra bonus for sustained forward movement
                if len(self.position_history) >= 3:
                    recent_positions = list(self.position_history)[-3:]
                    if self._is_moving_in_straight_line(recent_positions, position):
                        reward += self.reward_config['forward_progress_multiplier'] * linear_speed
            else:
                # Penalty for excessive turning while moving forward
                if angular_speed > 0.5:  # Spinning while moving
                    reward += self.reward_config['excessive_turning_penalty']
        
        # Penalize pure spinning (high angular, low linear)
        if angular_speed > 0.4 and linear_speed < 0.03:
            reward += self.reward_config['excessive_turning_penalty'] * 2.0  # Double penalty for pure spinning
        
        return reward
    
    def _calculate_wall_frontier_reward(self, depth_data: np.ndarray, action: np.ndarray, position: np.ndarray) -> float:
        """Reward wall-following behavior and frontier exploration"""
        reward = 0.0
        
        try:
            if depth_data.size == 0:
                return 0.0
                
            height, width = depth_data.shape
            linear_speed = abs(action[0])
            
            # Analyze depth data for wall following
            # Left side depths (for right wall following)
            left_roi = depth_data[:, :width//3]
            # Right side depths (for left wall following)  
            right_roi = depth_data[:, 2*width//3:]
            # Front center (for obstacle detection)
            center_roi = depth_data[height//3:2*height//3, width//3:2*width//3]
            
            # Calculate distances to walls
            left_valid = left_roi[(left_roi > 0.1) & (left_roi < 3.0)]
            right_valid = right_roi[(right_roi > 0.1) & (right_roi < 3.0)]
            center_valid = center_roi[(center_roi > 0.1) & (center_roi < 3.0)]
            
            # Wall following rewards
            if len(left_valid) > 0 or len(right_valid) > 0:
                # Find closest wall
                left_dist = np.min(left_valid) if len(left_valid) > 0 else float('inf')
                right_dist = np.min(right_valid) if len(right_valid) > 0 else float('inf')
                wall_dist = min(left_dist, right_dist)
                
                # Reward maintaining optimal distance from walls while moving forward
                if linear_speed > 0.05:  # Moving forward
                    optimal_dist = self.reward_config['optimal_wall_distance']
                    tolerance = self.reward_config['wall_distance_tolerance']
                    
                    if abs(wall_dist - optimal_dist) < tolerance:
                        # Perfect wall following distance
                        reward += self.reward_config['wall_following_bonus'] * linear_speed
                    elif wall_dist > optimal_dist + tolerance:
                        # Too far from wall, small bonus for moving toward unexplored areas
                        reward += self.reward_config['frontier_approach_bonus'] * 0.5 * linear_speed
            
            # Frontier exploration reward
            if len(center_valid) > 0:
                front_distance = np.min(center_valid)
                
                # Reward approaching frontiers (open spaces) while maintaining forward movement
                if front_distance > 2.0 and linear_speed > 0.08:  # Open space ahead
                    reward += self.reward_config['frontier_approach_bonus'] * linear_speed
                    
                    # Extra bonus for sustained forward movement into open areas
                    if len(self.action_history) >= 3:
                        recent_actions = list(self.action_history)[-3:]
                        if all(abs(a[0]) > 0.05 and abs(a[1]) < 0.2 for a in recent_actions):
                            reward += self.reward_config['exploration_efficiency_bonus']
            
        except Exception as e:
            # Don't let depth processing errors crash the reward calculation
            pass
        
        return reward
    
    def _is_moving_in_straight_line(self, recent_positions: List[np.ndarray], current_position: np.ndarray) -> bool:
        """Check if robot is moving in a relatively straight line"""
        if len(recent_positions) < 2:
            return False
            
        try:
            # Calculate movement vectors
            positions = recent_positions + [current_position]
            vectors = []
            for i in range(1, len(positions)):
                vec = positions[i] - positions[i-1]
                if np.linalg.norm(vec) > 0.02:  # Ignore very small movements
                    vectors.append(vec / np.linalg.norm(vec))  # Normalize
            
            if len(vectors) < 2:
                return False
            
            # Check if vectors are roughly aligned (dot product close to 1)
            alignments = []
            for i in range(1, len(vectors)):
                dot_product = np.dot(vectors[i-1], vectors[i])
                alignments.append(dot_product)
            
            # Consider it straight line if most movements are aligned
            return np.mean(alignments) > 0.7  # Cosine of ~45 degrees
            
        except Exception:
            return False
    
    def _calculate_differential_drive_reward(self, action: np.ndarray, wheel_velocities: Tuple[float, float]) -> float:
        """Reward efficient differential drive control based on actual wheel encoder feedback"""
        reward = 0.0
        left_vel, right_vel = wheel_velocities
        
        # Store wheel velocities for tracking
        self.wheel_velocity_history.append((left_vel, right_vel))
        
        commanded_linear = action[0]   # Desired linear velocity
        commanded_angular = action[1]  # Desired angular velocity
        
        # Calculate expected wheel velocities from commanded actions
        # Tank steering: left = linear - angular*width/2, right = linear + angular*width/2
        wheel_separation = 0.5  # From motor driver parameters
        expected_left = commanded_linear - (commanded_angular * wheel_separation / 2.0)
        expected_right = commanded_linear + (commanded_angular * wheel_separation / 2.0)
        
        # 1. Reward accurate differential drive control
        left_error = abs(left_vel - expected_left)
        right_error = abs(right_vel - expected_right)
        total_error = left_error + right_error
        
        # Strong reward for low tracking error (good control)
        if total_error < 0.1:  # Very accurate
            reward += self.reward_config['track_efficiency_bonus'] * 2.0
        elif total_error < 0.3:  # Reasonably accurate
            reward += self.reward_config['track_efficiency_bonus']
        elif total_error > 0.8:  # Poor tracking
            reward += self.reward_config['wheel_slip_penalty']
        
        # 2. Reward coordinated movement for straight lines
        if abs(commanded_angular) < 0.1:  # Trying to go straight
            wheel_velocity_difference = abs(left_vel - right_vel)
            if wheel_velocity_difference < 0.2:  # Both wheels moving similarly
                reward += self.reward_config['coordinated_turning_bonus'] * 2.0  # Double bonus for straight
            elif wheel_velocity_difference > 0.5:  # Wheels fighting each other
                reward += self.reward_config['wheel_slip_penalty']
        
        # 3. Penalty for opposite wheel directions when trying to go straight
        if abs(commanded_angular) < 0.1 and commanded_linear > 0.05:  # Should go straight forward
            if (left_vel > 0 and right_vel < 0) or (left_vel < 0 and right_vel > 0):
                # Wheels spinning in opposite directions = spinning in place
                reward += self.reward_config['excessive_turning_penalty'] * 3.0  # Triple penalty
        
        # 4. Reward both wheels moving forward together
        if commanded_linear > 0.05:  # Commanded forward movement
            if left_vel > 0.05 and right_vel > 0.05:  # Both wheels actually moving forward
                average_forward_speed = (left_vel + right_vel) / 2.0
                reward += self.reward_config['coordinated_turning_bonus'] * average_forward_speed
        
        # 5. Penalty for wheel slip/stall detection
        if abs(commanded_linear) > 0.05:  # Commanding movement
            if abs(left_vel) < 0.02 and abs(right_vel) < 0.02:  # But wheels not moving
                reward += self.reward_config['wheel_slip_penalty'] * 2.0  # Stall penalty
        
        return reward
    
    def _update_tracking(self, action: np.ndarray, position: np.ndarray, wheel_velocities: Optional[Tuple[float, float]] = None):
        """Update internal tracking state"""
        self.action_history.append(action.copy())
        self.position_history.append(position.copy())
        if wheel_velocities is not None:
            self.wheel_velocity_history.append(wheel_velocities)
        self.step_count += 1
    
    def get_reward_statistics(self) -> Dict[str, float]:
        """Get statistics about recent rewards"""
        if not self.reward_history:
            return {}
        
        rewards = list(self.reward_history)
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'recent_mean': np.mean(rewards[-50:]) if len(rewards) >= 50 else np.mean(rewards),
            'explored_areas': len(self.visited_areas),
            'total_steps': self.step_count
        }
    
    def reset_exploration_tracking(self):
        """Reset exploration tracking (useful for new sessions)"""
        self.visited_areas.clear()
        self.position_history.clear()
        self.action_history.clear()
        self.step_count = 0
        self.last_movement_time = time.time()
