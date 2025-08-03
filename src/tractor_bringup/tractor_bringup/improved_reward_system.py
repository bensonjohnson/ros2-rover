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
        
        # Reward scaling factors
        self.reward_config = {
            # Movement rewards
            'base_movement_reward': 15.0,  # Increased from 10.0
            'speed_bonus_threshold': 0.05,  # Minimum speed for bonus
            'optimal_speed_range': (0.08, 0.25),  # Sweet spot for speed
            'speed_bonus_multiplier': 3.0,
            
            # Exploration rewards
            'new_area_bonus': 8.0,  # Reward for visiting new areas
            'exploration_streak_bonus': 2.0,  # Bonus for continuous exploration
            
            # Safety penalties
            'collision_penalty': -25.0,  # Reduced from -50.0
            'near_collision_penalty': -5.0,  # Penalty for getting too close
            'stationary_penalty': -2.0,  # Penalty for not moving
            
            # Control smoothness
            'smooth_control_bonus': 1.0,
            'jerky_control_penalty': -1.0,
            
            # Time-based rewards
            'continuous_movement_bonus': 1.5,
            'stagnation_penalty': -3.0
        }
    
    def calculate_comprehensive_reward(self, 
                                     action: np.ndarray,
                                     position: np.ndarray,
                                     collision: bool,
                                     near_collision: bool,
                                     progress: float,
                                     depth_data: Optional[np.ndarray] = None) -> Tuple[float, Dict[str, float]]:
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
        
        # 6. Curiosity/Obstacle Interaction (if depth data available)
        if depth_data is not None:
            curiosity_reward = self._calculate_curiosity_reward(depth_data, action)
            reward_breakdown['curiosity'] = curiosity_reward
            total_reward += curiosity_reward
        
        # Update internal state
        self._update_tracking(action, position)
        
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
    
    def _update_tracking(self, action: np.ndarray, position: np.ndarray):
        """Update internal tracking state"""
        self.action_history.append(action.copy())
        self.position_history.append(position.copy())
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
