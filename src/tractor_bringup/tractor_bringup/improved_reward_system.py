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
    
    def __init__(self, **kwargs):
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
        
        # Enhanced reward scaling factors to encourage forward movement and penalize spinning
        self.reward_config = {
            # Core movement rewards (enhanced to encourage forward movement)
            'base_movement_reward': 8.0,  # Increased to encourage any movement
            'forward_progress_bonus': 12.0,  # Increased to reward forward progress
            'exploration_bonus': 10.0,  # Combined exploration reward
            
            # Anti-spinning rewards and penalties
            'straight_line_bonus': 10.0,  # Reward for moving straight
            'spinning_penalty': -8.0,  # Penalty for excessive spinning
            'forward_bias_bonus': 6.0,  # Bonus for forward movement over turning
            
            # Safety (kept simple and clear)
            'collision_penalty': -20.0,
            'near_collision_penalty': -5.0,
            'unsafe_behavior_penalty': -3.0,
            
            # Behavioral shaping (enhanced to prevent spinning)
            'smooth_movement_bonus': 1.0,
            'goal_oriented_bonus': 8.0,  # Increased reward for consistent direction
            'stagnation_penalty': -3.0,  # Increased penalty for not moving
        }
        
        # Reward noise and clipping (anti-overtraining measures)
        self.reward_noise_std = kwargs.get('reward_noise_std', 0.1)  # Add noise to prevent overfitting
        self.reward_clip_range = kwargs.get('reward_clip_range', (-30.0, 30.0))  # Clip extreme rewards
        self.reward_smoothing_alpha = kwargs.get('reward_smoothing_alpha', 0.1)  # Smooth reward changes
        
        # Anti-gaming mechanisms
        self.max_area_revisit_bonus = kwargs.get('max_area_revisit_bonus', 3)  # Limit area bonus exploitation
        self.spinning_detection_threshold = kwargs.get('spinning_threshold', 0.5)  # Detect and penalize spinning
        self.behavior_diversity_window = 20  # Track behavior diversity
        
        # Additional parameters (simplified)
        self.enable_reward_smoothing = kwargs.get('enable_reward_smoothing', True)
        self.enable_anti_gaming = kwargs.get('enable_anti_gaming', True)
        self.enable_diversity_tracking = kwargs.get('enable_diversity_tracking', True)
        
        # Tracking for anti-overtraining
        self.behavior_patterns = deque(maxlen=self.behavior_diversity_window)
        self.area_visit_count = {}  # Track how many times areas are visited
        self.smoothed_reward = 0.0  # Running average of rewards
    
    def calculate_comprehensive_reward(self, 
                                     action: np.ndarray,
                                     position: np.ndarray,
                                     collision: bool,
                                     near_collision: bool,
                                     progress: float,
                                     depth_data: Optional[np.ndarray] = None,
                                     wheel_velocities: Optional[Tuple[float, float]] = None) -> Tuple[float, Dict[str, float]]:
        """
        Calculate a simplified, robust reward that resists overtraining
        
        Returns:
            tuple: (total_reward, reward_breakdown)
        """
        
        reward_breakdown = {}
        total_reward = 0.0
        
        # 1. Core Movement Reward (simplified)
        movement_reward = self._calculate_core_movement_reward(action, progress)
        reward_breakdown['movement'] = movement_reward
        total_reward += movement_reward
        
        # 2. Exploration Reward (simplified and anti-gaming)
        exploration_reward = self._calculate_robust_exploration_reward(position)
        reward_breakdown['exploration'] = exploration_reward
        total_reward += exploration_reward
        
        # 3. Safety Penalties (clear and simple)
        safety_penalty = self._calculate_safety_penalty(collision, near_collision)
        reward_breakdown['safety'] = safety_penalty
        total_reward += safety_penalty
        
        # 4. Anti-Gaming Penalties
        if self.enable_anti_gaming:
            anti_gaming_penalty = self._calculate_anti_gaming_penalty(action, position)
            reward_breakdown['anti_gaming'] = anti_gaming_penalty
            total_reward += anti_gaming_penalty
        
        # 5. Behavior Diversity Bonus (encourage varied behaviors)
        if self.enable_diversity_tracking:
            diversity_bonus = self._calculate_diversity_bonus(action)
            reward_breakdown['diversity'] = diversity_bonus
            total_reward += diversity_bonus
        
        # Apply anti-overtraining measures
        total_reward = self._apply_anti_overtraining_measures(total_reward)
        
        # Update internal state
        self._update_tracking(action, position, wheel_velocities)
        
        # Store reward for analysis
        self.reward_history.append(total_reward)
        
        return total_reward, reward_breakdown
    
    def _calculate_core_movement_reward(self, action: np.ndarray, progress: float) -> float:
        """Enhanced movement reward that specifically encourages forward movement and penalizes spinning"""
        reward = 0.0
        
        linear_speed = action[0]  # Forward/backward speed (positive = forward)
        angular_speed = abs(action[1])  # Turning speed (absolute value)
        
        # Basic movement reward (encourage any forward movement)
        if linear_speed > 0.02:
            reward += self.reward_config['base_movement_reward'] * min(linear_speed, 0.3)  # Cap to prevent gaming
        
        # Forward progress bonus (based on actual displacement)
        if progress > 0:
            reward += self.reward_config['forward_progress_bonus'] * min(progress, 0.5)  # Cap progress reward
        
        # Forward bias bonus (reward forward movement over backward movement)
        if linear_speed > 0:
            reward += self.reward_config['forward_bias_bonus'] * linear_speed
        
        # Straight line bonus (reward moving straight over turning)
        if linear_speed > 0.05 and angular_speed < 0.2:  # Moving forward with minimal turning
            reward += self.reward_config['straight_line_bonus'] * linear_speed * (1.0 - angular_speed/0.2)
        
        # Smooth turning bonus (reward small, gradual turns over sharp turns)
        if linear_speed > 0.02:  # Only when moving forward
            if angular_speed < 0.1:
                # Very smooth turning - bonus
                reward += 3.0
            elif angular_speed < 0.3:
                # Moderate turning - small bonus
                reward += 1.0
            else:
                # Sharp turning - penalty
                reward -= 2.0
        
        # Proactive avoidance bonus (reward small corrective turns before getting too close)
        if linear_speed > 0.05 and 0.1 < angular_speed < 0.4:
            reward += 2.0  # Encourage gentle course corrections
        
        # Penalize excessive spinning without forward movement
        if angular_speed > self.spinning_detection_threshold and linear_speed < 0.02:
            reward += self.reward_config['spinning_penalty']
        
        # Stagnation penalty (penalize not moving)
        if abs(linear_speed) < 0.02 and angular_speed < 0.1:
            reward += self.reward_config['stagnation_penalty']
        
        # Goal-oriented bonus (reward consistent directional movement)
        if linear_speed > 0.05 and angular_speed < 0.3:  # Moving forward with minimal turning
            reward += self.reward_config['goal_oriented_bonus'] * linear_speed
        
        return reward
    
    def _calculate_robust_exploration_reward(self, position: np.ndarray) -> float:
        """Anti-gaming exploration reward with diminishing returns"""
        reward = 0.0
        
        # Discretize position into grid
        grid_x = int(position[0] / self.grid_size)
        grid_y = int(position[1] / self.grid_size)
        grid_cell = (grid_x, grid_y)
        
        # Track visit count for this area
        if grid_cell not in self.area_visit_count:
            self.area_visit_count[grid_cell] = 0
        
        visit_count = self.area_visit_count[grid_cell]
        
        # Diminishing returns for revisiting areas (prevents gaming)
        if visit_count < self.max_area_revisit_bonus:
            # Full reward for first visit, diminishing for subsequent visits
            exploration_multiplier = 1.0 / (1.0 + visit_count * 0.5)
            reward += self.reward_config['exploration_bonus'] * exploration_multiplier
        
        # Update visit count
        self.area_visit_count[grid_cell] += 1
        
        return reward
    
    def _calculate_anti_gaming_penalty(self, action: np.ndarray, position: np.ndarray) -> float:
        """Detect and penalize reward gaming behaviors"""
        penalty = 0.0
        
        linear_speed = abs(action[0])
        angular_speed = abs(action[1])
        
        # 1. Detect spinning behavior
        if self._detect_spinning_behavior(action):
            penalty += self.reward_config['unsafe_behavior_penalty'] * 2
        
        # 2. Detect oscillatory movement (back and forth gaming)
        if self._detect_oscillatory_behavior(position):
            penalty += self.reward_config['unsafe_behavior_penalty']
        
        # 3. Penalize extreme actions (likely gaming attempts)
        if angular_speed > 1.0:  # Very fast spinning
            penalty += self.reward_config['unsafe_behavior_penalty']
        
        return penalty
    
    def _calculate_diversity_bonus(self, action: np.ndarray) -> float:
        """Reward behavioral diversity to prevent overfitting"""
        reward = 0.0
        
        # Quantize action for pattern matching
        action_pattern = self._quantize_action(action)
        self.behavior_patterns.append(action_pattern)
        
        if len(self.behavior_patterns) >= 10:
            # Calculate diversity in recent behaviors
            recent_patterns = list(self.behavior_patterns)[-10:]
            unique_behaviors = len(set(recent_patterns))
            diversity_ratio = unique_behaviors / 10.0
            
            # Reward behavioral diversity
            if diversity_ratio > 0.5:  # Good diversity
                reward += self.reward_config['smooth_movement_bonus'] * diversity_ratio
        
        return reward
    
    def _apply_anti_overtraining_measures(self, raw_reward: float) -> float:
        """Apply measures to prevent overtraining"""
        
        # 1. Clip extreme rewards
        clipped_reward = np.clip(raw_reward, self.reward_clip_range[0], self.reward_clip_range[1])
        
        # 2. Add noise to prevent overfitting
        if self.reward_noise_std > 0:
            noise = np.random.normal(0, self.reward_noise_std)
            clipped_reward += noise
        
        # 3. Apply reward smoothing
        if self.enable_reward_smoothing:
            alpha = self.reward_smoothing_alpha
            self.smoothed_reward = alpha * clipped_reward + (1 - alpha) * self.smoothed_reward
            final_reward = self.smoothed_reward
        else:
            final_reward = clipped_reward
        
        return final_reward
    
    def _detect_spinning_behavior(self, action: np.ndarray) -> bool:
        """Detect if robot is spinning excessively"""
        if len(self.action_history) < 5:
            return False
        
        recent_actions = list(self.action_history)[-5:]
        recent_actions.append(action)
        
        # Check if most recent actions are high angular velocity with low linear velocity
        spinning_count = 0
        for act in recent_actions:
            if abs(act[1]) > self.spinning_detection_threshold and abs(act[0]) < 0.03:
                spinning_count += 1
        
        return spinning_count >= 4  # 4 out of 6 actions are spinning
    
    def _detect_oscillatory_behavior(self, position: np.ndarray) -> bool:
        """Detect back-and-forth oscillatory movement"""
        if len(self.position_history) < 6:
            return False
        
        recent_positions = list(self.position_history)[-6:] + [position]
        
        # Check if robot is moving back and forth in small area
        position_range = np.max(recent_positions, axis=0) - np.min(recent_positions, axis=0)
        movement_range = np.linalg.norm(position_range)
        
        # If movement is confined to small area but there's activity, likely oscillating
        if movement_range < 0.2:  # Less than 20cm total range
            # Check if there's actual movement (not just stationary)
            total_movement = 0
            for i in range(1, len(recent_positions)):
                total_movement += np.linalg.norm(recent_positions[i] - recent_positions[i-1])
            
            # If high activity in small space, likely oscillating
            return total_movement > 0.5  # More than 50cm of movement in 20cm space
        
        return False
    
    def _quantize_action(self, action: np.ndarray) -> str:
        """Convert action to discrete pattern for diversity tracking"""
        linear_speed = action[0]
        angular_speed = action[1]
        
        # Quantize to discrete bins
        if linear_speed > 0.1:
            linear_bin = 'fast_forward'
        elif linear_speed > 0.02:
            linear_bin = 'slow_forward'
        elif linear_speed < -0.02:
            linear_bin = 'backward'
        else:
            linear_bin = 'stationary'
        
        if angular_speed > 0.3:
            angular_bin = 'right_turn'
        elif angular_speed < -0.3:
            angular_bin = 'left_turn'
        else:
            angular_bin = 'straight'
        
        return f"{linear_bin}_{angular_bin}"
    
    def _calculate_safety_penalty(self, collision: bool, near_collision: bool) -> float:
        """Calculate safety-related penalties"""
        penalty = 0.0
        
        if collision:
            penalty += self.reward_config['collision_penalty']
        elif near_collision:
            penalty += self.reward_config['near_collision_penalty']
        
        return penalty
    
    # Legacy methods removed to prevent overtraining - these were too complex
    # and encouraged reward gaming behaviors. The new simplified system above
    # provides better generalization and robustness.
    
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
            'unique_areas': len(self.area_visit_count),
            'total_steps': self.step_count,
            'behavioral_diversity': self._calculate_behavioral_diversity(),
            'potential_overtraining_score': self._calculate_overtraining_risk()
        }
    
    def _calculate_behavioral_diversity(self) -> float:
        """Calculate behavioral diversity score (higher = more diverse)"""
        if len(self.behavior_patterns) < 10:
            return 0.0
        
        unique_patterns = len(set(self.behavior_patterns))
        total_patterns = len(self.behavior_patterns)
        return unique_patterns / total_patterns
    
    def _calculate_overtraining_risk(self) -> float:
        """Calculate a score indicating potential overtraining risk"""
        risk_score = 0.0
        
        if len(self.reward_history) < 100:
            return 0.0  # Not enough data
        
        recent_rewards = list(self.reward_history)[-50:]
        earlier_rewards = list(self.reward_history)[-100:-50]
        
        # Check for reward saturation without behavioral diversity
        reward_improvement = np.mean(recent_rewards) - np.mean(earlier_rewards)
        behavioral_diversity = self._calculate_behavioral_diversity()
        
        # High reward with low diversity = potential overtraining
        if reward_improvement > 5.0 and behavioral_diversity < 0.3:
            risk_score += 0.5
        
        # Check for repetitive patterns
        if len(self.behavior_patterns) >= 20:
            recent_patterns = list(self.behavior_patterns)[-20:]
            if len(set(recent_patterns)) < 5:  # Very low diversity
                risk_score += 0.3
        
        # Check for area revisiting patterns
        if len(self.area_visit_count) > 0:
            max_visits = max(self.area_visit_count.values())
            if max_visits > 10:  # Excessive revisiting
                risk_score += 0.2
        
        return min(risk_score, 1.0)  # Cap at 1.0
    
    def get_validation_metrics(self, test_positions: List[np.ndarray], test_actions: List[np.ndarray]) -> Dict[str, float]:
        """Calculate validation metrics to check for overtraining"""
        if not test_positions or not test_actions:
            return {}
        
        # Simulate rewards on test data
        test_rewards = []
        for pos, action in zip(test_positions, test_actions):
            # Simplified reward calculation for testing
            reward, _ = self.calculate_comprehensive_reward(
                action=action,
                position=pos,
                collision=False,
                near_collision=False,
                progress=0.1  # Dummy progress
            )
            test_rewards.append(reward)
        
        return {
            'test_mean_reward': np.mean(test_rewards),
            'test_std_reward': np.std(test_rewards),
            'training_test_gap': abs(np.mean(list(self.reward_history)[-50:]) - np.mean(test_rewards)),
            'generalization_score': 1.0 / (1.0 + abs(np.mean(list(self.reward_history)[-50:]) - np.mean(test_rewards)))
        }
    
    def reset_exploration_tracking(self):
        """Reset exploration tracking (useful for new sessions)"""
        self.visited_areas.clear()
        self.area_visit_count.clear()
        self.position_history.clear()
        self.action_history.clear()
        self.behavior_patterns.clear()
        self.step_count = 0
        self.last_movement_time = time.time()
        self.smoothed_reward = 0.0
    
    def enable_curriculum_mode(self, difficulty_level: float = 0.5):
        """Enable curriculum learning mode with adjustable difficulty"""
        # Adjust reward parameters based on difficulty
        base_multiplier = 0.5 + difficulty_level * 0.5  # 0.5 to 1.0
        
        self.reward_config['base_movement_reward'] *= base_multiplier
        self.reward_config['exploration_bonus'] *= base_multiplier
        self.reward_config['goal_oriented_bonus'] *= base_multiplier
        
        # Increase noise for lower difficulty (more exploration)
        self.reward_noise_std = 0.2 * (1.0 - difficulty_level)
    
    def should_stop_training(self, patience: int = 100) -> bool:
        """Early stopping criteria to prevent overtraining"""
        if len(self.reward_history) < patience * 2:
            return False
        
        # Check if reward has plateaued
        recent_rewards = list(self.reward_history)[-patience:]
        earlier_rewards = list(self.reward_history)[-patience*2:-patience]
        
        recent_mean = np.mean(recent_rewards)
        earlier_mean = np.mean(earlier_rewards)
        improvement = recent_mean - earlier_mean
        
        # Stop if minimal improvement and high overtraining risk
        if improvement < 1.0 and self._calculate_overtraining_risk() > 0.7:
            return True
        
        return False
