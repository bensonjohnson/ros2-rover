#!/usr/bin/env python3
"""
Adaptive Reward System for Enhanced Neural Network Learning
Implements curiosity-driven rewards, multi-objective optimization, and anti-gaming mechanisms
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, NamedTuple
from collections import deque
from dataclasses import dataclass
import json

@dataclass
class RewardComponents:
    """Structure for tracking individual reward components"""
    exploration: float = 0.0
    efficiency: float = 0.0
    safety: float = 0.0
    smoothness: float = 0.0
    curiosity: float = 0.0
    progress: float = 0.0
    total: float = 0.0

class CuriosityDrivenRewardCalculator:
    """
    Advanced reward calculator with curiosity-driven exploration,
    adaptive scaling, and multi-objective optimization support
    """
    
    def __init__(self, enable_curiosity: bool = True, enable_adaptive_scaling: bool = True,
                 enable_diversity_rewards: bool = True, **kwargs):
        
        # Core configuration
        self.enable_curiosity = enable_curiosity
        self.enable_adaptive_scaling = enable_adaptive_scaling
        self.enable_diversity_rewards = enable_diversity_rewards
        
        # Adaptive reward weights (will be adjusted during training)
        self.reward_weights = {
            'exploration_bonus': kwargs.get('exploration_bonus', 12.0),
            'forward_progress_bonus': kwargs.get('forward_progress_bonus', 18.0),
            'efficiency_bonus': kwargs.get('efficiency_bonus', 10.0),
            'curiosity_bonus': kwargs.get('curiosity_bonus', 8.0),
            'safety_bonus': kwargs.get('safety_bonus', 15.0),
            'smoothness_bonus': kwargs.get('smoothness_bonus', 6.0),
            
            # Penalties
            'collision_penalty': kwargs.get('collision_penalty', -30.0),
            'unsafe_behavior_penalty': kwargs.get('unsafe_behavior_penalty', -8.0),
            'spinning_penalty': kwargs.get('spinning_penalty', -12.0),
            'stagnation_penalty': kwargs.get('stagnation_penalty', -5.0),
        }
        
        # Adaptive scaling parameters
        self.performance_window = 50
        self.reward_history = deque(maxlen=1000)
        self.performance_metrics = deque(maxlen=self.performance_window)
        self.scaling_factors = {key: 1.0 for key in self.reward_weights.keys()}
        
        # Exploration and curiosity tracking
        self.visited_states = set()
        self.state_visit_counts = {}
        self.grid_size = 0.4  # Finer grid for better exploration tracking
        self.novelty_threshold = 0.3
        
        # Curiosity mechanism (prediction error based)
        self.curiosity_enabled = enable_curiosity
        self.prediction_errors = deque(maxlen=200)
        self.curiosity_lr = 0.01
        self.curiosity_scale = 1.0
        
        # Forward dynamics prediction (simplified)
        self.forward_model_enabled = True
        self.predicted_positions = deque(maxlen=100)
        self.actual_positions = deque(maxlen=100)
        
        # Behavioral diversity tracking
        self.action_patterns = deque(maxlen=100)
        self.behavior_diversity_window = 20
        self.diversity_bonus_scale = 3.0
        
        # Multi-objective reward decomposition
        self.objective_components = ['exploration', 'efficiency', 'safety', 'smoothness']
        self.component_history = {comp: deque(maxlen=200) for comp in self.objective_components}
        
        # Anti-gaming and overtraining prevention
        self.gaming_detection_enabled = True
        self.oscillation_detection_window = 10
        self.position_history = deque(maxlen=self.oscillation_detection_window)
        self.action_history = deque(maxlen=20)
        
        # Adaptive curriculum parameters
        self.curriculum_stage = 0
        self.curriculum_progression_threshold = 100  # Steps before evaluating progression
        self.steps_in_current_stage = 0
        self.stage_performance_threshold = 0.7
        
        # Performance tracking for adaptation
        self.success_rate_window = 30
        self.recent_successes = deque(maxlen=self.success_rate_window)
        self.adaptation_interval = 25
        self.last_adaptation_step = 0
        
        # Reward statistics for debugging
        self.reward_stats = {
            'mean_reward': 0.0,
            'reward_variance': 0.0,
            'success_rate': 0.0,
            'exploration_rate': 0.0,
            'diversity_score': 0.0
        }
        
        print("✓ Adaptive Curiosity-Driven Reward System initialized")
        print(f"  Curiosity: {'Enabled' if enable_curiosity else 'Disabled'}")
        print(f"  Adaptive scaling: {'Enabled' if enable_adaptive_scaling else 'Disabled'}")
        print(f"  Diversity rewards: {'Enabled' if enable_diversity_rewards else 'Disabled'}")
    
    def calculate_comprehensive_reward(self, 
                                     action: np.ndarray,
                                     position: np.ndarray,
                                     collision: bool,
                                     near_collision: bool,
                                     progress: float,
                                     depth_data: Optional[np.ndarray] = None,
                                     wheel_velocities: Optional[Tuple[float, float]] = None,
                                     step_count: int = 0) -> Tuple[float, RewardComponents]:
        """
        Calculate comprehensive reward with curiosity, adaptation, and multi-objective optimization
        
        Returns:
            Tuple of (total_reward, reward_components)
        """
        
        components = RewardComponents()
        
        # Core movement and exploration rewards
        components.exploration = self._calculate_exploration_reward(position, action)
        components.efficiency = self._calculate_efficiency_reward(action, progress, wheel_velocities)
        components.safety = self._calculate_safety_reward(collision, near_collision, depth_data)
        components.smoothness = self._calculate_smoothness_reward(action)
        
        # Advanced curiosity-driven rewards
        if self.curiosity_enabled:
            components.curiosity = self._calculate_curiosity_reward(position, action, depth_data)
        
        # Progress tracking
        components.progress = self._calculate_progress_reward(progress, position)
        
        # Apply adaptive scaling
        if self.enable_adaptive_scaling:
            components = self._apply_adaptive_scaling(components, step_count)
        
        # Anti-gaming detection and penalties
        gaming_penalty = self._detect_and_penalize_gaming(action, position)
        
        # Calculate total reward
        components.total = (
            components.exploration + 
            components.efficiency + 
            components.safety + 
            components.smoothness + 
            components.curiosity + 
            components.progress + 
            gaming_penalty
        )
        
        # Update tracking and adaptation
        self._update_tracking(components, position, action, step_count)
        
        # Adaptive reward scaling based on curriculum
        components.total = self._apply_curriculum_scaling(components.total, step_count)
        
        return components.total, components
    
    def _calculate_exploration_reward(self, position: np.ndarray, action: np.ndarray) -> float:
        """Calculate exploration reward with novelty detection"""
        reward = 0.0
        
        # Grid-based novelty
        grid_x = int(position[0] / self.grid_size)
        grid_y = int(position[1] / self.grid_size)
        grid_cell = (grid_x, grid_y)
        
        # Track visit counts
        if grid_cell not in self.state_visit_counts:
            self.state_visit_counts[grid_cell] = 0
            reward += self.reward_weights['exploration_bonus'] * 1.5  # First visit bonus
        
        visit_count = self.state_visit_counts[grid_cell]
        
        # Diminishing returns for revisiting
        if visit_count < 5:
            exploration_multiplier = max(0.1, 1.0 / (1.0 + visit_count * 0.3))
            reward += self.reward_weights['exploration_bonus'] * exploration_multiplier
        
        self.state_visit_counts[grid_cell] += 1
        
        # Frontier exploration bonus (areas adjacent to unknown)
        frontier_bonus = self._calculate_frontier_bonus(grid_cell)
        reward += frontier_bonus
        
        return reward * self.scaling_factors.get('exploration_bonus', 1.0)
    
    def _calculate_efficiency_reward(self, action: np.ndarray, progress: float, 
                                   wheel_velocities: Optional[Tuple[float, float]]) -> float:
        """Calculate efficiency reward for effective movement"""
        reward = 0.0
        
        # Forward progress reward
        if progress > 0:
            progress_multiplier = min(progress * 15.0, self.reward_weights['forward_progress_bonus'])
            reward += progress_multiplier
        
        # Speed efficiency
        linear_speed = abs(action[0])
        angular_speed = abs(action[1])
        
        # Optimal speed range bonus
        if 0.08 < linear_speed < 0.25:
            speed_efficiency = 1.0 - abs(linear_speed - 0.15) / 0.15
            reward += self.reward_weights['efficiency_bonus'] * speed_efficiency
        
        # Penalize excessive spinning while stationary
        if linear_speed < 0.03 and angular_speed > 0.3:
            reward += self.reward_weights['spinning_penalty'] * 0.5
        
        # Wheel velocity efficiency (if available)
        if wheel_velocities is not None:
            left_vel, right_vel = wheel_velocities
            # Penalize excessive wheel speed differences (inefficient turning)
            if abs(left_vel - right_vel) > 0.3 and linear_speed < 0.05:
                reward += self.reward_weights['unsafe_behavior_penalty'] * 0.3
        
        return reward * self.scaling_factors.get('efficiency_bonus', 1.0)
    
    def _calculate_safety_reward(self, collision: bool, near_collision: bool, 
                               depth_data: Optional[np.ndarray]) -> float:
        """Calculate safety reward with predictive collision avoidance"""
        reward = 0.0
        
        # Collision penalties
        if collision:
            reward += self.reward_weights['collision_penalty']
        elif near_collision:
            reward += self.reward_weights['unsafe_behavior_penalty']
        else:
            # Safety bonus for maintaining safe distance
            reward += self.reward_weights['safety_bonus'] * 0.3
        
        # Predictive safety reward based on depth data
        if depth_data is not None:
            safety_margin = self._calculate_predictive_safety(depth_data)
            if safety_margin > 0.5:  # Good safety margin
                reward += self.reward_weights['safety_bonus'] * safety_margin * 0.4
            elif safety_margin < 0.2:  # Dangerous situation
                reward += self.reward_weights['unsafe_behavior_penalty'] * (0.2 - safety_margin) * 2.0
        
        return reward * self.scaling_factors.get('safety_bonus', 1.0)
    
    def _calculate_smoothness_reward(self, action: np.ndarray) -> float:
        """Calculate smoothness reward for realistic robot behavior"""
        reward = 0.0
        
        # Action magnitude reasonableness
        action_magnitude = np.linalg.norm(action)
        if action_magnitude < 1.0:  # Reasonable action
            reward += self.reward_weights['smoothness_bonus'] * 0.5
        else:  # Extreme action
            reward += self.reward_weights['unsafe_behavior_penalty'] * (action_magnitude - 1.0)
        
        # Action consistency (if we have history)
        if len(self.action_history) > 0:
            last_action = self.action_history[-1]
            action_change = np.linalg.norm(action - last_action)
            
            # Reward smooth transitions
            if action_change < 0.2:
                reward += self.reward_weights['smoothness_bonus'] * 0.3
            elif action_change > 0.8:  # Jerky movement
                reward += self.reward_weights['unsafe_behavior_penalty'] * 0.2
        
        return reward * self.scaling_factors.get('smoothness_bonus', 1.0)
    
    def _calculate_curiosity_reward(self, position: np.ndarray, action: np.ndarray, 
                                  depth_data: Optional[np.ndarray]) -> float:
        """Calculate curiosity reward based on prediction error"""
        if not self.curiosity_enabled:
            return 0.0
        
        reward = 0.0
        
        # Position prediction error (simplified forward model)
        if len(self.predicted_positions) > 0 and len(self.actual_positions) > 0:
            predicted_pos = self.predicted_positions[-1]
            actual_pos = position
            
            prediction_error = np.linalg.norm(predicted_pos - actual_pos)
            
            # Curiosity is proportional to prediction error (up to a limit)
            curiosity_bonus = min(prediction_error / 0.5, 1.0) * self.reward_weights['curiosity_bonus']
            reward += curiosity_bonus * self.curiosity_scale
            
            # Store prediction error for learning
            self.prediction_errors.append(prediction_error)
            
            # Adaptive curiosity scale (reduce if errors are consistently high)
            if len(self.prediction_errors) >= 20:
                avg_error = np.mean(list(self.prediction_errors)[-20:])
                if avg_error > 0.3:  # High prediction errors - reduce curiosity scale
                    self.curiosity_scale *= 0.98
                elif avg_error < 0.1:  # Low prediction errors - increase curiosity scale
                    self.curiosity_scale *= 1.01
                
                self.curiosity_scale = np.clip(self.curiosity_scale, 0.1, 3.0)
        
        # Predict next position (simple kinematic model)
        predicted_next_pos = position + action[:2] * 0.1  # Assuming ~0.1s timestep
        self.predicted_positions.append(predicted_next_pos)
        self.actual_positions.append(position.copy())
        
        # Scene complexity curiosity (if depth data available)
        if depth_data is not None:
            scene_complexity = self._calculate_scene_complexity(depth_data)
            complexity_bonus = scene_complexity * self.reward_weights['curiosity_bonus'] * 0.2
            reward += complexity_bonus
        
        return reward * self.scaling_factors.get('curiosity_bonus', 1.0)
    
    def _calculate_progress_reward(self, progress: float, position: np.ndarray) -> float:
        """Calculate overall progress and goal-oriented reward"""
        reward = 0.0
        
        # Distance-based progress
        if progress > 0:
            reward += progress * self.reward_weights['forward_progress_bonus']
        
        # Exploration progress (new areas discovered)
        exploration_progress = len(self.state_visit_counts) / max(1, len(self.state_visit_counts) + 50)
        reward += exploration_progress * self.reward_weights['exploration_bonus'] * 0.3
        
        # Avoid stagnation
        if len(self.position_history) >= 5:
            recent_positions = list(self.position_history)[-5:]
            movement_variance = np.var([np.linalg.norm(pos - position) for pos in recent_positions])
            
            if movement_variance < 0.01:  # Very little movement
                reward += self.reward_weights['stagnation_penalty']
        
        return reward
    
    def _calculate_frontier_bonus(self, current_cell: Tuple[int, int]) -> float:
        """Calculate bonus for exploring frontier areas (edges of known space)"""
        bonus = 0.0
        
        # Check adjacent cells
        adjacent_cells = [
            (current_cell[0] + dx, current_cell[1] + dy)
            for dx in [-1, 0, 1] for dy in [-1, 0, 1]
            if not (dx == 0 and dy == 0)
        ]
        
        unknown_adjacent = sum(1 for cell in adjacent_cells if cell not in self.state_visit_counts)
        
        # Frontier bonus proportional to unknown adjacent cells
        if unknown_adjacent > 0:
            bonus = (unknown_adjacent / 8.0) * self.reward_weights['exploration_bonus'] * 0.4
        
        return bonus
    
    def _calculate_predictive_safety(self, depth_data: np.ndarray) -> float:
        """Calculate predictive safety margin from depth data"""
        try:
            # Focus on forward-facing region
            h, w = depth_data.shape
            forward_region = depth_data[h//3:2*h//3, w//4:3*w//4]
            
            # Filter valid depth values
            valid_depths = forward_region[(forward_region > 0.1) & (forward_region < 5.0)]
            
            if len(valid_depths) == 0:
                return 0.5  # Default moderate safety
            
            # Calculate safety metrics
            min_distance = np.min(valid_depths)
            mean_distance = np.mean(valid_depths)
            percentile_10 = np.percentile(valid_depths, 10)
            
            # Safety margin calculation
            safety_margin = min(min_distance / 1.5, 1.0) * 0.4 + min(percentile_10 / 2.0, 1.0) * 0.6
            
            return np.clip(safety_margin, 0.0, 1.0)
        
        except Exception:
            return 0.5  # Default safety margin
    
    def _calculate_scene_complexity(self, depth_data: np.ndarray) -> float:
        """Calculate scene complexity for curiosity bonus"""
        try:
            # Edge density
            edges = np.abs(np.gradient(depth_data)[0]) + np.abs(np.gradient(depth_data)[1])
            edge_density = np.mean(edges[edges > 0.1])
            
            # Depth variance
            valid_depths = depth_data[(depth_data > 0.1) & (depth_data < 5.0)]
            depth_variance = np.var(valid_depths) if len(valid_depths) > 0 else 0
            
            # Obstacle density
            close_obstacles = np.sum((depth_data > 0) & (depth_data < 1.0))
            total_valid = np.sum(depth_data > 0)
            obstacle_density = close_obstacles / max(total_valid, 1)
            
            # Combine complexity metrics
            complexity = (
                min(edge_density / 0.5, 1.0) * 0.4 +
                min(depth_variance / 2.0, 1.0) * 0.3 +
                obstacle_density * 0.3
            )
            
            return np.clip(complexity, 0.0, 1.0)
        
        except Exception:
            return 0.3  # Default complexity
    
    def _detect_and_penalize_gaming(self, action: np.ndarray, position: np.ndarray) -> float:
        """Detect and penalize reward gaming behaviors"""
        penalty = 0.0
        
        if not self.gaming_detection_enabled:
            return penalty
        
        # Oscillation detection
        self.position_history.append(position.copy())
        
        if len(self.position_history) >= self.oscillation_detection_window:
            positions = np.array(list(self.position_history))
            
            # Check for back-and-forth movement
            movement_directions = np.diff(positions, axis=0)
            if len(movement_directions) > 4:
                # Calculate direction changes
                direction_changes = 0
                for i in range(1, len(movement_directions)):
                    dot_product = np.dot(movement_directions[i], movement_directions[i-1])
                    if dot_product < -0.5:  # Opposite directions
                        direction_changes += 1
                
                # Penalize excessive oscillation
                if direction_changes > len(movement_directions) * 0.6:
                    penalty += self.reward_weights['unsafe_behavior_penalty'] * 0.5
        
        # Spinning detection
        linear_speed = abs(action[0])
        angular_speed = abs(action[1])
        
        if angular_speed > 0.8 and linear_speed < 0.05:
            penalty += self.reward_weights['spinning_penalty'] * 0.3
        
        # Action repetition detection
        self.action_history.append(action.copy())
        
        if len(self.action_history) >= 10:
            recent_actions = np.array(list(self.action_history)[-10:])
            action_variance = np.var(recent_actions, axis=0)
            
            # Penalize highly repetitive actions
            if np.all(action_variance < 0.01):
                penalty += self.reward_weights['stagnation_penalty'] * 0.5
        
        return penalty
    
    def _apply_adaptive_scaling(self, components: RewardComponents, step_count: int) -> RewardComponents:
        """Apply adaptive scaling based on recent performance"""
        if not self.enable_adaptive_scaling:
            return components
        
        # Only adapt every N steps to avoid instability
        if step_count - self.last_adaptation_step < self.adaptation_interval:
            return components
        
        self.last_adaptation_step = step_count
        
        # Analyze recent performance
        if len(self.reward_history) >= 20:
            recent_rewards = list(self.reward_history)[-20:]
            avg_reward = np.mean(recent_rewards)
            reward_trend = np.mean(recent_rewards[-10:]) - np.mean(recent_rewards[-20:-10])
            
            # Adapt scaling factors based on performance
            
            # If exploration is stagnating, increase exploration rewards
            exploration_rate = len(self.state_visit_counts) / max(step_count / 10, 1)
            if exploration_rate < 0.5:  # Low exploration rate
                self.scaling_factors['exploration_bonus'] *= 1.1
                self.scaling_factors['curiosity_bonus'] *= 1.05
            elif exploration_rate > 2.0:  # High exploration rate
                self.scaling_factors['exploration_bonus'] *= 0.95
                self.scaling_factors['efficiency_bonus'] *= 1.1
            
            # If rewards are decreasing, increase safety focus
            if reward_trend < -1.0:
                self.scaling_factors['safety_bonus'] *= 1.15
                self.scaling_factors['collision_penalty'] *= 0.9  # Less harsh penalties
            elif reward_trend > 2.0:  # Good progress
                self.scaling_factors['efficiency_bonus'] *= 1.05
            
            # Clamp scaling factors to reasonable ranges
            for key in self.scaling_factors:
                self.scaling_factors[key] = np.clip(self.scaling_factors[key], 0.2, 3.0)
        
        # Apply scaling to components
        scaled_components = RewardComponents()
        scaled_components.exploration = components.exploration * self.scaling_factors.get('exploration_bonus', 1.0)
        scaled_components.efficiency = components.efficiency * self.scaling_factors.get('efficiency_bonus', 1.0)
        scaled_components.safety = components.safety * self.scaling_factors.get('safety_bonus', 1.0)
        scaled_components.smoothness = components.smoothness * self.scaling_factors.get('smoothness_bonus', 1.0)
        scaled_components.curiosity = components.curiosity * self.scaling_factors.get('curiosity_bonus', 1.0)
        scaled_components.progress = components.progress
        
        return scaled_components
    
    def _apply_curriculum_scaling(self, total_reward: float, step_count: int) -> float:
        """Apply curriculum learning scaling"""
        self.steps_in_current_stage += 1
        
        # Basic curriculum progression (can be enhanced)
        if self.steps_in_current_stage > self.curriculum_progression_threshold:
            # Evaluate performance in current stage
            if len(self.reward_history) >= 20:
                recent_performance = np.mean(list(self.reward_history)[-20:])
                
                if recent_performance > 5.0 and self.curriculum_stage < 3:  # Good performance
                    self.curriculum_stage += 1
                    self.steps_in_current_stage = 0
                    print(f"📈 Advanced to curriculum stage {self.curriculum_stage}")
        
        # Apply stage-specific scaling
        stage_multipliers = [1.2, 1.0, 0.9, 0.8]  # Easier early stages
        stage_multiplier = stage_multipliers[min(self.curriculum_stage, len(stage_multipliers) - 1)]
        
        return total_reward * stage_multiplier
    
    def _update_tracking(self, components: RewardComponents, position: np.ndarray, 
                        action: np.ndarray, step_count: int):
        """Update internal tracking and statistics"""
        
        # Update reward history
        self.reward_history.append(components.total)
        
        # Update component history
        self.component_history['exploration'].append(components.exploration)
        self.component_history['efficiency'].append(components.efficiency)
        self.component_history['safety'].append(components.safety)
        self.component_history['smoothness'].append(components.smoothness)
        
        # Update action patterns for diversity tracking
        action_pattern = self._quantize_action(action)
        self.action_patterns.append(action_pattern)
        
        # Update performance metrics
        success = components.total > 0  # Simple success criterion
        self.recent_successes.append(success)
        
        # Update statistics
        if len(self.reward_history) >= 10:
            self.reward_stats['mean_reward'] = float(np.mean(list(self.reward_history)[-50:]))
            self.reward_stats['reward_variance'] = float(np.var(list(self.reward_history)[-50:]))
        
        if len(self.recent_successes) >= 10:
            self.reward_stats['success_rate'] = float(np.mean(list(self.recent_successes)[-30:]))
        
        # Calculate exploration rate
        self.reward_stats['exploration_rate'] = float(len(self.state_visit_counts) / max(step_count / 10, 1))
        
        # Calculate behavioral diversity
        if len(self.action_patterns) >= self.behavior_diversity_window:
            recent_patterns = list(self.action_patterns)[-self.behavior_diversity_window:]
            unique_patterns = len(set(recent_patterns))
            self.reward_stats['diversity_score'] = float(unique_patterns / self.behavior_diversity_window)
    
    def _quantize_action(self, action: np.ndarray) -> Tuple[int, int]:
        """Quantize action for pattern tracking"""
        linear_bin = min(int((action[0] + 1.0) * 5), 9)  # 10 bins for linear
        angular_bin = min(int((action[1] + 1.0) * 5), 9)  # 10 bins for angular
        return (linear_bin, angular_bin)
    
    def get_reward_breakdown(self) -> Dict[str, float]:
        """Get detailed breakdown of recent reward components"""
        breakdown = {}
        
        for component, history in self.component_history.items():
            if len(history) > 0:
                breakdown[f'avg_{component}'] = float(np.mean(list(history)[-20:]))
                breakdown[f'std_{component}'] = float(np.std(list(history)[-20:]))
        
        # Add scaling factors
        for key, value in self.scaling_factors.items():
            breakdown[f'scale_{key}'] = float(value)
        
        # Add statistics
        breakdown.update(self.reward_stats)
        
        # Add curiosity metrics
        if self.curiosity_enabled and len(self.prediction_errors) > 0:
            breakdown['avg_prediction_error'] = float(np.mean(list(self.prediction_errors)[-20:]))
            breakdown['curiosity_scale'] = float(self.curiosity_scale)
        
        breakdown['curriculum_stage'] = self.curriculum_stage
        breakdown['visited_states'] = len(self.state_visit_counts)
        
        return breakdown
    
    def get_optimization_suggestions(self) -> List[str]:
        """Get suggestions for optimizing the reward system"""
        suggestions = []
        
        # Analyze recent performance
        if len(self.reward_history) >= 50:
            recent_rewards = list(self.reward_history)[-50:]
            avg_reward = np.mean(recent_rewards)
            reward_trend = np.mean(recent_rewards[-25:]) - np.mean(recent_rewards[-50:-25])
            
            if avg_reward < 2.0:
                suggestions.append("Low average reward - consider increasing exploration bonus")
            
            if reward_trend < -0.5:
                suggestions.append("Declining performance - consider reducing penalty harshness")
            
            if self.reward_stats['exploration_rate'] < 0.3:
                suggestions.append("Low exploration rate - increase curiosity and novelty bonuses")
            
            if self.reward_stats['diversity_score'] < 0.4:
                suggestions.append("Low behavioral diversity - add diversity bonuses")
            
            if self.reward_stats['success_rate'] < 0.3:
                suggestions.append("Low success rate - consider curriculum learning adjustments")
        
        # Check for scaling issues
        extreme_scales = [k for k, v in self.scaling_factors.items() if v < 0.3 or v > 2.5]
        if extreme_scales:
            suggestions.append(f"Extreme scaling factors detected: {extreme_scales}")
        
        return suggestions
    
    def save_configuration(self, filepath: str):
        """Save current configuration and statistics"""
        config = {
            'reward_weights': self.reward_weights,
            'scaling_factors': self.scaling_factors,
            'curriculum_stage': self.curriculum_stage,
            'reward_stats': self.reward_stats,
            'state_visit_count': len(self.state_visit_counts),
            'curiosity_scale': self.curiosity_scale,
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"✓ Reward configuration saved to {filepath}")
        except Exception as e:
            print(f"✗ Failed to save configuration: {e}")
    
    def load_configuration(self, filepath: str):
        """Load configuration from file"""
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            
            self.reward_weights.update(config.get('reward_weights', {}))
            self.scaling_factors.update(config.get('scaling_factors', {}))
            self.curriculum_stage = config.get('curriculum_stage', 0)
            self.curiosity_scale = config.get('curiosity_scale', 1.0)
            
            print(f"✓ Reward configuration loaded from {filepath}")
            
        except Exception as e:
            print(f"✗ Failed to load configuration: {e}")

# Factory function for creating adaptive reward calculators
def create_adaptive_reward_calculator(mode: str = "balanced", **kwargs) -> CuriosityDrivenRewardCalculator:
    """
    Create adaptive reward calculator with predefined configurations
    
    Args:
        mode: "exploration", "balanced", "safety", or "efficiency"
        **kwargs: Additional configuration parameters
    
    Returns:
        Configured reward calculator
    """
    
    mode_configs = {
        "exploration": {
            "exploration_bonus": 15.0,
            "curiosity_bonus": 12.0,
            "forward_progress_bonus": 10.0,
            "enable_curiosity": True,
            "enable_diversity_rewards": True
        },
        "balanced": {
            "exploration_bonus": 12.0,
            "curiosity_bonus": 8.0,
            "forward_progress_bonus": 15.0,
            "safety_bonus": 12.0,
            "enable_curiosity": True,
            "enable_diversity_rewards": True
        },
        "safety": {
            "safety_bonus": 20.0,
            "collision_penalty": -40.0,
            "exploration_bonus": 8.0,
            "enable_curiosity": False,
            "enable_diversity_rewards": False
        },
        "efficiency": {
            "forward_progress_bonus": 25.0,
            "efficiency_bonus": 15.0,
            "exploration_bonus": 8.0,
            "enable_curiosity": True,
            "enable_diversity_rewards": False
        }
    }
    
    config = mode_configs.get(mode, mode_configs["balanced"])
    config.update(kwargs)
    
    return CuriosityDrivenRewardCalculator(**config)