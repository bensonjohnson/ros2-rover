#!/usr/bin/env python3
"""
Multi-Metric Fitness Evaluation System
Provides comprehensive evaluation of rover performance across multiple objectives
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass
import time

@dataclass
class PerformanceMetrics:
    """Container for all performance metrics collected during evaluation"""
    # Core performance metrics
    avg_reward: float = 0.0
    exploration_coverage: float = 0.0  # Fraction of available area explored
    movement_efficiency: float = 0.0   # Distance traveled / energy used
    
    # Safety metrics
    collision_rate: float = 0.0        # Collisions per unit time
    near_collision_rate: float = 0.0   # Near misses per unit time
    safety_margin_avg: float = 0.0     # Average distance to obstacles
    
    # Learning efficiency metrics
    training_loss: float = 0.0
    convergence_rate: float = 0.0      # Rate of improvement
    behavioral_diversity: float = 0.0   # Anti-overtraining measure
    
    # Energy efficiency metrics  
    energy_per_distance: float = 0.0   # Energy consumption per meter
    idle_time_ratio: float = 0.0       # Fraction of time not moving
    
    # Robustness metrics
    performance_variance: float = 0.0   # Consistency across episodes
    recovery_success_rate: float = 0.0  # Success rate after getting stuck

@dataclass
class ObjectiveWeights:
    """Weights for different optimization objectives"""
    performance: float = 0.4    # Exploration performance and reward
    safety: float = 0.3         # Collision avoidance and safety margins
    efficiency: float = 0.2     # Energy and movement efficiency  
    robustness: float = 0.1     # Consistency and recovery capability

class MultiMetricEvaluator:
    """
    Comprehensive multi-objective fitness evaluator for rover performance.
    
    Evaluates rover performance across multiple competing objectives:
    - Performance: Exploration efficiency, reward accumulation
    - Safety: Collision avoidance, safety margins
    - Efficiency: Energy consumption, movement efficiency
    - Robustness: Consistency, recovery from failures
    """
    
    def __init__(self, 
                 evaluation_window: int = 100,
                 enable_debug: bool = False,
                 objective_weights: Optional[ObjectiveWeights] = None):
        
        self.evaluation_window = evaluation_window
        self.enable_debug = enable_debug
        self.objective_weights = objective_weights or ObjectiveWeights()
        
        # Metric history storage
        self.reward_history = deque(maxlen=evaluation_window)
        self.collision_history = deque(maxlen=evaluation_window)
        self.position_history = deque(maxlen=evaluation_window * 10)  # Higher resolution
        self.action_history = deque(maxlen=evaluation_window)
        self.energy_history = deque(maxlen=evaluation_window)
        self.loss_history = deque(maxlen=evaluation_window)
        
        # Exploration tracking
        self.explored_positions = set()
        self.grid_resolution = 0.1  # 10cm grid for exploration tracking
        
        # Episode tracking
        self.episode_rewards = deque(maxlen=20)
        self.episode_collisions = deque(maxlen=20)
        self.episode_distances = deque(maxlen=20)
        
        # Timing
        self.evaluation_start_time = time.time()
        self.total_distance_traveled = 0.0
        self.last_position = np.array([0.0, 0.0])
        
        if self.enable_debug:
            print("[MultiMetric] Initialized multi-objective evaluator")
            print(f"[MultiMetric] Objective weights: P={self.objective_weights.performance:.1f}, "
                  f"S={self.objective_weights.safety:.1f}, E={self.objective_weights.efficiency:.1f}, "
                  f"R={self.objective_weights.robustness:.1f}")
    
    def update_metrics(self, 
                      reward: float,
                      position: np.ndarray,
                      action: np.ndarray,
                      collision: bool = False,
                      near_collision: bool = False,
                      safety_margin: float = 1.0,
                      energy_used: float = 0.0,
                      training_loss: Optional[float] = None):
        """Update metrics with new observation"""
        
        # Update basic metrics
        self.reward_history.append(reward)
        self.collision_history.append(1.0 if collision else 0.0)
        self.position_history.append(position.copy())
        self.action_history.append(action.copy())
        self.energy_history.append(energy_used)
        
        if training_loss is not None:
            self.loss_history.append(training_loss)
        
        # Update exploration tracking
        grid_pos = tuple(np.round(position / self.grid_resolution).astype(int))
        self.explored_positions.add(grid_pos)
        
        # Update distance tracking
        if len(self.position_history) > 1:
            distance = np.linalg.norm(position - self.last_position)
            self.total_distance_traveled += distance
        self.last_position = position.copy()
    
    def calculate_comprehensive_fitness(self) -> Tuple[float, PerformanceMetrics, Dict[str, float]]:
        """
        Calculate comprehensive fitness score across all objectives.
        
        Returns:
            Tuple of (overall_fitness, detailed_metrics, objective_scores)
        """
        
        # Calculate detailed metrics
        metrics = self._calculate_detailed_metrics()
        
        # Calculate objective scores
        performance_score = self._calculate_performance_score(metrics)
        safety_score = self._calculate_safety_score(metrics)
        efficiency_score = self._calculate_efficiency_score(metrics)
        robustness_score = self._calculate_robustness_score(metrics)
        
        objective_scores = {
            'performance': performance_score,
            'safety': safety_score,
            'efficiency': efficiency_score,
            'robustness': robustness_score
        }
        
        # Weighted combination for overall fitness
        overall_fitness = (
            self.objective_weights.performance * performance_score +
            self.objective_weights.safety * safety_score +
            self.objective_weights.efficiency * efficiency_score +
            self.objective_weights.robustness * robustness_score
        )
        
        if self.enable_debug:
            print(f"[MultiMetric] Overall fitness: {overall_fitness:.4f}")
            print(f"[MultiMetric] Objectives - P: {performance_score:.3f}, S: {safety_score:.3f}, "
                  f"E: {efficiency_score:.3f}, R: {robustness_score:.3f}")
        
        return overall_fitness, metrics, objective_scores
    
    def _calculate_detailed_metrics(self) -> PerformanceMetrics:
        """Calculate detailed performance metrics from collected data"""
        
        metrics = PerformanceMetrics()
        
        if not self.reward_history:
            return metrics
        
        # Core performance metrics
        metrics.avg_reward = np.mean(self.reward_history)
        metrics.exploration_coverage = self._calculate_exploration_coverage()
        metrics.movement_efficiency = self._calculate_movement_efficiency()
        
        # Safety metrics
        metrics.collision_rate = self._calculate_collision_rate()
        metrics.near_collision_rate = 0.0  # Would need near-collision tracking
        metrics.safety_margin_avg = 1.0    # Would need safety margin tracking
        
        # Learning efficiency metrics
        if self.loss_history:
            metrics.training_loss = np.mean(self.loss_history)
            metrics.convergence_rate = self._calculate_convergence_rate()
        
        metrics.behavioral_diversity = self._calculate_behavioral_diversity()
        
        # Energy efficiency metrics
        metrics.energy_per_distance = self._calculate_energy_efficiency()
        metrics.idle_time_ratio = self._calculate_idle_time_ratio()
        
        # Robustness metrics
        metrics.performance_variance = self._calculate_performance_variance()
        metrics.recovery_success_rate = 0.8  # Would need recovery tracking
        
        return metrics
    
    def _calculate_performance_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate normalized performance objective score [0, 1]"""
        
        # Normalize reward (assuming typical range -10 to 30)
        reward_component = np.clip((metrics.avg_reward + 10) / 40, 0, 1)
        
        # Exploration coverage is already normalized
        exploration_component = metrics.exploration_coverage
        
        # Movement efficiency (higher is better, normalize to [0,1])
        efficiency_component = np.clip(metrics.movement_efficiency, 0, 1)
        
        # Weighted combination
        performance_score = (0.5 * reward_component + 
                           0.3 * exploration_component +
                           0.2 * efficiency_component)
        
        return performance_score
    
    def _calculate_safety_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate normalized safety objective score [0, 1]"""
        
        # Lower collision rate is better (invert)
        collision_component = max(0, 1.0 - metrics.collision_rate * 5.0)  # Assume max 0.2 collisions/step
        
        # Higher safety margin is better  
        margin_component = np.clip(metrics.safety_margin_avg / 2.0, 0, 1)  # Normalize to 2m max
        
        # Weighted combination
        safety_score = (0.7 * collision_component + 
                       0.3 * margin_component)
        
        return safety_score
    
    def _calculate_efficiency_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate normalized efficiency objective score [0, 1]"""
        
        # Lower energy per distance is better (invert and normalize)
        energy_component = max(0, 1.0 - metrics.energy_per_distance / 10.0)  # Assume max 10 units/meter
        
        # Lower idle time ratio is better (invert)
        idle_component = max(0, 1.0 - metrics.idle_time_ratio)
        
        # Weighted combination
        efficiency_score = (0.6 * energy_component + 
                          0.4 * idle_component)
        
        return efficiency_score
    
    def _calculate_robustness_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate normalized robustness objective score [0, 1]"""
        
        # Lower performance variance is better (invert)
        variance_component = max(0, 1.0 - metrics.performance_variance / 100.0)  # Normalize variance
        
        # Higher recovery success rate is better
        recovery_component = metrics.recovery_success_rate
        
        # Higher behavioral diversity is better (anti-overtraining)
        diversity_component = metrics.behavioral_diversity
        
        # Weighted combination
        robustness_score = (0.4 * variance_component +
                          0.3 * recovery_component +
                          0.3 * diversity_component)
        
        return robustness_score
    
    def _calculate_exploration_coverage(self) -> float:
        """Calculate fraction of environment explored"""
        
        if not self.explored_positions:
            return 0.0
        
        # Estimate total explorable area (this would be environment-specific)
        # For now, assume a reasonable exploration area
        total_grid_cells = (20 / self.grid_resolution) ** 2  # 20m x 20m area
        explored_cells = len(self.explored_positions)
        
        coverage = min(1.0, explored_cells / total_grid_cells)
        return coverage
    
    def _calculate_movement_efficiency(self) -> float:
        """Calculate movement efficiency based on path taken"""
        
        if len(self.position_history) < 2:
            return 0.0
        
        # Calculate straight-line distance vs actual distance
        start_pos = self.position_history[0]
        end_pos = self.position_history[-1]
        straight_distance = np.linalg.norm(end_pos - start_pos)
        
        if self.total_distance_traveled == 0:
            return 0.0
        
        # Efficiency = straight_line_distance / actual_distance
        efficiency = straight_distance / max(self.total_distance_traveled, 0.1)
        return min(1.0, efficiency)
    
    def _calculate_collision_rate(self) -> float:
        """Calculate collisions per time step"""
        
        if not self.collision_history:
            return 0.0
        
        return np.mean(self.collision_history)
    
    def _calculate_convergence_rate(self) -> float:
        """Calculate rate of training loss improvement"""
        
        if len(self.loss_history) < 10:
            return 0.0
        
        # Calculate slope of loss reduction (negative slope is good)
        recent_losses = list(self.loss_history)[-10:]
        x = np.arange(len(recent_losses))
        slope = np.polyfit(x, recent_losses, 1)[0]
        
        # Convert to positive improvement rate
        improvement_rate = max(0, -slope / max(recent_losses[0], 0.1))
        return min(1.0, improvement_rate)
    
    def _calculate_behavioral_diversity(self) -> float:
        """Calculate behavioral diversity (anti-overtraining measure)"""
        
        if len(self.action_history) < 10:
            return 0.5  # Default medium diversity
        
        # Calculate action diversity using standard deviation
        recent_actions = np.array(list(self.action_history)[-20:])
        
        # Diversity in linear velocity
        linear_diversity = np.std(recent_actions[:, 0])
        # Diversity in angular velocity  
        angular_diversity = np.std(recent_actions[:, 1])
        
        # Normalize and combine (higher std = higher diversity)
        diversity = (linear_diversity + angular_diversity) / 2.0
        return min(1.0, diversity * 5.0)  # Scale appropriately
    
    def _calculate_energy_efficiency(self) -> float:
        """Calculate energy consumption per unit distance"""
        
        if not self.energy_history or self.total_distance_traveled == 0:
            return 5.0  # Default moderate consumption
        
        total_energy = sum(self.energy_history)
        energy_per_distance = total_energy / max(self.total_distance_traveled, 0.1)
        
        return energy_per_distance
    
    def _calculate_idle_time_ratio(self) -> float:
        """Calculate fraction of time spent idle (not moving)"""
        
        if not self.action_history:
            return 0.0
        
        # Count timesteps with very low action magnitudes
        recent_actions = np.array(list(self.action_history)[-50:])
        action_magnitudes = np.linalg.norm(recent_actions, axis=1)
        idle_steps = np.sum(action_magnitudes < 0.05)
        
        idle_ratio = idle_steps / len(recent_actions)
        return idle_ratio
    
    def _calculate_performance_variance(self) -> float:
        """Calculate variance in performance across recent episodes"""
        
        if len(self.episode_rewards) < 3:
            return 50.0  # Default moderate variance
        
        return np.var(self.episode_rewards)
    
    def end_episode(self, episode_reward: float, episode_collisions: int, episode_distance: float):
        """Mark end of episode and record episode-level metrics"""
        
        self.episode_rewards.append(episode_reward)
        self.episode_collisions.append(episode_collisions)
        self.episode_distances.append(episode_distance)
        
        if self.enable_debug:
            print(f"[MultiMetric] Episode complete: Reward={episode_reward:.2f}, "
                  f"Collisions={episode_collisions}, Distance={episode_distance:.2f}m")
    
    def get_current_metrics_summary(self) -> Dict[str, float]:
        """Get current metrics summary for monitoring"""
        
        fitness, metrics, objectives = self.calculate_comprehensive_fitness()
        
        return {
            'overall_fitness': fitness,
            'performance_objective': objectives['performance'],
            'safety_objective': objectives['safety'],
            'efficiency_objective': objectives['efficiency'],
            'robustness_objective': objectives['robustness'],
            'avg_reward': metrics.avg_reward,
            'collision_rate': metrics.collision_rate,
            'exploration_coverage': metrics.exploration_coverage,
            'behavioral_diversity': metrics.behavioral_diversity
        }
    
    def reset(self):
        """Reset all metrics for new evaluation period"""
        
        self.reward_history.clear()
        self.collision_history.clear()
        self.position_history.clear()
        self.action_history.clear()
        self.energy_history.clear()
        self.loss_history.clear()
        
        self.explored_positions.clear()
        self.total_distance_traveled = 0.0
        self.last_position = np.array([0.0, 0.0])
        self.evaluation_start_time = time.time()
        
        if self.enable_debug:
            print("[MultiMetric] Metrics reset for new evaluation period")
    
    def update_objective_weights(self, new_weights: ObjectiveWeights):
        """Update objective weights for different optimization priorities"""
        
        self.objective_weights = new_weights
        
        if self.enable_debug:
            print(f"[MultiMetric] Updated objective weights: P={new_weights.performance:.1f}, "
                  f"S={new_weights.safety:.1f}, E={new_weights.efficiency:.1f}, R={new_weights.robustness:.1f}")