"""Phase Manager for SAC Training Curriculum.

Manages training phases based on multi-metric performance evaluation.
Supports phase transitions and regression detection.
"""

from collections import deque
from typing import Dict, Optional
import numpy as np


class PhaseManager:
    """Manages SAC training phases with performance-based transitions."""
    
    PHASES = ['exploration', 'learning', 'refinement']
    
    # Transition thresholds
    EXPLORATION_TO_LEARNING = {
        'avg_reward': 0.15,           # Minimum average eval reward
        'collision_rate': 0.35,       # Maximum collision rate
        'avg_length': 45,             # Minimum average episode length (steps)
        'window_size': 10             # Minimum eval episodes required
    }
    
    LEARNING_TO_REFINEMENT = {
        'avg_reward': 0.25,           # Minimum average eval reward
        'collision_rate': 0.20,       # Maximum collision rate
        'avg_length': 90,             # Minimum average episode length
        'reward_std': 0.20,           # Maximum reward standard deviation
        'window_size': 10             # Minimum eval episodes required
    }
    
    # Regression thresholds (downgrade if performance drops)
    REFINEMENT_TO_LEARNING = {
        'avg_reward': 0.15,           # If reward drops below this
        'collision_rate': 0.35        # Or collision rate exceeds this
    }
    
    LEARNING_TO_EXPLORATION = {
        'avg_reward': 0.05,           # If reward drops below this
        'collision_rate': 0.50        # Or collision rate exceeds this
    }
    
    def __init__(self, initial_phase: str = 'exploration'):
        """Initialize phase manager.
        
        Args:
            initial_phase: Starting phase name
        """
        if initial_phase not in self.PHASES:
            raise ValueError(f"Invalid phase: {initial_phase}")
        
        self._phase = initial_phase
        self._phase_entry_step = 0
        self._phase_durations = {p: 0 for p in self.PHASES}
        
        # Metrics window for evaluation episodes
        self._eval_window = deque(maxlen=20)
        
        # Training episode tracking (for collision rate, etc.)
        self._training_window = deque(maxlen=50)
        
        # Statistics
        self._total_transitions = 0
        self._phase_history = []
    
    @property
    def phase(self) -> str:
        """Get current phase name."""
        return self._phase
    
    @property
    def phase_index(self) -> int:
        """Get numeric phase index (0=exploration, 1=learning, 2=refinement)."""
        return self.PHASES.index(self._phase)
    
    def record_eval_episode(self, reward: float, collided: bool, length: int):
        """Record an evaluation episode result.
        
        Args:
            reward: Total episode reward
            collided: Whether episode ended in collision
            length: Number of steps in episode
        """
        self._eval_window.append({
            'reward': reward,
            'collided': collided,
            'length': length
        })
    
    def record_training_episode(self, reward: float, collided: bool, length: int):
        """Record a training episode result.
        
        Args:
            reward: Total episode reward
            collided: Whether episode ended in collision
            length: Number of steps in episode
        """
        self._training_window.append({
            'reward': reward,
            'collided': collided,
            'length': length
        })
    
    def get_metrics(self) -> Dict:
        """Get current performance metrics.
        
        Returns:
            Dictionary with current metrics
        """
        if len(self._eval_window) == 0:
            return {
                'avg_reward': 0.0,
                'collision_rate': 1.0,
                'avg_length': 0,
                'reward_std': 0.0,
                'sample_count': 0
            }
        
        metrics = list(self._eval_window)
        rewards = [m['reward'] for m in metrics]
        
        return {
            'avg_reward': np.mean(rewards),
            'collision_rate': np.mean([m['collided'] for m in metrics]),
            'avg_length': int(np.mean([m['length'] for m in metrics])),
            'reward_std': np.std(rewards),
            'sample_count': len(metrics)
        }
    
    def check_transition(self, current_step: int = 0) -> Optional[str]:
        """Check if phase transition is warranted.
        
        Args:
            current_step: Current training step (for logging)
            
        Returns:
            New phase name if transition occurs, None otherwise
        """
        metrics = self.get_metrics()
        
        if metrics['sample_count'] < 10:
            return None
        
        # Check for upgrade (next phase)
        if self._phase == 'exploration':
            if self._meets_exploration_to_learning(metrics):
                new_phase = 'learning'
                self._log_transition('exploration', 'learning', metrics, current_step)
                return new_phase
                
        elif self._phase == 'learning':
            if self._meets_learning_to_refinement(metrics):
                new_phase = 'refinement'
                self._log_transition('learning', 'refinement', metrics, current_step)
                return new_phase
        
        # Check for regression (previous phase)
        if self._phase == 'refinement':
            if self._meets_refinement_to_learning(metrics):
                new_phase = 'learning'
                self._log_regression('refinement', 'learning', metrics, current_step)
                return new_phase
                
        elif self._phase == 'learning':
            if self._meets_learning_to_exploration(metrics):
                new_phase = 'exploration'
                self._log_regression('learning', 'exploration', metrics, current_step)
                return new_phase
        
        return None
    
    def apply_transition(self, new_phase: str, current_step: int):
        """Apply a phase transition.
        
        Args:
            new_phase: New phase name
            current_step: Current training step
        """
        old_phase = self._phase
        
        # Update phase duration tracking
        if old_phase in self._phase_durations:
            self._phase_durations[old_phase] += (current_step - self._phase_entry_step)
        
        # Apply transition
        self._phase = new_phase
        self._phase_entry_step = current_step
        self._total_transitions += 1
        
        # Record in history
        self._phase_history.append({
            'from': old_phase,
            'to': new_phase,
            'step': current_step,
            'duration': current_step - self._phase_entry_step
        })
    
    def get_phase_config(self) -> Dict:
        """Get configuration for current phase.
        
        Returns:
            Dictionary with phase-specific configuration
        """
        configs = {
            'exploration': {
                'alive_bonus': 0.08,
                'alive_penalty': 0.05,
                'forward_bonus_mult': 1.5,
                'spin_penalty_scale': 0.3,
                'intent_reward_scale': 0.12,
                'smoothness_mult': 0.02,
                'exploration_bonus_magnitude': 0.20,
                'diversity_bonus_scale': 0.05,
                'wall_avoidance_base': 0.20,
                'noise_scale': 0.15,
                'eval_frequency': 5,  # Every 5 episodes
            },
            'learning': {
                'alive_bonus': 0.04,
                'alive_penalty': 0.08,
                'forward_bonus_mult': 1.0,
                'spin_penalty_scale': 0.6,
                'intent_reward_scale': 0.04,
                'smoothness_mult': 0.03,
                'exploration_bonus_magnitude': 0.12,
                'diversity_bonus_scale': 0.05,
                'wall_avoidance_base': 0.25,
                'noise_scale': 0.10,
                'eval_frequency': 5,
            },
            'refinement': {
                'alive_bonus': 0.0,
                'alive_penalty': 0.10,
                'forward_bonus_mult': 0.8,
                'spin_penalty_scale': 1.0,
                'intent_reward_scale': 0.0,
                'smoothness_mult': 0.05,
                'exploration_bonus_magnitude': 0.10,
                'diversity_bonus_scale': 0.05,
                'wall_avoidance_base': 0.30,
                'noise_scale': 0.0,  # Deterministic
                'eval_frequency': 5,
            }
        }
        
        return configs[self._phase]
    
    def get_phase_config_value(self, config_key: str, default=None):
        """Get a specific configuration value for current phase.
        
        Args:
            config_key: Configuration key to look up
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self.get_phase_config().get(config_key, default)
    
    def _meets_exploration_to_learning(self, metrics: Dict) -> bool:
        """Check if metrics meet exploration→learning criteria."""
        thresh = self.EXPLORATION_TO_LEARNING
        return (
            metrics['avg_reward'] > thresh['avg_reward'] and
            metrics['collision_rate'] < thresh['collision_rate'] and
            metrics['avg_length'] > thresh['avg_length']
        )
    
    def _meets_learning_to_refinement(self, metrics: Dict) -> bool:
        """Check if metrics meet learning→refinement criteria."""
        thresh = self.LEARNING_TO_REFINEMENT
        return (
            metrics['avg_reward'] > thresh['avg_reward'] and
            metrics['collision_rate'] < thresh['collision_rate'] and
            metrics['avg_length'] > thresh['avg_length'] and
            metrics['reward_std'] < thresh['reward_std']
        )
    
    def _meets_refinement_to_learning(self, metrics: Dict) -> bool:
        """Check if metrics indicate refinement→learning regression."""
        thresh = self.REFINEMENT_TO_LEARNING
        return (
            metrics['avg_reward'] < thresh['avg_reward'] or
            metrics['collision_rate'] > thresh['collision_rate']
        )
    
    def _meets_learning_to_exploration(self, metrics: Dict) -> bool:
        """Check if metrics indicate learning→exploration regression."""
        thresh = self.LEARNING_TO_EXPLORATION
        return (
            metrics['avg_reward'] < thresh['avg_reward'] or
            metrics['collision_rate'] > thresh['collision_rate']
        )
    
    def _log_transition(self, from_phase: str, to_phase: str, 
                        metrics: Dict, step: int):
        """Log a phase upgrade transition."""
        print(f"⭐ PHASE TRANSITION: {from_phase} → {to_phase}")
        print(f"   Step: {step}")
        print(f"   Metrics: avg_reward={metrics['avg_reward']:.3f}, "
              f"collision_rate={metrics['collision_rate']:.3f}, "
              f"avg_length={metrics['avg_length']}")
    
    def _log_regression(self, from_phase: str, to_phase: str,
                        metrics: Dict, step: int):
        """Log a phase regression."""
        print(f"📉 PHASE REGRESSION: {from_phase} → {to_phase}")
        print(f"   Step: {step}")
        print(f"   Metrics: avg_reward={metrics['avg_reward']:.3f}, "
              f"collision_rate={metrics['collision_rate']:.3f}, "
              f"avg_length={metrics['avg_length']}")
        print(f"   ⚠️ Performance degraded, reverting to earlier phase")
    
    def get_statistics(self) -> Dict:
        """Get phase management statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'current_phase': self._phase,
            'total_transitions': self._total_transitions,
            'phase_durations': self._phase_durations.copy(),
            'phase_history': self._phase_history.copy(),
            'metrics': self.get_metrics()
        }
    
    def reset(self):
        """Reset phase manager to initial state."""
        self._phase = 'exploration'
        self._phase_entry_step = 0
        self._phase_durations = {p: 0 for p in self.PHASES}
        self._eval_window.clear()
        self._training_window.clear()
        self._total_transitions = 0
        self._phase_history.clear()