#!/usr/bin/env python3
"""
Bayesian Optimization for Reward System Parameters
Optimizes reward weights and anti-overtraining parameters for improved learning dynamics
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

try:
    import botorch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood
    BOTORCH_AVAILABLE = True
    print("BoTorch successfully imported for reward optimization")
except ImportError as e:
    BOTORCH_AVAILABLE = False
    print(f"BoTorch not available for reward optimization: {e}")

@dataclass
class RewardHyperparameters:
    """Container for reward system hyperparameters that will be optimized"""
    # Core reward weights
    base_movement_reward: float = 5.0
    forward_progress_bonus: float = 8.0
    exploration_bonus: float = 10.0
    collision_penalty: float = -20.0
    near_collision_penalty: float = -5.0
    smooth_movement_bonus: float = 1.0
    goal_oriented_bonus: float = 5.0
    stagnation_penalty: float = -2.0
    
    # Anti-overtraining parameters
    reward_noise_std: float = 0.1
    reward_smoothing_alpha: float = 0.1
    max_area_revisit_bonus: int = 3
    spinning_threshold: float = 0.5
    
    # Behavioral parameters
    forward_movement_weight: float = 2.0
    angular_movement_penalty: float = 0.5
    backward_movement_penalty: float = 0.2

class BayesianRewardOptimizer:
    """
    Bayesian optimization for reward system hyperparameters.
    
    Uses BoTorch to optimize reward weights and anti-overtraining parameters
    for improved learning performance and behavioral diversity.
    """
    
    def __init__(self, 
                 initial_params: Optional[RewardHyperparameters] = None,
                 optimization_steps: int = 25,
                 enable_debug: bool = False):
        
        if not BOTORCH_AVAILABLE:
            raise ImportError("BoTorch is required for Bayesian reward optimization. Install with: pip install botorch")
        
        self.enable_debug = enable_debug
        self.optimization_steps = optimization_steps
        
        # Define reward hyperparameter bounds (min, max) for optimization
        self.param_bounds = {
            # Core reward weights
            'base_movement_reward': (1.0, 15.0),
            'forward_progress_bonus': (3.0, 20.0),
            'exploration_bonus': (5.0, 25.0),
            'collision_penalty': (-50.0, -5.0),
            'near_collision_penalty': (-15.0, -1.0),
            'smooth_movement_bonus': (0.5, 5.0),
            'goal_oriented_bonus': (1.0, 15.0),
            'stagnation_penalty': (-10.0, -0.5),
            
            # Anti-overtraining parameters
            'reward_noise_std': (0.01, 0.5),
            'reward_smoothing_alpha': (0.05, 0.3),
            'max_area_revisit_bonus': (1, 8),
            'spinning_threshold': (0.2, 1.0),
            
            # Behavioral parameters
            'forward_movement_weight': (1.0, 4.0),
            'angular_movement_penalty': (0.1, 1.0),
            'backward_movement_penalty': (0.05, 0.8)
        }
        
        # Parameter names in optimization order
        self.param_names = list(self.param_bounds.keys())
        self.n_params = len(self.param_names)
        
        # Convert bounds to torch tensors for BoTorch
        bounds_array = np.array([self.param_bounds[name] for name in self.param_names])
        self.bounds = torch.tensor(bounds_array.T, dtype=torch.float64)
        
        # Initialize with provided parameters or defaults
        self.current_params = initial_params or RewardHyperparameters()
        
        # Storage for optimization history
        self.param_history: List[torch.Tensor] = []
        self.fitness_history: List[float] = []
        
        # BoTorch model components
        self.gp_model: Optional[SingleTaskGP] = None
        self.mll = None
        
        # Acquisition function parameters
        self.acquisition_restarts = 12
        self.acquisition_raw_samples = 384
        
        if self.enable_debug:
            print(f"[BayesianReward] Initialized with {self.n_params} parameters to optimize")
            print(f"[BayesianReward] Parameter bounds: {self.param_bounds}")
    
    def _params_to_tensor(self, params: RewardHyperparameters) -> torch.Tensor:
        """Convert RewardHyperparameters to normalized tensor for BoTorch"""
        values = []
        for name in self.param_names:
            value = getattr(params, name)
            min_val, max_val = self.param_bounds[name]
            # Normalize to [0, 1]
            normalized = (value - min_val) / (max_val - min_val)
            values.append(normalized)
        
        return torch.tensor(values, dtype=torch.float64)
    
    def _tensor_to_params(self, tensor: torch.Tensor) -> RewardHyperparameters:
        """Convert normalized tensor back to RewardHyperparameters"""
        params_dict = {}
        
        for i, name in enumerate(self.param_names):
            normalized_val = tensor[i].item()
            min_val, max_val = self.param_bounds[name]
            # Denormalize from [0, 1]
            actual_val = normalized_val * (max_val - min_val) + min_val
            
            # Handle integer parameters
            if name == 'max_area_revisit_bonus':
                actual_val = int(round(actual_val))
            
            params_dict[name] = actual_val
        
        return RewardHyperparameters(**params_dict)
    
    def update_fitness(self, params: RewardHyperparameters, fitness_score: float):
        """Update the Bayesian optimization model with new fitness observation"""
        
        # Convert parameters to normalized tensor
        param_tensor = self._params_to_tensor(params).unsqueeze(0)  # Add batch dimension
        
        # Store the observation
        self.param_history.append(param_tensor.squeeze(0))
        self.fitness_history.append(fitness_score)
        
        if self.enable_debug:
            print(f"[BayesianReward] Updated with fitness {fitness_score:.4f}, total observations: {len(self.fitness_history)}")
        
        # Rebuild GP model with new data
        self._update_gp_model()
    
    def _update_gp_model(self):
        """Update the Gaussian Process model with current observations"""
        
        if len(self.param_history) == 0:
            return
        
        # Stack observations into tensors
        X = torch.stack(self.param_history)  # Shape: (n_obs, n_params)
        y = torch.tensor(self.fitness_history, dtype=torch.float64).unsqueeze(-1)  # Shape: (n_obs, 1)
        
        try:
            # Suppress BoTorch warnings about small datasets
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Create and fit GP model
                self.gp_model = SingleTaskGP(X, y)
                self.mll = ExactMarginalLogLikelihood(self.gp_model.likelihood, self.gp_model)
                fit_gpytorch_mll(self.mll)
                
                if self.enable_debug:
                    print(f"[BayesianReward] Updated GP model with {len(self.fitness_history)} observations")
                    
        except Exception as e:
            if self.enable_debug:
                print(f"[BayesianReward] Failed to update GP model: {e}")
            self.gp_model = None
            self.mll = None
    
    def suggest_next_params(self, use_ucb: bool = True) -> RewardHyperparameters:
        """Suggest next reward hyperparameters using Bayesian optimization"""
        
        # If we don't have enough data or model failed, use random exploration
        if self.gp_model is None or len(self.fitness_history) < 3:
            return self._random_exploration()
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Choose acquisition function
                if use_ucb:
                    # UCB balances exploration and exploitation
                    acquisition_func = UpperConfidenceBound(self.gp_model, beta=2.5)
                else:
                    # Expected Improvement
                    best_f = torch.tensor(max(self.fitness_history), dtype=torch.float64)
                    acquisition_func = ExpectedImprovement(self.gp_model, best_f=best_f)
                
                # Optimize acquisition function to find next point
                candidates, _ = optimize_acqf(
                    acquisition_func,
                    bounds=self.bounds,
                    q=1,  # Number of candidates
                    num_restarts=self.acquisition_restarts,
                    raw_samples=self.acquisition_raw_samples,
                )
                
                # Clamp candidates to bounds and convert back to hyperparameters  
                clamped_candidates = torch.clamp(candidates.squeeze(0), 0.0, 1.0)
                suggested_params = self._tensor_to_params(clamped_candidates)
                
                if self.enable_debug:
                    print(f"[BayesianReward] Suggested parameters: {suggested_params}")
                
                return suggested_params
                
        except Exception as e:
            if self.enable_debug:
                print(f"[BayesianReward] Acquisition optimization failed: {e}, using random exploration")
            return self._random_exploration()
    
    def _random_exploration(self) -> RewardHyperparameters:
        """Generate random reward hyperparameters within bounds for exploration"""
        
        params_dict = {}
        for name in self.param_names:
            min_val, max_val = self.param_bounds[name]
            
            if name == 'max_area_revisit_bonus':
                # Integer parameter
                value = np.random.randint(int(min_val), int(max_val) + 1)
            else:
                # Float parameter
                value = np.random.uniform(min_val, max_val)
            
            params_dict[name] = value
        
        random_params = RewardHyperparameters(**params_dict)
        
        if self.enable_debug:
            print(f"[BayesianReward] Random exploration: {random_params}")
        
        return random_params
    
    def get_best_params(self) -> Tuple[RewardHyperparameters, float]:
        """Return the reward hyperparameters that achieved the best fitness"""
        
        if not self.fitness_history:
            return self.current_params, -float('inf')
        
        best_idx = np.argmax(self.fitness_history)
        best_param_tensor = self.param_history[best_idx]
        best_params = self._tensor_to_params(best_param_tensor)
        best_fitness = self.fitness_history[best_idx]
        
        return best_params, best_fitness
    
    def get_optimization_stats(self) -> Dict[str, float]:
        """Get statistics about the reward optimization process"""
        
        if not self.fitness_history:
            return {
                "n_evaluations": 0,
                "best_fitness": -float('inf'),
                "avg_fitness": 0.0,
                "fitness_improvement": 0.0
            }
        
        n_evals = len(self.fitness_history)
        best_fitness = max(self.fitness_history)
        avg_fitness = np.mean(self.fitness_history)
        
        # Calculate improvement
        if n_evals >= 8:
            early_avg = np.mean(self.fitness_history[:n_evals//4])
            recent_avg = np.mean(self.fitness_history[-n_evals//4:])
            fitness_improvement = recent_avg - early_avg
        else:
            fitness_improvement = 0.0
        
        return {
            "n_evaluations": n_evals,
            "best_fitness": float(best_fitness),
            "avg_fitness": float(avg_fitness),
            "fitness_improvement": float(fitness_improvement),
            "model_ready": self.gp_model is not None
        }
    
    def export_config_dict(self, params: RewardHyperparameters) -> Dict:
        """Export reward parameters in the format expected by anti_overtraining_config.py"""
        
        return {
            # Core reward parameters
            'base_movement_reward': params.base_movement_reward,
            'forward_progress_bonus': params.forward_progress_bonus,
            'exploration_bonus': params.exploration_bonus,
            'collision_penalty': params.collision_penalty,
            'near_collision_penalty': params.near_collision_penalty,
            'smooth_movement_bonus': params.smooth_movement_bonus,
            'goal_oriented_bonus': params.goal_oriented_bonus,
            'stagnation_penalty': params.stagnation_penalty,
            
            # Anti-overtraining measures
            'reward_noise_std': params.reward_noise_std,
            'reward_smoothing_alpha': params.reward_smoothing_alpha,
            'max_area_revisit_bonus': params.max_area_revisit_bonus,
            'spinning_threshold': params.spinning_threshold,
            
            # Feature toggles (keep defaults)
            'enable_reward_smoothing': True,
            'enable_anti_gaming': True,
            'enable_diversity_tracking': True,
        }
    
    def save_state(self, filepath: str):
        """Save optimization state to file"""
        try:
            state = {
                'param_history': [p.numpy() for p in self.param_history],
                'fitness_history': self.fitness_history,
                'current_params': self.current_params.__dict__,
                'param_bounds': self.param_bounds,
                'param_names': self.param_names
            }
            
            torch.save(state, filepath)
            
            if self.enable_debug:
                print(f"[BayesianReward] Saved optimization state to {filepath}")
                
        except Exception as e:
            if self.enable_debug:
                print(f"[BayesianReward] Failed to save state: {e}")
    
    def load_state(self, filepath: str):
        """Load optimization state from file"""
        try:
            state = torch.load(filepath, map_location='cpu')
            
            self.param_history = [torch.tensor(p, dtype=torch.float64) for p in state['param_history']]
            self.fitness_history = state['fitness_history']
            self.current_params = RewardHyperparameters(**state['current_params'])
            
            # Rebuild GP model if we have data
            if len(self.fitness_history) > 0:
                self._update_gp_model()
            
            if self.enable_debug:
                print(f"[BayesianReward] Loaded optimization state from {filepath}")
                print(f"[BayesianReward] Loaded {len(self.fitness_history)} evaluations")
                
        except Exception as e:
            if self.enable_debug:
                print(f"[BayesianReward] Failed to load state: {e}")


class AdaptiveBayesianRewardWrapper:
    """
    Wrapper that integrates BayesianRewardOptimizer with the existing reward system.
    
    This class handles the scheduling and application of Bayesian-optimized reward parameters
    to the training process.
    """
    
    def __init__(self, 
                 reward_calculator,  # ImprovedRewardCalculator instance
                 optimization_interval: int = 500,  # Optimize every N training steps
                 min_observations: int = 5,         # Minimum data before optimization
                 enable_debug: bool = False):
        
        if not BOTORCH_AVAILABLE:
            print("[AdaptiveBayesianReward] BoTorch not available - using default reward parameters")
            self.bayesian_optimizer = None
            self.reward_calculator = reward_calculator
            self.enable_debug = enable_debug
            return
        
        self.reward_calculator = reward_calculator
        self.optimization_interval = optimization_interval
        self.min_observations = min_observations
        self.enable_debug = enable_debug
        
        # Initialize Bayesian optimizer with current reward parameters
        # Extract current parameters from the reward calculator
        current_params = RewardHyperparameters(
            # These would need to be extracted from the actual reward calculator
            base_movement_reward=5.0,
            forward_progress_bonus=8.0,
            exploration_bonus=10.0,
            collision_penalty=-20.0,
            # ... etc
        )
        
        self.bayesian_optimizer = BayesianRewardOptimizer(
            initial_params=current_params,
            optimization_steps=25,
            enable_debug=enable_debug
        )
        
        # Track when to apply new parameters
        self.last_optimization_step = 0
        self.current_suggested_params = current_params
        
        if self.enable_debug:
            print(f"[AdaptiveBayesianReward] Initialized with optimization every {optimization_interval} steps")
    
    def should_optimize_reward_params(self, step: int) -> bool:
        """Check if it's time to optimize reward parameters"""
        if self.bayesian_optimizer is None:
            return False
        
        steps_since_last = step - self.last_optimization_step
        has_enough_data = len(self.bayesian_optimizer.fitness_history) >= self.min_observations
        
        return steps_since_last >= self.optimization_interval and has_enough_data
    
    def apply_bayesian_optimization(self, step: int, performance_metrics: Dict[str, float]):
        """Apply Bayesian optimization to suggest new reward parameters"""
        if self.bayesian_optimizer is None:
            return
        
        # Calculate fitness score from performance metrics
        fitness_score = self._calculate_reward_fitness(performance_metrics)
        
        # Update Bayesian optimizer with recent performance
        self.bayesian_optimizer.update_fitness(self.current_suggested_params, fitness_score)
        
        # Get new hyperparameter suggestions
        if self.should_optimize_reward_params(step):
            new_params = self.bayesian_optimizer.suggest_next_params(use_ucb=True)
            self._apply_params_to_reward_system(new_params)
            self.current_suggested_params = new_params
            self.last_optimization_step = step
            
            if self.enable_debug:
                print(f"[AdaptiveBayesianReward] Applied new parameters at step {step}")
                print(f"[AdaptiveBayesianReward] Exploration bonus: {new_params.exploration_bonus:.2f}, "
                      f"Collision penalty: {new_params.collision_penalty:.2f}")
    
    def _calculate_reward_fitness(self, metrics: Dict[str, float]) -> float:
        """Calculate fitness score for reward parameters based on performance metrics"""
        
        # Extract key metrics
        avg_reward = metrics.get('avg_reward', 0.0)
        behavioral_diversity = metrics.get('behavioral_diversity', 0.5)
        collision_rate = metrics.get('collision_rate', 0.1)
        exploration_progress = metrics.get('exploration_progress', 0.0)
        
        # Normalize and combine metrics
        # Higher average reward is better
        reward_component = max(0.0, avg_reward / 30.0)
        
        # Higher behavioral diversity is better (anti-overtraining)  
        diversity_component = behavioral_diversity
        
        # Lower collision rate is better
        safety_component = max(0.0, 1.0 - collision_rate * 2.0)
        
        # Higher exploration progress is better
        exploration_component = max(0.0, exploration_progress)
        
        # Weighted combination
        fitness = (0.3 * reward_component +
                  0.3 * diversity_component + 
                  0.2 * safety_component +
                  0.2 * exploration_component)
        
        return fitness
    
    def _apply_params_to_reward_system(self, params: RewardHyperparameters):
        """Apply optimized reward parameters to the reward calculator"""
        
        try:
            # This would need to be implemented based on the actual reward calculator interface
            if hasattr(self.reward_calculator, 'update_parameters'):
                config_dict = self.bayesian_optimizer.export_config_dict(params)
                self.reward_calculator.update_parameters(config_dict)
            
            if self.enable_debug:
                print(f"[AdaptiveBayesianReward] Applied reward parameters:")
                print(f"  Exploration bonus: {params.exploration_bonus:.2f}")
                print(f"  Collision penalty: {params.collision_penalty:.2f}")
                print(f"  Forward progress bonus: {params.forward_progress_bonus:.2f}")
                
        except Exception as e:
            if self.enable_debug:
                print(f"[AdaptiveBayesianReward] Failed to apply reward parameters: {e}")
    
    def get_optimization_summary(self) -> Dict[str, float]:
        """Get summary of Bayesian reward optimization performance"""
        if self.bayesian_optimizer is None:
            return {"bayesian_reward_optimization": "disabled"}
        
        stats = self.bayesian_optimizer.get_optimization_stats()
        best_params, best_fitness = self.bayesian_optimizer.get_best_params()
        
        return {
            **stats,
            "best_exploration_bonus": best_params.exploration_bonus,
            "best_collision_penalty": best_params.collision_penalty,
            "best_forward_progress_bonus": best_params.forward_progress_bonus,
            "steps_since_optimization": self.last_optimization_step
        }