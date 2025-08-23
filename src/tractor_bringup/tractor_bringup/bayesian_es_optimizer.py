#!/usr/bin/env python3
"""
Bayesian Optimization wrapper for Evolutionary Strategy hyperparameter tuning
Uses BoTorch to intelligently optimize ES parameters like sigma, evolution frequency, etc.
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
    print("BoTorch successfully imported")
except ImportError as e:
    BOTORCH_AVAILABLE = False
    print(f"BoTorch not available: {e}")
    print("Install with: pip install botorch")

@dataclass
class ESHyperparameters:
    """Container for ES hyperparameters that will be optimized"""
    sigma: float = 0.1                    # Mutation strength
    evolution_frequency: int = 50         # Steps between evolutions  
    sigma_decay_rate: float = 0.99        # Rate of sigma decay when improving
    sigma_increase_rate: float = 1.05     # Rate of sigma increase when stagnating
    elite_ratio: float = 0.2              # Fraction of population to preserve as elites
    diversity_injection_ratio: float = 0.25  # Fraction to replace when diversity low

@dataclass
class TrainingHyperparameters:
    """Container for PyTorch training hyperparameters that will be optimized"""
    learning_rate: float = 0.001          # Adam optimizer learning rate
    batch_size: int = 32                  # Training batch size
    prioritized_replay_alpha: float = 0.6  # Priority strength for experience replay
    prioritized_replay_beta: float = 0.4   # Importance sampling correction
    reward_clip_min: float = -30.0        # Minimum reward clipping
    reward_clip_max: float = 30.0         # Maximum reward clipping
    dropout_rate: float = 0.3             # Dropout rate in neural network
    weight_decay: float = 1e-5            # L2 regularization strength
    gradient_clip_norm: float = 1.0       # Gradient clipping threshold

@dataclass  
class CombinedHyperparameters:
    """Container combining ES and training hyperparameters"""
    es_params: ESHyperparameters
    training_params: TrainingHyperparameters

class BayesianESOptimizer:
    """
    Bayesian optimization wrapper for ES hyperparameter tuning.
    
    Uses BoTorch to model the relationship between ES hyperparameters and fitness outcomes,
    then suggests optimal hyperparameter combinations for the next ES generation.
    """
    
    def __init__(self, 
                 initial_params: Optional[ESHyperparameters] = None,
                 optimization_steps: int = 20,
                 enable_debug: bool = False):
        
        if not BOTORCH_AVAILABLE:
            raise ImportError("BoTorch is required for Bayesian ES optimization. Install with: pip install botorch")
        
        self.enable_debug = enable_debug
        self.optimization_steps = optimization_steps
        
        # Define hyperparameter bounds (min, max) for optimization
        self.param_bounds = {
            'sigma': (0.01, 0.5),
            'evolution_frequency': (10, 200),
            'sigma_decay_rate': (0.95, 0.999),
            'sigma_increase_rate': (1.01, 1.2),
            'elite_ratio': (0.1, 0.4),
            'diversity_injection_ratio': (0.1, 0.5)
        }
        
        # Parameter names in optimization order
        self.param_names = list(self.param_bounds.keys())
        self.n_params = len(self.param_names)
        
        # Convert bounds to torch tensors for BoTorch
        bounds_array = np.array([self.param_bounds[name] for name in self.param_names])
        self.bounds = torch.tensor(bounds_array.T, dtype=torch.float64)
        
        # Initialize with provided parameters or defaults
        self.current_params = initial_params or ESHyperparameters()
        
        # Storage for optimization history
        self.param_history: List[torch.Tensor] = []
        self.fitness_history: List[float] = []
        
        # BoTorch model components
        self.gp_model: Optional[SingleTaskGP] = None
        self.mll = None
        
        # Acquisition function parameters
        self.acquisition_restarts = 10
        self.acquisition_raw_samples = 512
        
        if self.enable_debug:
            print(f"[BayesianES] Initialized with {self.n_params} parameters to optimize")
            print(f"[BayesianES] Parameter bounds: {self.param_bounds}")
    
    def _params_to_tensor(self, params: ESHyperparameters) -> torch.Tensor:
        """Convert ESHyperparameters to normalized tensor for BoTorch"""
        values = []
        for name in self.param_names:
            value = getattr(params, name)
            min_val, max_val = self.param_bounds[name]
            # Normalize to [0, 1]
            normalized = (value - min_val) / (max_val - min_val)
            values.append(normalized)
        
        return torch.tensor(values, dtype=torch.float64)
    
    def _tensor_to_params(self, tensor: torch.Tensor) -> ESHyperparameters:
        """Convert normalized tensor back to ESHyperparameters"""
        params_dict = {}
        
        for i, name in enumerate(self.param_names):
            normalized_val = tensor[i].item()
            min_val, max_val = self.param_bounds[name]
            # Denormalize from [0, 1]
            actual_val = normalized_val * (max_val - min_val) + min_val
            
            # Handle integer parameters
            if name == 'evolution_frequency':
                actual_val = int(round(actual_val))
            
            params_dict[name] = actual_val
        
        return ESHyperparameters(**params_dict)
    
    def update_fitness(self, params: ESHyperparameters, fitness_score: float):
        """Update the Bayesian optimization model with new fitness observation"""
        
        # Convert parameters to normalized tensor
        param_tensor = self._params_to_tensor(params).unsqueeze(0)  # Add batch dimension
        
        # Store the observation
        self.param_history.append(param_tensor.squeeze(0))
        self.fitness_history.append(fitness_score)
        
        if self.enable_debug:
            print(f"[BayesianES] Updated with fitness {fitness_score:.4f}, total observations: {len(self.fitness_history)}")
        
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
                    print(f"[BayesianES] Updated GP model with {len(self.fitness_history)} observations")
                    
        except Exception as e:
            if self.enable_debug:
                print(f"[BayesianES] Failed to update GP model: {e}")
            self.gp_model = None
            self.mll = None
    
    def suggest_next_params(self, use_ucb: bool = True) -> ESHyperparameters:
        """
        Suggest next hyperparameters to try using Bayesian optimization.
        
        Args:
            use_ucb: If True, use Upper Confidence Bound acquisition function.
                    If False, use Expected Improvement.
        
        Returns:
            ESHyperparameters object with suggested values
        """
        
        # If we don't have enough data or model failed, use random exploration
        if self.gp_model is None or len(self.fitness_history) < 2:
            return self._random_exploration()
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Choose acquisition function
                if use_ucb:
                    # UCB balances exploration and exploitation
                    acquisition_func = UpperConfidenceBound(self.gp_model, beta=2.0)
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
                    print(f"[BayesianES] Suggested parameters: {suggested_params}")
                
                return suggested_params
                
        except Exception as e:
            if self.enable_debug:
                print(f"[BayesianES] Acquisition optimization failed: {e}, using random exploration")
            return self._random_exploration()
    
    def _random_exploration(self) -> ESHyperparameters:
        """Generate random hyperparameters within bounds for exploration"""
        
        params_dict = {}
        for name in self.param_names:
            min_val, max_val = self.param_bounds[name]
            
            if name == 'evolution_frequency':
                # Integer parameter
                value = np.random.randint(int(min_val), int(max_val) + 1)
            else:
                # Float parameter
                value = np.random.uniform(min_val, max_val)
            
            params_dict[name] = value
        
        random_params = ESHyperparameters(**params_dict)
        
        if self.enable_debug:
            print(f"[BayesianES] Random exploration: {random_params}")
        
        return random_params
    
    def get_best_params(self) -> Tuple[ESHyperparameters, float]:
        """
        Return the hyperparameters that achieved the best fitness so far.
        
        Returns:
            Tuple of (best_params, best_fitness)
        """
        
        if not self.fitness_history:
            return self.current_params, -float('inf')
        
        best_idx = np.argmax(self.fitness_history)
        best_param_tensor = self.param_history[best_idx]
        best_params = self._tensor_to_params(best_param_tensor)
        best_fitness = self.fitness_history[best_idx]
        
        return best_params, best_fitness
    
    def get_optimization_stats(self) -> Dict[str, float]:
        """Get statistics about the optimization process"""
        
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
        
        # Calculate improvement (compare first 25% to last 25%)
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
                print(f"[BayesianES] Saved optimization state to {filepath}")
                
        except Exception as e:
            if self.enable_debug:
                print(f"[BayesianES] Failed to save state: {e}")
    
    def load_state(self, filepath: str):
        """Load optimization state from file"""
        try:
            state = torch.load(filepath, map_location='cpu')
            
            self.param_history = [torch.tensor(p, dtype=torch.float64) for p in state['param_history']]
            self.fitness_history = state['fitness_history']
            self.current_params = ESHyperparameters(**state['current_params'])
            
            # Rebuild GP model if we have data
            if len(self.fitness_history) > 0:
                self._update_gp_model()
            
            if self.enable_debug:
                print(f"[BayesianES] Loaded optimization state from {filepath}")
                print(f"[BayesianES] Loaded {len(self.fitness_history)} evaluations")
                
        except Exception as e:
            if self.enable_debug:
                print(f"[BayesianES] Failed to load state: {e}")


class AdaptiveBayesianESWrapper:
    """
    Wrapper that integrates BayesianESOptimizer with the existing ES trainer.
    
    This class handles the scheduling and application of Bayesian-optimized hyperparameters
    to the ES training process.
    """
    
    def __init__(self, 
                 es_trainer,  # EvolutionaryStrategyTrainer instance
                 optimization_interval: int = 5,  # Optimize every N generations
                 min_observations: int = 3,       # Minimum data before optimization
                 enable_debug: bool = False):
        
        if not BOTORCH_AVAILABLE:
            print("[AdaptiveBayesianES] BoTorch not available - using default ES parameters")
            self.bayesian_optimizer = None
            self.es_trainer = es_trainer
            self.enable_debug = enable_debug
            return
        
        self.es_trainer = es_trainer
        self.optimization_interval = optimization_interval
        self.min_observations = min_observations
        self.enable_debug = enable_debug
        
        # Initialize Bayesian optimizer with current ES parameters
        current_params = ESHyperparameters(
            sigma=es_trainer.sigma,
            evolution_frequency=es_trainer.current_evolution_frequency,
            sigma_decay_rate=es_trainer.sigma_decay_rate,
            sigma_increase_rate=es_trainer.sigma_increase_rate,
            elite_ratio=es_trainer.elite_ratio,
            diversity_injection_ratio=es_trainer.diversity_injection_ratio
        )
        
        self.bayesian_optimizer = BayesianESOptimizer(
            initial_params=current_params,
            enable_debug=enable_debug
        )
        
        # Track when to apply new parameters
        self.last_optimization_generation = 0
        self.current_suggested_params = current_params
        
        if self.enable_debug:
            print(f"[AdaptiveBayesianES] Initialized with optimization every {optimization_interval} generations")
    
    def should_optimize_hyperparams(self, generation: int) -> bool:
        """Check if it's time to optimize hyperparameters"""
        if self.bayesian_optimizer is None:
            return False
        
        generations_since_last = generation - self.last_optimization_generation
        has_enough_data = len(self.bayesian_optimizer.fitness_history) >= self.min_observations
        
        return generations_since_last >= self.optimization_interval and has_enough_data
    
    def apply_bayesian_optimization(self, generation: int, recent_fitness: float):
        """Apply Bayesian optimization to suggest new ES hyperparameters"""
        if self.bayesian_optimizer is None:
            return
        
        # Update Bayesian optimizer with recent performance
        self.bayesian_optimizer.update_fitness(self.current_suggested_params, recent_fitness)
        
        # Get new hyperparameter suggestions
        if self.should_optimize_hyperparams(generation):
            new_params = self.bayesian_optimizer.suggest_next_params(use_ucb=True)
            self._apply_params_to_es_trainer(new_params)
            self.current_suggested_params = new_params
            self.last_optimization_generation = generation
            
            if self.enable_debug:
                print(f"[AdaptiveBayesianES] Applied new parameters at generation {generation}")
                print(f"[AdaptiveBayesianES] New params: sigma={new_params.sigma:.4f}, "
                      f"freq={new_params.evolution_frequency}, "
                      f"decay={new_params.sigma_decay_rate:.4f}")
    
    def _apply_params_to_es_trainer(self, params: ESHyperparameters):
        """Apply optimized hyperparameters to the ES trainer"""
        
        # Apply parameters to ES trainer
        self.es_trainer.sigma = params.sigma
        self.es_trainer.current_evolution_frequency = params.evolution_frequency
        self.es_trainer.sigma_decay_rate = params.sigma_decay_rate
        self.es_trainer.sigma_increase_rate = params.sigma_increase_rate
        self.es_trainer.elite_ratio = params.elite_ratio
        self.es_trainer.diversity_injection_ratio = params.diversity_injection_ratio
        
        # Recalculate dependent parameters
        self.es_trainer.n_elites = max(1, int(self.es_trainer.population_size * params.elite_ratio))
        
        # Update sigma bounds based on new parameters
        self.es_trainer.min_sigma = params.sigma * 0.1
        self.es_trainer.max_sigma = params.sigma * 3.0
    
    def get_optimization_summary(self) -> Dict[str, float]:
        """Get summary of Bayesian optimization performance"""
        if self.bayesian_optimizer is None:
            return {"bayesian_optimization": "disabled"}
        
        stats = self.bayesian_optimizer.get_optimization_stats()
        best_params, best_fitness = self.bayesian_optimizer.get_best_params()
        
        return {
            **stats,
            "best_sigma": best_params.sigma,
            "best_evolution_frequency": float(best_params.evolution_frequency),
            "best_elite_ratio": best_params.elite_ratio,
            "generations_since_optimization": self.es_trainer.generation - self.last_optimization_generation
        }


class BayesianTrainingOptimizer:
    """
    Bayesian optimization for PyTorch training hyperparameters.
    
    Optimizes learning rate, batch size, replay parameters, and regularization settings
    for improved training performance and stability.
    """
    
    def __init__(self, 
                 initial_params: Optional[TrainingHyperparameters] = None,
                 optimization_steps: int = 15,
                 enable_debug: bool = False):
        
        if not BOTORCH_AVAILABLE:
            raise ImportError("BoTorch is required for Bayesian training optimization. Install with: pip install botorch")
        
        self.enable_debug = enable_debug
        self.optimization_steps = optimization_steps
        
        # Define training hyperparameter bounds (min, max) for optimization
        self.param_bounds = {
            'learning_rate': (1e-5, 1e-2),
            'batch_size': (8, 128),
            'prioritized_replay_alpha': (0.4, 0.8),
            'prioritized_replay_beta': (0.2, 1.0),
            'reward_clip_min': (-50.0, -10.0),
            'reward_clip_max': (10.0, 50.0),
            'dropout_rate': (0.1, 0.5),
            'weight_decay': (1e-6, 1e-3),
            'gradient_clip_norm': (0.5, 2.0)
        }
        
        # Parameter names in optimization order
        self.param_names = list(self.param_bounds.keys())
        self.n_params = len(self.param_names)
        
        # Convert bounds to torch tensors for BoTorch
        bounds_array = np.array([self.param_bounds[name] for name in self.param_names])
        self.bounds = torch.tensor(bounds_array.T, dtype=torch.float64)
        
        # Initialize with provided parameters or defaults
        self.current_params = initial_params or TrainingHyperparameters()
        
        # Storage for optimization history
        self.param_history: List[torch.Tensor] = []
        self.fitness_history: List[float] = []
        
        # BoTorch model components
        self.gp_model: Optional[SingleTaskGP] = None
        self.mll = None
        
        # Acquisition function parameters
        self.acquisition_restarts = 8
        self.acquisition_raw_samples = 256
        
        if self.enable_debug:
            print(f"[BayesianTraining] Initialized with {self.n_params} parameters to optimize")
            print(f"[BayesianTraining] Parameter bounds: {self.param_bounds}")
    
    def _params_to_tensor(self, params: TrainingHyperparameters) -> torch.Tensor:
        """Convert TrainingHyperparameters to normalized tensor for BoTorch"""
        values = []
        for name in self.param_names:
            value = getattr(params, name)
            min_val, max_val = self.param_bounds[name]
            # Normalize to [0, 1]
            normalized = (value - min_val) / (max_val - min_val)
            values.append(normalized)
        
        return torch.tensor(values, dtype=torch.float64)
    
    def _tensor_to_params(self, tensor: torch.Tensor) -> TrainingHyperparameters:
        """Convert normalized tensor back to TrainingHyperparameters"""
        params_dict = {}
        
        for i, name in enumerate(self.param_names):
            normalized_val = tensor[i].item()
            min_val, max_val = self.param_bounds[name]
            # Denormalize from [0, 1]
            actual_val = normalized_val * (max_val - min_val) + min_val
            
            # Handle integer and special parameters
            if name == 'batch_size':
                actual_val = int(round(actual_val))
                # Ensure batch size is power of 2 for efficiency
                actual_val = max(8, 2 ** round(np.log2(actual_val)))
            elif name in ['learning_rate', 'weight_decay']:
                # Use log scale for learning rate and weight decay
                actual_val = 10 ** (np.log10(min_val) + normalized_val * (np.log10(max_val) - np.log10(min_val)))
            
            params_dict[name] = actual_val
        
        return TrainingHyperparameters(**params_dict)
    
    def update_fitness(self, params: TrainingHyperparameters, fitness_score: float):
        """Update the Bayesian optimization model with new fitness observation"""
        
        # Convert parameters to normalized tensor
        param_tensor = self._params_to_tensor(params).unsqueeze(0)  # Add batch dimension
        
        # Store the observation
        self.param_history.append(param_tensor.squeeze(0))
        self.fitness_history.append(fitness_score)
        
        if self.enable_debug:
            print(f"[BayesianTraining] Updated with fitness {fitness_score:.4f}, total observations: {len(self.fitness_history)}")
        
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
                    print(f"[BayesianTraining] Updated GP model with {len(self.fitness_history)} observations")
                    
        except Exception as e:
            if self.enable_debug:
                print(f"[BayesianTraining] Failed to update GP model: {e}")
            self.gp_model = None
            self.mll = None
    
    def suggest_next_params(self, use_ucb: bool = True) -> TrainingHyperparameters:
        """Suggest next training hyperparameters using Bayesian optimization"""
        
        # If we don't have enough data or model failed, use random exploration
        if self.gp_model is None or len(self.fitness_history) < 2:
            return self._random_exploration()
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Choose acquisition function
                if use_ucb:
                    # UCB balances exploration and exploitation
                    acquisition_func = UpperConfidenceBound(self.gp_model, beta=2.0)
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
                    print(f"[BayesianTraining] Suggested parameters: {suggested_params}")
                
                return suggested_params
                
        except Exception as e:
            if self.enable_debug:
                print(f"[BayesianTraining] Acquisition optimization failed: {e}, using random exploration")
            return self._random_exploration()
    
    def _random_exploration(self) -> TrainingHyperparameters:
        """Generate random training hyperparameters within bounds for exploration"""
        
        params_dict = {}
        for name in self.param_names:
            min_val, max_val = self.param_bounds[name]
            
            if name == 'batch_size':
                # Integer parameter - prefer powers of 2
                log_min, log_max = np.log2(min_val), np.log2(max_val)
                log_val = np.random.uniform(log_min, log_max)
                value = max(8, 2 ** round(log_val))
            elif name in ['learning_rate', 'weight_decay']:
                # Log scale parameters
                log_min, log_max = np.log10(min_val), np.log10(max_val)
                log_val = np.random.uniform(log_min, log_max)
                value = 10 ** log_val
            else:
                # Linear scale parameters
                value = np.random.uniform(min_val, max_val)
            
            params_dict[name] = value
        
        random_params = TrainingHyperparameters(**params_dict)
        
        if self.enable_debug:
            print(f"[BayesianTraining] Random exploration: {random_params}")
        
        return random_params
    
    def get_best_params(self) -> Tuple[TrainingHyperparameters, float]:
        """Return the training hyperparameters that achieved the best fitness"""
        
        if not self.fitness_history:
            return self.current_params, -float('inf')
        
        best_idx = np.argmax(self.fitness_history)
        best_param_tensor = self.param_history[best_idx]
        best_params = self._tensor_to_params(best_param_tensor)
        best_fitness = self.fitness_history[best_idx]
        
        return best_params, best_fitness
    
    def get_optimization_stats(self) -> Dict[str, float]:
        """Get statistics about the training optimization process"""
        
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
        if n_evals >= 6:
            early_avg = np.mean(self.fitness_history[:n_evals//3])
            recent_avg = np.mean(self.fitness_history[-n_evals//3:])
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