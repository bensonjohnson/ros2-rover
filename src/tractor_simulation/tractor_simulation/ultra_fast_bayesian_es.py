#!/usr/bin/env python3
"""
Ultra-Fast Bayesian ES Hyperparameter Optimization
Mirrors your rover's AdaptiveBayesianESWrapper for perfect compatibility
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
import logging

try:
    import botorch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.utils.transforms import normalize, unnormalize
    BOTORCH_AVAILABLE = True
except ImportError as e:
    BOTORCH_AVAILABLE = False
    print(f"BoTorch not available for hyperparameter optimization: {e}")

@dataclass
class UltraFastESHyperparameters:
    """ES hyperparameters that will be optimized by Bayesian optimization"""
    sigma: float = 0.1                    # Mutation strength
    learning_rate: float = 0.01           # ES parameter update rate  
    population_size: int = 32             # ES population size
    elite_ratio: float = 0.2              # Fraction of population to preserve as elites
    sigma_decay_rate: float = 0.99        # Rate of sigma decay when improving
    sigma_increase_rate: float = 1.05     # Rate of sigma increase when stagnating
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor for BoTorch"""
        return torch.tensor([
            self.sigma, 
            self.learning_rate,
            float(self.population_size) / 100.0,  # Normalize to 0-1 range
            self.elite_ratio,
            self.sigma_decay_rate,
            self.sigma_increase_rate
        ], dtype=torch.float64)
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'UltraFastESHyperparameters':
        """Create from tensor"""
        return cls(
            sigma=float(tensor[0]),
            learning_rate=float(tensor[1]), 
            population_size=int(tensor[2] * 100),  # Denormalize
            elite_ratio=float(tensor[3]),
            sigma_decay_rate=float(tensor[4]),
            sigma_increase_rate=float(tensor[5])
        )

class UltraFastBayesianESOptimizer:
    """
    Bayesian optimization of ES hyperparameters for ultra-fast training
    Mirrors your rover's implementation for perfect compatibility
    """
    
    def __init__(self, bounds: Optional[torch.Tensor] = None, enable_debug: bool = False):
        self.enable_debug = enable_debug
        self.logger = logging.getLogger(__name__)
        
        # Hyperparameter bounds [lower, upper] for each parameter
        if bounds is None:
            self.bounds = torch.tensor([
                [0.01, 0.5],     # sigma
                [0.001, 0.1],    # learning_rate  
                [0.08, 1.0],     # population_size (normalized)
                [0.1, 0.4],      # elite_ratio
                [0.95, 0.999],   # sigma_decay_rate
                [1.001, 1.1]     # sigma_increase_rate
            ], dtype=torch.float64).T  # Shape: (2, n_params)
        else:
            self.bounds = bounds
            
        # Training data
        self.train_X = torch.empty(0, self.bounds.shape[1], dtype=torch.float64)
        self.train_Y = torch.empty(0, 1, dtype=torch.float64)
        
        # Model
        self.model = None
        
        # Optimization history
        self.optimization_history = []
        self.best_hyperparams = None
        self.best_fitness = -float('inf')
        
    def add_observation(self, hyperparams: UltraFastESHyperparameters, fitness: float):
        """Add a new hyperparameter observation"""
        x = hyperparams.to_tensor().unsqueeze(0)  # (1, n_params)
        y = torch.tensor([[fitness]], dtype=torch.float64)
        
        self.train_X = torch.cat([self.train_X, x], dim=0)
        self.train_Y = torch.cat([self.train_Y, y], dim=0)
        
        # Update best
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_hyperparams = hyperparams
            
        if self.enable_debug:
            self.logger.info(f"Added observation: fitness={fitness:.4f}, total_obs={len(self.train_Y)}")
    
    def fit_model(self):
        """Fit GP model to current observations"""
        if len(self.train_X) < 2:
            return False
            
        try:
            # Normalize inputs
            normalized_X = normalize(self.train_X, bounds=self.bounds)
            
            # Fit GP
            self.model = SingleTaskGP(normalized_X, self.train_Y)
            mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            fit_gpytorch_mll(mll)
            
            return True
            
        except Exception as e:
            if self.enable_debug:
                self.logger.error(f"Model fitting failed: {e}")
            return False
    
    def optimize_hyperparameters(self, n_candidates: int = 1) -> List[UltraFastESHyperparameters]:
        """Optimize hyperparameters using acquisition function"""
        if self.model is None or len(self.train_X) < 3:
            # Not enough data - return random samples
            return [self._sample_random_hyperparams() for _ in range(n_candidates)]
        
        try:
            # Set up acquisition function
            acq_func = ExpectedImprovement(self.model, best_f=self.train_Y.max())
            
            # Optimize acquisition function
            normalized_bounds = torch.stack([torch.zeros(self.bounds.shape[1]), 
                                           torch.ones(self.bounds.shape[1])])
            
            candidates_normalized, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=normalized_bounds,
                q=n_candidates,
                num_restarts=5,
                raw_samples=50
            )
            
            # Unnormalize and convert to hyperparameters
            candidates = unnormalize(candidates_normalized, bounds=self.bounds)
            
            return [UltraFastESHyperparameters.from_tensor(candidate) 
                   for candidate in candidates]
            
        except Exception as e:
            if self.enable_debug:
                self.logger.error(f"Hyperparameter optimization failed: {e}")
            # Fallback to random samples
            return [self._sample_random_hyperparams() for _ in range(n_candidates)]
    
    def _sample_random_hyperparams(self) -> UltraFastESHyperparameters:
        """Sample random hyperparameters within bounds"""
        random_tensor = torch.rand(self.bounds.shape[1], dtype=torch.float64)
        scaled = self.bounds[0] + random_tensor * (self.bounds[1] - self.bounds[0])
        return UltraFastESHyperparameters.from_tensor(scaled)
    
    def get_best_hyperparameters(self) -> Optional[UltraFastESHyperparameters]:
        """Get the best hyperparameters found so far"""
        return self.best_hyperparams
    
    def get_optimization_summary(self) -> Dict:
        """Get summary of optimization progress"""
        return {
            "total_observations": len(self.train_Y),
            "best_fitness": self.best_fitness,
            "best_hyperparams": self.best_hyperparams.__dict__ if self.best_hyperparams else None,
            "model_fitted": self.model is not None,
            "optimization_history": self.optimization_history
        }


class UltraFastBayesianESWrapper:
    """
    Wrapper that integrates Bayesian hyperparameter optimization with ultra-fast ES trainer
    Mirrors your rover's AdaptiveBayesianESWrapper
    """
    
    def __init__(self, 
                 es_trainer,  # UltraFastRobotTrainer instance
                 optimization_interval: int = 5,  # Optimize every N generations  
                 min_observations: int = 3,       # Need at least N data points
                 enable_debug: bool = False):
        
        self.es_trainer = es_trainer
        self.optimization_interval = optimization_interval
        self.min_observations = min_observations
        self.enable_debug = enable_debug
        self.logger = logging.getLogger(__name__)
        
        # Initialize Bayesian optimizer
        self.bayesian_optimizer = UltraFastBayesianESOptimizer(enable_debug=enable_debug)
        
        # Tracking
        self.last_optimization_generation = 0
        self.current_hyperparams = UltraFastESHyperparameters()  # Default hyperparams
        
        # Apply initial hyperparameters
        self._apply_hyperparameters(self.current_hyperparams)
        
        if self.enable_debug:
            self.logger.info("UltraFast Bayesian ES wrapper initialized")
    
    def _apply_hyperparameters(self, hyperparams: UltraFastESHyperparameters):
        """Apply hyperparameters to the ES trainer"""
        try:
            # Update trainer hyperparameters
            if hasattr(self.es_trainer, 'sigma'):
                self.es_trainer.sigma = hyperparams.sigma
            if hasattr(self.es_trainer, 'learning_rate'): 
                self.es_trainer.learning_rate = hyperparams.learning_rate
            if hasattr(self.es_trainer, 'elite_ratio'):
                self.es_trainer.elite_ratio = hyperparams.elite_ratio
                self.es_trainer.n_elites = max(1, int(self.es_trainer.population_size * hyperparams.elite_ratio))
            if hasattr(self.es_trainer, 'sigma_decay_rate'):
                self.es_trainer.sigma_decay_rate = hyperparams.sigma_decay_rate
            if hasattr(self.es_trainer, 'sigma_increase_rate'):
                self.es_trainer.sigma_increase_rate = hyperparams.sigma_increase_rate
                
            if self.enable_debug:
                self.logger.info(f"Applied hyperparameters: sigma={hyperparams.sigma:.4f}, "
                               f"lr={hyperparams.learning_rate:.4f}")
                
        except Exception as e:
            self.logger.error(f"Failed to apply hyperparameters: {e}")
    
    def should_optimize(self, generation: int) -> bool:
        """Check if we should run hyperparameter optimization"""
        if not BOTORCH_AVAILABLE:
            return False
            
        return (generation - self.last_optimization_generation) >= self.optimization_interval
    
    def apply_bayesian_optimization(self, generation: int, current_fitness: float):
        """Apply Bayesian optimization for hyperparameters (mirrors rover implementation)"""
        if not self.should_optimize(generation):
            return
        
        # Record current hyperparameter performance
        self.bayesian_optimizer.add_observation(self.current_hyperparams, current_fitness)
        
        # Fit model and optimize
        if self.bayesian_optimizer.fit_model():
            # Get new hyperparameters
            new_hyperparams_list = self.bayesian_optimizer.optimize_hyperparameters(n_candidates=1)
            if new_hyperparams_list:
                self.current_hyperparams = new_hyperparams_list[0]
                self._apply_hyperparameters(self.current_hyperparams)
                
                if self.enable_debug:
                    self.logger.info(f"Bayesian optimization applied at generation {generation}")
        
        self.last_optimization_generation = generation
    
    def get_optimization_summary(self) -> Dict:
        """Get optimization summary"""
        return self.bayesian_optimizer.get_optimization_summary()


if __name__ == "__main__":
    # Test the Bayesian ES optimization
    if BOTORCH_AVAILABLE:
        print("Testing Ultra-Fast Bayesian ES Optimization...")
        
        optimizer = UltraFastBayesianESOptimizer(enable_debug=True)
        
        # Simulate some observations
        for i in range(10):
            # Random hyperparameters
            hyperparams = UltraFastESHyperparameters(
                sigma=np.random.uniform(0.05, 0.2),
                learning_rate=np.random.uniform(0.005, 0.05)
            )
            
            # Fake fitness (higher sigma = better fitness for this test)
            fitness = hyperparams.sigma * 100 + np.random.normal(0, 5)
            
            optimizer.add_observation(hyperparams, fitness)
        
        # Test optimization
        optimizer.fit_model()
        best_hyperparams = optimizer.optimize_hyperparameters(n_candidates=3)
        
        print(f"Best hyperparameters found:")
        for i, hp in enumerate(best_hyperparams):
            print(f"  Candidate {i+1}: sigma={hp.sigma:.4f}, lr={hp.learning_rate:.4f}")
            
        print("✓ Ultra-Fast Bayesian ES optimization working!")
    else:
        print("BoTorch not available - Bayesian optimization will be disabled")