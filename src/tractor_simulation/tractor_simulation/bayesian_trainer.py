#!/usr/bin/env python3
"""
Bayesian Optimization Trainer using BoTorch
Much more efficient than evolutionary strategies
"""

import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval
from botorch.utils.transforms import normalize, unnormalize

class BayesianOptimizationTrainer:
    def __init__(self, model_dim=100, bounds=None, batch_size=3, n_init=10):
        """
        Args:
            model_dim: Number of neural network parameters to optimize
            bounds: Tensor of shape (2, model_dim) with [lower, upper] bounds
            batch_size: Number of points to evaluate per iteration  
            n_init: Number of random initial points
        """
        self.model_dim = model_dim
        self.batch_size = batch_size
        self.n_init = n_init
        
        # Use double precision for better numerical stability (BoTorch recommendation)
        self.dtype = torch.float64
        
        # Set reasonable bounds for neural network weights
        if bounds is None:
            self.bounds = torch.stack([
                -2.0 * torch.ones(model_dim, dtype=self.dtype),  # Lower bounds
                2.0 * torch.ones(model_dim, dtype=self.dtype)    # Upper bounds  
            ])
        else:
            self.bounds = bounds.to(dtype=self.dtype)
            
        # Initialize storage
        self.train_X = None
        self.train_Y = None
        self.model = None
        
    def initialize_random_points(self):
        """Generate initial random points"""
        X = torch.rand(self.n_init, self.model_dim, dtype=self.dtype)
        X = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * X
        return X
        
    def fit_model(self):
        """Fit GP model to current data"""
        if self.train_X is None or len(self.train_X) < 2:
            return None
            
        # Normalize inputs for better GP performance
        normalized_X = normalize(self.train_X, bounds=self.bounds)
        
        # Fit Single Task GP
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        self.model = SingleTaskGP(normalized_X, self.train_Y, likelihood=likelihood)
        
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
        
        return self.model
        
    def propose_next_candidates(self):
        """Propose next batch of candidates using Expected Improvement"""
        if self.model is None:
            return None
            
        # Create acquisition function (using qLogEI for better numerical stability)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        acq_func = qLogExpectedImprovement(
            model=self.model,
            best_f=self.train_Y.max(),
            sampler=sampler
        )
        
        # Optimize acquisition function
        normalized_bounds = torch.stack([torch.zeros(self.model_dim), torch.ones(self.model_dim)])
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=normalized_bounds,
            q=self.batch_size,
            num_restarts=10,
            raw_samples=256,
            options={"batch_limit": 5, "maxiter": 200}
        )
        
        # Unnormalize back to original space
        candidates = unnormalize(candidates, bounds=self.bounds)
        return candidates
        
    def add_observations(self, X, Y):
        """Add new observations to training set"""
        # Ensure correct dtype
        X = X.to(dtype=self.dtype)
        Y = Y.to(dtype=self.dtype)
        
        if self.train_X is None:
            self.train_X = X.clone()
            self.train_Y = Y.clone()
        else:
            self.train_X = torch.cat([self.train_X, X])
            self.train_Y = torch.cat([self.train_Y, Y])
            
    def get_best_parameters(self):
        """Get the best parameters found so far"""
        if self.train_Y is None:
            return None
        best_idx = self.train_Y.argmax()
        return self.train_X[best_idx]
        
    def optimize(self, objective_fn, n_iterations=50):
        """
        Main optimization loop
        
        Args:
            objective_fn: Function that takes parameter vector and returns fitness
            n_iterations: Number of BO iterations
        """
        print(f"Starting Bayesian Optimization for {n_iterations} iterations...")
        
        # Initialize with random points
        X_init = self.initialize_random_points()
        Y_init = torch.tensor([objective_fn(x) for x in X_init], dtype=self.dtype).unsqueeze(-1)
        self.add_observations(X_init, Y_init)
        
        print(f"Initial best: {Y_init.max().item():.4f}")
        
        # BO loop
        for iteration in range(n_iterations):
            # Fit GP model
            self.fit_model()
            
            # Propose next candidates
            candidates = self.propose_next_candidates()
            if candidates is None:
                continue
                
            # Evaluate candidates 
            Y_new = torch.tensor([objective_fn(x) for x in candidates], dtype=self.dtype).unsqueeze(-1)
            
            # Add to training set
            self.add_observations(candidates, Y_new)
            
            # Print progress
            current_best = self.train_Y.max().item()
            print(f"Iteration {iteration+1:2d}: Best so far = {current_best:.4f}")
            
        return self.get_best_parameters()