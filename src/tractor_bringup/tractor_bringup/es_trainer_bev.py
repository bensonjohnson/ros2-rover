#!/usr/bin/env python3
"""
Evolutionary Strategy Training System for Autonomous Rover Exploration (BEV Version)
Implements evolutionary algorithms for neural network optimization using Bird's Eye View maps
"""

import numpy as np
import time
import os
import pickle
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from collections import deque
import cv2

try:
    from rknn.api import RKNN
    RKNN_AVAILABLE = True
except ImportError:
    RKNN_AVAILABLE = False

try:
    from .improved_reward_system import ImprovedRewardCalculator
    IMPROVED_REWARDS = True
except ImportError:
    IMPROVED_REWARDS = False
    print("Improved reward system not available - using basic rewards")

try:
    from .bayesian_es_optimizer import AdaptiveBayesianESWrapper
    BAYESIAN_OPTIMIZATION = True
except ImportError:
    BAYESIAN_OPTIMIZATION = False
    print("Bayesian ES optimization not available - using standard ES")

from .rknn_trainer_bev import BEVExplorationNet

class EvolutionaryStrategyTrainerBEV:
    """
    Handles model training using evolutionary strategies, data collection, and RKNN conversion for BEV images
    """
    
    def __init__(self, model_dir="models", bev_channels: int = 4, enable_debug: bool = False, 
                 population_size: int = 10, sigma: float = 0.1, learning_rate: float = 0.01,
                 enable_bayesian_optimization: bool = True, extra_proprio: int = 13):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.enable_debug = enable_debug
        self.bev_channels = bev_channels
        self.bev_height = 200
        self.bev_width = 200
        
        # Extended proprio feature count now: base(3) + extras
        self.extra_proprio = int(extra_proprio)
        
        # Evolutionary Strategy parameters
        self.population_size = population_size
        self.sigma = sigma  # Standard deviation of parameter perturbations
        self.initial_sigma = sigma  # Store initial sigma for reference
        self.learning_rate = learning_rate
        self.generation = 0
        self.population = []
        self.fitness_history = deque(maxlen=1000)
        self.best_fitness = -float('inf')
        self.best_model_state = None
        
        # Adaptive sigma parameters
        self.fitness_improvement_window = 10  # Look at last N generations for improvement
        self.sigma_decay_rate = 0.99  # How fast to decay sigma when improving
        self.sigma_increase_rate = 1.05  # How fast to increase sigma when stagnating
        self.min_sigma = sigma * 0.1  # Minimum sigma to maintain exploration
        self.max_sigma = sigma * 3.0  # Maximum sigma to prevent chaos
        self.fitness_improvement_threshold = 0.01  # Minimum improvement to consider progress
        self.stagnation_counter = 0  # Track generations without improvement
        
        # Elite preservation parameters
        self.elite_ratio = 0.2  # Preserve top 20% of population
        self.n_elites = max(1, int(self.population_size * self.elite_ratio))
        self.elite_individuals = []  # Store elite (perturbation, fitness) pairs
        self.elite_fitness_scores = []
        
        # Momentum-based parameter updates (Adam-like)
        self.momentum_decay = 0.9  # Momentum decay rate
        self.velocity_decay = 0.999  # Second moment decay rate
        self.epsilon = 1e-8  # Small constant for numerical stability
        self.momentum = None  # First moment estimate
        self.velocity = None  # Second moment estimate
        self.update_step = 0  # Track number of parameter updates for bias correction
        
        # Population diversity monitoring
        self.min_diversity_threshold = 0.001  # Minimum diversity to maintain
        self.diversity_history = deque(maxlen=20)  # Track diversity over generations
        self.diversity_injection_ratio = 0.25  # Fraction of population to replace when diversity is low
        
        # Adaptive evolution frequency
        self.base_evolution_frequency = 50  # Base frequency (every N steps)
        self.min_evolution_frequency = 25   # Minimum frequency (most frequent)
        self.max_evolution_frequency = 300  # Maximum frequency (least frequent, mature model)
        self.current_evolution_frequency = self.base_evolution_frequency
        self.data_quality_window = 10  # Window to assess data quality
        self.recent_rewards = deque(maxlen=self.data_quality_window)
        self.last_evolution_step = 0
        
        # Neural network (same architecture as RL version)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Optimize CPU threading for ES population evaluation on RK3588
        # Use fewer threads to leave cores available for real-time control
        torch.set_num_threads(4)  # Use 4 cores, leave 4 for system/control
        torch.set_num_interop_threads(2)  # Parallel operations
        
        self.model = BEVExplorationNet(bev_channels=bev_channels, extra_proprio=self.extra_proprio).to(self.device)
        
        # Initialize population
        self._initialize_population()
        
        # Experience replay (simplified for ES). Store BEV as uint8 to reduce RAM.
        self.buffer_capacity = 50000
        self.buffer_size = 0
        self.insert_ptr = 0
        self.proprio_dim = 3 + self.extra_proprio
        self.bev_store = np.zeros((self.buffer_capacity, self.bev_channels, self.bev_height, self.bev_width), dtype=np.uint8)
        self.proprio_store = np.zeros((self.buffer_capacity, self.proprio_dim), dtype=np.float32)
        self.action_store = np.zeros((self.buffer_capacity, 2), dtype=np.float32)  # linear, angular
        self.reward_store = np.zeros((self.buffer_capacity,), dtype=np.float32)
        self.done_store = np.zeros((self.buffer_capacity,), dtype=np.uint8)
        
        # Data collection
        self.reward_history = deque(maxlen=1000)
        self.action_history = deque(maxlen=100)
        
        # Dataset collection for RKNN quantization
        self.dataset_samples_collected = 0
        self.target_dataset_samples = 100  # Number of samples to collect for quantization
        self.dataset_collection_interval = 50  # Collect every N training steps
        
        # Initialize improved reward calculator if available
        if IMPROVED_REWARDS:
            self.reward_calculator = ImprovedRewardCalculator()
            print("Using improved reward system")
        else:
            self.reward_calculator = None
            print("Using basic reward system")
        
        # Initialize Bayesian optimization for ES hyperparameters
        self.bayesian_es_wrapper = None
        if enable_bayesian_optimization and BAYESIAN_OPTIMIZATION:
            try:
                self.bayesian_es_wrapper = AdaptiveBayesianESWrapper(
                    es_trainer=self,
                    optimization_interval=5,  # Optimize every 5 generations
                    min_observations=3,       # Need at least 3 data points
                    enable_debug=enable_debug
                )
                print("Using Bayesian optimization for ES hyperparameters")
            except Exception as e:
                print(f"Failed to initialize Bayesian optimization: {e}")
                self.bayesian_es_wrapper = None
        elif enable_bayesian_optimization and not BAYESIAN_OPTIMIZATION:
            print("Bayesian optimization requested but BoTorch not available")
        else:
            print("Using standard ES hyperparameter adaptation")
        
        # Initialize dataset.txt for quantization
        self.init_dataset_file()
        
        # Load existing model if available
        self.load_latest_model()
        # Runtime inference flags
        self.use_rknn_inference = False
        self.rknn_runtime = None
        print(f"[ESTrainerInit] Using trainer file: {__file__} expected_proprio={3 + self.extra_proprio}")

    def _initialize_population(self):
        """Initialize population of parameter perturbations"""
        # Get the current model parameters as a flat vector
        params = self._get_flat_params()
        self.param_shape = params.shape
        
        # Initialize population with random perturbations
        self.population = []
        
        # First, add elite individuals from previous generation if available
        n_elites_added = 0
        if self.elite_individuals:
            for elite_perturbation, elite_fitness in self.elite_individuals:
                if n_elites_added < self.n_elites and n_elites_added < self.population_size:
                    self.population.append(elite_perturbation.copy())
                    n_elites_added += 1
        
        # Fill remaining slots with random perturbations
        for i in range(n_elites_added, self.population_size):
            # Random perturbation
            perturbation = np.random.randn(*self.param_shape) * self.sigma
            self.population.append(perturbation)
        
        if self.enable_debug:
            print(f"[ES] Initialized population with {self.population_size} individuals ({n_elites_added} elites preserved)")

    def _get_flat_params(self) -> np.ndarray:
        """Get model parameters as a flat numpy array"""
        params = []
        for param in self.model.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def _set_flat_params(self, params: np.ndarray):
        """Set model parameters from a flat numpy array"""
        idx = 0
        for param in self.model.parameters():
            param_shape = param.data.shape
            param_size = param.data.numel()
            param_values = params[idx:idx+param_size].reshape(param_shape)
            param.data = torch.from_numpy(param_values).float().to(self.device)
            idx += param_size

    def add_experience(self,
                      bev_image: np.ndarray,
                      proprioceptive: np.ndarray,
                      action: np.ndarray,
                      reward: float,
                      next_bev_image: np.ndarray = None,
                      done: bool = False,
                      collision: bool = False,
                      in_recovery: bool = False):
        """Add experience to replay buffer."""
        # Preprocess BEV image to (C,H,W)
        processed = self.preprocess_bev_for_storage(bev_image)
        # Defensive padding/truncation for proprioceptive data
        expected = self.proprio_dim
        if proprioceptive.shape[0] < expected:
            if self.enable_debug:
                print(f"[AddExp] Padding proprio {proprioceptive.shape[0]} -> {expected}")
            proprioceptive = np.concatenate([proprioceptive, np.zeros(expected - proprioceptive.shape[0], dtype=proprioceptive.dtype)])
        elif proprioceptive.shape[0] > expected:
            if self.enable_debug:
                print(f"[AddExp] Truncating proprio {proprioceptive.shape[0]} -> {expected}")
            proprioceptive = proprioceptive[:expected]
        # Ring buffer insert
        i = self.insert_ptr
        # Store as uint8
        self.bev_store[i] = processed
        self.proprio_store[i] = proprioceptive.astype(np.float32)
        # Action expected length 2 (linear, angular); trim/pad if needed
        if action.shape[0] < 2:
            act = np.zeros(2, dtype=np.float32)
            act[:action.shape[0]] = action
        else:
            act = action[:2]
        self.action_store[i] = act.astype(np.float32)
        self.reward_store[i] = float(reward)
        self.done_store[i] = 1 if done else 0
        # Advance pointer / size
        self.insert_ptr = (i + 1) % self.buffer_capacity
        if self.buffer_size < self.buffer_capacity:
            self.buffer_size += 1
        self.reward_history.append(reward)
        self.recent_rewards.append(reward)
        
        # Collect samples for RKNN quantization dataset
        if (self.dataset_samples_collected < self.target_dataset_samples and 
            self.generation % self.dataset_collection_interval == 0):
            self.save_quantization_sample(bev_image, proprioceptive)

    def calculate_reward(self, 
                        action: np.ndarray,
                        collision: bool,
                        progress: float,
                        exploration_bonus: float,
                        position: Optional[np.ndarray] = None,
                        bev_data: Optional[np.ndarray] = None,
                        wheel_velocities: Optional[Tuple[float, float]] = None) -> float:
        """Calculate reward for evolutionary strategy (same as RL version)"""
        
        # Use improved reward system if available
        if self.reward_calculator is not None and position is not None:
            # Use comprehensive reward calculation
            near_collision = False  # You can enhance this based on BEV data
            if bev_data is not None:
                try:
                    # Check for obstacles in the near field
                    max_height_channel = bev_data[:, :, 0]  # Max height channel
                    # Check for high obstacles in the front area
                    front_area = max_height_channel[:max_height_channel.shape[0]//3, :]  # Front third
                    if np.any(front_area > 0.3):  # Obstacle higher than 30cm
                        near_collision = True
                except:
                    pass
            
            total_reward, reward_breakdown = self.reward_calculator.calculate_comprehensive_reward(
                action=action,
                position=position,
                collision=collision,
                near_collision=near_collision,
                progress=progress,
                depth_data=None,  # Not using depth data anymore
                wheel_velocities=wheel_velocities
            )
            
            # Log reward breakdown occasionally for debugging
            if self.generation % 100 == 0:
                print(f"Reward breakdown: {reward_breakdown}")
                
            return total_reward
        
        # Fallback to basic reward system
        reward = 0.0
        
        # Progress reward (forward movement is good)
        reward += progress * 10.0
        
        # Collision penalty
        if collision:
            reward -= 50.0
            
        # Exploration bonus (new areas are good)
        reward += exploration_bonus * 5.0
        
        # Smooth control reward (avoid jerky movements)
        if len(self.action_history) > 0:
            last_action = self.action_history[-1]
            action_smoothness = -np.linalg.norm(action - last_action) * 2.0
            reward += action_smoothness
            
        # Speed efficiency and magnitude bonus (encourage meaningful movement)
        speed = abs(action[0])
        if 0.05 < speed < 0.3:
            reward += 2.0
        elif speed < 0.02:
            reward -= 5.0  # Penalize being too slow
            
        # Speed magnitude bonus (reward higher speeds to prevent tiny movements)
        if speed > 0.03:  # Only for meaningful movement
            speed_factor = min(speed, 0.25)  # Cap at 0.25 m/s
            speed_bonus = 15.0 * (speed_factor ** 1.5)  # Similar to improved reward system
            reward += speed_bonus
            
        self.action_history.append(action)
        return reward

    def evaluate_individual(self, perturbation: np.ndarray) -> float:
        """Evaluate fitness of an individual in the population"""
        # Save original parameters
        original_params = self._get_flat_params()
        
        # Apply perturbation
        perturbed_params = original_params + perturbation
        self._set_flat_params(perturbed_params)
        
        # Evaluate fitness on recent experiences
        if self.buffer_size < 32:
            # Not enough data, return low fitness
            self._set_flat_params(original_params)  # Restore parameters
            return -100.0
        
        # Sample a batch of experiences
        batch_size = min(64, self.buffer_size)
        indices = np.random.choice(self.buffer_size, batch_size, replace=False)
        
        total_reward = 0.0
        total_similarity = 0.0
        valid_samples = 0
        
        # Evaluate on batch
        for idx in indices:
            bev_data = self.bev_store[idx]
            proprio_data = self.proprio_store[idx]
            actual_action = self.action_store[idx]
            actual_reward = self.reward_store[idx]
            
            # Get model prediction
            with torch.no_grad():
                bev_tensor = torch.from_numpy(bev_data).unsqueeze(0).float().to(self.device)
                proprio_tensor = torch.from_numpy(proprio_data).unsqueeze(0).float().to(self.device)
                predicted_output = self.model(bev_tensor, proprio_tensor)
                predicted_action = torch.tanh(predicted_output[0, :2]).cpu().numpy()
            
            # Calculate how close prediction is to actual action
            action_distance = np.linalg.norm(predicted_action - actual_action)
            action_similarity = np.exp(-action_distance)  # Convert distance to similarity (0-1)
            
            # Fitness is weighted by reward and similarity
            # Higher rewards and better similarity lead to higher fitness
            fitness = actual_reward * action_similarity
            
            total_reward += actual_reward
            total_similarity += action_similarity
            valid_samples += 1
        
        # Calculate average fitness with bonus for good similarity
        avg_reward = total_reward / max(valid_samples, 1)
        avg_similarity = total_similarity / max(valid_samples, 1)
        
        # Combined fitness: reward + moderate similarity bonus + exploration bonus
        exploration_bonus = 0.0
        if valid_samples > 0:
            # Bonus for diverse actions (encourage exploration)
            action_magnitudes = [np.linalg.norm(self.action_store[idx]) for idx in indices]
            avg_action_magnitude = np.mean(action_magnitudes)
            exploration_bonus = min(avg_action_magnitude * 2.0, 1.0)  # Cap exploration bonus
        
        avg_fitness = avg_reward + avg_similarity * 1.0 + exploration_bonus  # Reduced similarity weight, added exploration
        
        # Restore original parameters
        self._set_flat_params(original_params)
        
        if self.enable_debug:
            print(f"[ES] Evaluated fitness: {avg_fitness:.4f} (reward: {avg_reward:.4f}, similarity: {avg_similarity:.4f})")
        
        return avg_fitness

    def _adapt_sigma(self, current_best_fitness: float):
        """Adapt sigma based on fitness improvement"""
        # Check if we have enough history to assess improvement
        if len(self.fitness_history) < self.fitness_improvement_window:
            return
        
        # Calculate average fitness over the improvement window
        recent_fitness = list(self.fitness_history)[-self.fitness_improvement_window:]
        avg_recent_fitness = np.mean(recent_fitness)
        
        # Compare with earlier fitness
        if len(self.fitness_history) >= 2 * self.fitness_improvement_window:
            earlier_fitness = list(self.fitness_history)[-2 * self.fitness_improvement_window:-self.fitness_improvement_window]
            avg_earlier_fitness = np.mean(earlier_fitness)
            
            fitness_improvement = avg_recent_fitness - avg_earlier_fitness
            
            if fitness_improvement >= self.fitness_improvement_threshold:
                # Good improvement - reduce sigma to exploit current region
                self.sigma *= self.sigma_decay_rate
                self.stagnation_counter = 0
                if self.enable_debug:
                    print(f"[ES] Fitness improving ({fitness_improvement:.4f}), reducing sigma to {self.sigma:.6f}")
            else:
                # Stagnation - increase sigma to explore more
                self.stagnation_counter += 1
                if self.stagnation_counter >= 3:  # Only increase after multiple stagnant generations
                    self.sigma *= self.sigma_increase_rate
                    self.stagnation_counter = 0
                    if self.enable_debug:
                        print(f"[ES] Fitness stagnating, increasing sigma to {self.sigma:.6f}")
        
        # Clamp sigma to reasonable bounds
        self.sigma = np.clip(self.sigma, self.min_sigma, self.max_sigma)
    
    def _update_elites(self, fitness_scores: np.ndarray):
        """Update elite individuals based on current generation fitness"""
        # Combine current population with their fitness scores
        population_fitness_pairs = list(zip(self.population, fitness_scores))
        
        # Add previous elites to the mix (they compete with current generation)
        if self.elite_individuals:
            population_fitness_pairs.extend(self.elite_individuals)
        
        # Sort by fitness (descending)
        population_fitness_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Select top performers as new elites
        self.elite_individuals = population_fitness_pairs[:self.n_elites]
        self.elite_fitness_scores = [fitness for _, fitness in self.elite_individuals]
        
        if self.enable_debug and self.elite_individuals:
            elite_fitnesses = [f for _, f in self.elite_individuals]
            print(f"[ES] Updated elites: {len(self.elite_individuals)} individuals, "
                  f"fitness range [{min(elite_fitnesses):.4f}, {max(elite_fitnesses):.4f}]")
    
    def _calculate_population_diversity(self) -> float:
        """Calculate the average pairwise distance between individuals in the population"""
        if len(self.population) < 2:
            return 0.0
        
        total_distance = 0.0
        num_pairs = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = np.linalg.norm(self.population[i] - self.population[j])
                total_distance += distance
                num_pairs += 1
        
        return total_distance / num_pairs if num_pairs > 0 else 0.0
    
    def _maintain_diversity(self):
        """Inject random individuals if population diversity is too low"""
        current_diversity = self._calculate_population_diversity()
        self.diversity_history.append(current_diversity)
        
        if current_diversity < self.min_diversity_threshold:
            # Replace worst individuals with random ones
            n_replace = max(1, int(self.population_size * self.diversity_injection_ratio))
            
            # Find indices of worst individuals (note: we don't have fitness here, so replace random ones)
            replace_indices = np.random.choice(self.population_size, n_replace, replace=False)
            
            for idx in replace_indices:
                # Create new random perturbation
                self.population[idx] = np.random.randn(*self.param_shape) * self.sigma
            
            if self.enable_debug:
                print(f"[ES] Diversity too low ({current_diversity:.6f}), injected {n_replace} random individuals")
        
        elif self.enable_debug:
            avg_diversity = np.mean(self.diversity_history) if self.diversity_history else 0.0
            print(f"[ES] Population diversity: {current_diversity:.6f} (avg: {avg_diversity:.6f})")
    
    def should_evolve(self, current_step: int) -> bool:
        """Determine if evolution should occur based on adaptive frequency"""
        # Always check minimum buffer requirement
        if self.buffer_size < 50:
            return False
        
        # Check if enough steps have passed since last evolution
        steps_since_evolution = current_step - self.last_evolution_step
        if steps_since_evolution < self.current_evolution_frequency:
            return False
        
        # Adapt evolution frequency based on data quality and learning progress
        self._adapt_evolution_frequency()
        
        # Update last evolution step
        self.last_evolution_step = current_step
        return True
    
    def _adapt_evolution_frequency(self):
        """Adapt evolution frequency based on data quality and learning progress"""
        if len(self.recent_rewards) < self.data_quality_window:
            return
        
        # Assess data quality based on reward variance and progress
        reward_variance = np.var(self.recent_rewards)
        reward_trend = np.mean(list(self.recent_rewards)[5:]) - np.mean(list(self.recent_rewards)[:5]) if len(self.recent_rewards) >= 10 else 0
        
        # High variance or positive trend = evolve more frequently (good learning signal)
        # Low variance or negative trend = evolve less frequently (poor learning signal)
        
        if reward_variance > 5.0 or reward_trend > 1.0:
            # Good learning signal - evolve less frequently (model is learning well, let it collect more experience)
            self.current_evolution_frequency = min(
                self.max_evolution_frequency,
                int(self.current_evolution_frequency * 1.2)
            )
            if self.enable_debug:
                print(f"[ES] Decreasing evolution frequency to {self.current_evolution_frequency} (good signal, collecting more experience)")
        elif reward_variance < 1.0 and reward_trend < -0.5:
            # Poor learning signal - evolve more frequently (model needs more exploration)
            self.current_evolution_frequency = max(
                self.min_evolution_frequency,
                int(self.current_evolution_frequency * 0.8)
            )
            if self.enable_debug:
                print(f"[ES] Increasing evolution frequency to {self.current_evolution_frequency} (poor signal, needs more exploration)")
        
        # Also consider stagnation
        if self.stagnation_counter >= 3:
            # Stagnating - try evolving more frequently to break out of local optimum
            self.current_evolution_frequency = max(
                self.min_evolution_frequency,
                int(self.current_evolution_frequency * 0.7)
            )
            if self.enable_debug:
                print(f"[ES] Stagnation detected, increasing evolution frequency to {self.current_evolution_frequency}")

    def evolve_population(self) -> Dict[str, float]:
        """Perform one generation of evolution"""
        # Evaluate fitness for each individual
        fitness_scores = []
        for i, perturbation in enumerate(self.population):
            fitness = self.evaluate_individual(perturbation)
            fitness_scores.append(fitness)
            
            if self.enable_debug and i % 5 == 0:
                print(f"[ES] Evaluated individual {i+1}/{self.population_size}, fitness: {fitness:.4f}")
        
        # Convert to numpy arrays
        fitness_scores = np.array(fitness_scores)
        
        # Update fitness history
        self.fitness_history.extend(fitness_scores)
        
        # Update best model if needed
        best_idx = np.argmax(fitness_scores)
        current_best_fitness = fitness_scores[best_idx]
        best_model_updated = False
        if current_best_fitness > self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_model_state = self._get_flat_params() + self.population[best_idx]
            best_model_updated = True
            print(f"[ES] New best fitness: {self.best_fitness:.4f}")
        
        # Adapt sigma based on fitness progress
        self._adapt_sigma(current_best_fitness)
        
        # Rank-based fitness selection (more robust than standardization)
        fitness_ranks = np.argsort(np.argsort(-fitness_scores))  # Higher rank = better fitness
        
        # Calculate rank-based weights (log-normal distribution)
        weights = np.maximum(0, np.log(self.population_size/2 + 1) - np.log(fitness_ranks + 1))
        
        # Normalize weights and center them
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
            weights = weights - 1.0 / self.population_size
        else:
            # Fallback to uniform weights if all weights are zero
            weights = np.zeros(self.population_size)
            weights[best_idx] = 1.0 / self.population_size
        
        # Update parameters using rank-weighted gradient
        original_params = self._get_flat_params()
        
        # Calculate gradient estimate using rank-based weights
        grad_estimate = np.zeros_like(original_params)
        for i in range(self.population_size):
            grad_estimate += weights[i] * self.population[i]
        grad_estimate /= self.sigma
        
        if self.enable_debug:
            top_3_indices = np.argsort(-fitness_scores)[:3]
            top_3_weights = weights[top_3_indices]
            print(f"[ES] Top 3 individuals - Fitness: {fitness_scores[top_3_indices]}, Weights: {top_3_weights}")
        
        # Apply momentum-based parameter update (Adam-like optimizer)
        self.update_step += 1
        
        # Initialize momentum and velocity if first update
        if self.momentum is None:
            self.momentum = np.zeros_like(grad_estimate)
            self.velocity = np.zeros_like(grad_estimate)
        
        # Update momentum (first moment)
        self.momentum = self.momentum_decay * self.momentum + (1 - self.momentum_decay) * grad_estimate
        
        # Update velocity (second moment)
        self.velocity = self.velocity_decay * self.velocity + (1 - self.velocity_decay) * (grad_estimate ** 2)
        
        # Bias correction
        momentum_corrected = self.momentum / (1 - self.momentum_decay ** self.update_step)
        velocity_corrected = self.velocity / (1 - self.velocity_decay ** self.update_step)
        
        # Adaptive learning rate update
        adaptive_update = self.learning_rate * momentum_corrected / (np.sqrt(velocity_corrected) + self.epsilon)
        
        # Update parameters
        new_params = original_params + adaptive_update
        self._set_flat_params(new_params)
        
        if self.enable_debug:
            grad_norm = np.linalg.norm(grad_estimate)
            update_norm = np.linalg.norm(adaptive_update)
            print(f"[ES] Gradient norm: {grad_norm:.6f}, Update norm: {update_norm:.6f}")
        
        # Update elite preservation
        self._update_elites(fitness_scores)
        
        # Only apply best model if it was updated and it's better than current
        if best_model_updated and self.best_model_state is not None:
            # Compare current model fitness with best model fitness
            current_fitness = self.evaluate_individual(np.zeros_like(new_params))
            if self.best_fitness > current_fitness:
                self._set_flat_params(self.best_model_state)
                if self.enable_debug:
                    print(f"[ES] Applied best model parameters (fitness: {self.best_fitness:.4f})")
        
        # Maintain population diversity before generating new population
        self._maintain_diversity()
        
        # Generate new population based on current model parameters (using adapted sigma)
        self._initialize_population()
        
        # Update generation counter
        self.generation += 1
        
        # Apply Bayesian optimization for hyperparameters
        if self.bayesian_es_wrapper is not None:
            self.bayesian_es_wrapper.apply_bayesian_optimization(self.generation, current_best_fitness)
        
        # Save model periodically (more frequent during early training)
        save_frequency = 10 if self.generation < 100 else 50  # Save every 10 gens for first 100, then every 50
        if self.generation % save_frequency == 0 or self.generation == 1:  # Always save first generation
            self.save_model()
        
        # Convert to RKNN with adaptive frequency - more frequent early training
        rknn_conversion_frequency = self._get_rknn_conversion_frequency()
        if self.generation % rknn_conversion_frequency == 0:
            try:
                self.convert_to_rknn()
                print(f"[ES] RKNN model updated at generation {self.generation}")
            except Exception as e:
                print(f"RKNN conversion failed: {e}")
        
        avg_fitness = float(np.mean(fitness_scores))
        fitness_std = float(np.std(fitness_scores))
        print(f"[ES] Generation {self.generation} completed - Avg Fitness: {avg_fitness:.4f}, Best: {self.best_fitness:.4f}, Std: {fitness_std:.4f}")
        
        # Calculate current population diversity
        current_diversity = self._calculate_population_diversity()
        
        # Prepare return statistics
        stats = {
            "generation": self.generation,
            "avg_fitness": avg_fitness,
            "best_fitness": float(self.best_fitness),
            "fitness_std": fitness_std,
            "samples": self.buffer_size,
            "sigma": float(self.sigma),
            "stagnation_counter": self.stagnation_counter,
            "n_elites": len(self.elite_individuals),
            "elite_avg_fitness": float(np.mean(self.elite_fitness_scores)) if self.elite_fitness_scores else 0.0,
            "grad_norm": float(np.linalg.norm(grad_estimate)),
            "population_diversity": float(current_diversity),
            "avg_diversity": float(np.mean(self.diversity_history)) if self.diversity_history else 0.0,
            "evolution_frequency": self.current_evolution_frequency,
            "reward_variance": float(np.var(self.recent_rewards)) if self.recent_rewards else 0.0
        }
        
        # Add Bayesian optimization statistics if available
        if self.bayesian_es_wrapper is not None:
            bayesian_stats = self.bayesian_es_wrapper.get_optimization_summary()
            stats.update({f"bayesian_{k}": v for k, v in bayesian_stats.items()})
        
        return stats

    def preprocess_bev_for_model(self, bev_image: np.ndarray) -> np.ndarray:
        """Normalize and prepare BEV image for model input.
        Input bev_image: (H,W,C) or (C,H,W).
        Returns (C,H,W) float32."""
        try:
            # Ensure correct shape (C,H,W)
            if bev_image.ndim == 3:
                if bev_image.shape[0] == self.bev_height and bev_image.shape[1] == self.bev_width:
                    # Already in (H,W,C) format, transpose to (C,H,W)
                    bev_processed = np.transpose(bev_image, (2, 0, 1))
                elif bev_image.shape[1] == self.bev_height and bev_image.shape[2] == self.bev_width:
                    # Already in (C,H,W) format
                    bev_processed = bev_image
                else:
                    # Unexpected shape, create default
                    bev_processed = np.zeros((self.bev_channels, self.bev_height, self.bev_width), dtype=np.float32)
            else:
                # Unexpected dimensions, create default
                bev_processed = np.zeros((self.bev_channels, self.bev_height, self.bev_width), dtype=np.float32)
                
            return bev_processed.astype(np.float32) / 255.0 if bev_processed.dtype == np.uint8 else bev_processed.astype(np.float32)
        except Exception:
            # Fallback zero tensor
            return np.zeros((self.bev_channels, self.bev_height, self.bev_width), dtype=np.float32)
    
    def preprocess_bev_for_storage(self, bev_image: np.ndarray) -> np.ndarray:
        """Preprocess BEV image for storage as uint8 [0,255] in replay buffer."""
        try:
            # Ensure correct shape (C,H,W)
            if bev_image.ndim == 3:
                if bev_image.shape[0] == self.bev_height and bev_image.shape[1] == self.bev_width:
                    # Already in (H,W,C) format, transpose to (C,H,W)
                    bev_processed = np.transpose(bev_image, (2, 0, 1))
                elif bev_image.shape[1] == self.bev_height and bev_image.shape[2] == self.bev_width:
                    # Already in (C,H,W) format
                    bev_processed = bev_image
                else:
                    # Unexpected shape, create default
                    bev_processed = np.zeros((self.bev_channels, self.bev_height, self.bev_width), dtype=np.uint8)
            else:
                # Unexpected dimensions, create default
                bev_processed = np.zeros((self.bev_channels, self.bev_height, self.bev_width), dtype=np.uint8)

            # If not already uint8, scale/clip
            if bev_processed.dtype != np.uint8:
                arr = bev_processed.astype(np.float32)
                scale = 255.0 if np.percentile(arr, 99) <= 1.5 else 1.0
                arr = np.clip(arr * scale, 0.0, 255.0)
                bev_u8 = arr.astype(np.uint8)
            else:
                bev_u8 = bev_processed

            return bev_u8
        except Exception:
            return np.zeros((self.bev_channels, self.bev_height, self.bev_width), dtype=np.uint8)
        
    def enable_rknn_inference(self):
        """Enable RKNN runtime inference if model file exists."""
        if not RKNN_AVAILABLE:
            if self.enable_debug:
                print("[RKNN] Toolkit not available - cannot enable RKNN inference")
            return False
        rknn_path = os.path.join(self.model_dir, "exploration_model_bev_es.rknn")
        if not os.path.exists(rknn_path):
            # Try fallback to RL model
            rknn_path = os.path.join(self.model_dir, "exploration_model_bev.rknn")
            if not os.path.exists(rknn_path):
                if self.enable_debug:
                    print(f"[RKNN] No RKNN file found at {rknn_path}")
                return False
        try:
            # Release previous runtime
            if self.rknn_runtime is not None:
                try:
                    self.rknn_runtime.release()
                except Exception:
                    pass
                self.rknn_runtime = None
            r = RKNN(verbose=self.enable_debug)
            if self.enable_debug:
                print(f"[RKNN] Loading RKNN runtime from {rknn_path}")
            ret = r.load_rknn(rknn_path)
            if ret != 0:
                print("[RKNN] load_rknn failed (ret != 0)")
                return False
            # Always specify target to avoid simulator warning
            if self.enable_debug:
                print("[RKNN] Initializing runtime target=rk3588")
            ret = r.init_runtime(target='rk3588')
            if ret != 0:
                print(f"[RKNN] init_runtime failed (ret={ret}), retrying once without explicit target")
                try:
                    ret2 = r.init_runtime()
                    if ret2 != 0:
                        print(f"[RKNN] init_runtime retry failed (ret={ret2})")
                        r.release()
                        return False
                except Exception as e2:
                    print(f"[RKNN] init_runtime retry exception: {e2}")
                    r.release()
                    return False
            self.rknn_runtime = r
            self.use_rknn_inference = True
            if self.enable_debug:
                print("[RKNN] Runtime initialized and inference enabled")
            return True
        except Exception as e:
            print(f"[RKNN] Failed to enable runtime: {e}")
            return False

    def disable_rknn_inference(self):
        self.use_rknn_inference = False
        if self.rknn_runtime is not None:
            try:
                self.rknn_runtime.release()
            except Exception:
                pass
            self.rknn_runtime = None
            if self.enable_debug:
                print("[RKNN] Runtime released")

    def inference(self, bev_image: np.ndarray, proprioceptive: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run inference to get action and confidence.
        Uses RKNN runtime when enabled; otherwise PyTorch."""
        if self.use_rknn_inference and self.rknn_runtime is not None:
            try:
                processed = self.preprocess_bev_for_model(bev_image)  # (C,H,W)
                bev_input = processed[np.newaxis, ...].astype(np.float32)
                sensor_input = proprioceptive.astype(np.float32)[np.newaxis, ...]
                # RKNN expects list of inputs in same order as export
                outputs = self.rknn_runtime.inference(inputs=[bev_input, sensor_input])
                if not outputs:
                    raise RuntimeError("RKNN inference returned no outputs")
                out = outputs[0]  # (1,3)
                if out is None:
                    raise RuntimeError("RKNN output None")
                raw = out[0]
                # Apply same post-processing as PyTorch path
                action = np.tanh(raw[:2])
                confidence = self._stable_sigmoid(raw[2])
                return action.astype(np.float32), float(confidence)
            except Exception as e:
                if self.enable_debug:
                    print(f"[RKNN] Inference failed, falling back to PyTorch: {e}")
                # Fallback to PyTorch
        # Default PyTorch path
        self.model.eval()
        with torch.no_grad():
            processed = self.preprocess_bev_for_model(bev_image)
            bev_tensor = torch.from_numpy(processed).unsqueeze(0).float().to(self.device)
            sensor_tensor = torch.from_numpy(proprioceptive).float().unsqueeze(0).to(self.device)
            output = self.model(bev_tensor, sensor_tensor)
            action = torch.tanh(output[0, :2]).cpu().numpy()
            confidence = torch.sigmoid(output[0, 2]).item()
        self.model.train()
        return action, float(confidence)
        
    def _stable_sigmoid(self, x):
        """Numerically stable sigmoid function to avoid overflow warnings"""
        x = np.float64(x)  # Use higher precision
        if x >= 0:
            exp_neg_x = np.exp(-x)
            return 1.0 / (1.0 + exp_neg_x)
        else:
            exp_x = np.exp(x)
            return exp_x / (1.0 + exp_x)
        
    def _get_rknn_conversion_frequency(self) -> int:
        """Determine RKNN conversion frequency based on training progress.
        More frequent updates early when model might be poor."""
        if self.generation <= 20:
            return 5   # Very frequent early updates (every 5 generations)
        elif self.generation <= 50:
            return 10  # Frequent updates in early training
        elif self.generation <= 100:
            return 25  # Moderate frequency 
        else:
            return 100  # Standard frequency for mature training
    
    def save_model(self):
        """Save PyTorch model and training state"""
        try:
            # Ensure model directory exists
            os.makedirs(self.model_dir, exist_ok=True)
            
            timestamp = int(time.time())
            
            # Save PyTorch model
            model_path = os.path.join(self.model_dir, f"exploration_model_bev_es_{timestamp}.pth")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'generation': self.generation,
                'best_fitness': self.best_fitness,
                'fitness_history': list(self.fitness_history),
                'elite_individuals': self.elite_individuals,
                'elite_fitness_scores': self.elite_fitness_scores,
                'sigma': self.sigma,
                'stagnation_counter': self.stagnation_counter,
                'momentum': self.momentum,
                'velocity': self.velocity,
                'update_step': self.update_step,
                'diversity_history': list(self.diversity_history)
            }, model_path)
            
            # Save latest symlink
            latest_path = os.path.join(self.model_dir, "exploration_model_bev_es_latest.pth")
            if os.path.exists(latest_path):
                os.remove(latest_path)
            os.symlink(os.path.basename(model_path), latest_path)
            
            print(f"ES Model saved: {model_path}")
        except Exception as e:
            print(f"Failed to save ES model: {e}")
        
    def load_latest_model(self):
        """Load the latest saved model"""
        try:
            latest_path = os.path.join(self.model_dir, "exploration_model_bev_es_latest.pth")
            if os.path.exists(latest_path) and os.path.islink(latest_path):
                actual_model_path = os.path.join(self.model_dir, os.readlink(latest_path))
            elif os.path.exists(latest_path):
                actual_model_path = latest_path
            else:
                # Try to load RL model as starting point
                latest_path_rl = os.path.join(self.model_dir, "exploration_model_bev_latest.pth")
                if os.path.exists(latest_path_rl) and os.path.islink(latest_path_rl):
                    actual_model_path = os.path.join(self.model_dir, os.readlink(latest_path_rl))
                elif os.path.exists(latest_path_rl):
                    actual_model_path = latest_path_rl
                else:
                    print("No saved model found, starting with fresh model")
                    return
            if not os.path.exists(actual_model_path):
                print(f"Model file {actual_model_path} not found")
                return
            checkpoint = torch.load(actual_model_path, map_location=self.device)
            state_dict = checkpoint['model_state_dict']
            # Load model
            self.model.load_state_dict(state_dict)
            self.generation = checkpoint.get('generation', 0)
            self.best_fitness = checkpoint.get('best_fitness', -float('inf'))
            if 'fitness_history' in checkpoint:
                self.fitness_history = deque(checkpoint['fitness_history'], maxlen=1000)
            
            # Load elite preservation data if available
            if 'elite_individuals' in checkpoint and 'elite_fitness_scores' in checkpoint:
                self.elite_individuals = checkpoint['elite_individuals']
                self.elite_fitness_scores = checkpoint['elite_fitness_scores']
                print(f"Loaded {len(self.elite_individuals)} elite individuals")
            
            # Load adaptive sigma parameters if available
            if 'sigma' in checkpoint:
                self.sigma = checkpoint['sigma']
            if 'stagnation_counter' in checkpoint:
                self.stagnation_counter = checkpoint['stagnation_counter']
            
            # Load momentum parameters if available
            if 'momentum' in checkpoint and checkpoint['momentum'] is not None:
                self.momentum = checkpoint['momentum']
            if 'velocity' in checkpoint and checkpoint['velocity'] is not None:
                self.velocity = checkpoint['velocity']
            if 'update_step' in checkpoint:
                self.update_step = checkpoint['update_step']
            
            # Load diversity history if available
            if 'diversity_history' in checkpoint:
                self.diversity_history = deque(checkpoint['diversity_history'], maxlen=20)
            
            print(f"Loaded ES model from {actual_model_path}")
            print(f"Generation: {self.generation}, Best Fitness: {self.best_fitness}, Sigma: {self.sigma:.6f}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Starting with fresh model")
            
    def _proprio_feature_size(self):
        """Return proprioceptive feature size consistent with network construction.
        Matches 'proprio_inputs = 3 + extra_proprio' used in __init__."""
        return 3 + self.extra_proprio

    def convert_to_rknn(self):
        """Convert PyTorch model to RKNN format for NPU inference (fixed input shapes) with detailed debug."""
        if not RKNN_AVAILABLE:
            print("RKNN not available - skipping conversion")
            # Still export ONNX for inspection
            try:
                onnx_path = os.path.join(self.model_dir, "exploration_model_bev_es.onnx")
                dummy_bev = torch.randn(1, self.bev_channels, self.bev_height, self.bev_width).to(self.device)
                dummy_sensor = torch.randn(1, self._proprio_feature_size()).to(self.device)
                torch.onnx.export(
                    self.model,
                    (dummy_bev, dummy_sensor),
                    onnx_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['bev_image', 'sensor'],
                    output_names=['action_confidence'],
                    dynamic_axes=None
                )
                print(f"ONNX exported (without RKNN toolkit): {onnx_path}")
            except Exception as e:
                print(f"ONNX export failed (no RKNN toolkit): {e}")
            return
        try:
            if self.enable_debug:
                print("[RKNN] Starting conversion pipeline")
            os.makedirs(self.model_dir, exist_ok=True)
            dummy_bev = torch.randn(1, self.bev_channels, self.bev_height, self.bev_width).to(self.device)
            proprio_size = self._proprio_feature_size()
            dummy_sensor = torch.randn(1, proprio_size).to(self.device)
            if self.enable_debug:
                print(f"[RKNN] Dummy tensors created bev_shape={tuple(dummy_bev.shape)} sensor_shape={tuple(dummy_sensor.shape)}")
            # Sanity check flatten size
            with torch.no_grad():
                self.model.eval()
                bev_features = self.model.bev_conv(dummy_bev)
                # bev_conv ends with Flatten(), so bev_features is (B, F)
                flat_dim = bev_features.shape[1]
                expected = self.model.bev_fc.in_features
                if self.enable_debug:
                    print(f"[RKNN] BEV feature flatten size={flat_dim} expected={expected}")
                if flat_dim != expected:
                    print(f"[RKNN Export] WARNING: bev flatten size {flat_dim} != expected {expected}. Aborting export.")
                    self.model.train()
                    return
            onnx_path = os.path.join(self.model_dir, "exploration_model_bev_es.onnx")
            if self.enable_debug:
                print(f"[RKNN] Exporting ONNX to {onnx_path}")
            torch.onnx.export(
                self.model,
                (dummy_bev, dummy_sensor),
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['bev_image', 'sensor'],
                output_names=['action_confidence'],
                dynamic_axes=None
            )
            if self.enable_debug:
                print("[RKNN] ONNX export complete")
            dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'dataset.txt')
            dataset_path = os.path.abspath(dataset_path)
            if self.enable_debug:
                print(f"[RKNN] Dataset path: {dataset_path}")
            rknn = RKNN(verbose=self.enable_debug)
            # First try simplest config (some toolkit versions choke on multi-input mean/std lists)
            try:
                if self.enable_debug:
                    print("[RKNN] Calling config (simple)")
                rknn.config(target_platform='rk3588')
            except Exception as e:
                print(f"[RKNN] Simple config failed: {e}. Trying bev-only mean/std.")
                try:
                    mean_values = [[0.0] * self.bev_channels]
                    std_values = [[1.0] * self.bev_channels]
                    rknn.config(mean_values=mean_values, std_values=std_values, target_platform='rk3588')
                except Exception as e2:
                    print(f"[RKNN] BEV-only config failed: {e2}. Aborting.")
                    rknn.release()
                    self.model.train()
                    return
            if self.enable_debug:
                print("[RKNN] Loading ONNX model")
            ret = rknn.load_onnx(model=onnx_path)
            if ret != 0:
                print("[RKNN] Failed to load ONNX model (ret != 0)")
                rknn.release()
                self.model.train()
                return
            if self.enable_debug:
                print("[RKNN] ONNX loaded successfully")
            do_quantization = False
            if os.path.exists(dataset_path):
                try:
                    with open(dataset_path, 'r') as f:
                        if f.read().strip():
                            do_quantization = True
                except Exception as e:
                    print(f"[RKNN] Error reading dataset file: {e}")
            if self.enable_debug:
                print(f"[RKNN] Building (quantization={do_quantization})")
            if do_quantization:
                ret = rknn.build(do_quantization=True, dataset=dataset_path)
            else:
                ret = rknn.build(do_quantization=False)
            if ret != 0:
                print("[RKNN] Failed to build RKNN model (ret != 0)")
                rknn.release()
                self.model.train()
                return
            if self.enable_debug:
                print("[RKNN] Build successful, exporting RKNN")
            rknn_path = os.path.join(self.model_dir, "exploration_model_bev_es.rknn")
            ret = rknn.export_rknn(rknn_path)
            if ret != 0:
                print("[RKNN] Failed to export RKNN model")
            else:
                print(f"RKNN model saved: {rknn_path}")
                # Hot-reload runtime if currently using RKNN inference
                if getattr(self, 'use_rknn_inference', False):
                    if self.enable_debug:
                        print("[RKNN] Reloading runtime with new model")
                    self.enable_rknn_inference()
            rknn.release()
            if self.enable_debug:
                print("[RKNN] Pipeline complete")
            self.model.train()
        except Exception as e:
            print(f"RKNN conversion failed (exception): {e}")
            import traceback
            traceback.print_exc()
            self.model.train()
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get current training statistics"""
        
        return {
            "generation": self.generation,
            "buffer_size": self.buffer_size,
            "buffer_capacity": self.buffer_capacity,
            "avg_fitness": np.mean(self.fitness_history) if self.fitness_history else 0.0,
            "best_fitness": self.best_fitness if self.best_fitness != -float('inf') else -999.99,
            "buffer_full": self.buffer_size / self.buffer_capacity,
            "evolution_frequency": self.current_evolution_frequency,
            "sigma": self.sigma
        }
    
    def init_dataset_file(self):
        """Initialize dataset.txt file for RKNN quantization"""
        try:
            dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'dataset.txt')
            dataset_path = os.path.abspath(dataset_path)
            
            # Create empty dataset.txt file (clear any existing content)
            with open(dataset_path, 'w') as f:
                pass  # Create empty file, no header needed for RKNN toolkit
            
            if self.enable_debug:
                print(f"[Dataset] Initialized dataset file: {dataset_path}")
                
        except Exception as e:
            if self.enable_debug:
                print(f"Failed to initialize dataset file: {e}")

    def save_quantization_sample(self, bev_image: np.ndarray, proprioceptive: np.ndarray):
        """Save a sample to dataset.txt for RKNN quantization"""
        try:
            # Get the dataset path (same as used in convert_to_rknn)
            dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'dataset.txt')
            dataset_path = os.path.abspath(dataset_path)
            
            # Preprocess the sample to match inference format
            processed_bev = self.preprocess_bev_for_model(bev_image)  # (C,H,W)
            bev_input = processed_bev[np.newaxis, ...].astype(np.float32)  # (1,C,H,W)
            
            # Pad/truncate proprioceptive data to match expected size
            expected = self._proprio_feature_size()
            if proprioceptive.shape[0] < expected:
                padded_proprio = np.concatenate([proprioceptive, np.zeros(expected - proprioceptive.shape[0], dtype=proprioceptive.dtype)])
            elif proprioceptive.shape[0] > expected:
                padded_proprio = proprioceptive[:expected]
            else:
                padded_proprio = proprioceptive
            sensor_input = padded_proprio.astype(np.float32)[np.newaxis, ...]  # (1,features)
            
            # For RKNN quantization, we need to save each input separately for multi-input models
            # Create separate files for each input in /tmp directory
            import tempfile
            temp_bev_file = tempfile.mktemp(suffix='.npy')
            temp_sensor_file = tempfile.mktemp(suffix='.npy')
            
            # Save inputs to temporary files
            np.save(temp_bev_file, bev_input)
            np.save(temp_sensor_file, sensor_input)
            
            # Write paths to dataset.txt (one line per sample with paths to input files)
            with open(dataset_path, 'a') as f:
                f.write(f"{temp_bev_file} {temp_sensor_file}\n")
            
            self.dataset_samples_collected += 1
            
            if self.enable_debug and self.dataset_samples_collected % 10 == 0:
                print(f"[Dataset] Collected {self.dataset_samples_collected}/{self.target_dataset_samples} quantization samples")
                
        except Exception as e:
            if self.enable_debug:
                print(f"Failed to save quantization sample: {e}")

    def safe_save(self):
        try:
            self.save_model()
        except Exception as e:
            print(f"Model save failed during shutdown: {e}")
