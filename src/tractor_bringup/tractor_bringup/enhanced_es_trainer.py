#!/usr/bin/env python3
"""
Enhanced Evolutionary Strategy Trainer with Multi-Objective Optimization
Implements advanced ES techniques for improved learning in robotics environments
"""

import numpy as np
import time
import os
from typing import Dict, List, Tuple, Optional, NamedTuple
import torch
import torch.nn as nn
from collections import deque
import cv2

try:
    from .optimized_depth_network import OptimizedDepthExplorationNet, DynamicInferenceController
    OPTIMIZED_NETWORK_AVAILABLE = True
except ImportError:
    from .rknn_trainer_depth import DepthImageExplorationNet as OptimizedDepthExplorationNet
    OPTIMIZED_NETWORK_AVAILABLE = False
    print("Using fallback to original network architecture")

try:
    from .bayesian_es_optimizer import AdaptiveBayesianESWrapper
    BAYESIAN_OPTIMIZATION = True
except ImportError:
    BAYESIAN_OPTIMIZATION = False

try:
    from .improved_reward_system import ImprovedRewardCalculator
    IMPROVED_REWARDS = True
except ImportError:
    IMPROVED_REWARDS = False

class MultiObjectiveESTrainer:
    """
    Enhanced Evolutionary Strategy trainer with multi-objective optimization,
    curriculum learning, and advanced exploration strategies
    """
    
    def __init__(self, model_dir="models", stacked_frames: int = 1, enable_debug: bool = False,
                 population_size: int = 15, sigma: float = 0.08, learning_rate: float = 0.015,
                 enable_bayesian_optimization: bool = True, performance_mode: str = "balanced"):
        
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.enable_debug = enable_debug
        self.performance_mode = performance_mode
        
        # Image processing parameters
        self.target_h = 160
        self.target_w = 288
        self.clip_max_distance = 4.0
        self.stacked_frames = stacked_frames
        self.frame_stack: deque = deque(maxlen=stacked_frames)
        
        # Enhanced proprioceptive features
        self.extra_proprio = 13
        self.proprio_dim = 3 + self.extra_proprio
        
        # Multi-objective ES parameters
        self.population_size = max(population_size, 10)  # Minimum viable population
        self.sigma = sigma
        self.initial_sigma = sigma
        self.learning_rate = learning_rate
        self.generation = 0
        
        # Multi-objective weights and tracking
        self.objective_weights = {
            "exploration": 0.4,      # Encourage exploration and novelty
            "efficiency": 0.3,       # Reward efficient movement and goal reaching
            "safety": 0.2,           # Avoid collisions and unsafe behaviors
            "smoothness": 0.1        # Encourage smooth, realistic actions
        }
        
        # Adaptive objective weight adjustment
        self.objective_history = {obj: deque(maxlen=100) for obj in self.objective_weights.keys()}
        self.objective_improvement_thresholds = {
            "exploration": 0.02,
            "efficiency": 0.05,
            "safety": 0.01,
            "smoothness": 0.03
        }
        
        # Enhanced population management
        self.elite_ratio = 0.25  # Preserve top 25%
        self.n_elites = max(2, int(self.population_size * self.elite_ratio))
        self.diversity_injection_ratio = 0.2
        self.novelty_archive_size = 50
        self.novelty_archive = []
        
        # Curriculum learning parameters
        self.curriculum_enabled = True
        self.curriculum_stage = 0
        self.curriculum_progression_threshold = 20  # Generations before advancing
        self.curriculum_stages = [
            {"name": "basic_movement", "complexity": 0.3, "safety_weight": 0.4},
            {"name": "obstacle_avoidance", "complexity": 0.6, "safety_weight": 0.3},
            {"name": "complex_exploration", "complexity": 1.0, "safety_weight": 0.2}
        ]
        
        # Device setup optimized for RK3588
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(6)  # Optimal for RK3588 8-core ARM
        torch.set_num_interop_threads(2)
        
        # Create optimized neural network
        if OPTIMIZED_NETWORK_AVAILABLE:
            self.model = OptimizedDepthExplorationNet(
                stacked_frames=stacked_frames,
                extra_proprio=self.extra_proprio,
                performance_mode=performance_mode,
                enable_temporal=False  # Disable for ES simplicity
            ).to(self.device)
            print(f"✓ Using optimized network architecture ({performance_mode} mode)")
        else:
            self.model = OptimizedDepthExplorationNet(
                stacked_frames=stacked_frames,
                extra_proprio=self.extra_proprio
            ).to(self.device)
            print("✓ Using fallback network architecture")
        
        # Dynamic inference controller
        if OPTIMIZED_NETWORK_AVAILABLE:
            self.inference_controller = DynamicInferenceController(target_fps=30.0)
        else:
            self.inference_controller = None
        
        # Enhanced experience buffer with priority and meta-learning
        self.buffer_capacity = 75000  # Larger buffer for better diversity
        self.buffer_size = 0
        self.insert_ptr = 0
        
        # Multi-modal storage
        self.depth_store = np.zeros((self.buffer_capacity, self.stacked_frames, self.target_h, self.target_w), dtype=np.float32)
        self.proprio_store = np.zeros((self.buffer_capacity, self.proprio_dim), dtype=np.float32)
        self.action_store = np.zeros((self.buffer_capacity, 2), dtype=np.float32)
        self.reward_store = np.zeros((self.buffer_capacity,), dtype=np.float32)
        self.objective_scores_store = np.zeros((self.buffer_capacity, len(self.objective_weights)), dtype=np.float32)
        self.done_store = np.zeros((self.buffer_capacity,), dtype=np.uint8)
        self.episode_store = np.zeros((self.buffer_capacity,), dtype=np.int32)
        
        # Population and evolution tracking
        self.population = []
        self.population_fitness = []
        self.population_objectives = []  # Multi-objective scores for each individual
        
        # Advanced evolution mechanisms
        self.momentum_decay = 0.9
        self.velocity_decay = 0.999
        self.epsilon = 1e-8
        self.momentum = None
        self.velocity = None
        self.update_step = 0
        
        # Meta-learning and adaptation
        self.meta_learning_enabled = True
        self.adaptation_history = deque(maxlen=20)
        self.performance_trend_window = 10
        
        # Initialize ES components
        self._initialize_population()
        self._setup_reward_system()
        self._setup_bayesian_optimization(enable_bayesian_optimization)
        
        # Load existing model if available
        self.load_latest_model()
        
        # Performance monitoring
        self.training_metrics = {
            "generation": 0,
            "avg_fitness": 0.0,
            "best_fitness": -float('inf'),
            "diversity": 0.0,
            "curriculum_stage": 0,
            "adaptation_rate": 0.0
        }
        
        print(f"✓ Enhanced ES Trainer initialized:")
        print(f"  Population: {self.population_size}, Sigma: {self.sigma:.4f}")
        print(f"  Multi-objective: {list(self.objective_weights.keys())}")
        print(f"  Curriculum: {'Enabled' if self.curriculum_enabled else 'Disabled'}")
        print(f"  Performance mode: {performance_mode}")
    
    def _initialize_population(self):
        """Initialize population with strategic diversity"""
        self.population = []
        self.population_fitness = []
        self.population_objectives = []
        
        # Get parameter template
        params = self._get_flat_params()
        self.param_shape = params.shape
        
        # Strategic initialization: combination of methods
        for i in range(self.population_size):
            if i < self.population_size // 3:
                # Random initialization with different scales
                scale = 0.5 + (i / (self.population_size // 3)) * 1.0
                perturbation = np.random.randn(*self.param_shape) * self.sigma * scale
            elif i < 2 * self.population_size // 3:
                # Orthogonal initialization for diversity
                perturbation = self._generate_orthogonal_perturbation(i - self.population_size // 3)
            else:
                # Small perturbations around current model
                perturbation = np.random.randn(*self.param_shape) * self.sigma * 0.5
            
            self.population.append(perturbation)
            self.population_fitness.append(-float('inf'))
            self.population_objectives.append(np.zeros(len(self.objective_weights)))
        
        if self.enable_debug:
            print(f"✓ Initialized population with {len(self.population)} individuals")
    
    def _generate_orthogonal_perturbation(self, index: int) -> np.ndarray:
        """Generate orthogonal perturbation for better diversity"""
        base_perturbation = np.random.randn(*self.param_shape) * self.sigma
        
        # Apply orthogonalization if we have previous perturbations
        if len(self.population) > 0:
            for existing_pert in self.population[-min(5, len(self.population)):]:
                # Gram-Schmidt-like orthogonalization
                projection = np.dot(base_perturbation, existing_pert) / (np.linalg.norm(existing_pert) + 1e-8)
                base_perturbation -= projection * existing_pert / (np.linalg.norm(existing_pert) + 1e-8)
        
        # Normalize and scale
        norm = np.linalg.norm(base_perturbation)
        if norm > 1e-8:
            base_perturbation = base_perturbation / norm * self.sigma * np.sqrt(len(base_perturbation))
        
        return base_perturbation
    
    def _setup_reward_system(self):
        """Setup enhanced reward calculation system"""
        if IMPROVED_REWARDS:
            self.reward_calculator = ImprovedRewardCalculator(
                enable_anti_gaming=True,
                enable_diversity_tracking=True,
                enable_reward_smoothing=False  # Handle smoothing in ES layer
            )
            print("✓ Enhanced reward system initialized")
        else:
            self.reward_calculator = None
            print("⚠ Using basic reward system")
    
    def _setup_bayesian_optimization(self, enable_bayesian: bool):
        """Setup Bayesian optimization for hyperparameters"""
        self.bayesian_es_wrapper = None
        if enable_bayesian and BAYESIAN_OPTIMIZATION:
            try:
                self.bayesian_es_wrapper = AdaptiveBayesianESWrapper(
                    es_trainer=self,
                    optimization_interval=8,  # More frequent optimization
                    min_observations=5,
                    enable_debug=self.enable_debug
                )
                print("✓ Bayesian hyperparameter optimization enabled")
            except Exception as e:
                print(f"⚠ Bayesian optimization failed to initialize: {e}")
    
    def calculate_multi_objective_fitness(self, 
                                       individual_idx: int,
                                       depth_batch: np.ndarray,
                                       proprio_batch: np.ndarray,
                                       action_batch: np.ndarray,
                                       reward_batch: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Calculate multi-objective fitness for an individual
        Returns: (combined_fitness, objective_scores)
        """
        
        # Apply individual to model
        original_params = self._get_flat_params()
        perturbed_params = original_params + self.population[individual_idx]
        self._set_flat_params(perturbed_params)
        
        batch_size = len(depth_batch)
        total_objectives = np.zeros(len(self.objective_weights))
        
        with torch.no_grad():
            # Batch inference for efficiency
            depth_tensor = torch.from_numpy(depth_batch).float().to(self.device)
            proprio_tensor = torch.from_numpy(proprio_batch).float().to(self.device)
            
            predicted_output = self.model(depth_tensor, proprio_tensor)
            predicted_actions = torch.tanh(predicted_output[:, :2]).cpu().numpy()
        
        # Calculate objective components
        for i in range(batch_size):
            pred_action = predicted_actions[i]
            actual_action = action_batch[i]
            reward = reward_batch[i]
            
            # 1. Exploration objective (novelty and diversity)
            exploration_score = self._calculate_exploration_objective(pred_action, actual_action, reward)
            
            # 2. Efficiency objective (task performance and speed)
            efficiency_score = self._calculate_efficiency_objective(pred_action, actual_action, reward)
            
            # 3. Safety objective (collision avoidance and stability)
            safety_score = self._calculate_safety_objective(pred_action, actual_action, reward)
            
            # 4. Smoothness objective (action consistency and realism)
            smoothness_score = self._calculate_smoothness_objective(pred_action, actual_action)
            
            objectives = np.array([exploration_score, efficiency_score, safety_score, smoothness_score])
            total_objectives += objectives
        
        # Average objectives over batch
        avg_objectives = total_objectives / batch_size
        
        # Apply curriculum learning weights
        curriculum_weights = self._get_curriculum_weights()
        weighted_objectives = avg_objectives * curriculum_weights
        
        # Combine objectives into single fitness
        combined_fitness = np.sum(weighted_objectives * list(self.objective_weights.values()))
        
        # Restore original parameters
        self._set_flat_params(original_params)
        
        return combined_fitness, avg_objectives
    
    def _calculate_exploration_objective(self, pred_action: np.ndarray, actual_action: np.ndarray, reward: float) -> float:
        """Calculate exploration/novelty objective"""
        # Action similarity (how well we predict actual exploratory actions)
        action_similarity = np.exp(-np.linalg.norm(pred_action - actual_action))
        
        # Novelty bonus (encourage unique behaviors)
        novelty_score = self._calculate_action_novelty(pred_action)
        
        # Exploration reward component
        exploration_component = max(0, reward) if reward > 0 else 0
        
        return (action_similarity * 0.4 + novelty_score * 0.3 + exploration_component * 0.3)
    
    def _calculate_efficiency_objective(self, pred_action: np.ndarray, actual_action: np.ndarray, reward: float) -> float:
        """Calculate task efficiency objective"""
        # Forward movement preference (efficiency in exploration)
        forward_component = max(0, pred_action[0]) * 2.0
        
        # Speed appropriateness (not too slow, not too fast)
        speed = abs(pred_action[0])
        speed_efficiency = 1.0 if 0.05 < speed < 0.25 else max(0, 1.0 - abs(speed - 0.15) / 0.15)
        
        # Reward-based efficiency
        efficiency_reward = max(0, reward) * 0.1
        
        return (forward_component * 0.5 + speed_efficiency * 0.3 + efficiency_reward * 0.2)
    
    def _calculate_safety_objective(self, pred_action: np.ndarray, actual_action: np.ndarray, reward: float) -> float:
        """Calculate safety objective"""
        # Penalize extreme actions
        action_magnitude = np.linalg.norm(pred_action)
        safety_penalty = max(0, action_magnitude - 1.0) * 2.0
        
        # Collision avoidance (negative rewards indicate danger)
        collision_penalty = max(0, -reward) if reward < 0 else 0
        
        # Angular velocity reasonableness
        angular_penalty = max(0, abs(pred_action[1]) - 0.8) * 1.5
        
        safety_score = 1.0 - (safety_penalty + collision_penalty * 0.1 + angular_penalty)
        return max(0, safety_score)
    
    def _calculate_smoothness_objective(self, pred_action: np.ndarray, actual_action: np.ndarray) -> float:
        """Calculate action smoothness objective"""
        # Action consistency with actual behavior
        consistency = 1.0 / (1.0 + np.linalg.norm(pred_action - actual_action))
        
        # Internal action smoothness (no extreme changes)
        internal_smoothness = 1.0 / (1.0 + abs(pred_action[0] - pred_action[1]))
        
        return (consistency * 0.7 + internal_smoothness * 0.3)
    
    def _calculate_action_novelty(self, action: np.ndarray) -> float:
        """Calculate novelty of action compared to archive"""
        if len(self.novelty_archive) == 0:
            return 1.0
        
        # Calculate distance to nearest neighbor in novelty archive
        distances = [np.linalg.norm(action - archived_action) for archived_action in self.novelty_archive]
        min_distance = min(distances)
        
        # Convert distance to novelty score
        novelty = min(min_distance / 0.5, 1.0)  # Normalize by expected action range
        
        # Add to archive if sufficiently novel
        if novelty > 0.3:
            self.novelty_archive.append(action.copy())
            if len(self.novelty_archive) > self.novelty_archive_size:
                self.novelty_archive.pop(0)
        
        return novelty
    
    def _get_curriculum_weights(self) -> np.ndarray:
        """Get curriculum learning weights for current stage"""
        if not self.curriculum_enabled or self.curriculum_stage >= len(self.curriculum_stages):
            return np.ones(len(self.objective_weights))
        
        stage = self.curriculum_stages[self.curriculum_stage]
        
        # Adjust weights based on curriculum stage
        weights = np.ones(len(self.objective_weights))
        if "safety_weight" in stage:
            # Increase safety weight in early stages
            weights[2] *= stage["safety_weight"] / self.objective_weights["safety"]  # Safety is index 2
        
        return weights / np.sum(weights) * len(weights)  # Normalize while maintaining scale
    
    def evolve_population(self) -> Dict[str, float]:
        """Enhanced evolution with multi-objective optimization"""
        if self.buffer_size < 64:  # Need sufficient data
            return {"error": "insufficient_data", "buffer_size": self.buffer_size}
        
        # Sample diverse batch for evaluation
        batch_size = min(128, self.buffer_size)
        indices = self._sample_diverse_batch(batch_size)
        
        depth_batch = self.depth_store[indices]
        proprio_batch = self.proprio_store[indices]
        action_batch = self.action_store[indices]
        reward_batch = self.reward_store[indices]
        
        # Evaluate all individuals
        fitness_scores = []
        objective_scores_list = []
        
        for i in range(self.population_size):
            fitness, objectives = self.calculate_multi_objective_fitness(
                i, depth_batch, proprio_batch, action_batch, reward_batch
            )
            fitness_scores.append(fitness)
            objective_scores_list.append(objectives)
        
        fitness_scores = np.array(fitness_scores)
        objective_scores_matrix = np.array(objective_scores_list)
        
        # Update tracking
        self.population_fitness = fitness_scores.tolist()
        self.population_objectives = objective_scores_list
        
        # Track objective improvements
        for i, obj_name in enumerate(self.objective_weights.keys()):
            avg_obj_score = np.mean(objective_scores_matrix[:, i])
            self.objective_history[obj_name].append(avg_obj_score)
        
        # Update best model if improved
        best_idx = np.argmax(fitness_scores)
        current_best_fitness = fitness_scores[best_idx]
        
        if current_best_fitness > self.training_metrics["best_fitness"]:
            self.training_metrics["best_fitness"] = current_best_fitness
            # Apply best individual to model
            best_params = self._get_flat_params() + self.population[best_idx]
            self._set_flat_params(best_params)
            print(f"✓ New best fitness: {current_best_fitness:.4f}")
        
        # Adaptive objective weight adjustment
        self._adapt_objective_weights()
        
        # Multi-objective evolution update
        self._apply_multi_objective_update(fitness_scores, objective_scores_matrix)
        
        # Curriculum progression check
        self._update_curriculum_stage()
        
        # Generate new population
        self._generate_next_population(fitness_scores, objective_scores_matrix)
        
        # Update generation counter
        self.generation += 1
        
        # Apply Bayesian optimization
        if self.bayesian_es_wrapper is not None:
            self.bayesian_es_wrapper.apply_bayesian_optimization(self.generation, current_best_fitness)
        
        # Prepare return statistics
        stats = self._compile_evolution_stats(fitness_scores, objective_scores_matrix)
        self.training_metrics.update(stats)
        
        return stats
    
    def _sample_diverse_batch(self, batch_size: int) -> np.ndarray:
        """Sample diverse batch prioritizing recent and diverse experiences"""
        if self.buffer_size <= batch_size:
            return np.arange(self.buffer_size)
        
        # Prioritize recent experiences (80% recent, 20% random)
        recent_size = int(batch_size * 0.8)
        random_size = batch_size - recent_size
        
        # Recent experiences
        recent_start = max(0, self.buffer_size - recent_size * 2)
        recent_indices = np.random.choice(
            np.arange(recent_start, self.buffer_size),
            size=recent_size,
            replace=False
        )
        
        # Random diverse experiences
        remaining_indices = np.arange(0, recent_start)
        if len(remaining_indices) > 0:
            random_indices = np.random.choice(
                remaining_indices,
                size=min(random_size, len(remaining_indices)),
                replace=False
            )
        else:
            random_indices = np.array([])
        
        return np.concatenate([recent_indices, random_indices])
    
    def _adapt_objective_weights(self):
        """Adaptively adjust objective weights based on progress"""
        if not self.meta_learning_enabled:
            return
        
        for obj_name in self.objective_weights.keys():
            history = self.objective_history[obj_name]
            if len(history) >= 20:
                # Calculate improvement trend
                recent_avg = np.mean(list(history)[-10:])
                earlier_avg = np.mean(list(history)[-20:-10])
                improvement = recent_avg - earlier_avg
                
                threshold = self.objective_improvement_thresholds[obj_name]
                
                if improvement < -threshold:  # Declining performance
                    self.objective_weights[obj_name] *= 1.1  # Increase weight
                elif improvement > threshold * 2:  # Excellent performance
                    self.objective_weights[obj_name] *= 0.95  # Slightly decrease weight
        
        # Renormalize weights
        total_weight = sum(self.objective_weights.values())
        for obj_name in self.objective_weights.keys():
            self.objective_weights[obj_name] /= total_weight
        
        if self.enable_debug and self.generation % 20 == 0:
            print(f"📊 Objective weights: {self.objective_weights}")
    
    def _apply_multi_objective_update(self, fitness_scores: np.ndarray, objective_scores: np.ndarray):
        """Apply sophisticated parameter update using multi-objective information"""
        # Rank individuals by fitness
        fitness_ranks = np.argsort(np.argsort(-fitness_scores))
        
        # Calculate rank-based weights
        weights = np.maximum(0, np.log(self.population_size/2 + 1) - np.log(fitness_ranks + 1))
        weights = weights / np.sum(weights) - 1.0 / self.population_size
        
        # Multi-objective gradient estimation
        original_params = self._get_flat_params()
        grad_estimate = np.zeros_like(original_params)
        
        # Primary fitness gradient
        for i in range(self.population_size):
            grad_estimate += weights[i] * self.population[i]
        grad_estimate /= self.sigma
        
        # Multi-objective diversity gradient (encourage diverse solutions)
        diversity_grad = self._calculate_diversity_gradient(objective_scores)
        
        # Combine gradients
        total_grad = grad_estimate + 0.1 * diversity_grad
        
        # Apply momentum-based update
        self.update_step += 1
        
        if self.momentum is None:
            self.momentum = np.zeros_like(total_grad)
            self.velocity = np.zeros_like(total_grad)
        
        # Update momentum and velocity
        self.momentum = self.momentum_decay * self.momentum + (1 - self.momentum_decay) * total_grad
        self.velocity = self.velocity_decay * self.velocity + (1 - self.velocity_decay) * (total_grad ** 2)
        
        # Bias correction
        momentum_corrected = self.momentum / (1 - self.momentum_decay ** self.update_step)
        velocity_corrected = self.velocity / (1 - self.velocity_decay ** self.update_step)
        
        # Adaptive update
        adaptive_update = self.learning_rate * momentum_corrected / (np.sqrt(velocity_corrected) + self.epsilon)
        
        # Apply update
        new_params = original_params + adaptive_update
        self._set_flat_params(new_params)
    
    def _calculate_diversity_gradient(self, objective_scores: np.ndarray) -> np.ndarray:
        """Calculate gradient to encourage population diversity in objective space"""
        # Find individuals that are too similar in objective space
        diversity_grad = np.zeros(self.param_shape)
        
        for i in range(self.population_size):
            for j in range(i + 1, self.population_size):
                # Objective space distance
                obj_distance = np.linalg.norm(objective_scores[i] - objective_scores[j])
                
                # If too similar in objective space, push apart in parameter space
                if obj_distance < 0.1:
                    param_diff = self.population[i] - self.population[j]
                    param_distance = np.linalg.norm(param_diff)
                    
                    if param_distance > 1e-8:
                        # Push in direction of parameter difference
                        diversity_grad += param_diff / param_distance * (0.1 - obj_distance)
        
        return diversity_grad / self.population_size
    
    def _update_curriculum_stage(self):
        """Update curriculum learning stage based on performance"""
        if not self.curriculum_enabled or self.curriculum_stage >= len(self.curriculum_stages):
            return
        
        # Check if ready to advance curriculum
        if self.generation > (self.curriculum_stage + 1) * self.curriculum_progression_threshold:
            current_stage = self.curriculum_stages[self.curriculum_stage]
            
            # Check if performance criteria are met
            if len(self.objective_history["safety"]) >= 10:
                recent_safety = np.mean(list(self.objective_history["safety"])[-10:])
                if recent_safety > 0.7:  # Good safety performance
                    self.curriculum_stage += 1
                    print(f"📈 Advanced to curriculum stage {self.curriculum_stage}")
    
    def _generate_next_population(self, fitness_scores: np.ndarray, objective_scores: np.ndarray):
        """Generate next generation with enhanced diversity maintenance"""
        # Elite preservation with multi-objective considerations
        elite_indices = np.argsort(-fitness_scores)[:self.n_elites]
        
        # Generate new population
        new_population = []
        
        # Preserve elites
        for i in range(self.n_elites):
            elite_idx = elite_indices[i]
            new_population.append(self.population[elite_idx].copy())
        
        # Fill remaining slots
        for i in range(self.n_elites, self.population_size):
            if i < self.population_size * 0.8:  # 80% from evolution
                # Select parents based on fitness and diversity
                parent_idx = self._select_diverse_parent(fitness_scores, objective_scores)
                offspring = self._generate_offspring(parent_idx)
            else:  # 20% random injection for diversity
                offspring = np.random.randn(*self.param_shape) * self.sigma
            
            new_population.append(offspring)
        
        self.population = new_population
    
    def _select_diverse_parent(self, fitness_scores: np.ndarray, objective_scores: np.ndarray) -> int:
        """Select parent considering both fitness and diversity"""
        # Tournament selection with diversity bonus
        tournament_size = 3
        candidates = np.random.choice(self.population_size, tournament_size, replace=False)
        
        best_score = -float('inf')
        best_candidate = candidates[0]
        
        for candidate in candidates:
            # Base fitness
            score = fitness_scores[candidate]
            
            # Diversity bonus (favor individuals with unique objective profiles)
            diversity_bonus = self._calculate_objective_diversity(candidate, objective_scores)
            combined_score = score + 0.2 * diversity_bonus
            
            if combined_score > best_score:
                best_score = combined_score
                best_candidate = candidate
        
        return best_candidate
    
    def _calculate_objective_diversity(self, individual_idx: int, objective_scores: np.ndarray) -> float:
        """Calculate diversity bonus for individual in objective space"""
        individual_objectives = objective_scores[individual_idx]
        
        # Calculate distance to all other individuals in objective space
        distances = []
        for i in range(self.population_size):
            if i != individual_idx:
                distance = np.linalg.norm(individual_objectives - objective_scores[i])
                distances.append(distance)
        
        # Return average distance (higher = more diverse)
        return np.mean(distances) if distances else 0.0
    
    def _generate_offspring(self, parent_idx: int) -> np.ndarray:
        """Generate offspring with adaptive mutation"""
        parent = self.population[parent_idx]
        
        # Adaptive mutation strength based on performance
        fitness = self.population_fitness[parent_idx]
        avg_fitness = np.mean(self.population_fitness)
        
        if fitness > avg_fitness:
            # Good individual - small mutations
            mutation_strength = self.sigma * 0.8
        else:
            # Poor individual - larger mutations
            mutation_strength = self.sigma * 1.2
        
        # Generate offspring
        offspring = parent + np.random.randn(*self.param_shape) * mutation_strength
        
        return offspring
    
    def _compile_evolution_stats(self, fitness_scores: np.ndarray, objective_scores: np.ndarray) -> Dict[str, float]:
        """Compile comprehensive evolution statistics"""
        stats = {
            "generation": self.generation,
            "avg_fitness": float(np.mean(fitness_scores)),
            "best_fitness": float(np.max(fitness_scores)),
            "fitness_std": float(np.std(fitness_scores)),
            "population_diversity": float(self._calculate_population_diversity()),
            "curriculum_stage": self.curriculum_stage,
            "buffer_size": self.buffer_size,
            "sigma": self.sigma
        }
        
        # Add objective-specific statistics
        for i, obj_name in enumerate(self.objective_weights.keys()):
            stats[f"avg_{obj_name}"] = float(np.mean(objective_scores[:, i]))
            stats[f"best_{obj_name}"] = float(np.max(objective_scores[:, i]))
        
        # Add objective weights
        for obj_name, weight in self.objective_weights.items():
            stats[f"weight_{obj_name}"] = float(weight)
        
        return stats
    
    def _calculate_population_diversity(self) -> float:
        """Calculate population diversity in parameter space"""
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
    
    # Additional methods for compatibility with existing codebase
    def _get_flat_params(self) -> np.ndarray:
        """Get model parameters as flat numpy array"""
        params = []
        for param in self.model.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)
    
    def _set_flat_params(self, params: np.ndarray):
        """Set model parameters from flat numpy array"""
        idx = 0
        for param in self.model.parameters():
            param_shape = param.data.shape
            param_size = param.data.numel()
            param_values = params[idx:idx+param_size].reshape(param_shape)
            param.data = torch.from_numpy(param_values).float().to(self.device)
            idx += param_size
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get current training statistics"""
        return self.training_metrics.copy()
    
    def save_model(self):
        """Save model with enhanced metadata"""
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            timestamp = int(time.time())
            
            model_path = os.path.join(self.model_dir, f"enhanced_es_model_{timestamp}.pth")
            
            # Save comprehensive state
            save_dict = {
                'model_state_dict': self.model.state_dict(),
                'generation': self.generation,
                'population': self.population,
                'population_fitness': self.population_fitness,
                'population_objectives': self.population_objectives,
                'objective_weights': self.objective_weights,
                'curriculum_stage': self.curriculum_stage,
                'training_metrics': self.training_metrics,
                'sigma': self.sigma,
                'momentum': self.momentum,
                'velocity': self.velocity,
                'update_step': self.update_step,
                'performance_mode': self.performance_mode
            }
            
            torch.save(save_dict, model_path)
            
            # Create latest symlink
            latest_path = os.path.join(self.model_dir, "enhanced_es_model_latest.pth")
            if os.path.exists(latest_path):
                os.remove(latest_path)
            os.symlink(os.path.basename(model_path), latest_path)
            
            print(f"✓ Enhanced ES model saved: {model_path}")
            
        except Exception as e:
            print(f"✗ Failed to save model: {e}")
    
    def load_latest_model(self):
        """Load latest model with enhanced state recovery"""
        try:
            latest_path = os.path.join(self.model_dir, "enhanced_es_model_latest.pth")
            
            if os.path.exists(latest_path) and os.path.islink(latest_path):
                actual_path = os.path.join(self.model_dir, os.readlink(latest_path))
            elif os.path.exists(latest_path):
                actual_path = latest_path
            else:
                print("No saved enhanced model found")
                return
            
            checkpoint = torch.load(actual_path, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Restore training state
            self.generation = checkpoint.get('generation', 0)
            self.population = checkpoint.get('population', [])
            self.population_fitness = checkpoint.get('population_fitness', [])
            self.population_objectives = checkpoint.get('population_objectives', [])
            self.objective_weights = checkpoint.get('objective_weights', self.objective_weights)
            self.curriculum_stage = checkpoint.get('curriculum_stage', 0)
            self.training_metrics = checkpoint.get('training_metrics', self.training_metrics)
            self.sigma = checkpoint.get('sigma', self.sigma)
            self.momentum = checkpoint.get('momentum', None)
            self.velocity = checkpoint.get('velocity', None)
            self.update_step = checkpoint.get('update_step', 0)
            
            print(f"✓ Enhanced ES model loaded: Generation {self.generation}")
            print(f"  Curriculum stage: {self.curriculum_stage}")
            print(f"  Best fitness: {self.training_metrics.get('best_fitness', 'N/A')}")
            
        except Exception as e:
            print(f"⚠ Failed to load enhanced model: {e}")
    
    # Interface methods for compatibility
    def add_experience(self, depth_image: np.ndarray, proprioceptive: np.ndarray, 
                      action: np.ndarray, reward: float, next_depth_image: np.ndarray = None,
                      done: bool = False, collision: bool = False, in_recovery: bool = False):
        """Add experience to buffer with multi-objective decomposition"""
        
        # Preprocess depth image
        processed = self.preprocess_depth_for_storage(depth_image)
        
        # Ensure proprioceptive data has correct dimension
        if proprioceptive.shape[0] < self.proprio_dim:
            padded = np.zeros(self.proprio_dim, dtype=proprioceptive.dtype)
            padded[:proprioceptive.shape[0]] = proprioceptive
            proprioceptive = padded
        elif proprioceptive.shape[0] > self.proprio_dim:
            proprioceptive = proprioceptive[:self.proprio_dim]
        
        # Ensure action has correct dimension
        if action.shape[0] < 2:
            padded_action = np.zeros(2, dtype=np.float32)
            padded_action[:action.shape[0]] = action
            action = padded_action
        else:
            action = action[:2]
        
        # Store in ring buffer
        i = self.insert_ptr
        self.depth_store[i] = processed.astype(np.float32)
        self.proprio_store[i] = proprioceptive.astype(np.float32)
        self.action_store[i] = action.astype(np.float32)
        self.reward_store[i] = float(reward)
        self.done_store[i] = 1 if done else 0
        
        # Decompose reward into objectives (simplified)
        exploration_component = max(0, reward) * 0.4
        efficiency_component = max(0, reward - 2) * 0.3 if reward > 2 else 0
        safety_component = 1.0 if reward > -5 else max(0, (reward + 10) / 5)
        smoothness_component = 1.0 / (1.0 + np.linalg.norm(action)) * 0.5
        
        objective_scores = np.array([exploration_component, efficiency_component, 
                                   safety_component, smoothness_component], dtype=np.float32)
        self.objective_scores_store[i] = objective_scores
        
        # Update pointers
        self.insert_ptr = (i + 1) % self.buffer_capacity
        if self.buffer_size < self.buffer_capacity:
            self.buffer_size += 1
    
    def preprocess_depth_for_storage(self, depth_image: np.ndarray) -> np.ndarray:
        """Preprocess depth image for storage"""
        try:
            depth = np.nan_to_num(depth_image, nan=0.0, posinf=self.clip_max_distance, neginf=0.0)
            depth = np.clip(depth, 0.0, self.clip_max_distance)
            depth_resized = cv2.resize(depth, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)
            depth_norm = depth_resized / self.clip_max_distance
            
            if self.stacked_frames == 1:
                return depth_norm.astype(np.float32)[None, ...]
            else:
                return np.repeat(depth_norm[None, ...], self.stacked_frames, axis=0)
        except Exception:
            return np.zeros((self.stacked_frames, self.target_h, self.target_w), dtype=np.float32)
    
    def should_evolve(self, current_step: int) -> bool:
        """Determine if evolution should occur"""
        if self.buffer_size < 64:
            return False
        
        # Adaptive evolution frequency based on curriculum stage
        base_frequency = 50
        stage_modifier = 1.0 + self.curriculum_stage * 0.2  # Slower evolution in later stages
        evolution_frequency = int(base_frequency * stage_modifier)
        
        return current_step % evolution_frequency == 0
    
    def inference(self, depth_image: np.ndarray, proprioceptive: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run inference using current model"""
        self.model.eval()
        
        with torch.no_grad():
            # Preprocess inputs
            processed_depth = self.preprocess_depth_for_model(depth_image)
            depth_tensor = torch.from_numpy(processed_depth).unsqueeze(0).float().to(self.device)
            
            # Handle proprioceptive padding/truncation
            if proprioceptive.shape[0] < self.proprio_dim:
                padded = np.zeros(self.proprio_dim, dtype=proprioceptive.dtype)
                padded[:proprioceptive.shape[0]] = proprioceptive
                proprioceptive = padded
            elif proprioceptive.shape[0] > self.proprio_dim:
                proprioceptive = proprioceptive[:self.proprio_dim]
            
            sensor_tensor = torch.from_numpy(proprioceptive).float().unsqueeze(0).to(self.device)
            
            # Forward pass
            output = self.model(depth_tensor, sensor_tensor)
            action = torch.tanh(output[0, :2]).cpu().numpy()
            confidence = torch.sigmoid(output[0, 2]).item()
        
        self.model.train()
        return action, float(confidence)
    
    def preprocess_depth_for_model(self, depth_image: np.ndarray) -> np.ndarray:
        """Preprocess depth image for model inference"""
        try:
            depth = np.nan_to_num(depth_image, nan=0.0, posinf=self.clip_max_distance, neginf=0.0)
            depth = np.clip(depth, 0.0, self.clip_max_distance)
            depth_resized = cv2.resize(depth, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)
            depth_norm = depth_resized / self.clip_max_distance
            
            # Update frame stack
            self.frame_stack.append(depth_norm.astype(np.float32))
            while len(self.frame_stack) < self.stacked_frames:
                self.frame_stack.append(self.frame_stack[0] if self.frame_stack else depth_norm.astype(np.float32))
            
            stacked = np.stack(list(self.frame_stack), axis=0)
            return stacked
        except Exception:
            return np.zeros((self.stacked_frames, self.target_h, self.target_w), dtype=np.float32)