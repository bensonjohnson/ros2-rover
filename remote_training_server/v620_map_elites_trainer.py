#!/usr/bin/env python3
"""MAP-Elites trainer for rover on V620 with ROCm.

Quality-Diversity algorithm that maintains an archive of diverse driving behaviors.
Each cell in the archive represents a different behavior profile (speed × safety).
"""

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import zmq
import zstandard as zstd

from model_architectures import RGBDEncoder, PolicyHead  # Reuse network components


class ActorNetwork(nn.Module):
    """Actor-only network for MAP-Elites (no value head needed)."""

    def __init__(self, proprio_dim: int = 6):
        super().__init__()
        self.encoder = RGBDEncoder()
        self.policy_head = PolicyHead(self.encoder.output_dim, proprio_dim)

    def forward(self, rgb, depth, proprio):
        """Forward pass.

        Args:
            rgb: (B, 3, H, W) RGB image
            depth: (B, 1, H, W) Depth image
            proprio: (B, 6) Proprioception

        Returns:
            action: (B, 2) [linear_vel, angular_vel] in [-1, 1] range
        """
        features = self.encoder(rgb, depth)
        action = self.policy_head(features, proprio)
        return torch.tanh(action)  # Squash to [-1, 1] range


class PopulationTracker:
    """Simple population tracker for single-population evolution with adaptive sizing."""

    def __init__(self, initial_population_size: int = 10, max_population_size: int = 25):
        """Initialize population tracker.

        Args:
            initial_population_size: Starting population size (for aggressive early culling)
            max_population_size: Maximum population size (reached as training progresses)
        """
        self.initial_population_size = initial_population_size
        self.max_population_size = max_population_size
        self.population_size = initial_population_size  # Will grow adaptively

        # Population: list of {'model': state_dict, 'fitness': float, 'metrics': dict}
        # Sorted by fitness (best first)
        self.population = []

        # Statistics
        self.total_evaluations = 0
        self.improvements = 0

    def get_adaptive_population_size(self) -> int:
        """Compute adaptive population size based on training progress.

        Start small (10) for frequent tournaments, grow to preserve diversity.
        """
        if self.total_evaluations < 50:
            return self.initial_population_size  # 10: aggressive early culling
        elif self.total_evaluations < 200:
            return (self.initial_population_size + self.max_population_size) // 2  # 15-17: mid-stage
        else:
            return self.max_population_size  # 25: preserve diversity

    def update_population_size(self):
        """Update population size based on training progress."""
        self.population_size = self.get_adaptive_population_size()

    def add(
        self,
        model_state: dict,
        fitness: float,
        avg_speed: float,
        avg_clearance: float,
        metrics: dict
    ) -> Tuple[bool, float, int]:
        """Try to add model to population.

        Returns:
            (was_added, fitness_improvement, rank)
        """
        self.total_evaluations += 1

        # Update adaptive population size
        self.update_population_size()

        entry = {
            'model': copy.deepcopy(model_state),
            'fitness': fitness,
            'avg_speed': avg_speed,
            'avg_clearance': avg_clearance,
            'metrics': metrics,
        }

        # If population not full, always add
        if len(self.population) < self.population_size:
            self.population.append(entry)
            self.population.sort(key=lambda x: x['fitness'], reverse=True)
            self.improvements += 1
            rank = self.population.index(entry) + 1
            return True, float('inf'), rank

        # Check if better than worst in population
        worst_fitness = self.population[-1]['fitness']

        if fitness > worst_fitness:
            improvement = (fitness - worst_fitness) / max(abs(worst_fitness), 1e-6)

            # Replace worst model
            self.population[-1] = entry
            self.population.sort(key=lambda x: x['fitness'], reverse=True)
            self.improvements += 1
            rank = self.population.index(entry) + 1
            return True, improvement, rank

        return False, 0.0, -1

    def get_random_elite(self) -> Optional[dict]:
        """Sample random model from top half of population."""
        if not self.population:
            return None

        import random
        # Sample from top 50% (elites)
        elite_size = max(1, len(self.population) // 2)
        elite = random.choice(self.population[:elite_size])
        return elite['model']

    def get_best(self) -> Optional[dict]:
        """Get the best model."""
        if not self.population:
            return None
        return self.population[0]

    def calculate_behavior_novelty(self, avg_speed: float, avg_clearance: float) -> float:
        """Calculate how novel this behavior is compared to population.

        Returns novelty score in [0, 1] where 1 = very novel, 0 = common.
        """
        if len(self.population) < 3:
            return 0.5  # Not enough data, neutral novelty

        # Get existing behaviors
        speeds = [entry['avg_speed'] for entry in self.population]
        clearances = [entry['avg_clearance'] for entry in self.population]

        # Calculate minimum distance to existing behaviors (k-nearest neighbor style)
        distances = []
        for i in range(len(self.population)):
            # Normalize speed (0-0.3 m/s typical) and clearance (0-5m typical)
            speed_dist = abs(avg_speed - speeds[i]) / 0.3
            clearance_dist = abs(avg_clearance - clearances[i]) / 5.0
            euclidean_dist = np.sqrt(speed_dist**2 + clearance_dist**2)
            distances.append(euclidean_dist)

        # Use 3-nearest neighbor average distance as novelty
        distances.sort()
        k = min(3, len(distances))
        novelty = np.mean(distances[:k])

        # Clamp to [0, 1]
        return min(1.0, novelty)

    def get_stats(self) -> dict:
        """Get population statistics."""
        if not self.population:
            return {
                'population_size': 0,
                'total_evaluations': self.total_evaluations,
                'improvements': self.improvements,
            }

        fitnesses = [entry['fitness'] for entry in self.population if np.isfinite(entry['fitness'])]
        speeds = [entry['avg_speed'] for entry in self.population]
        clearances = [entry['avg_clearance'] for entry in self.population]

        stats = {
            'population_size': len(self.population),
            'total_evaluations': self.total_evaluations,
            'improvements': self.improvements,
            'speed_mean': np.mean(speeds) if speeds else 0.0,
            'clearance_mean': np.mean(clearances) if clearances else 0.0,
        }

        if fitnesses:
            stats.update({
                'fitness_mean': np.mean(fitnesses),
                'fitness_max': np.max(fitnesses),
                'fitness_min': np.min(fitnesses),
                'fitness_best': self.population[0]['fitness'],
            })

        return stats

    def save(self, filepath: str):
        """Save population to disk."""
        import json

        # Prepare metadata (without model weights)
        population_metadata = []
        for idx, entry in enumerate(self.population):
            population_metadata.append({
                'rank': idx + 1,
                'fitness': entry['fitness'],
                'avg_speed': entry['avg_speed'],
                'avg_clearance': entry['avg_clearance'],
                'metrics': entry['metrics']
            })

        metadata = {
            'population_size': self.population_size,
            'population': population_metadata,
            'total_evaluations': self.total_evaluations,
            'improvements': self.improvements,
        }

        print(f"  💾 Saving population with {len(population_metadata)} models", flush=True)

        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save models separately
        models_dir = Path(filepath).parent / 'evolution_models'
        models_dir.mkdir(exist_ok=True)

        for idx, entry in enumerate(self.population):
            model_path = models_dir / f'rank_{idx+1}.pt'
            torch.save(entry['model'], model_path)


class MAPElitesTrainer:
    """Single-population evolution trainer with adaptive strategies."""

    def __init__(
        self,
        port: int = 5556,
        checkpoint_dir: str = './checkpoints',
        device: str = 'cuda',
        initial_population_size: int = 10,
        max_population_size: int = 25,
    ):
        self.port = port
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Initialize adaptive population tracker
        self.population = PopulationTracker(
            initial_population_size=initial_population_size,
            max_population_size=max_population_size
        )

        # Create template model
        self.template_model = ActorNetwork().to(self.device)

        # Mutation parameters
        self.mutation_std = 0.02  # Gaussian noise std for weight perturbation

        # ZeroMQ REP socket for bidirectional communication with rover
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")

        # Model ID tracking
        self.next_model_id = 0
        self.pending_evaluations = {}  # model_id -> (model_state, generation_type)
        self.refinement_pending = {}  # model_id -> model_state for models waiting refinement
        self.multi_tournament_pending = []  # Queue for multi-tournament candidates

        # Tournament selection cache
        self.last_refined_model = None  # (model_state, trajectory_data) for tournament selection

        # Adaptive tournament parameters
        self.tournament_sizes = {
            'champion': 500,  # New best model
            'elite': 300,     # Top 5
            'good': 150,      # Top 10
            'marginal': 75    # Made it in
        }

        # Feature flags
        self.enable_diversity_bonus = True
        self.enable_multi_tournament = True
        self.enable_warmup_acceleration = True

        print(f"✓ Adaptive evolution trainer initialized on {self.device}")
        print(f"✓ Population: {initial_population_size} → {max_population_size} (adaptive)")
        print(f"✓ Tournament sizes: 75-500 (adaptive)")
        print(f"✓ REP socket listening on port {port}")

    def generate_random_model(self) -> dict:
        """Generate random model (for initial population)."""
        model = ActorNetwork().to(self.device)
        # Random initialization is already done by PyTorch
        return model.state_dict()

    def generate_heuristic_model(self, policy_type: str) -> dict:
        """Generate model with heuristic behavior injected.

        Args:
            policy_type: One of 'cautious', 'forward', 'explorer'

        Returns:
            Model state dict biased toward the heuristic behavior
        """
        model = ActorNetwork().to(self.device)

        # Start with small random weights
        with torch.no_grad():
            for param in model.parameters():
                param.data *= 0.1  # Scale down random init

        # Inject bias into policy head output layer
        # policy_head.policy is a Sequential, last layer (-1) outputs [linear_vel, angular_vel]
        final_layer = model.policy_head.policy[-1]  # Last Linear layer

        with torch.no_grad():
            if policy_type == 'cautious':
                # Bias: slow forward, moderate turning
                final_layer.bias[0] = 0.2  # linear: slow
                final_layer.bias[1] = 0.0  # angular: neutral
            elif policy_type == 'forward':
                # Bias: fast forward, minimal turning
                final_layer.bias[0] = 0.6  # linear: fast
                final_layer.bias[1] = 0.0  # angular: neutral
            elif policy_type == 'explorer':
                # Bias: moderate forward, slight turning tendency
                final_layer.bias[0] = 0.4  # linear: moderate
                final_layer.bias[1] = 0.1  # angular: slight turn

        return model.state_dict()

    def get_adaptive_mutation_std(self) -> float:
        """Compute adaptive mutation strength based on population maturity.

        Returns:
            Mutation standard deviation
        """
        pop_size = len(self.population.population)
        target_size = self.population.population_size

        # Early exploration (population not full): larger mutations
        if pop_size < target_size * 0.5:
            return 0.03
        # Mid exploration (population growing): medium mutations
        elif pop_size < target_size:
            return 0.02
        # Mature population (full): smaller fine-tuning mutations
        else:
            return 0.015

    def mutate_model(self, parent_state: dict, mutation_std: Optional[float] = None) -> dict:
        """Create mutated copy of model.

        Args:
            parent_state: Parent model to mutate
            mutation_std: Optional override for mutation strength (uses adaptive if None)
        """
        # Use adaptive mutation if not specified
        if mutation_std is None:
            mutation_std = self.get_adaptive_mutation_std()

        # Load parent weights
        child_model = ActorNetwork().to(self.device)
        child_model.load_state_dict(parent_state)

        # Add Gaussian noise to all parameters
        with torch.no_grad():
            for param in child_model.parameters():
                noise = torch.randn_like(param) * mutation_std
                param.add_(noise)

        return child_model.state_dict()

    def tournament_selection(
        self,
        parent_state: dict,
        trajectory_data: dict,
        num_candidates: int = 5,
        mutation_std_range: Tuple[float, float] = (0.01, 0.05)
    ) -> Tuple[dict, float]:
        """Create multiple mutations and select best via fitness test on trajectory data.

        Args:
            parent_state: Refined model to mutate
            trajectory_data: Trajectory data to test mutations on
            num_candidates: Number of mutations to test
            mutation_std_range: Range of mutation strengths to try

        Returns:
            (best_mutation_state, fitness_score)
        """
        print(f"  🏆 Running tournament selection ({num_candidates} candidates)...", flush=True)

        # Prepare trajectory data for evaluation
        rgb = torch.from_numpy(trajectory_data['rgb'].copy()).permute(0, 3, 1, 2).to(self.device).float() / 255.0
        depth = torch.from_numpy(trajectory_data['depth'].copy()).unsqueeze(1).to(self.device).float()
        proprio = torch.from_numpy(trajectory_data['proprio']).to(self.device).float()
        actions = torch.from_numpy(trajectory_data['actions']).to(self.device).float()

        best_fitness = float('-inf')
        best_mutation = None

        candidates = []

        # Generate candidate mutations with varying mutation strengths
        for i in range(num_candidates):
            # Vary mutation strength across candidates
            mutation_std = np.random.uniform(*mutation_std_range)

            # Create mutation
            candidate = ActorNetwork().to(self.device)
            candidate.load_state_dict(parent_state)

            with torch.no_grad():
                for param in candidate.parameters():
                    noise = torch.randn_like(param) * mutation_std
                    param.add_(noise)

            candidates.append((candidate, mutation_std))

        # Evaluate all candidates in parallel batches (GPU efficient)
        batch_eval_size = 10  # Evaluate 10 models simultaneously
        fitness_scores = []

        with torch.no_grad():
            for batch_start in range(0, num_candidates, batch_eval_size):
                batch_end = min(batch_start + batch_eval_size, num_candidates)
                batch_candidates = candidates[batch_start:batch_end]

                # Collect predictions from all models in batch
                batch_predictions = []
                for candidate, _ in batch_candidates:
                    candidate.eval()
                    pred_actions = candidate(rgb, depth, proprio)
                    batch_predictions.append(pred_actions)

                # Compute fitness for all in batch
                for i, (pred_actions, (_, mutation_std)) in enumerate(zip(batch_predictions, batch_candidates)):
                    # Goal-oriented fitness: test if mutation achieves fitness objectives
                    fitness = self.compute_tournament_fitness(
                        pred_actions=pred_actions,
                        target_actions=actions,
                        proprio=proprio
                    )

                    fitness_scores.append((fitness, mutation_std, batch_start + i))

                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_mutation = batch_candidates[i][0].state_dict()
                        best_std = mutation_std

                # Log batch progress
                if batch_end % max(10, num_candidates // 5) == 0 or batch_end == num_candidates:
                    print(f"    Evaluated {batch_end}/{num_candidates} candidates...", flush=True)

        # Show top 5 candidates
        fitness_scores.sort(reverse=True, key=lambda x: x[0])
        print(f"    Top 5 mutations:", flush=True)
        for rank, (fitness, std, idx) in enumerate(fitness_scores[:5], 1):
            marker = "★" if rank == 1 else " "
            print(f"      {marker} #{idx+1}: fitness={fitness:.6f} (std={std:.4f})", flush=True)

        print(f"  ✓ Best mutation selected: fitness={best_fitness:.6f} (std={best_std:.4f})", flush=True)

        return best_mutation, best_fitness

    def compute_tournament_fitness(
        self,
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
        proprio: torch.Tensor
    ) -> float:
        """Compute goal-oriented tournament fitness.

        Instead of just measuring action imitation (MSE), test if mutation
        achieves the actual fitness objectives:
        - High speed when clearance is high
        - Low speed when clearance is low
        - Prefer forward motion over spinning
        - Smooth action transitions

        Args:
            pred_actions: Predicted actions (N, 2) [linear, angular]
            target_actions: Original actions from trajectory (N, 2)
            proprio: Proprioception data (N, 6) [lin_vel, ang_vel, roll, pitch, accel, clearance]

        Returns:
            Fitness score (higher is better)
        """
        fitness = 0.0

        # Extract clearance from proprioception (index 5)
        clearance = proprio[:, 5]  # (N,)

        # 1. IMITATION BASELINE (30% weight)
        #    Still reward staying close to successful behavior, but not as dominant
        mse_loss = torch.nn.functional.mse_loss(pred_actions, target_actions)
        imitation_score = -mse_loss.item()
        fitness += imitation_score * 0.3

        # 2. SPEED-CLEARANCE CORRELATION (30% weight)
        #    Reward high forward velocity when safe, low velocity when risky
        high_clearance_mask = clearance > 2.0
        low_clearance_mask = clearance < 0.5

        if high_clearance_mask.any():
            # When clearance is high (>2m), should go forward confidently
            safe_linear = pred_actions[high_clearance_mask, 0].mean().item()
            fitness += safe_linear * 15.0  # Strongly reward forward motion when safe

        if low_clearance_mask.any():
            # When clearance is low (<0.5m), should be cautious (low linear)
            risky_linear = pred_actions[low_clearance_mask, 0].mean().item()
            fitness -= abs(risky_linear) * 10.0  # Penalize fast motion when risky

        # 3. FORWARD BIAS (25% weight)
        #    Tank steering: forward motion is good, spinning is bad
        linear_magnitude = torch.abs(pred_actions[:, 0]).mean().item()
        angular_magnitude = torch.abs(pred_actions[:, 1]).mean().item()
        total_action = linear_magnitude + angular_magnitude + 1e-6
        forward_ratio = linear_magnitude / total_action

        # Reward forward-biased actions
        if forward_ratio > 0.6:
            fitness += (forward_ratio - 0.6) * 25.0  # Strong reward
        elif forward_ratio < 0.3:
            fitness -= (0.3 - forward_ratio) * 30.0  # Strong penalty for spinning

        # 4. ACTION SMOOTHNESS (15% weight)
        #    Smooth actions indicate stable, confident behavior
        if pred_actions.shape[0] > 1:
            action_diffs = torch.diff(pred_actions, dim=0)
            angular_smoothness = torch.std(action_diffs[:, 1]).item()

            if angular_smoothness < 0.1:
                fitness += (0.1 - angular_smoothness) * 20.0  # Reward smooth
            elif angular_smoothness > 0.3:
                fitness -= (angular_smoothness - 0.3) * 15.0  # Penalize jerky

        # 5. AVOID EXTREME ACTIONS (bonus)
        #    Penalize saturation at action limits
        action_magnitude = torch.abs(pred_actions).mean().item()
        if action_magnitude > 0.9:
            fitness -= (action_magnitude - 0.9) * 10.0  # Saturated actions are risky

        return fitness

    def refine_model_with_gradients(
        self,
        model_state: dict,
        trajectory_data: dict,
        learning_rate: float = 1e-4,
        num_epochs: int = 10,
        batch_size: int = 32,
        use_replay_buffer: bool = True
    ) -> dict:
        """Refine model using gradient descent on trajectory data (behavioral cloning).

        Args:
            model_state: Model state dict to refine
            trajectory_data: Dictionary with 'rgb', 'depth', 'proprio', 'actions'
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            use_replay_buffer: Mix in data from trajectory buffer

        Returns:
            Refined model state dict
        """
        # Load model
        model = ActorNetwork().to(self.device)
        model.load_state_dict(model_state)
        model.train()

        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Prepare current trajectory data
        rgb_list = [trajectory_data['rgb'].copy()]
        depth_list = [trajectory_data['depth'].copy()]
        proprio_list = [trajectory_data['proprio']]
        actions_list = [trajectory_data['actions']]

        # Mix in replay buffer data (prevents catastrophic forgetting)
        if use_replay_buffer and self.trajectory_buffer:
            num_replay_samples = int(len(self.trajectory_buffer) * self.replay_buffer_mix_ratio)
            num_replay_samples = max(1, min(num_replay_samples, len(self.trajectory_buffer)))

            # Sample random trajectories from buffer
            import random
            buffer_cells = list(self.trajectory_buffer.keys())
            replay_cells = random.sample(buffer_cells, num_replay_samples)

            for cell in replay_cells:
                replay_traj = self.trajectory_buffer[cell]
                rgb_list.append(replay_traj['rgb'])
                depth_list.append(replay_traj['depth'])
                proprio_list.append(replay_traj['proprio'])
                actions_list.append(replay_traj['actions'])

            print(f"    Mixed in {num_replay_samples} trajectories from buffer", flush=True)

        # Concatenate all data
        rgb_combined = np.concatenate(rgb_list, axis=0)
        depth_combined = np.concatenate(depth_list, axis=0)
        proprio_combined = np.concatenate(proprio_list, axis=0)
        actions_combined = np.concatenate(actions_list, axis=0)

        # Convert to tensors
        rgb = torch.from_numpy(rgb_combined).permute(0, 3, 1, 2).to(self.device).float() / 255.0  # (N, 3, H, W)
        depth = torch.from_numpy(depth_combined).unsqueeze(1).to(self.device).float()  # (N, 1, H, W)
        proprio = torch.from_numpy(proprio_combined).to(self.device).float()
        actions = torch.from_numpy(actions_combined).to(self.device).float()

        num_samples = rgb.shape[0]

        # Training loop
        total_loss = 0.0
        for epoch in range(num_epochs):
            # Shuffle indices
            indices = torch.randperm(num_samples)
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:min(i+batch_size, num_samples)]

                # Get batch
                rgb_batch = rgb[batch_indices]
                depth_batch = depth[batch_indices]
                proprio_batch = proprio[batch_indices]
                actions_batch = actions[batch_indices]

                # Forward pass
                pred_actions = model(rgb_batch, depth_batch, proprio_batch)

                # Behavioral cloning loss (MSE)
                loss = torch.nn.functional.mse_loss(pred_actions, actions_batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            total_loss += avg_loss

            if (epoch + 1) % 5 == 0:
                print(f"    Refinement epoch {epoch+1}/{num_epochs}, loss: {avg_loss:.6f}")

        avg_total_loss = total_loss / num_epochs
        print(f"  ✓ Gradient refinement complete, avg loss: {avg_total_loss:.6f}")

        # Return refined model
        return model.state_dict()

    def generate_model_for_evaluation(self) -> Tuple[dict, str]:
        """Generate next model to evaluate with adaptive strategies.

        Returns:
            (model_state_dict, generation_type)
        """
        # Warmup acceleration: first 10 evaluations use heuristic-seeded models
        if self.enable_warmup_acceleration and self.population.total_evaluations < 10:
            heuristic_types = ['cautious', 'forward', 'explorer']
            # Cycle through heuristics, with some random
            if self.population.total_evaluations < 6:
                policy_type = heuristic_types[self.population.total_evaluations % 3]
                return self.generate_heuristic_model(policy_type), f'heuristic_{policy_type}'
            else:
                # Last few: mix of random and heuristic
                if np.random.random() < 0.5:
                    return self.generate_random_model(), 'random'
                else:
                    policy_type = np.random.choice(heuristic_types)
                    return self.generate_heuristic_model(policy_type), f'heuristic_{policy_type}'

        # Multi-tournament queue: if we have pending tournament candidates, send those first
        if self.multi_tournament_pending:
            candidate = self.multi_tournament_pending.pop(0)
            return candidate, 'multi_tournament'

        # Single tournament: if we have a cached model with trajectory
        if self.last_refined_model is not None:
            parent_state, trajectory_data, tournament_size = self.last_refined_model
            print(f"  🏆 Running tournament ({tournament_size} candidates)", flush=True)

            best_mutation, fitness = self.tournament_selection(
                parent_state=parent_state,
                trajectory_data=trajectory_data,
                num_candidates=tournament_size
            )

            # Clear cache after use
            self.last_refined_model = None

            return best_mutation, 'tournament'

        # Standard generation: 30% random, 70% mutation of population elite
        if np.random.random() < 0.3:
            return self.generate_random_model(), 'random'
        else:
            parent_state = self.population.get_random_elite()
            if parent_state is None:
                return self.generate_random_model(), 'random'
            return self.mutate_model(parent_state), 'mutation'

    def compute_fitness(self, episode_data: dict) -> float:
        """Compute behavior-neutral fitness with optional diversity bonus.

        Goals:
        - Maximize collision-free exploration distance
        - Reward smooth, efficient motion
        - Reward forward bias (tank steering)
        - BE NEUTRAL to speed (let model decide when to be fast/cautious)
        - Reward path quality and obstacle clearance
        - (Optional) Bonus for exploring novel behaviors
        """
        distance = episode_data['total_distance']
        collisions = episode_data['collision_count']
        avg_speed = episode_data.get('avg_speed', 0.0)
        avg_clearance = episode_data['avg_clearance']
        duration = max(episode_data['duration'], 1.0)
        action_smoothness = episode_data.get('action_smoothness', 0.0)
        avg_linear_action = episode_data.get('avg_linear_action', 0.0)
        avg_angular_action = episode_data.get('avg_angular_action', 0.0)

        # Base fitness: exploration distance (PRIMARY goal)
        # This is speed-neutral - slow+safe and fast+aggressive both maximize distance
        fitness = distance * 3.0

        # 1. Collision penalty: Critical for safety
        #    Collisions indicate poor decision-making
        if collisions > 0:
            collision_penalty = 15.0 * (collisions ** 1.5)  # 1→15, 2→42, 3→78
            fitness -= collision_penalty

        # 2. Path quality bonus (reward maintaining good clearance)
        #    This rewards finding open paths without requiring speed
        if avg_clearance > 0.5:
            # Reward smart path selection
            clearance_bonus = min((avg_clearance - 0.5) * 2.0, 8.0)
            fitness += clearance_bonus

        # 3. Smooth motion reward
        #    Smooth actions indicate confident, deliberate behavior
        #    Works for both slow-cautious and fast-aggressive
        if action_smoothness > 0:
            if action_smoothness < 0.1:  # Very smooth
                smoothness_bonus = (0.1 - action_smoothness) * 15.0  # Up to +1.5
                fitness += smoothness_bonus
            elif action_smoothness > 0.3:  # Very jerky
                jerkiness_penalty = (action_smoothness - 0.3) * 8.0
                fitness -= jerkiness_penalty

        # 4. Forward movement bias (CRITICAL for tank steering)
        #    Penalize spinning in place, reward making progress
        #    This is behavior-neutral: applies to slow and fast equally
        if avg_linear_action > 0 or avg_angular_action > 0:
            total_action = avg_linear_action + avg_angular_action + 1e-6
            forward_bias = avg_linear_action / total_action

            if forward_bias > 0.6:  # Mostly forward movement
                forward_bonus = (forward_bias - 0.6) * 20.0  # Up to +8.0
                fitness += forward_bonus
            elif forward_bias < 0.3:  # Mostly spinning in place
                spinning_penalty = (0.3 - forward_bias) * 25.0  # Up to -7.5
                fitness -= spinning_penalty

            # Extra penalty for spinning with little progress
            if avg_angular_action > 0.5 and distance < 2.0:
                fitness -= 12.0  # Severe penalty for unproductive spinning

        # 5. Efficiency: distance per time
        #    This is BEHAVIOR-NEUTRAL: rewards making progress efficiently
        #    A slow cautious model can be just as efficient as a fast aggressive one
        #    We only penalize if robot is stuck/stagnant
        actual_speed = distance / duration

        if distance < 1.0 and collisions == 0:
            # Barely moved and no obstacles - likely frozen/stuck
            fitness *= 0.3  # Harsh penalty for stagnation
        elif actual_speed < 0.03:
            # Moving but very inefficiently
            fitness *= 0.7  # Moderate penalty

        # 6. Diversity bonus (optional): reward exploring novel behaviors
        #    This helps prevent premature convergence and maintains exploration
        if self.enable_diversity_bonus and fitness > 0:
            novelty_score = self.population.calculate_behavior_novelty(avg_speed, avg_clearance)
            if novelty_score > 0.5:  # Significantly different from existing population
                diversity_bonus = novelty_score * 3.0  # Up to +3.0 bonus
                fitness += diversity_bonus
                # Note: this is logged separately in the main loop

        return fitness

    def run(self, num_evaluations: int = 1000):
        """Run single-population evolution training with REQ-REP protocol."""
        print("=" * 60)
        print("Starting Single-Population Evolution Training")
        print("=" * 60)
        print()

        # Resume from checkpoint if loaded
        evaluation_count = self.population.total_evaluations

        if evaluation_count > 0:
            print(f"Resuming from evaluation {evaluation_count}")
            print(f"Target evaluations: {num_evaluations} (need {num_evaluations - evaluation_count} more)")
        else:
            print(f"Target evaluations: {num_evaluations}")

        print(f"Population size: {self.population.population_size}")
        print()
        print("Waiting for rover to connect...")
        print()

        while evaluation_count < num_evaluations:
            try:
                # Wait for request from rover (either model request or episode result)
                message = self.socket.recv_pyobj()

                if message['type'] == 'request_model':
                    # Rover is requesting a new model to evaluate
                    model_state, gen_type = self.generate_model_for_evaluation()
                    model_id = self.next_model_id
                    self.next_model_id += 1

                    # Store for later when we get results
                    self.pending_evaluations[model_id] = (model_state, gen_type)

                    # Send model to rover
                    # Serialize model_state to bytes using torch.save
                    import io
                    buffer = io.BytesIO()
                    torch.save(model_state, buffer)
                    model_bytes = buffer.getvalue()

                    response = {
                        'type': 'model',
                        'model_id': model_id,
                        'model_bytes': model_bytes,  # Send as bytes, not pickled object
                        'generation_type': gen_type
                    }
                    self.socket.send_pyobj(response)

                    print(f"→ Sent model #{model_id} ({gen_type}) to rover", flush=True)

                elif message['type'] == 'episode_result':
                    # Rover is sending back episode results
                    model_id = message['model_id']
                    total_distance = message['total_distance']
                    collision_count = message['collision_count']
                    avg_speed = message['avg_speed']
                    avg_clearance = message['avg_clearance']
                    episode_duration = message['duration']
                    action_smoothness = message.get('action_smoothness', 0.0)
                    avg_linear_action = message.get('avg_linear_action', 0.0)
                    avg_angular_action = message.get('avg_angular_action', 0.0)

                    # Get the model state we sent earlier
                    if model_id not in self.pending_evaluations:
                        print(f"⚠ Received result for unknown model ID {model_id}")
                        self.socket.send_pyobj({'type': 'ack'})
                        continue

                    model_state, gen_type = self.pending_evaluations.pop(model_id)

                    evaluation_count += 1

                    # Compute fitness
                    fitness = self.compute_fitness(message)

                    # Try to add to population
                    added, improvement, rank = self.population.add(
                        model_state=model_state,
                        fitness=fitness,
                        avg_speed=avg_speed,
                        avg_clearance=avg_clearance,
                        metrics={
                            'distance': total_distance,
                            'collisions': collision_count,
                            'duration': episode_duration,
                        }
                    )

                    # GREEDY COMPUTE: Collect trajectory from ALL additions (not just top performers)
                    # Tournament size scales with importance
                    should_collect = False
                    collection_reason = ""
                    tournament_size = 0
                    is_champion = False

                    if added:
                        # Always collect from population additions (maximize V620 compute)
                        should_collect = True

                        # Determine adaptive tournament size based on rank
                        if rank == 1:
                            # NEW BEST MODEL - champion treatment!
                            tournament_size = self.tournament_sizes['champion']  # 500
                            collection_reason = f"🏆 NEW CHAMPION (rank #1)"
                            is_champion = True
                        elif rank <= 5:
                            # Elite performer
                            tournament_size = self.tournament_sizes['elite']  # 300
                            collection_reason = f"Elite (rank #{rank})"
                        elif rank <= 10:
                            # Good performer
                            tournament_size = self.tournament_sizes['good']  # 150
                            collection_reason = f"Good (rank #{rank})"
                        else:
                            # Marginal addition
                            tournament_size = self.tournament_sizes['marginal']  # 75
                            collection_reason = f"Marginal (rank #{rank})"

                    # Calculate variable episode duration suggestion
                    best_fitness = self.population.get_best()['fitness'] if self.population.get_best() else 0
                    pop_size = len(self.population.population)

                    if pop_size < 5:
                        suggested_duration = 30.0  # Early: fail fast
                    elif best_fitness < 10:
                        suggested_duration = 50.0  # Mid: standard
                    else:
                        suggested_duration = 75.0  # Late: deep data

                    # If added and should collect, request trajectory data for tournament selection
                    if added and should_collect:
                        # Store tournament metadata for when trajectory arrives
                        self.refinement_pending[model_id] = (model_state, tournament_size, is_champion)

                        # Send acknowledgment with trajectory request and episode duration suggestion
                        self.socket.send_pyobj({
                            'type': 'ack',
                            'collect_trajectory': True,
                            'model_id': model_id,  # Re-run same model
                            'suggested_episode_duration': suggested_duration
                        })
                        print(f"  → Requesting trajectory for {collection_reason} "
                              f"(tournament: {tournament_size} candidates, "
                              f"next episode: {suggested_duration:.0f}s)...", flush=True)
                    else:
                        # Normal acknowledgment with episode duration suggestion
                        self.socket.send_pyobj({
                            'type': 'ack',
                            'suggested_episode_duration': suggested_duration
                        })

                    # Log progress
                    if added:
                        status = f"✓ ADDED rank #{rank} ({collection_reason})"
                    else:
                        status = "rejected (below threshold)"

                    print(f"← Eval {evaluation_count}/{num_evaluations} | "
                          f"Model #{model_id} ({gen_type}) | "
                          f"Fitness: {fitness:.2f} | "
                          f"Speed: {avg_speed:.3f} m/s | "
                          f"Clear: {avg_clearance:.2f} m | "
                          f"{status}",
                          flush=True)

                    # Checkpoint every 10 evaluations
                    if evaluation_count % 10 == 0:
                        self.save_checkpoint(evaluation_count)
                        stats = self.population.get_stats()
                        best_fitness = stats.get('fitness_best', 0.0)
                        pop_size = stats['population_size']
                        print(f"  Population: {pop_size} models | Best fitness: {best_fitness:.2f}",
                              flush=True)

                elif message['type'] == 'trajectory_data':
                    # Rover is sending trajectory data for tournament selection
                    model_id = message['model_id']
                    trajectory_raw = message['trajectory']
                    compressed = message.get('compressed', False)

                    print(f"← Received trajectory data for model #{model_id}", flush=True)

                    # Get the model and tournament metadata from pending tracking
                    if model_id not in self.refinement_pending:
                        print(f"  ⚠ Model #{model_id} not pending tournament", flush=True)
                        self.socket.send_pyobj({'type': 'ack'})
                        continue

                    original_model_state, tournament_size, is_champion = self.refinement_pending.pop(model_id)

                    # Decompress trajectory data
                    if compressed:
                        # Decompress RGB and depth using Zstandard
                        dctx = zstd.ZstdDecompressor()
                        rgb_bytes = dctx.decompress(trajectory_raw['rgb'])
                        depth_bytes = dctx.decompress(trajectory_raw['depth'])

                        # Reconstruct numpy arrays
                        rgb_array = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(trajectory_raw['rgb_shape'])
                        depth_array = np.frombuffer(depth_bytes, dtype=np.float32).reshape(trajectory_raw['depth_shape'])

                        trajectory_data = {
                            'rgb': rgb_array,
                            'depth': depth_array,
                            'proprio': trajectory_raw['proprio'],
                            'actions': trajectory_raw['actions'],
                        }
                    else:
                        trajectory_data = trajectory_raw

                    print(f"  📦 Caching model #{model_id} ({len(trajectory_data['actions'])} samples) for tournament", flush=True)

                    # MULTI-TOURNAMENT for champions: run 3 parallel tournaments
                    if self.enable_multi_tournament and is_champion:
                        print(f"  🏆🏆🏆 CHAMPION detected! Running 3 parallel tournaments...", flush=True)

                        # Run 3 independent tournaments with varying mutation ranges
                        mutation_ranges = [
                            (0.005, 0.03),  # Conservative
                            (0.01, 0.05),   # Standard
                            (0.02, 0.08),   # Aggressive
                        ]

                        for i, mut_range in enumerate(mutation_ranges):
                            print(f"    Running tournament {i+1}/3 (mutation: {mut_range})...", flush=True)
                            best_mutation, fitness = self.tournament_selection(
                                parent_state=original_model_state,
                                trajectory_data=trajectory_data,
                                num_candidates=tournament_size // 3,  # Divide compute across 3 tournaments
                                mutation_std_range=mut_range
                            )
                            # Queue this candidate for evaluation
                            self.multi_tournament_pending.append(best_mutation)

                        print(f"  ✓ 3 tournament candidates queued for evaluation!", flush=True)

                    else:
                        # Standard single tournament
                        # Cache original model + trajectory for goal-oriented tournament selection
                        self.last_refined_model = (original_model_state, trajectory_data, tournament_size)

                        print(f"  ✓ Ready for tournament selection ({tournament_size} candidates)", flush=True)

                    # Send acknowledgment (instant - no gradient descent wait!)
                    self.socket.send_pyobj({'type': 'ack'})

                else:
                    print(f"⚠ Unknown message type: {message.get('type')}")
                    self.socket.send_pyobj({'type': 'error', 'message': 'Unknown type'})

            except KeyboardInterrupt:
                print("\n🛑 Training interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error in training loop: {e}")
                import traceback
                traceback.print_exc()
                # Try to send error response
                try:
                    self.socket.send_pyobj({'type': 'error', 'message': str(e)})
                except:
                    pass
                continue

        print()
        print("=" * 60)
        print("✅ Single-Population Evolution Training Complete!")
        print("=" * 60)
        self.save_checkpoint(evaluation_count, final=True)
        self.print_population_summary()

    def save_checkpoint(self, evaluation: int, final: bool = False):
        """Save population checkpoint."""
        suffix = 'final' if final else f'eval_{evaluation}'
        checkpoint_path = self.checkpoint_dir / f'evolution_{suffix}.json'

        self.population.save(str(checkpoint_path))

        # Export best model
        self.export_best_model(suffix)

        print(f"✓ Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load population from checkpoint."""
        import json

        checkpoint_path = Path(checkpoint_path)

        # Load metadata
        with open(checkpoint_path, 'r') as f:
            metadata = json.load(f)

        # Create new population with same size
        self.population = PopulationTracker(
            population_size=metadata.get('population_size', 20)
        )
        self.population.total_evaluations = metadata['total_evaluations']
        self.population.improvements = metadata.get('improvements', 0)

        # Load models
        models_dir = checkpoint_path.parent / 'evolution_models'
        if not models_dir.exists():
            print(f"⚠ Models directory not found: {models_dir}")
            return

        # Load population metadata
        population_metadata = metadata.get('population', [])
        print(f"  Population metadata has {len(population_metadata)} entries")

        # Load model files
        model_files = sorted(models_dir.glob('rank_*.pt'),
                            key=lambda p: int(p.stem.split('_')[1]))
        print(f"  Found {len(model_files)} model files in {models_dir}")

        loaded_count = 0
        for model_file in model_files:
            # Parse rank from filename: rank_1.pt
            rank = int(model_file.stem.split('_')[1]) - 1  # Convert to 0-indexed

            if rank >= len(population_metadata):
                print(f"  ⚠ No metadata for rank {rank+1}")
                continue

            # Load model
            model_state = torch.load(model_file, map_location='cpu')
            entry_meta = population_metadata[rank]

            # Reconstruct population entry
            entry = {
                'model': model_state,
                'fitness': entry_meta['fitness'],
                'avg_speed': entry_meta['avg_speed'],
                'avg_clearance': entry_meta['avg_clearance'],
                'metrics': entry_meta['metrics'],
            }
            self.population.population.append(entry)
            loaded_count += 1

        print(f"  Loaded {loaded_count}/{len(model_files)} models from checkpoint")

    def export_best_model(self, suffix: str):
        """Export the best model from population."""
        export_dir = self.checkpoint_dir / 'best_models'
        export_dir.mkdir(exist_ok=True)

        best_entry = self.population.get_best()
        if best_entry:
            model_path = export_dir / f'best_{suffix}.pt'
            torch.save(best_entry['model'], model_path)
            print(f"  ✓ Exported best model: fitness={best_entry['fitness']:.2f}")

    def print_population_summary(self):
        """Print detailed population summary."""
        stats = self.population.get_stats()

        print()
        print("Population Summary:")
        print(f"  Population size: {stats['population_size']}")
        print(f"  Total evaluations: {stats['total_evaluations']}")
        print(f"  Improvements: {stats['improvements']}")

        if 'fitness_best' in stats:
            print(f"  Fitness - Best: {stats['fitness_best']:.2f}, "
                  f"Mean: {stats['fitness_mean']:.2f}, "
                  f"Min: {stats['fitness_min']:.2f}")
        else:
            print(f"  Fitness - (no valid fitness values yet)")

        print(f"  Avg speed: {stats.get('speed_mean', 0):.3f} m/s")
        print(f"  Avg clearance: {stats.get('clearance_mean', 0):.2f} m")


def main():
    parser = argparse.ArgumentParser(description='Adaptive evolution trainer with greedy compute')
    parser.add_argument('--port', type=int, default=5556,
                        help='Port to listen for episode results')
    parser.add_argument('--num-evaluations', type=int, default=1000,
                        help='Number of episodes to evaluate')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory for checkpoints')
    parser.add_argument('--initial-population', type=int, default=10,
                        help='Initial population size (default: 10)')
    parser.add_argument('--max-population', type=int, default=25,
                        help='Maximum population size (default: 25)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from specific checkpoint (e.g., "eval_100" or "final")')
    parser.add_argument('--fresh', action='store_true',
                        help='Start fresh, ignore existing checkpoints')
    parser.add_argument('--disable-diversity', action='store_true',
                        help='Disable diversity bonus in fitness')
    parser.add_argument('--disable-multi-tournament', action='store_true',
                        help='Disable multi-tournament for champions')
    parser.add_argument('--disable-warmup', action='store_true',
                        help='Disable warmup acceleration')
    args = parser.parse_args()

    # Check ROCm
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print("⚠ No GPU detected - will use CPU (slow)")

    print()

    # Create trainer with adaptive features
    trainer = MAPElitesTrainer(
        port=args.port,
        checkpoint_dir=args.checkpoint_dir,
        initial_population_size=args.initial_population,
        max_population_size=args.max_population,
    )

    # Configure features
    trainer.enable_diversity_bonus = not args.disable_diversity
    trainer.enable_multi_tournament = not args.disable_multi_tournament
    trainer.enable_warmup_acceleration = not args.disable_warmup

    print(f"  Features enabled:")
    print(f"    Diversity bonus: {trainer.enable_diversity_bonus}")
    print(f"    Multi-tournament: {trainer.enable_multi_tournament}")
    print(f"    Warmup acceleration: {trainer.enable_warmup_acceleration}")
    print()

    # Auto-resume from latest checkpoint (or explicit resume)
    checkpoint_to_load = None

    if args.fresh:
        # User explicitly wants to start fresh
        print("🆕 Starting fresh (--fresh flag)")
        print()
    elif args.resume:
        # Explicit resume request
        checkpoint_to_load = Path(args.checkpoint_dir) / f'evolution_{args.resume}.json'
    else:
        # Auto-resume: find latest checkpoint
        checkpoint_dir = Path(args.checkpoint_dir)
        if checkpoint_dir.exists():
            # Look for checkpoints in order: final, then highest eval_N
            final_checkpoint = checkpoint_dir / 'evolution_final.json'
            if final_checkpoint.exists():
                checkpoint_to_load = final_checkpoint
            else:
                # Find highest eval_N checkpoint
                eval_checkpoints = list(checkpoint_dir.glob('evolution_eval_*.json'))
                if eval_checkpoints:
                    # Extract numbers and find max
                    def get_eval_num(path):
                        try:
                            return int(path.stem.split('_')[-1])
                        except:
                            return 0
                    latest = max(eval_checkpoints, key=get_eval_num)
                    checkpoint_to_load = latest

    if checkpoint_to_load and checkpoint_to_load.exists():
        print(f"🔄 Resuming from checkpoint: {checkpoint_to_load.name}")
        trainer.load_checkpoint(str(checkpoint_to_load))
        print(f"✓ Checkpoint loaded")
        trainer.print_population_summary()
        print()
    else:
        print("Starting fresh (no checkpoint found)")
        print()

    # Run training
    trainer.run(num_evaluations=args.num_evaluations)


if __name__ == '__main__':
    main()
