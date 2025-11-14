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


class MAPElitesArchive:
    """Archive maintaining best models for each behavior cell."""

    def __init__(
        self,
        speed_bins: List[float] = [0.0, 0.05, 0.10, 0.15, 0.20],
        clearance_bins: List[float] = [0.2, 0.5, 1.0, 2.0, 5.0],
    ):
        """Initialize archive.

        Args:
            speed_bins: Bin edges for average speed (m/s)
            clearance_bins: Bin edges for average obstacle clearance (m)
        """
        self.speed_bins = speed_bins
        self.clearance_bins = clearance_bins

        # Archive: (speed_idx, clearance_idx) -> {'model': state_dict, 'fitness': float, 'metrics': dict}
        self.archive = {}

        # Statistics
        self.total_evaluations = 0
        self.archive_additions = 0

    def get_cell_index(self, avg_speed: float, avg_clearance: float) -> Tuple[int, int]:
        """Map behavior to archive cell indices."""
        speed_idx = np.digitize(avg_speed, self.speed_bins) - 1
        clearance_idx = np.digitize(avg_clearance, self.clearance_bins) - 1

        # Clamp to valid range
        speed_idx = max(0, min(len(self.speed_bins) - 2, speed_idx))
        clearance_idx = max(0, min(len(self.clearance_bins) - 2, clearance_idx))

        # Convert to Python int to avoid numpy types in string keys
        return (int(speed_idx), int(clearance_idx))

    def add(
        self,
        model_state: dict,
        fitness: float,
        avg_speed: float,
        avg_clearance: float,
        metrics: dict
    ) -> Tuple[bool, bool, float]:
        """Try to add model to archive.

        Returns:
            (was_added, is_new_cell, fitness_improvement_ratio)
        """
        self.total_evaluations += 1

        cell_idx = self.get_cell_index(avg_speed, avg_clearance)

        is_new_cell = cell_idx not in self.archive
        fitness_improvement = 0.0

        # Check if this cell is empty or if new model is better
        if is_new_cell:
            self.archive[cell_idx] = {
                'model': copy.deepcopy(model_state),
                'fitness': fitness,
                'avg_speed': avg_speed,
                'avg_clearance': avg_clearance,
                'metrics': metrics,
            }
            self.archive_additions += 1
            return True, True, float('inf')  # New cell = infinite improvement
        elif fitness > self.archive[cell_idx]['fitness']:
            old_fitness = self.archive[cell_idx]['fitness']
            fitness_improvement = (fitness - old_fitness) / max(abs(old_fitness), 1e-6)

            self.archive[cell_idx] = {
                'model': copy.deepcopy(model_state),
                'fitness': fitness,
                'avg_speed': avg_speed,
                'avg_clearance': avg_clearance,
                'metrics': metrics,
            }
            self.archive_additions += 1
            return True, False, fitness_improvement

        return False, False, 0.0

    def get_random_elite(self) -> Optional[dict]:
        """Sample random model from archive."""
        if not self.archive:
            return None

        import random
        cell_idx = random.choice(list(self.archive.keys()))
        return self.archive[cell_idx]['model']

    def get_best_in_cell(self, speed_idx: int, clearance_idx: int) -> Optional[dict]:
        """Get best model for specific behavior cell."""
        return self.archive.get((speed_idx, clearance_idx))

    def coverage(self) -> float:
        """Fraction of archive cells filled."""
        total_cells = (len(self.speed_bins) - 1) * (len(self.clearance_bins) - 1)
        return len(self.archive) / total_cells

    def get_stats(self) -> dict:
        """Get archive statistics."""
        if not self.archive:
            return {
                'coverage': 0.0,
                'filled_cells': 0,
                'total_cells': (len(self.speed_bins) - 1) * (len(self.clearance_bins) - 1),
                'total_evaluations': self.total_evaluations,
                'archive_additions': self.archive_additions,
            }

        # Filter out -inf fitness values (from models awaiting re-evaluation)
        fitnesses = [entry['fitness'] for entry in self.archive.values() if np.isfinite(entry['fitness'])]
        speeds = [entry['avg_speed'] for entry in self.archive.values()]
        clearances = [entry['avg_clearance'] for entry in self.archive.values()]

        stats = {
            'coverage': self.coverage(),
            'filled_cells': len(self.archive),
            'total_cells': (len(self.speed_bins) - 1) * (len(self.clearance_bins) - 1),
            'total_evaluations': self.total_evaluations,
            'archive_additions': self.archive_additions,
            'speed_mean': np.mean(speeds) if speeds else 0.0,
            'clearance_mean': np.mean(clearances) if clearances else 0.0,
        }

        # Only include fitness stats if we have valid fitness values
        if fitnesses:
            stats.update({
                'fitness_mean': np.mean(fitnesses),
                'fitness_max': np.max(fitnesses),
                'fitness_min': np.min(fitnesses),
            })

        return stats

    def save(self, filepath: str):
        """Save archive to disk."""
        import json

        # Prepare archive metadata (without model weights)
        archive_metadata = {}
        for cell_idx, entry in self.archive.items():
            archive_metadata[str(cell_idx)] = {
                'fitness': entry['fitness'],
                'avg_speed': entry['avg_speed'],
                'avg_clearance': entry['avg_clearance'],
                'metrics': entry['metrics']
            }

        metadata = {
            'speed_bins': self.speed_bins,
            'clearance_bins': self.clearance_bins,
            'archive': archive_metadata,  # Include archive metadata
            'total_evaluations': self.total_evaluations,
            'archive_additions': self.archive_additions,
        }

        # Debug: verify archive is being saved
        print(f"  💾 Saving archive with {len(archive_metadata)} cells of metadata", flush=True)

        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save models separately
        models_dir = Path(filepath).parent / 'map_elites_models'
        models_dir.mkdir(exist_ok=True)

        for cell_idx, entry in self.archive.items():
            model_path = models_dir / f'cell_{cell_idx[0]}_{cell_idx[1]}.pt'
            torch.save(entry['model'], model_path)


class MAPElitesTrainer:
    """MAP-Elites trainer for rover navigation."""

    def __init__(
        self,
        port: int = 5556,
        checkpoint_dir: str = './checkpoints',
        device: str = 'cuda',
    ):
        self.port = port
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Initialize archive
        self.archive = MAPElitesArchive()

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
        self.refinement_pending = {}  # model_id -> (cell_idx, model_state) for models waiting refinement

        # Tournament selection cache
        self.last_refined_model = None  # (model_state, trajectory_data) for tournament selection
        self.tournament_candidates = 400  # Number of mutations to test (V620 has 32GB VRAM, no gradient descent)

        # Tournament trigger parameters (no gradient descent)
        self.refinement_fitness_threshold = 0.10  # Collect trajectory if >10% improvement
        self.always_refine_new_cells = True  # Always collect trajectory for new cells

        print(f"✓ MAP-Elites trainer initialized on {self.device}")
        print(f"✓ REP socket listening on port {port}")

    def generate_random_model(self) -> dict:
        """Generate random model (for initial population)."""
        model = ActorNetwork().to(self.device)
        # Random initialization is already done by PyTorch
        return model.state_dict()

    def get_adaptive_mutation_std(self) -> float:
        """Compute adaptive mutation strength based on archive density.

        Returns:
            Mutation standard deviation
        """
        coverage = self.archive.coverage()

        # Early exploration (< 30% coverage): larger mutations
        if coverage < 0.3:
            return 0.03
        # Mid exploration (30-70% coverage): medium mutations
        elif coverage < 0.7:
            return 0.02
        # Dense archive (> 70% coverage): smaller fine-tuning mutations
        else:
            return 0.01

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
        """Generate next model to evaluate.

        Returns:
            (model_state_dict, generation_type)
        """
        # Quick warmup: first 3 evaluations are random for initial diversity
        # With goal-oriented tournament, we don't need long warmup like gradient descent did
        if self.archive.total_evaluations < 3:
            return self.generate_random_model(), 'random'

        # If we have a recently cached model with trajectory, use goal-oriented tournament
        if self.last_refined_model is not None:
            parent_state, trajectory_data = self.last_refined_model
            print(f"  🏆 Running goal-oriented tournament ({self.tournament_candidates} candidates)", flush=True)

            best_mutation, fitness = self.tournament_selection(
                parent_state=parent_state,
                trajectory_data=trajectory_data,
                num_candidates=self.tournament_candidates
            )

            # Clear cache after use
            self.last_refined_model = None

            return best_mutation, 'tournament'

        # 30% random, 70% mutation of archive elite
        if np.random.random() < 0.3:
            return self.generate_random_model(), 'random'
        else:
            parent_state = self.archive.get_random_elite()
            if parent_state is None:
                return self.generate_random_model(), 'random'
            return self.mutate_model(parent_state), 'mutation'

    def compute_fitness(self, episode_data: dict) -> float:
        """Compute fitness encouraging safe exploration with smooth, confident motion.

        Goals:
        - Explore via open paths (high clearance preferred)
        - Increase speed with confidence (reward fast + safe combinations)
        - Smooth, efficient motion (penalize erratic behavior)
        - Discover optimal paths regardless of indoor/outdoor
        - Encourage forward movement (tank steering = turning in place wastes time)
        """
        distance = episode_data['total_distance']
        collisions = episode_data['collision_count']
        avg_speed = episode_data['avg_speed']
        avg_clearance = episode_data['avg_clearance']
        duration = max(episode_data['duration'], 1.0)
        action_smoothness = episode_data.get('action_smoothness', 0.0)
        avg_linear_action = episode_data.get('avg_linear_action', 0.0)
        avg_angular_action = episode_data.get('avg_angular_action', 0.0)

        # Base fitness: exploration distance (primary goal)
        # For tank steering, forward progress is everything
        fitness = distance * 2.0  # Double weight on actual distance covered

        # 1. Collision penalty: Catastrophic for safety
        #    Exponential penalty - multiple collisions indicate poor behavior
        if collisions > 0:
            collision_penalty = 10.0 * (collisions ** 1.5)  # 1→10, 2→28, 3→52
            fitness -= collision_penalty

        # 2. Open path exploration bonus
        #    Reward finding and using open spaces (clearance > 1m is good)
        if avg_clearance > 0.5:
            # Scale bonus with clearance: 0.5m→0, 1.0m→0.5, 2.0m→2.0, 5.0m→4.5
            openness_bonus = min((avg_clearance - 0.5) * 1.0, 5.0)
            fitness += openness_bonus

        # 3. Confident speed bonus (speed is good when combined with safety)
        #    Reward high speed ONLY when clearance is also high
        #    This encourages "speed up in open areas, slow down in tight spaces"
        if collisions == 0 and avg_clearance > 0.8:
            # Confidence factor: higher clearance allows higher speed bonus
            confidence_multiplier = min(avg_clearance / 2.0, 1.5)  # 0.8m→0.4x, 2m→1.0x, 4m+→1.5x
            speed_bonus = avg_speed * 5.0 * confidence_multiplier  # 0.2m/s @ 2m clear → +1.0
            fitness += speed_bonus

        # 4. Smooth motion reward (low action jerkiness)
        #    Reward smooth turns and steady motion
        #    action_smoothness is std of angular velocity changes (lower = smoother)
        if action_smoothness > 0:
            # Typical smoothness range: 0.01 (very smooth) to 0.3+ (jerky)
            if action_smoothness < 0.1:  # Very smooth
                smoothness_bonus = (0.1 - action_smoothness) * 10.0  # Up to +1.0 bonus
                fitness += smoothness_bonus
            elif action_smoothness > 0.3:  # Very jerky
                jerkiness_penalty = (action_smoothness - 0.3) * 5.0  # Increasing penalty
                fitness -= jerkiness_penalty

        # 5. Efficiency reward (distance/time)
        #    Penalize if robot took too long (indicates erratic motion, spinning, backtracking)
        #    Expected: ~0.08 m/s = reasonable exploration speed
        expected_speed = 0.08  # m/s baseline for exploration
        actual_speed = distance / duration

        if actual_speed < expected_speed * 0.5:  # Very slow = likely stuck/spinning
            fitness *= 0.7  # 30% penalty for inefficient motion
        elif actual_speed > expected_speed * 1.5:  # Fast and smooth
            fitness *= 1.1  # 10% bonus for efficient exploration

        # 6. Stagnation penalty
        #    If barely moving, heavily penalize (robot gave up or stuck)
        if avg_speed < 0.02 and collisions == 0:  # Nearly stationary without collisions
            fitness *= 0.3  # Harsh penalty - robot is doing nothing useful

        # 7. Forward movement bias (CRITICAL for tank steering)
        #    Tank steering means turning in place = zero progress but wastes time
        #    Heavily penalize spinning, strongly reward forward movement
        if avg_linear_action > 0 or avg_angular_action > 0:
            # Forward bias ratio: linear_action / (linear_action + angular_action)
            # 1.0 = pure forward, 0.5 = equal mix, 0.0 = pure spinning
            total_action = avg_linear_action + avg_angular_action + 1e-6  # Avoid divide by zero
            forward_bias = avg_linear_action / total_action

            if forward_bias > 0.6:  # Mostly forward movement (good!)
                forward_bonus = (forward_bias - 0.6) * 15.0  # Up to +6.0 bonus
                fitness += forward_bonus
            elif forward_bias < 0.3:  # Mostly spinning in place (bad!)
                spinning_penalty = (0.3 - forward_bias) * 20.0  # Up to -6.0 penalty
                fitness -= spinning_penalty

            # Extra harsh penalty if lots of rotation but little distance
            if avg_angular_action > 0.5 and distance < 2.0:
                # High rotation, low distance = spinning in place
                fitness -= 10.0  # Severe penalty

        return fitness

    def run(self, num_evaluations: int = 1000):
        """Run MAP-Elites training with REQ-REP protocol."""
        print("=" * 60)
        print("Starting MAP-Elites Training")
        print("=" * 60)
        print()
        print(f"Target evaluations: {num_evaluations}")
        print(f"Archive dimensions: {len(self.archive.speed_bins)-1} × {len(self.archive.clearance_bins)-1}")
        print()
        print("Waiting for rover to connect...")
        print()

        evaluation_count = 0

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

                    # Try to add to archive
                    added, is_new_cell, improvement_ratio = self.archive.add(
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

                    # Request trajectory for tournament selection if significant improvement or new cell
                    should_collect = False
                    collection_reason = ""

                    if added:
                        if is_new_cell and self.always_refine_new_cells:
                            should_collect = True
                            collection_reason = "new behavior cell"
                        elif improvement_ratio >= self.refinement_fitness_threshold:
                            should_collect = True
                            collection_reason = f"{improvement_ratio*100:.1f}% fitness improvement"
                        else:
                            collection_reason = f"skipped (only {improvement_ratio*100:.1f}% improvement)"

                    # If added and should collect, request trajectory data for tournament selection
                    if added and should_collect:
                        # Get the cell index where this model was added
                        cell_idx = self.archive.get_cell_index(avg_speed, avg_clearance)

                        # Store for tournament when trajectory data arrives
                        self.refinement_pending[model_id] = (cell_idx, model_state)

                        # Send acknowledgment with trajectory request
                        self.socket.send_pyobj({
                            'type': 'ack',
                            'collect_trajectory': True,
                            'model_id': model_id  # Re-run same model
                        })
                        print(f"  → Requesting trajectory for tournament ({collection_reason})...", flush=True)
                    else:
                        # Normal acknowledgment
                        self.socket.send_pyobj({'type': 'ack'})

                    # Log progress
                    status = f"✓ ADDED ({collection_reason})" if added else "rejected"
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
                        stats = self.archive.get_stats()
                        print(f"  Coverage: {stats['coverage']:.1%} ({stats['filled_cells']}/{stats['total_cells']} cells)",
                              flush=True)

                elif message['type'] == 'trajectory_data':
                    # Rover is sending trajectory data for tournament selection
                    model_id = message['model_id']
                    trajectory_raw = message['trajectory']
                    compressed = message.get('compressed', False)

                    print(f"← Received trajectory data for model #{model_id}", flush=True)

                    # Get the cell and model from pending tracking
                    if model_id not in self.refinement_pending:
                        print(f"  ⚠ Model #{model_id} not pending tournament", flush=True)
                        self.socket.send_pyobj({'type': 'ack'})
                        continue

                    cell_idx, original_model_state = self.refinement_pending.pop(model_id)

                    # Decompress trajectory data
                    if compressed:
                        import lz4.frame
                        # Decompress RGB and depth
                        rgb_bytes = lz4.frame.decompress(trajectory_raw['rgb'])
                        depth_bytes = lz4.frame.decompress(trajectory_raw['depth'])

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

                    print(f"  📦 Caching model #{model_id} (cell {cell_idx}, {len(trajectory_data['actions'])} samples) for tournament", flush=True)

                    # Cache original model + trajectory for goal-oriented tournament selection
                    # NO gradient descent - tournament will directly optimize for goals!
                    self.last_refined_model = (original_model_state, trajectory_data)

                    print(f"  ✓ Ready for tournament selection ({self.tournament_candidates} candidates)", flush=True)

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
        print("✅ MAP-Elites Training Complete!")
        print("=" * 60)
        self.save_checkpoint(evaluation_count, final=True)
        self.print_archive_summary()

    def save_checkpoint(self, evaluation: int, final: bool = False):
        """Save archive checkpoint."""
        suffix = 'final' if final else f'eval_{evaluation}'
        archive_path = self.checkpoint_dir / f'map_elites_{suffix}.json'

        self.archive.save(str(archive_path))

        # Also export best models in each category
        self.export_best_models(suffix)

        print(f"✓ Checkpoint saved: {archive_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load archive from checkpoint."""
        import json

        checkpoint_path = Path(checkpoint_path)

        # Load metadata
        with open(checkpoint_path, 'r') as f:
            metadata = json.load(f)

        # Create new archive with same bins
        self.archive = MAPElitesArchive(
            speed_bins=metadata['speed_bins'],
            clearance_bins=metadata['clearance_bins']
        )
        self.archive.total_evaluations = metadata['total_evaluations']
        self.archive.archive_additions = metadata['archive_additions']

        # Load models
        models_dir = checkpoint_path.parent / 'map_elites_models'
        if not models_dir.exists():
            print(f"⚠ Models directory not found: {models_dir}")
            return

        # Debug: Check what's in the archive metadata
        archive_metadata = metadata.get('archive', {})
        archive_keys = list(archive_metadata.keys())
        print(f"  Archive metadata has {len(archive_keys)} entries")

        # Reconstruct archive
        model_files = list(models_dir.glob('cell_*.pt'))
        print(f"  Found {len(model_files)} model files in {models_dir}")

        loaded_count = 0
        for model_file in model_files:
            # Parse cell indices from filename: cell_0_1.pt
            parts = model_file.stem.split('_')
            speed_idx = int(parts[1])
            clearance_idx = int(parts[2])

            # Load model
            model_state = torch.load(model_file, map_location='cpu')

            # Get metadata from the archive dict stored in JSON
            # Try multiple key formats for compatibility with old checkpoints
            cell_key_formats = [
                f"({speed_idx}, {clearance_idx})",  # Standard format
                f"(np.int64({speed_idx}), {clearance_idx})",  # Old format with numpy type
                f"({speed_idx}, np.int64({clearance_idx}))",  # Old format (rare)
                f"(np.int64({speed_idx}), np.int64({clearance_idx}))",  # Old format both
            ]

            cell_data = None
            found_key = None
            for key_format in cell_key_formats:
                if key_format in archive_metadata:
                    cell_data = archive_metadata[key_format]
                    found_key = key_format
                    break

            if cell_data is not None:
                # Found metadata in checkpoint
                self.archive.archive[(speed_idx, clearance_idx)] = {
                    'model': model_state,
                    'fitness': cell_data['fitness'],
                    'avg_speed': cell_data['avg_speed'],
                    'avg_clearance': cell_data['avg_clearance'],
                    'metrics': cell_data['metrics'],
                }
                loaded_count += 1
            else:
                # Old checkpoint format without metadata - use -inf fitness so it gets replaced
                print(f"  ⚠ No metadata for cell ({speed_idx}, {clearance_idx}), will be re-evaluated")
                self.archive.archive[(speed_idx, clearance_idx)] = {
                    'model': model_state,
                    'fitness': float('-inf'),  # Ensures any real evaluation replaces this
                    'avg_speed': 0.0,
                    'avg_clearance': 0.0,
                    'metrics': {},
                }
                loaded_count += 1

        print(f"  Loaded {loaded_count}/{len(model_files)} models from checkpoint")

    def export_best_models(self, suffix: str):
        """Export best models for different behavior profiles."""
        export_dir = self.checkpoint_dir / 'map_elites_exports'
        export_dir.mkdir(exist_ok=True)

        # Find best models in different categories
        categories = {
            'cautious': (0, 3),      # Slow, high clearance
            'balanced': (2, 2),      # Medium speed, medium clearance
            'aggressive': (3, 1),    # Fast, low clearance
            'explorer': (2, 3),      # Medium speed, high clearance
        }

        for name, (speed_idx, clear_idx) in categories.items():
            entry = self.archive.get_best_in_cell(speed_idx, clear_idx)
            if entry:
                model_path = export_dir / f'{name}_{suffix}.pt'
                torch.save(entry['model'], model_path)
                print(f"  ✓ Exported '{name}' model: fitness={entry['fitness']:.2f}")

    def print_archive_summary(self):
        """Print detailed archive summary."""
        stats = self.archive.get_stats()

        print()
        print("Archive Summary:")
        print(f"  Coverage: {stats['coverage']:.1%} ({stats['filled_cells']}/{stats['total_cells']} cells)")
        print(f"  Total evaluations: {stats['total_evaluations']}")
        print(f"  Archive additions: {stats['archive_additions']}")

        if 'fitness_max' in stats:
            print(f"  Fitness - Max: {stats['fitness_max']:.2f}, "
                  f"Mean: {stats['fitness_mean']:.2f}, "
                  f"Min: {stats['fitness_min']:.2f}")
        else:
            print(f"  Fitness - (no valid fitness values yet)")

        print(f"  Avg speed: {stats.get('speed_mean', 0):.3f} m/s")
        print(f"  Avg clearance: {stats.get('clearance_mean', 0):.2f} m")


def main():
    parser = argparse.ArgumentParser(description='MAP-Elites trainer for rover')
    parser.add_argument('--port', type=int, default=5556,
                        help='Port to listen for episode results')
    parser.add_argument('--num-evaluations', type=int, default=1000,
                        help='Number of episodes to evaluate')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from specific checkpoint (e.g., "eval_100" or "final")')
    parser.add_argument('--fresh', action='store_true',
                        help='Start fresh, ignore existing checkpoints')
    parser.add_argument('--tournament-size', type=int, default=400,
                        help='Number of mutations to test in goal-oriented tournament (default: 400, no gradient descent)')
    args = parser.parse_args()

    # Check ROCm
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print("⚠ No GPU detected - will use CPU (slow)")

    print()

    # Create trainer
    trainer = MAPElitesTrainer(
        port=args.port,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Set tournament size from args
    trainer.tournament_candidates = args.tournament_size
    print(f"  Tournament selection: {args.tournament_size} candidates")
    print()

    # Auto-resume from latest checkpoint (or explicit resume)
    checkpoint_to_load = None

    if args.fresh:
        # User explicitly wants to start fresh
        print("🆕 Starting fresh (--fresh flag)")
        print()
    elif args.resume:
        # Explicit resume request
        checkpoint_to_load = Path(args.checkpoint_dir) / f'map_elites_{args.resume}.json'
    else:
        # Auto-resume: find latest checkpoint
        checkpoint_dir = Path(args.checkpoint_dir)
        if checkpoint_dir.exists():
            # Look for checkpoints in order: final, then highest eval_N
            final_checkpoint = checkpoint_dir / 'map_elites_final.json'
            if final_checkpoint.exists():
                checkpoint_to_load = final_checkpoint
            else:
                # Find highest eval_N checkpoint
                eval_checkpoints = list(checkpoint_dir.glob('map_elites_eval_*.json'))
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
        trainer.print_archive_summary()
        print()
    else:
        print("Starting fresh (no checkpoint found)")
        print()

    # Run training
    trainer.run(num_evaluations=args.num_evaluations)


if __name__ == '__main__':
    main()
