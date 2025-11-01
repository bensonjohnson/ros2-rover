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

from v620_ppo_trainer import ActorNetwork  # Reuse your existing network architecture


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

        return (speed_idx, clearance_idx)

    def add(
        self,
        model_state: dict,
        fitness: float,
        avg_speed: float,
        avg_clearance: float,
        metrics: dict
    ) -> bool:
        """Try to add model to archive.

        Returns:
            True if model was added (either new cell or better fitness)
        """
        self.total_evaluations += 1

        cell_idx = self.get_cell_index(avg_speed, avg_clearance)

        # Check if this cell is empty or if new model is better
        if cell_idx not in self.archive or fitness > self.archive[cell_idx]['fitness']:
            self.archive[cell_idx] = {
                'model': copy.deepcopy(model_state),
                'fitness': fitness,
                'avg_speed': avg_speed,
                'avg_clearance': avg_clearance,
                'metrics': metrics,
            }
            self.archive_additions += 1
            return True

        return False

    def get_random_elite(self) -> Optional[dict]:
        """Sample random model from archive."""
        if not self.archive:
            return None

        cell_idx = np.random.choice(list(self.archive.keys()))
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

        fitnesses = [entry['fitness'] for entry in self.archive.values()]
        speeds = [entry['avg_speed'] for entry in self.archive.values()]
        clearances = [entry['avg_clearance'] for entry in self.archive.values()]

        return {
            'coverage': self.coverage(),
            'filled_cells': len(self.archive),
            'total_cells': (len(self.speed_bins) - 1) * (len(self.clearance_bins) - 1),
            'total_evaluations': self.total_evaluations,
            'archive_additions': self.archive_additions,
            'fitness_mean': np.mean(fitnesses),
            'fitness_max': np.max(fitnesses),
            'fitness_min': np.min(fitnesses),
            'speed_mean': np.mean(speeds),
            'clearance_mean': np.mean(clearances),
        }

    def save(self, filepath: str):
        """Save archive to disk."""
        archive_data = {
            'speed_bins': self.speed_bins,
            'clearance_bins': self.clearance_bins,
            'archive': {str(k): v for k, v in self.archive.items()},
            'total_evaluations': self.total_evaluations,
            'archive_additions': self.archive_additions,
        }

        with open(filepath, 'w') as f:
            # Save metadata as JSON
            metadata = {k: v for k, v in archive_data.items() if k != 'archive'}
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

        # ZeroMQ for receiving episode results from rover
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(f"tcp://*:{port}")
        self.socket.setsockopt(zmq.RCVHWM, 1000)

        print(f"✓ MAP-Elites trainer initialized on {self.device}")
        print(f"✓ Listening for episode results on port {port}")

    def generate_random_model(self) -> dict:
        """Generate random model (for initial population)."""
        model = ActorNetwork().to(self.device)
        # Random initialization is already done by PyTorch
        return model.state_dict()

    def mutate_model(self, parent_state: dict) -> dict:
        """Create mutated copy of model."""
        # Load parent weights
        child_model = ActorNetwork().to(self.device)
        child_model.load_state_dict(parent_state)

        # Add Gaussian noise to all parameters
        with torch.no_grad():
            for param in child_model.parameters():
                noise = torch.randn_like(param) * self.mutation_std
                param.add_(noise)

        return child_model.state_dict()

    def generate_model_for_evaluation(self) -> Tuple[dict, str]:
        """Generate next model to evaluate.

        Returns:
            (model_state_dict, generation_type)
        """
        # First 20 evaluations: random initialization
        if self.archive.total_evaluations < 20:
            return self.generate_random_model(), 'random'

        # 30% random, 70% mutation of archive elite
        if np.random.random() < 0.3:
            return self.generate_random_model(), 'random'
        else:
            parent_state = self.archive.get_random_elite()
            if parent_state is None:
                return self.generate_random_model(), 'random'
            return self.mutate_model(parent_state), 'mutation'

    def compute_fitness(self, episode_data: dict) -> float:
        """Compute fitness from episode results.

        Fitness = distance traveled - collision penalty
        """
        distance = episode_data['total_distance']
        collisions = episode_data['collision_count']

        # Heavy penalty for collisions
        fitness = distance - (collisions * 5.0)

        return fitness

    def run(self, num_evaluations: int = 1000):
        """Run MAP-Elites training."""
        print("=" * 60)
        print("Starting MAP-Elites Training")
        print("=" * 60)
        print()
        print(f"Target evaluations: {num_evaluations}")
        print(f"Archive dimensions: {len(self.archive.speed_bins)-1} × {len(self.archive.clearance_bins)-1}")
        print()

        evaluation_count = 0

        while evaluation_count < num_evaluations:
            # Wait for episode result from rover
            try:
                # Receive with timeout
                if self.socket.poll(timeout=60000):  # 60 second timeout
                    episode_data = self.socket.recv_pyobj()
                else:
                    print("⚠ Timeout waiting for episode data")
                    continue

            except Exception as e:
                print(f"❌ Error receiving data: {e}")
                continue

            evaluation_count += 1

            # Extract episode results
            model_state = episode_data['model_state']
            total_distance = episode_data['total_distance']
            collision_count = episode_data['collision_count']
            avg_speed = episode_data['avg_speed']
            avg_clearance = episode_data['avg_clearance']
            episode_duration = episode_data['duration']

            # Compute fitness
            fitness = self.compute_fitness(episode_data)

            # Try to add to archive
            added = self.archive.add(
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

            # Log progress
            if evaluation_count % 10 == 0:
                stats = self.archive.get_stats()
                print(f"Eval {evaluation_count}/{num_evaluations} | "
                      f"Coverage: {stats['coverage']:.1%} ({stats['filled_cells']}/{stats['total_cells']}) | "
                      f"Fitness: {stats.get('fitness_max', 0):.2f} | "
                      f"Speed: {avg_speed:.3f} m/s | "
                      f"Clear: {avg_clearance:.2f} m | "
                      f"{'✓ ADDED' if added else 'rejected'}",
                      flush=True)

            # Checkpoint every 50 evaluations
            if evaluation_count % 50 == 0:
                self.save_checkpoint(evaluation_count)

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
        print(f"  Fitness - Max: {stats.get('fitness_max', 0):.2f}, "
              f"Mean: {stats.get('fitness_mean', 0):.2f}, "
              f"Min: {stats.get('fitness_min', 0):.2f}")
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

    # Run training
    trainer.run(num_evaluations=args.num_evaluations)


if __name__ == '__main__':
    main()
