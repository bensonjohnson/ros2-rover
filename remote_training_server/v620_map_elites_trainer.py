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
            action: (B, 2) [linear_vel, angular_vel]
        """
        features = self.encoder(rgb, depth)
        action = self.policy_head(features, proprio)
        return action


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

        # ZeroMQ REP socket for bidirectional communication with rover
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")

        # Model ID tracking
        self.next_model_id = 0
        self.pending_evaluations = {}  # model_id -> (model_state, generation_type)
        self.refinement_pending = {}  # model_id -> (cell_idx, model_state) for models waiting refinement

        print(f"✓ MAP-Elites trainer initialized on {self.device}")
        print(f"✓ REP socket listening on port {port}")

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

    def refine_model_with_gradients(
        self,
        model_state: dict,
        trajectory_data: dict,
        learning_rate: float = 1e-4,
        num_epochs: int = 10,
        batch_size: int = 32
    ) -> dict:
        """Refine model using gradient descent on trajectory data (behavioral cloning).

        Args:
            model_state: Model state dict to refine
            trajectory_data: Dictionary with 'rgb', 'depth', 'proprio', 'actions'
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
            batch_size: Batch size for training

        Returns:
            Refined model state dict
        """
        # Load model
        model = ActorNetwork().to(self.device)
        model.load_state_dict(model_state)
        model.train()

        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Prepare data
        # RGB comes as (N, H, W, 3), need to transpose to (N, 3, H, W)
        rgb_np = trajectory_data['rgb']  # (N, H, W, 3)
        rgb = torch.from_numpy(rgb_np).permute(0, 3, 1, 2).to(self.device).float() / 255.0  # (N, 3, H, W)

        # Depth comes as (N, H, W), need to add channel dim: (N, 1, H, W)
        depth_np = trajectory_data['depth']  # (N, H, W)
        depth = torch.from_numpy(depth_np).unsqueeze(1).to(self.device).float()  # (N, 1, H, W)

        proprio = torch.from_numpy(trajectory_data['proprio']).to(self.device).float()
        actions = torch.from_numpy(trajectory_data['actions']).to(self.device).float()

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

                    # If added to archive, request trajectory data for gradient refinement
                    if added:
                        # Get the cell index where this model was added
                        cell_idx = self.archive.get_cell_index(avg_speed, avg_clearance)

                        # Store for refinement when trajectory data arrives
                        self.refinement_pending[model_id] = (cell_idx, model_state)

                        # Send acknowledgment with trajectory request
                        self.socket.send_pyobj({
                            'type': 'ack',
                            'collect_trajectory': True,
                            'model_id': model_id  # Re-run same model
                        })
                        print(f"  → Requesting trajectory data for refinement...", flush=True)
                    else:
                        # Normal acknowledgment
                        self.socket.send_pyobj({'type': 'ack'})

                    # Log progress
                    print(f"← Eval {evaluation_count}/{num_evaluations} | "
                          f"Model #{model_id} ({gen_type}) | "
                          f"Fitness: {fitness:.2f} | "
                          f"Speed: {avg_speed:.3f} m/s | "
                          f"Clear: {avg_clearance:.2f} m | "
                          f"{'✓ ADDED' if added else 'rejected'}",
                          flush=True)

                    # Checkpoint every 50 evaluations
                    if evaluation_count % 50 == 0:
                        self.save_checkpoint(evaluation_count)
                        stats = self.archive.get_stats()
                        print(f"  Coverage: {stats['coverage']:.1%} ({stats['filled_cells']}/{stats['total_cells']} cells)",
                              flush=True)

                elif message['type'] == 'trajectory_data':
                    # Rover is sending trajectory data for gradient refinement
                    model_id = message['model_id']
                    trajectory_raw = message['trajectory']
                    compressed = message.get('compressed', False)

                    # Decompress if needed
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

                    print(f"← Received trajectory data for model #{model_id} "
                          f"({len(trajectory_data['actions'])} samples)", flush=True)

                    # Get the cell and model from refinement tracking
                    if model_id not in self.refinement_pending:
                        print(f"  ⚠ Model #{model_id} not pending refinement", flush=True)
                        self.socket.send_pyobj({'type': 'ack'})
                        continue

                    cell_idx, original_model_state = self.refinement_pending.pop(model_id)

                    # Send acknowledgment IMMEDIATELY so rover can continue
                    # (refinement happens asynchronously to the rover)
                    self.socket.send_pyobj({'type': 'ack'})

                    print(f"  Refining model #{model_id} (cell {cell_idx}) with gradient descent...", flush=True)

                    # Refine the model
                    refined_model_state = self.refine_model_with_gradients(
                        model_state=original_model_state,
                        trajectory_data=trajectory_data,
                        learning_rate=1e-4,
                        num_epochs=10,
                        batch_size=32
                    )

                    # Update the archive with refined model
                    if cell_idx in self.archive.archive:
                        self.archive.archive[cell_idx]['model'] = refined_model_state
                        print(f"  ✓ Archive cell {cell_idx} updated with refined model", flush=True)
                    else:
                        print(f"  ⚠ Cell {cell_idx} not found in archive", flush=True)

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

        # Reconstruct archive
        for model_file in models_dir.glob('cell_*.pt'):
            # Parse cell indices from filename: cell_0_1.pt
            parts = model_file.stem.split('_')
            speed_idx = int(parts[1])
            clearance_idx = int(parts[2])

            # Load model
            model_state = torch.load(model_file, map_location='cpu')

            # Get metadata from the archive dict stored in JSON
            cell_key = f"({speed_idx}, {clearance_idx})"
            if cell_key in metadata.get('archive', {}):
                cell_data = metadata['archive'][cell_key]
                self.archive.archive[(speed_idx, clearance_idx)] = {
                    'model': model_state,
                    'fitness': cell_data['fitness'],
                    'avg_speed': cell_data['avg_speed'],
                    'avg_clearance': cell_data['avg_clearance'],
                    'metrics': cell_data['metrics'],
                }

        print(f"  Loaded {len(self.archive.archive)} models from checkpoint")

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
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from specific checkpoint (e.g., "eval_100" or "final")')
    parser.add_argument('--fresh', action='store_true',
                        help='Start fresh, ignore existing checkpoints')
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
