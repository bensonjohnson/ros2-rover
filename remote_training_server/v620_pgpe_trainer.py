#!/usr/bin/env python3
"""PGPE trainer for rover using EvoTorch.

Converts from custom MAP-Elites implementation to EvoTorch's PGPE algorithm.
Maintains ZeroMQ communication with rover.
"""

import argparse
import copy
import io
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import zmq
from evotorch import Problem
from evotorch.algorithms import PGPE
from evotorch.logging import StdOutLogger

from model_architectures import RGBDEncoder, PolicyHead


class ActorNetwork(nn.Module):
    """Actor-only network for PGPE evolution."""

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


class RoverProblem(Problem):
    """EvoTorch Problem definition for rover policy optimization.

    This class bridges EvoTorch's PGPE algorithm with the rover's ZeroMQ protocol.
    """

    def __init__(
        self,
        port: int = 5556,
        checkpoint_dir: str = './checkpoints',
        device: str = 'cuda',
    ):
        # Create template model to get parameter count
        self.template_model = ActorNetwork().to(device)
        num_params = sum(p.numel() for p in self.template_model.parameters())

        print(f"ActorNetwork parameter count: {num_params:,}")

        # Initialize EvoTorch Problem
        super().__init__(
            objective_sense="max",  # Maximize fitness
            solution_length=num_params,
            dtype=torch.float32,
            device=device,
        )

        self.port = port
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.device_str = device

        # ZeroMQ REP socket for communication with rover
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")

        # Track evaluations
        self.total_evaluations = 0
        self.best_fitness_ever = float('-inf')
        self.best_model_ever = None

        # Model ID tracking
        self.next_model_id = 0
        self.pending_evaluations = {}  # model_id -> solution_tensor

        print(f"✓ PGPE Problem initialized")
        print(f"✓ Parameters: {num_params:,}")
        print(f"✓ Device: {device}")
        print(f"✓ REP socket listening on port {port}")

    def _evaluate_batch(self, solutions: torch.Tensor) -> torch.Tensor:
        """Evaluate a batch of solutions (policy parameter vectors).

        This is called by PGPE during optimization. We send each solution
        to the rover for real-world evaluation.

        Args:
            solutions: (popsize, num_params) tensor of parameter vectors

        Returns:
            fitness: (popsize,) tensor of fitness values
        """
        batch_size = solutions.shape[0]
        fitness_values = []

        print(f"\n{'='*60}")
        print(f"Evaluating batch of {batch_size} solutions")
        print(f"{'='*60}")

        for idx, solution in enumerate(solutions):
            # Convert solution (flat params) to model state dict
            model_state = self._solution_to_state_dict(solution)

            # Send to rover and get fitness
            fitness = self._evaluate_on_rover(model_state)
            fitness_values.append(fitness)

            self.total_evaluations += 1

            # Track best model
            if fitness > self.best_fitness_ever:
                self.best_fitness_ever = fitness
                self.best_model_ever = copy.deepcopy(model_state)
                print(f"  🏆 NEW BEST! Fitness: {fitness:.2f}")

            print(f"  [{idx+1}/{batch_size}] Fitness: {fitness:.2f}")

        return torch.tensor(fitness_values, dtype=torch.float32, device=self.device)

    def _solution_to_state_dict(self, solution: torch.Tensor) -> dict:
        """Convert flat parameter vector to model state dict.

        Args:
            solution: (num_params,) flat parameter tensor

        Returns:
            state_dict: PyTorch state dict
        """
        state_dict = {}
        offset = 0

        for name, param in self.template_model.named_parameters():
            numel = param.numel()
            # Extract parameters for this layer
            param_data = solution[offset:offset+numel].reshape(param.shape)
            state_dict[name] = param_data.detach().cpu()
            offset += numel

        return state_dict

    def _evaluate_on_rover(self, model_state: dict) -> float:
        """Send model to rover and get fitness from episode evaluation.

        Uses the existing ZeroMQ REQ-REP protocol.

        Args:
            model_state: PyTorch model state dict

        Returns:
            fitness: Fitness value from rover episode
        """
        try:
            # Wait for rover to request a model
            message = self.socket.recv_pyobj()

            if message['type'] != 'request_model':
                print(f"⚠ Unexpected message type: {message.get('type')}")
                self.socket.send_pyobj({'type': 'error', 'message': 'Expected request_model'})
                return 0.0

            # Assign model ID
            model_id = self.next_model_id
            self.next_model_id += 1

            # Serialize model to bytes
            buffer = io.BytesIO()
            torch.save(model_state, buffer)
            model_bytes = buffer.getvalue()

            # Send model to rover
            response = {
                'type': 'model',
                'model_id': model_id,
                'model_bytes': model_bytes,
                'generation_type': 'pgpe'
            }
            self.socket.send_pyobj(response)

            print(f"  → Sent model #{model_id} to rover")

            # Wait for episode result
            result_message = self.socket.recv_pyobj()

            if result_message['type'] != 'episode_result':
                print(f"⚠ Expected episode_result, got {result_message.get('type')}")
                self.socket.send_pyobj({'type': 'error'})
                return 0.0

            # Extract episode metrics
            total_distance = result_message['total_distance']
            collision_count = result_message['collision_count']
            avg_speed = result_message['avg_speed']
            avg_clearance = result_message['avg_clearance']
            episode_duration = result_message['duration']
            action_smoothness = result_message.get('action_smoothness', 0.0)
            avg_linear_action = result_message.get('avg_linear_action', 0.0)
            avg_angular_action = result_message.get('avg_angular_action', 0.0)

            # Compute fitness
            fitness = self._compute_fitness(result_message)

            # Send acknowledgment
            self.socket.send_pyobj({'type': 'ack'})

            print(f"  ← Episode result: dist={total_distance:.2f}m, "
                  f"cols={collision_count}, fitness={fitness:.2f}")

            return fitness

        except Exception as e:
            print(f"❌ Error evaluating on rover: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def _compute_fitness(self, episode_data: dict) -> float:
        """Compute tank-optimized fitness (same as original MAP-Elites).

        Args:
            episode_data: Episode metrics from rover

        Returns:
            fitness: Scalar fitness value
        """
        # Extract metrics
        distance = episode_data['total_distance']
        collisions = episode_data['collision_count']
        avg_speed = episode_data.get('avg_speed', 0.0)
        avg_clearance = episode_data['avg_clearance']
        duration = max(episode_data['duration'], 1.0)
        action_smoothness = episode_data.get('action_smoothness', 0.0)
        avg_linear_action = episode_data.get('avg_linear_action', 0.0)
        avg_angular_action = episode_data.get('avg_angular_action', 0.0)

        # Tank-specific metrics
        turn_efficiency = episode_data.get('turn_efficiency', 0.0)
        stationary_rotation = episode_data.get('stationary_rotation_time', 0.0)
        track_slip = episode_data.get('track_slip_detected', False)

        # Base fitness: exploration distance
        fitness = distance * 2.0

        # 1. Collision penalty
        if collisions > 0:
            collision_penalty = 6.0 * (collisions ** 1.2)
            fitness -= collision_penalty

        # 2. Path quality bonus
        if avg_clearance > 0.15:
            clearance_bonus = min((avg_clearance - 0.15) * 8.0, 15.0)
            fitness += clearance_bonus

            if avg_clearance > 0.3:
                open_space_bonus = min((avg_clearance - 0.3) * 5.0, 10.0)
                fitness += open_space_bonus

        # 3. Tank pivot turn reward
        if avg_speed < 0.05 and avg_angular_action > 0.2:
            if turn_efficiency > 0.5:
                fitness += 8.0
            else:
                fitness -= 3.0

            if avg_angular_action < 0.5:
                if action_smoothness < 0.15:
                    scanning_bonus = (0.15 - action_smoothness) * 10.0
                    fitness += scanning_bonus

        # 4. Penalize unproductive spinning
        if avg_linear_action < 0.1 and avg_angular_action > 0.4:
            if distance < 1.0:
                fitness -= 15.0
            elif avg_angular_action > 0.7:
                fitness -= 25.0

        # 5. Smooth motion reward
        if action_smoothness > 0:
            if action_smoothness < 0.08:
                smoothness_bonus = (0.08 - action_smoothness) * 20.0
                fitness += smoothness_bonus
            elif action_smoothness > 0.25:
                jerkiness_penalty = (action_smoothness - 0.25) * 12.0
                fitness -= jerkiness_penalty

        # 6. Track slippage penalty
        if track_slip:
            fitness -= 20.0

        # 7. Efficiency
        if distance < 0.5 and collisions == 0:
            fitness *= 0.4
        elif (distance / duration) < 0.02:
            fitness *= 0.7

        if distance > 2.0 and collisions == 0 and action_smoothness < 0.2:
            exploration_bonus = min(distance * 0.5, 10.0)
            fitness += exploration_bonus

        return fitness

    def save_best_model(self, filepath: str):
        """Save the best model found so far."""
        if self.best_model_ever is not None:
            torch.save(self.best_model_ever, filepath)
            print(f"✓ Saved best model (fitness={self.best_fitness_ever:.2f}): {filepath}")


def main():
    parser = argparse.ArgumentParser(description='PGPE trainer for rover evolution')
    parser.add_argument('--port', type=int, default=5556,
                        help='Port to listen for episode results')
    parser.add_argument('--num-evaluations', type=int, default=1000,
                        help='Number of episodes to evaluate')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory for checkpoints')
    parser.add_argument('--population-size', type=int, default=20,
                        help='PGPE population size (default: 20)')
    parser.add_argument('--center-lr', type=float, default=0.01,
                        help='PGPE center learning rate (default: 0.01)')
    parser.add_argument('--stdev-lr', type=float, default=0.001,
                        help='PGPE stdev learning rate (default: 0.001)')
    parser.add_argument('--stdev-init', type=float, default=0.02,
                        help='Initial standard deviation (default: 0.02)')
    args = parser.parse_args()

    # Check GPU
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        device = 'cuda'
    else:
        print("⚠ No GPU detected - will use CPU (slow)")
        device = 'cpu'

    print()

    # Create problem
    problem = RoverProblem(
        port=args.port,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
    )

    print()
    print("=" * 60)
    print("Initializing PGPE Algorithm")
    print("=" * 60)

    # Create PGPE searcher
    searcher = PGPE(
        problem,
        popsize=args.population_size,
        center_learning_rate=args.center_lr,
        stdev_learning_rate=args.stdev_lr,
        stdev_init=args.stdev_init,
        optimizer="adam",  # Use Adam for distribution updates
        optimizer_config={"lr": args.center_lr},
        ranking_method="centered",  # Centered ranking for better gradient estimates
    )

    # Add logger
    logger = StdOutLogger(searcher)

    print(f"✓ PGPE initialized")
    print(f"  Population size: {args.population_size}")
    print(f"  Center learning rate: {args.center_lr}")
    print(f"  Stdev learning rate: {args.stdev_lr}")
    print(f"  Initial stdev: {args.stdev_init}")
    print()

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    best_models_dir = checkpoint_dir / 'best_models'
    best_models_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Starting PGPE Evolution Training")
    print("=" * 60)
    print(f"Target evaluations: {args.num_evaluations}")
    print()
    print("Waiting for rover to connect...")
    print()

    # Training loop
    generation = 0
    while problem.total_evaluations < args.num_evaluations:
        try:
            # Run one generation (evaluates population, updates distribution)
            searcher.step()
            generation += 1

            # Get current best
            best_fitness = searcher.status['best'].evals[0].item()
            mean_fitness = searcher.status['pop_best'].evals.mean().item()

            print()
            print(f"{'='*60}")
            print(f"Generation {generation} Complete")
            print(f"{'='*60}")
            print(f"  Evaluations: {problem.total_evaluations}/{args.num_evaluations}")
            print(f"  Best (this gen): {best_fitness:.2f}")
            print(f"  Mean (this gen): {mean_fitness:.2f}")
            print(f"  Best (all time): {problem.best_fitness_ever:.2f}")
            print()

            # Save checkpoint every 10 generations
            if generation % 10 == 0:
                checkpoint_path = checkpoint_dir / f'pgpe_gen_{generation}.json'
                metadata = {
                    'generation': generation,
                    'evaluations': problem.total_evaluations,
                    'best_fitness': problem.best_fitness_ever,
                    'population_size': args.population_size,
                }
                with open(checkpoint_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

                # Save best model
                best_model_path = best_models_dir / f'best_gen_{generation}.pt'
                problem.save_best_model(str(best_model_path))

        except KeyboardInterrupt:
            print("\n🛑 Training interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error in training loop: {e}")
            import traceback
            traceback.print_exc()
            continue

    print()
    print("=" * 60)
    print("✅ PGPE Evolution Training Complete!")
    print("=" * 60)
    print(f"  Total evaluations: {problem.total_evaluations}")
    print(f"  Best fitness: {problem.best_fitness_ever:.2f}")
    print()

    # Save final best model
    final_model_path = best_models_dir / 'best_final.pt'
    problem.save_best_model(str(final_model_path))


if __name__ == '__main__':
    main()
