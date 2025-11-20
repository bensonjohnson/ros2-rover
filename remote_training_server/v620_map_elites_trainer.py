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
from tqdm import tqdm

from model_architectures import RGBDEncoder, PolicyHead  # Reuse network components


class ActorNetwork(nn.Module):
    """Actor-only network for MAP-Elites with LSTM memory (no value head needed)."""

    def __init__(self, proprio_dim: int = 6, use_lstm: bool = True):
        super().__init__()
        self.encoder = RGBDEncoder()
        self.policy_head = PolicyHead(self.encoder.output_dim, proprio_dim, use_lstm=use_lstm)
        self.use_lstm = use_lstm

    def forward(self, rgb, depth, proprio, hidden_state=None):
        """Forward pass with optional LSTM hidden state.

        Args:
            rgb: (B, 3, H, W) RGB image
            depth: (B, 1, H, W) Depth image
            proprio: (B, 6) Proprioception
            hidden_state: Optional (h, c) tuple for LSTM, each (1, B, 128)

        Returns:
            If use_lstm: (action, (h_new, c_new)) where action is (B, 2) [linear_vel, angular_vel] in [-1, 1] range
            Else: (action, None)
        """
        features = self.encoder(rgb, depth)
        action, hidden_state_new = self.policy_head(features, proprio, hidden_state)
        return torch.tanh(action), hidden_state_new  # Squash to [-1, 1] range


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
            # Find rank by object identity (avoid tensor comparison issues)
            rank = next(i+1 for i, x in enumerate(self.population) if x is entry)
            return True, float('inf'), rank

        # Check if better than worst in population
        worst_fitness = self.population[-1]['fitness']

        if fitness > worst_fitness:
            improvement = (fitness - worst_fitness) / max(abs(worst_fitness), 1e-6)

            # Replace worst model
            self.population[-1] = entry
            self.population.sort(key=lambda x: x['fitness'], reverse=True)
            self.improvements += 1
            # Find rank by object identity (avoid tensor comparison issues)
            rank = next(i+1 for i, x in enumerate(self.population) if x is entry)
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
            'population_size': self.population_size,  # Current adaptive size
            'initial_population_size': self.initial_population_size,
            'max_population_size': self.max_population_size,
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


class ReplayBuffer:
    """Experience Replay Buffer for storing successful trajectories.
    
    Prevents catastrophic forgetting by maintaining a diverse set of past experiences.
    """
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.buffer = []  # List of (fitness, trajectory_data) tuples
        
    def add(self, trajectory_data: dict, fitness: float):
        """Add trajectory to buffer if it's good enough or buffer not full."""
        # Always add if buffer not full
        if len(self.buffer) < self.max_size:
            self.buffer.append((fitness, trajectory_data))
            self.buffer.sort(key=lambda x: x[0], reverse=True)
            return
            
        # If full, replace worst entry if new one is better
        if fitness > self.buffer[-1][0]:
            self.buffer[-1] = (fitness, trajectory_data)
            self.buffer.sort(key=lambda x: x[0], reverse=True)
            
    def sample(self, n: int) -> List[dict]:
        """Sample n trajectories from buffer."""
        if not self.buffer:
            return []
        
        import random
        # Sample from top 50% to encourage high-quality replay
        elite_size = max(1, len(self.buffer) // 2)
        candidates = self.buffer[:elite_size]
        
        # If we need more samples than elites, sample from whole buffer
        if n > len(candidates):
            candidates = self.buffer
            
        samples = random.sample(candidates, min(n, len(candidates)))
        return [s[1] for s in samples]
    
    def __len__(self):
        return len(self.buffer)


class MAPElitesTrainer:
    """Single-population evolution trainer with adaptive strategies."""

    def __init__(
        self,
        port: int = 5556,
        checkpoint_dir: str = './checkpoints',
        device: str = 'cuda',
        initial_population_size: int = 10,
        max_population_size: int = 25,
        use_fp16: bool = True,
    ):
        self.port = port
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_fp16 = use_fp16 and self.device.type == 'cuda'

        # Detect ROCm and apply optimizations
        self.is_rocm = torch.cuda.is_available() and torch.version.hip is not None
        if self.is_rocm:
            print(f"✓ ROCm detected (HIP version: {torch.version.hip})")
            self._apply_rocm_optimizations()

        # NEW: Auto-detect and set optimal batch size based on GPU memory with dynamic profiling
        self.batch_size = self._calculate_optimal_batch_size()
        print(f"✓ Auto-selected batch size: {self.batch_size} (based on dynamic profiling)")
        
        if self.use_fp16:
            print(f"✓ Precision: fp16 mixed precision (2x faster, 50% less memory)")
            print(f"  Note: ONNX export will still be fp32 for RKNN compatibility")
            # Initialize gradient scaler for mixed precision training
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            print(f"✓ Precision: fp32 (full precision)")
            self.scaler = None

        # Initialize adaptive population tracker
        self.population = PopulationTracker(
            initial_population_size=initial_population_size,
            max_population_size=max_population_size
        )
        
        # Initialize Replay Buffer
        self.replay_buffer = ReplayBuffer(max_size=50)
        print(f"✓ Replay Buffer initialized (max size: 50)")

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
        print(f"✓ Batch size: {self.batch_size} (auto-scaled)")
        print(f"✓ REP socket listening on port {port}")

    def _apply_rocm_optimizations(self):
        """Apply ROCm-specific optimizations for AMD GPUs (ROCm 6.0+)."""
        import os
        import warnings

        print(f"  Applying ROCm 7.1+ optimizations...")

        # 1. Enable TF32 (TensorFloat-32) for significant speedup on RDNA3/CDNA2+
        # This allows FP32 matmuls to run at lower precision internally
        
        # Suppress the specific warning about TF32 API deprecation because we ARE using the new API
        # but PyTorch 2.x on ROCm seems to trigger it anyway when calling set_float32_matmul_precision
        warnings.filterwarnings("ignore", message=".*Please use the new API settings to control TF32 behavior.*")
        
        try:
            # New API (PyTorch 2.x+)
            torch.set_float32_matmul_precision('high')
            print(f"    ✓ TF32 enabled (matmul: high precision)")
        except AttributeError:
            # Old API
            torch.backends.cuda.matmul.allow_tf32 = True
            print(f"    ✓ TF32 enabled (matmul: allow_tf32)")

        try:
            # New API for cuDNN
            torch.backends.cudnn.conv.fp32_precision = 'tf32'
            print(f"    ✓ TF32 enabled (cudnn: tf32)")
        except AttributeError:
            # Old API
            torch.backends.cudnn.allow_tf32 = True
            print(f"    ✓ TF32 enabled (cudnn: allow_tf32)")

        # 2. MIOpen Optimizations
        # Enable MIOpen V8 API for better performance
        os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
        
        # Disable MIOpen auto-tuner to avoid SQLite database locking/crashes in multi-process
        # But allow it to use compiled kernels
        torch.backends.cudnn.benchmark = False  # CHANGED: Disable benchmark to fix slow startup
        print(f"    ✓ MIOpen V8 API enabled")
        print(f"    ✓ cuDNN/MIOpen benchmark disabled (fast startup)")
        # print(f"      (Note: You may see 'MIOpen(HIP): Warning [SearchImpl]' logs initially - this is normal benchmarking)")

        # 3. Memory Allocator
        # Use expandable segments to reduce fragmentation
        if 'PYTORCH_HIP_ALLOC_CONF' not in os.environ:
            os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.8'
            print(f"    ✓ HIP allocator tuned (max_split_size_mb:128)")

        # Check environment variables
        print(f"  Environment check:")
        if os.environ.get('MIOPEN_FIND_ENFORCE') == 'NONE':
            print(f"    ✓ MIOPEN_FIND_ENFORCE=NONE")
        if os.environ.get('MIOPEN_DISABLE_CACHE') == '1':
            print(f"    ✓ MIOPEN_DISABLE_CACHE=1")
        if os.environ.get('HSA_FORCE_FINE_GRAIN_PCIE') == '1':
            print(f"    ✓ HSA_FORCE_FINE_GRAIN_PCIE=1")

    def generate_random_model(self) -> dict:
        """Generate random model (for initial population)."""
        model = ActorNetwork().to(self.device)
        # Random initialization is already done by PyTorch
        return model.state_dict()

    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available GPU/CPU memory."""
        if self.device.type == 'cuda':
            return self._calculate_gpu_batch_size()
        elif self.device.type == 'mps':
            return self._calculate_mps_batch_size()
        else:
            return self._calculate_cpu_batch_size()
    
    def _calculate_gpu_batch_size(self) -> int:
        """Calculate batch size for CUDA GPUs using dynamic memory profiling."""
        try:
            # Get total GPU memory in GB
            total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  GPU Memory: {total_memory_gb:.1f}GB total")

            # Dynamic profiling: test actual memory usage with real model
            print(f"  Running dynamic memory profiling...")

            # Create test model and move to device
            test_model = ActorNetwork().to(self.device)
            test_model.train()
            optimizer = torch.optim.Adam(test_model.parameters(), lr=1e-4)

            # Test batch size (start small)
            test_batch_size = 8

            # Generate synthetic test data (typical image dimensions for rover)
            # Assuming 640x480 resolution or similar
            rgb_test = torch.randn(test_batch_size, 3, 480, 640, device=self.device)
            depth_test = torch.randn(test_batch_size, 1, 480, 640, device=self.device)
            proprio_test = torch.randn(test_batch_size, 6, device=self.device)
            actions_test = torch.randn(test_batch_size, 2, device=self.device)

            # Clear any existing allocations
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

            # Run test forward + backward pass (fp32) - LSTM returns (actions, hidden_state)
            optimizer.zero_grad()
            pred_actions, _ = test_model(rgb_test, depth_test, proprio_test)
            loss = torch.nn.functional.mse_loss(pred_actions, actions_test)
            loss.backward()
            optimizer.step()

            # Measure peak memory usage
            peak_memory_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)
            memory_per_sample_gb = peak_memory_gb / test_batch_size

            print(f"    Test batch ({test_batch_size} samples): {peak_memory_gb:.2f}GB used")
            print(f"    Memory per sample: {memory_per_sample_gb:.3f}GB")

            # Clean up test model
            del test_model, optimizer, rgb_test, depth_test, proprio_test, actions_test, pred_actions, loss
            torch.cuda.empty_cache()

            # Calculate optimal batch size with conservative margin
            # Use 85% of total memory (15% headroom for spikes and overhead)
            target_memory_gb = total_memory_gb * 0.85
            optimal_batch_size = int(target_memory_gb / memory_per_sample_gb)

            # Clamp to reasonable range
            # Increased max from 256 to 2048 for high-VRAM cards (like V620 32GB)
            batch_size = max(8, min(optimal_batch_size, 2048))

            # Predict actual usage at optimal batch size
            predicted_usage_gb = batch_size * memory_per_sample_gb
            utilization_pct = (predicted_usage_gb / total_memory_gb) * 100

            print(f"  Optimal batch size: {batch_size}")
            print(f"  Predicted usage: {predicted_usage_gb:.1f}GB / {total_memory_gb:.1f}GB ({utilization_pct:.0f}%)")

            return batch_size

        except Exception as e:
            print(f"  ⚠ Dynamic profiling failed: {e}, using conservative batch size 32")
            import traceback
            traceback.print_exc()
            return 32
    
    def _calculate_mps_batch_size(self) -> int:
        """Calculate batch size for Apple MPS."""
        try:
            # MPS uses unified memory, estimate based on system RAM
            import psutil
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            
            # MPS is less memory efficient than CUDA, use more conservative estimate
            memory_per_sample_gb = 0.8  # Higher overhead for MPS
            
            # Reserve 30% for system and safety
            available_memory_gb = total_memory_gb * 0.7
            
            # Calculate batch size
            theoretical_batch_size = int(available_memory_gb / memory_per_sample_gb)
            
            # Clamp to reasonable range (MPS works better with smaller batches)
            batch_size = max(4, min(theoretical_batch_size, 64))
            
            print(f"  MPS Memory: {total_memory_gb:.1f}GB → Batch size: {batch_size}")
            return batch_size
            
        except Exception as e:
            print(f"  ⚠ MPS memory detection failed: {e}, using default batch size 16")
            return 16
    
    def _calculate_cpu_batch_size(self) -> int:
        """Calculate batch size for CPU training."""
        try:
            # CPU training is much slower, use smaller batches
            import psutil
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            
            # CPU needs less memory per sample but is slower
            memory_per_sample_gb = 0.3
            
            # Reserve 40% for system
            available_memory_gb = total_memory_gb * 0.6
            
            # Calculate batch size
            theoretical_batch_size = int(available_memory_gb / memory_per_sample_gb)
            
            # Clamp to reasonable range for CPU
            batch_size = max(4, min(theoretical_batch_size, 32))
            
            print(f"  CPU Memory: {total_memory_gb:.1f}GB → Batch size: {batch_size}")
            return batch_size
            
        except Exception as e:
            print(f"  ⚠ CPU memory detection failed: {e}, using default batch size 8")
            return 8
    
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
                final_layer.bias[0] = 0.3  # linear: slow but increased (was 0.2)
                final_layer.bias[1] = 0.0  # angular: neutral
            elif policy_type == 'forward':
                # Bias: VERY fast forward, minimal turning
                final_layer.bias[0] = 0.75  # linear: very fast! (was 0.6)
                final_layer.bias[1] = 0.0  # angular: neutral
            elif policy_type == 'explorer':
                # Bias: moderate-high forward, slight turning tendency
                final_layer.bias[0] = 0.5  # linear: moderate-high (was 0.4)
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
            # Use tqdm for progress bar
            pbar = tqdm(range(0, num_candidates, batch_eval_size), 
                       desc="    Evaluating candidates", 
                       unit="batch",
                       leave=False)
            
            for batch_start in pbar:
                batch_end = min(batch_start + batch_eval_size, num_candidates)
                batch_candidates = candidates[batch_start:batch_end]

                # Collect predictions from all models in batch (fp32)
                batch_predictions = []
                for candidate, _ in batch_candidates:
                    candidate.eval()
                    pred_actions, _ = candidate(rgb, depth, proprio)  # LSTM returns (actions, hidden_state)
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
                
                # Update progress bar with current best fitness
                pbar.set_postfix({'best_fitness': f'{best_fitness:.4f}'})

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
        """Compute tank-optimized tournament fitness with reduced imitation weight.

        Instead of just measuring action imitation (MSE), test if mutation
        achieves the actual fitness objectives:
        - High speed when clearance is high
        - Low speed when clearance is low
        - Prefer forward motion over spinning
        - Smooth action transitions
        - Tank-specific: reward controlled pivot turns at low speeds

        Args:
            pred_actions: Predicted actions (N, 2) [linear, angular]
            target_actions: Original actions from trajectory (N, 2)
            proprio: Proprioception data (N, 6) [lin_vel, ang_vel, roll, pitch, accel, clearance]

        Returns:
            Fitness score (higher is better)
        """
        fitness = 0.0

        # Extract clearance and speed from proprioception
        clearance = proprio[:, 5]  # (N,)
        linear_speed = proprio[:, 0]  # (N,)

        # 1. IMITATION BASELINE (REDUCED to 15% weight)
        #    Reduced from 30% to encourage innovation over behavior cloning
        mse_loss = torch.nn.functional.mse_loss(pred_actions, target_actions)
        imitation_score = -mse_loss.item()
        fitness += imitation_score * 0.15  # Reduced from 0.30

        # 2. SPEED-CLEARANCE CORRELATION (INCREASED to 35% weight)
        #    Strongly reward appropriate speed based on clearance
        #    Tank needs to be very responsive to clearance due to tight collision distance
        high_clearance_mask = clearance > 1.5  # Reduced from 2.0 for tank environment
        low_clearance_mask = clearance < 0.3   # Reduced from 0.5 for tank environment

        if high_clearance_mask.any():
            # When clearance is high, should go forward confidently
            safe_linear = pred_actions[high_clearance_mask, 0].mean().item()
            fitness += safe_linear * 25.0  # Increased from 15.0 for tank responsiveness

        if low_clearance_mask.any():
            # When clearance is low, should be very cautious
            risky_linear = pred_actions[low_clearance_mask, 0].mean().item()
            fitness -= abs(risky_linear) * 15.0  # Increased from 10.0 for safety

        # 3. SMOOTH OBSTACLE AVOIDANCE: NEW
        #    Reward moving forward while smoothly turning near obstacles (the "flow" behavior)
        medium_clearance_mask = (clearance > 0.15) & (clearance < 0.8)  # Near obstacles
        forward_turning_mask = (torch.abs(pred_actions[:, 0]) > 0.3) & (torch.abs(pred_actions[:, 1]) > 0.15)

        if medium_clearance_mask.any() and forward_turning_mask.any():
            # Find timesteps with both medium clearance and forward+turning
            flow_mask = medium_clearance_mask & forward_turning_mask
            if flow_mask.any():
                # Reward smooth obstacle navigation
                flow_linear = pred_actions[flow_mask, 0].mean().item()
                flow_angular = torch.abs(pred_actions[flow_mask, 1]).mean().item()

                # Bonus scales with forward action magnitude
                if flow_linear > 0.4:
                    fitness += flow_linear * 18.0  # Strong reward for maintaining forward speed near obstacles

                # Extra bonus if turning is smooth and controlled
                if len(pred_actions[flow_mask]) >= 2:
                    flow_action_diffs = torch.diff(pred_actions[flow_mask], dim=0)
                    flow_smoothness = torch.std(flow_action_diffs).item()
                    if flow_smoothness < 0.15:
                        fitness += 12.0  # Bonus for smooth flow around obstacles

        # 4. TANK PIVOT TURN BONUS
        #    Reward controlled rotation at low speeds (critical for tank maneuvering)
        low_speed_mask = linear_speed < 0.05  # Very low linear speed
        moderate_angular_mask = torch.abs(pred_actions[:, 1]) > 0.2  # Some rotation

        if low_speed_mask.any() and moderate_angular_mask.any():
            # Find timesteps with both low speed and rotation
            pivot_mask = low_speed_mask & moderate_angular_mask
            if pivot_mask.any():
                # Reward smooth angular control during pivots
                pivot_angular_actions = pred_actions[pivot_mask, 1]

                # Only compute smoothness if we have enough samples
                if len(pivot_angular_actions) >= 2:
                    angular_smoothness = torch.std(pivot_angular_actions).item()

                    if angular_smoothness < 0.15:  # Smooth pivot control
                        fitness += 10.0  # Bonus for good pivot turns
                    else:
                        fitness -= 5.0   # Penalty for jerky pivots

        # 5. FORWARD BIAS (INCREASED to 30% weight)
        #    Tank steering: forward motion is good, spinning is bad
        linear_magnitude = torch.abs(pred_actions[:, 0]).mean().item()
        angular_magnitude = torch.abs(pred_actions[:, 1]).mean().item()
        total_action = linear_magnitude + angular_magnitude + 1e-6
        forward_ratio = linear_magnitude / total_action

        # Reward forward-biased actions MORE STRONGLY
        if forward_ratio > 0.6:
            fitness += (forward_ratio - 0.6) * 35.0  # INCREASED strong reward (was 25.0)
        elif forward_ratio < 0.3:
            fitness -= (0.3 - forward_ratio) * 40.0  # INCREASED penalty for spinning (was 30.0)

        # Additional bonus for high linear values (not just ratio)
        if linear_magnitude > 0.5:
            fitness += (linear_magnitude - 0.5) * 20.0  # Reward high forward actions

        # 6. ACTION SMOOTHNESS (15% weight) - Keep similar
        #    Smooth actions indicate stable, confident behavior
        if pred_actions.shape[0] > 1:
            action_diffs = torch.diff(pred_actions, dim=0)
            angular_smoothness = torch.std(action_diffs[:, 1]).item()

            if angular_smoothness < 0.1:
                fitness += (0.1 - angular_smoothness) * 20.0  # Reward smooth
            elif angular_smoothness > 0.3:
                fitness -= (angular_smoothness - 0.3) * 15.0  # Penalize jerky

        # 7. AVOID EXTREME ACTIONS (bonus) - Keep similar
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
        batch_size: int = None,  # Now optional, will use self.batch_size if None
        use_replay_buffer: bool = True
    ) -> dict:
        """Refine model using gradient descent on trajectory data (behavioral cloning).

        Args:
            model_state: Model state dict to refine
            trajectory_data: Dictionary with 'rgb', 'depth', 'proprio', 'actions'
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
            batch_size: Batch size for training (auto-scaled if not provided)
            use_replay_buffer: Mix in data from trajectory buffer

        Returns:
            Refined model state dict
        """
        # Use auto-scaled batch size if not specified
        if batch_size is None:
            batch_size = self.batch_size
        
        print(f"  📊 Gradient refinement with batch size: {batch_size}")
        
        # Load model
        model = ActorNetwork().to(self.device)
        model.load_state_dict(model_state)
        model.train()

        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Log memory usage before training
        if self.device.type == 'cuda':
            memory_before = torch.cuda.memory_allocated() / (1024**3)
            print(f"  💾 GPU Memory before: {memory_before:.2f}GB")

        # Prepare current trajectory data
        rgb_list = [trajectory_data['rgb'].copy()]
        depth_list = [trajectory_data['depth'].copy()]
        proprio_list = [trajectory_data['proprio']]
        actions_list = [trajectory_data['actions']]

        # Mix in replay buffer data (prevents catastrophic forgetting)
        if use_replay_buffer and len(self.replay_buffer) > 0:
            # Sample up to 5 past trajectories
            replay_trajectories = self.replay_buffer.sample(5)
            
            for replay_traj in replay_trajectories:
                rgb_list.append(replay_traj['rgb'])
                depth_list.append(replay_traj['depth'])
                proprio_list.append(replay_traj['proprio'])
                actions_list.append(replay_traj['actions'])

            print(f"    Mixed in {len(replay_trajectories)} trajectories from replay buffer", flush=True)

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

                optimizer.zero_grad()

                # Forward pass with automatic mixed precision if enabled
                if self.use_fp16:
                    with torch.cuda.amp.autocast():
                        # fp16 forward pass - LSTM returns (actions, hidden_state)
                        pred_actions, _ = model(rgb_batch, depth_batch, proprio_batch)
                        # Behavioral cloning loss (MSE)
                        loss = torch.nn.functional.mse_loss(pred_actions, actions_batch)
                    
                    # Scaled backward pass for fp16
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    # fp32 forward pass - LSTM returns (actions, hidden_state)
                    pred_actions, _ = model(rgb_batch, depth_batch, proprio_batch)
                    # Behavioral cloning loss (MSE)
                    loss = torch.nn.functional.mse_loss(pred_actions, actions_batch)
                    
                    # Standard backward pass for fp32
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
        
        # Log memory usage after training
        if self.device.type == 'cuda':
            memory_after = torch.cuda.memory_allocated() / (1024**3)
            print(f"  💾 GPU Memory after: {memory_after:.2f}GB")
            print(f"  💾 GPU Memory used: {memory_after - memory_before:.2f}GB")

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
        """Compute tank-optimized fitness with enhanced diversity for single evolution.

        Goals:
        - Maximize collision-free exploration distance (tank-speed adjusted)
        - Reward smooth, efficient motion (critical for track longevity)
        - Reward intelligent pivot turns (tank-specific)
        - Penalize track slippage and unproductive spinning
        - Reward path quality and obstacle clearance
        - Strong diversity bonus to prevent single-evolution convergence
        """
        # Extract standard metrics
        distance = episode_data['total_distance']
        collisions = episode_data['collision_count']
        avg_speed = episode_data.get('avg_speed', 0.0)
        avg_clearance = episode_data['avg_clearance']
        duration = max(episode_data['duration'], 1.0)
        action_smoothness = episode_data.get('action_smoothness', 0.0)
        avg_linear_action = episode_data.get('avg_linear_action', 0.0)
        avg_angular_action = episode_data.get('avg_angular_action', 0.0)
        
        # NEW: Tank-specific metrics
        turn_efficiency = episode_data.get('turn_efficiency', 0.0)
        stationary_rotation = episode_data.get('stationary_rotation_time', 0.0)
        track_slip = episode_data.get('track_slip_detected', False)
        
        # NEW: Coverage and Oscillation
        coverage_count = episode_data.get('coverage_count', 0)
        oscillation_count = episode_data.get('oscillation_count', 0)

        # Base fitness: exploration distance (INCREASED to encourage forward movement)
        # Tank max speed is 0.18 m/s, but we want to strongly reward distance covered
        fitness = distance * 3.5  # INCREASED from 2.0 to encourage more forward movement

        # 1. Collision penalty: Softer for tank learning
        # Tank collision distance is very tight (0.12m), needs more forgiveness
        if collisions > 0:
            collision_penalty = 6.0 * (collisions ** 1.2)  # Reduced from 15.0 * 1.5
            fitness -= collision_penalty

        # 2. Path quality bonus: Enhanced for indoor navigation
        # Reward maintaining clearance and moving towards open space
        if avg_clearance > 0.15:  # 0.12m collision + 0.03m buffer
            # Base clearance reward
            clearance_bonus = min((avg_clearance - 0.15) * 10.0, 18.0)  # INCREASED (was 8.0, cap 15.0)
            fitness += clearance_bonus

            # NEW: Bonus for being in "good" clearance zones (0.3m+ = comfortable indoor space)
            if avg_clearance > 0.3:
                open_space_bonus = min((avg_clearance - 0.3) * 6.0, 12.0)  # INCREASED (was 5.0, cap 10.0)
                fitness += open_space_bonus

        # NEW: Smooth obstacle avoidance bonus (moving forward while turning near obstacles)
        # Reward the tank for maintaining speed while smoothly turning around obstacles
        if avg_clearance > 0.15 and avg_clearance < 0.8:  # Near obstacles but not colliding
            if avg_linear_action > 0.3 and avg_angular_action > 0.15:  # Moving and turning
                if action_smoothness < 0.2:  # Smooth motion
                    # This is the "flow" behavior we want - smooth navigation around obstacles
                    obstacle_flow_bonus = 15.0
                    fitness += obstacle_flow_bonus

                    # Extra bonus if maintaining good speed while doing this
                    if avg_speed > 0.08:  # Good speed for tank (0.18 max)
                        fitness += 10.0

        # 3. Tank pivot turn reward: Enhanced for indoor navigation
        # Reward efficient stationary rotation and scanning behavior
        if avg_speed < 0.05 and avg_angular_action > 0.2:  # Low speed, high rotation
            if turn_efficiency > 0.5:  # Efficient pivot (good heading change per rotation)
                fitness += 8.0  # Strong bonus for good pivot turns
            else:
                fitness -= 3.0  # Penalty for sloppy pivots
            
            # NEW: Bonus for controlled scanning behavior (slow, deliberate turns)
            if avg_angular_action < 0.5:  # Moderate angular speed (not spinning wildly)
                if action_smoothness < 0.15:  # Smooth turning
                    scanning_bonus = (0.15 - action_smoothness) * 10.0
                    fitness += scanning_bonus

        # 4. Penalize unproductive spinning and reward forward motion: ENHANCED
        # Tank should not spin in place without making progress
        if avg_linear_action < 0.1 and avg_angular_action > 0.4:  # Mostly spinning
            if distance < 1.0:  # Little progress
                fitness -= 20.0  # INCREASED penalty for unproductive spinning (was 15.0)
            elif avg_angular_action > 0.7:  # Wild spinning
                fitness -= 35.0  # INCREASED penalty for chaotic behavior (was 25.0)

        # NEW: Explicit forward motion bonus
        # Strongly reward good linear action (forward movement)
        if avg_linear_action > 0.3:  # Decent forward action
            forward_bonus = (avg_linear_action - 0.3) * 15.0  # Scale: 0.3->0.0, 1.0->10.5
            fitness += forward_bonus

        # Additional bonus for high forward + low angular (straight forward movement)
        if avg_linear_action > 0.5 and avg_angular_action < 0.3:
            straight_forward_bonus = 12.0
            fitness += straight_forward_bonus

        # 5. Smooth motion reward: INCREASED (critical for track longevity)
        if action_smoothness > 0:
            if action_smoothness < 0.08:  # Very smooth (was 0.1)
                smoothness_bonus = (0.08 - action_smoothness) * 20.0  # Increased from 15.0
                fitness += smoothness_bonus
            elif action_smoothness > 0.25:  # Jerky (was 0.3)
                jerkiness_penalty = (action_smoothness - 0.25) * 12.0  # Increased from 8.0
                fitness -= jerkiness_penalty

        # 6. Track slippage penalty: NEW
        if track_slip:
            fitness -= 20.0  # Heavy penalty for track slippage (inefficient, damaging)

        # 7. Efficiency: Enhanced for indoor exploration with stronger distance rewards
        if distance < 0.5 and collisions == 0:  # Barely moved
            fitness *= 0.3  # HARSHER penalty for stagnation (was 0.4)
        elif (distance / duration) < 0.02:  # Very inefficient
            fitness *= 0.6  # STRONGER penalty (was 0.7)

        # NEW: Reward consistent exploration patterns (not just distance)
        if distance > 2.0 and collisions == 0 and action_smoothness < 0.2:
            exploration_bonus = min(distance * 0.8, 15.0)  # INCREASED bonus (was 0.5 and capped at 10.0)
            fitness += exploration_bonus

        # Additional distance milestone bonuses
        if distance > 5.0:
            milestone_bonus = 8.0  # Good progress
            fitness += milestone_bonus
        if distance > 10.0:
            milestone_bonus = 15.0  # Excellent progress
            fitness += milestone_bonus

        # NEW: Coverage bonus (visiting new areas)
        if coverage_count > 5:
            coverage_bonus = coverage_count * 2.0
            fitness += coverage_bonus
            
        # NEW: Oscillation penalty
        if oscillation_count > 0:
            fitness -= oscillation_count * 5.0

        # 8. Diversity bonus: SIGNIFICANTLY INCREASED for single evolution
        # Single evolution needs strong diversity to avoid premature convergence
        if self.enable_diversity_bonus and fitness > 0:
            novelty_score = self.population.calculate_behavior_novelty(avg_speed, avg_clearance)
            if novelty_score > 0.6:  # More selective (was 0.5)
                diversity_bonus = novelty_score * 12.0  # Massively increased from 3.0
                fitness += diversity_bonus

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

                    # Checkpoint strategy: every eval during warmup, then every 10 evals
                    should_checkpoint = False
                    if evaluation_count <= 50:
                        # During warmup: checkpoint EVERY evaluation (progress is slow/expensive)
                        should_checkpoint = True
                    else:
                        # After warmup: checkpoint every 10 evaluations
                        should_checkpoint = (evaluation_count % 10 == 0)

                    # Special checkpoint at warmup completion (eval 50)
                    if evaluation_count == 50:
                        self.save_checkpoint(evaluation_count, warmup_complete=True)
                        print(f"🎯 Warmup complete! Saved checkpoint: evolution_warmup_complete.json",
                              flush=True)
                    elif should_checkpoint:
                        self.save_checkpoint(evaluation_count)

                    # Print status at every checkpoint
                    if should_checkpoint:
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
                        # Support both Zstandard (new) and LZ4 (old) for backward compatibility
                        compression_type = message.get('compression', 'lz4')  # Default to lz4 for old clients

                        if compression_type == 'zstd':
                            # Zstandard decompression
                            dctx = zstd.ZstdDecompressor()
                            rgb_bytes = dctx.decompress(trajectory_raw['rgb'])
                            depth_bytes = dctx.decompress(trajectory_raw['depth'])
                        else:
                            # LZ4 decompression (backward compatibility)
                            import lz4.frame
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

                    print(f"  📦 Caching model #{model_id} ({len(trajectory_data['actions'])} samples) for tournament", flush=True)
                    
                    # Add to Replay Buffer if it was a good run (fitness > 0)
                    # We don't have the fitness here directly, but we can infer it or just add all valid trajectories
                    # The buffer will sort and keep the best ones
                    # We need to find the fitness for this model_id
                    # It was just evaluated, so we can look it up in population or just pass a placeholder
                    # Better: The episode_result message came before this. We should have stored the fitness.
                    # For now, we'll use a heuristic or just add it.
                    # Actually, let's look up the fitness from the population if possible
                    # But the model might not be in population yet if it's pending refinement
                    # Let's just add it with a default high priority if it's a champion
                    priority = 100.0 if is_champion else 10.0
                    self.replay_buffer.add(trajectory_data, priority)
                    print(f"  💾 Added to Replay Buffer (size: {len(self.replay_buffer)})", flush=True)

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

    def save_checkpoint(self, evaluation: int, final: bool = False, warmup_complete: bool = False):
        """Save population checkpoint."""
        if final:
            suffix = 'final'
        elif warmup_complete:
            suffix = 'warmup_complete'
        else:
            suffix = f'eval_{evaluation}'
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

        # Create new population tracker
        # Prefer saved initial/max sizes if available (new format), otherwise use current values
        initial_pop = metadata.get('initial_population_size', self.population.initial_population_size)
        max_pop = metadata.get('max_population_size', metadata.get('population_size', self.population.max_population_size))
        self.population = PopulationTracker(
            initial_population_size=initial_pop,
            max_population_size=max_pop
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
    parser = argparse.ArgumentParser(description='Adaptive evolution trainer with greedy code')
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
            # Look for checkpoints in order: final, warmup_complete, then highest eval_N
            final_checkpoint = checkpoint_dir / 'evolution_final.json'
            warmup_checkpoint = checkpoint_dir / 'evolution_warmup_complete.json'

            if final_checkpoint.exists():
                checkpoint_to_load = final_checkpoint
            else:
                # Find highest eval_N checkpoint
                eval_checkpoints = list(checkpoint_dir.glob('evolution_eval_*.json'))

                # Also consider warmup_complete as a candidate
                if warmup_checkpoint.exists():
                    eval_checkpoints.append(warmup_checkpoint)

                if eval_checkpoints:
                    # Extract numbers and find max (treat warmup_complete as eval_50)
                    def get_eval_num(path):
                        try:
                            if 'warmup_complete' in path.stem:
                                return 50
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
