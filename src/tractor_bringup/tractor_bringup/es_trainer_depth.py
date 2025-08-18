#!/usr/bin/env python3
"""
Evolutionary Strategy Training System for Autonomous Rover Exploration (Depth Image Version)
Implements evolutionary algorithms for neural network optimization using depth images
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

from .rknn_trainer_depth import DepthImageExplorationNet

class EvolutionaryStrategyTrainer:
    """
    Handles model training using evolutionary strategies, data collection, and RKNN conversion for depth images
    """
    
    def __init__(self, model_dir="models", stacked_frames: int = 1, enable_debug: bool = False, 
                 population_size: int = 10, sigma: float = 0.1, learning_rate: float = 0.01):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.enable_debug = enable_debug
        self.target_h = 160
        self.target_w = 288
        self.clip_max_distance = 4.0
        self.stacked_frames = stacked_frames
        self.frame_stack: deque = deque(maxlen=stacked_frames)
        
        # Extended proprio feature count now: base(3) + extras(13) = 16 total features
        self.extra_proprio = 13  # updated to match 16-element proprio vector (3 base + 13 extras)
        
        # Evolutionary Strategy parameters
        self.population_size = population_size
        self.sigma = sigma  # Standard deviation of parameter perturbations
        self.learning_rate = learning_rate
        self.generation = 0
        self.population = []
        self.fitness_history = deque(maxlen=1000)
        self.best_fitness = -float('inf')
        self.best_model_state = None
        
        # Neural network (same architecture as RL version)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DepthImageExplorationNet(stacked_frames=stacked_frames, extra_proprio=self.extra_proprio).to(self.device)
        
        # Initialize population
        self._initialize_population()
        
        # Experience replay (simplified for ES)
        self.buffer_capacity = 10000
        self.buffer_size = 0
        self.insert_ptr = 0
        self.proprio_dim = 3 + self.extra_proprio
        self.depth_store = np.zeros((self.buffer_capacity, self.stacked_frames, self.target_h, self.target_w), dtype=np.float32)
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
        for i in range(self.population_size):
            # Random perturbation
            perturbation = np.random.randn(*self.param_shape) * self.sigma
            self.population.append(perturbation)
        
        print(f"[ES] Initialized population with {self.population_size} individuals")

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
            param.data = torch.from_numpy(param_values).to(self.device)
            idx += param_size

    def add_experience(self,
                      depth_image: np.ndarray,
                      proprioceptive: np.ndarray,
                      action: np.ndarray,
                      reward: float,
                      next_depth_image: np.ndarray = None,
                      done: bool = False,
                      collision: bool = False,
                      in_recovery: bool = False):
        """Add experience to replay buffer."""
        # Preprocess depth image to (C,H,W)
        processed = depth_image if depth_image.ndim == 3 else self.preprocess_depth_for_storage(depth_image)
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
        self.depth_store[i] = processed.astype(np.float32)
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
        
        # Collect samples for RKNN quantization dataset
        if (self.dataset_samples_collected < self.target_dataset_samples and 
            self.generation % self.dataset_collection_interval == 0):
            self.save_quantization_sample(depth_image, proprioceptive)

    def calculate_reward(self, 
                        action: np.ndarray,
                        collision: bool,
                        progress: float,
                        exploration_bonus: float,
                        position: Optional[np.ndarray] = None,
                        depth_data: Optional[np.ndarray] = None,
                        wheel_velocities: Optional[Tuple[float, float]] = None) -> float:
        """Calculate reward for evolutionary strategy (same as RL version)"""
        
        # Use improved reward system if available
        if self.reward_calculator is not None and position is not None:
            # Use comprehensive reward calculation
            near_collision = False  # You can enhance this based on depth data
            if depth_data is not None:
                try:
                    valid_depths = depth_data[(depth_data > 0.1) & (depth_data < 5.0)]
                    if len(valid_depths) > 0:
                        min_distance = np.min(valid_depths)
                        near_collision = 0.3 < min_distance < 0.5  # Close but not colliding
                except:
                    pass
            
            total_reward, reward_breakdown = self.reward_calculator.calculate_comprehensive_reward(
                action=action,
                position=position,
                collision=collision,
                near_collision=near_collision,
                progress=progress,
                depth_data=depth_data,
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
            
        # Speed efficiency (not too slow, not too fast)
        speed = abs(action[0])
        if 0.05 < speed < 0.3:
            reward += 2.0
        elif speed < 0.02:
            reward -= 5.0  # Penalize being too slow
            
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
        valid_samples = 0
        
        # Evaluate on batch
        for idx in indices:
            depth_data = self.depth_store[idx]
            proprio_data = self.proprio_store[idx]
            actual_action = self.action_store[idx]
            actual_reward = self.reward_store[idx]
            
            # Get model prediction
            with torch.no_grad():
                depth_tensor = torch.from_numpy(depth_data).unsqueeze(0).float().to(self.device)
                proprio_tensor = torch.from_numpy(proprio_data).unsqueeze(0).float().to(self.device)
                predicted_output = self.model(depth_tensor, proprio_tensor)
                predicted_action = torch.tanh(predicted_output[0, :2]).cpu().numpy()
            
            # Calculate how close prediction is to actual action weighted by reward
            action_similarity = 1.0 - np.linalg.norm(predicted_action - actual_action)
            fitness = actual_reward * max(0, action_similarity)  # Only positive similarity contributes
            
            total_reward += fitness
            valid_samples += 1
        
        avg_fitness = total_reward / max(valid_samples, 1)
        
        # Restore original parameters
        self._set_flat_params(original_params)
        
        return avg_fitness

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
        if fitness_scores[best_idx] > self.best_fitness:
            self.best_fitness = fitness_scores[best_idx]
            self.best_model_state = self._get_flat_params() + self.population[best_idx]
            print(f"[ES] New best fitness: {self.best_fitness:.4f}")
        
        # Natural evolution strategies update
        # Standardize fitness scores
        fitness_std = np.std(fitness_scores)
        if fitness_std == 0:
            fitness_std = 1e-8  # Avoid division by zero
        standardized_fitness = (fitness_scores - np.mean(fitness_scores)) / fitness_std
        
        # Update parameters using natural gradient
        original_params = self._get_flat_params()
        
        # Calculate gradient estimate
        grad_estimate = np.zeros_like(original_params)
        for i in range(self.population_size):
            grad_estimate += standardized_fitness[i] * self.population[i]
        grad_estimate /= (self.population_size * self.sigma)
        
        # Update parameters
        new_params = original_params + self.learning_rate * grad_estimate
        self._set_flat_params(new_params)
        
        # Generate new population
        self._initialize_population()
        
        # Update generation counter
        self.generation += 1
        
        # Save model periodically
        if self.generation % 50 == 0:
            self.save_model()
        
        # Convert to RKNN periodically
        if self.generation % 100 == 0:
            try:
                self.convert_to_rknn()
            except Exception as e:
                print(f"RKNN conversion failed: {e}")
        
        return {
            "generation": self.generation,
            "avg_fitness": float(np.mean(fitness_scores)),
            "best_fitness": float(self.best_fitness),
            "fitness_std": float(fitness_std),
            "samples": self.buffer_size
        }

    def preprocess_depth_for_model(self, depth_image: np.ndarray) -> np.ndarray:
        """Normalize, resize, clip and optionally stack frames.
        Input depth_image: (H,W) meters.
        Returns (stacked_frames, target_h, target_w) float32 in [0,1]."""
        try:
            # Clip far ranges to reduce dynamic range then normalize
            depth = np.nan_to_num(depth_image, nan=0.0, posinf=self.clip_max_distance, neginf=0.0)
            depth = np.clip(depth, 0.0, self.clip_max_distance)
            # Resize to target
            depth_resized = cv2.resize(depth, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)
            # Scale to [0,1]
            depth_norm = depth_resized / self.clip_max_distance
            # Append to stack
            self.frame_stack.append(depth_norm.astype(np.float32))
            # If stack not full yet, pad with copies of first
            while len(self.frame_stack) < self.stacked_frames:
                self.frame_stack.append(self.frame_stack[0])
            stacked = np.stack(list(self.frame_stack), axis=0)  # (C,H,W)
            return stacked
        except Exception:
            # Fallback zero tensor
            return np.zeros((self.stacked_frames, self.target_h, self.target_w), dtype=np.float32)
    
    def preprocess_depth_for_storage(self, depth_image: np.ndarray) -> np.ndarray:
        """Preprocess without mutating frame stack (single frame replicated)."""
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
        
    def enable_rknn_inference(self):
        """Enable RKNN runtime inference if model file exists."""
        if not RKNN_AVAILABLE:
            if self.enable_debug:
                print("[RKNN] Toolkit not available - cannot enable RKNN inference")
            return False
        rknn_path = os.path.join(self.model_dir, "exploration_model_depth_es.rknn")
        if not os.path.exists(rknn_path):
            # Try fallback to RL model
            rknn_path = os.path.join(self.model_dir, "exploration_model_depth.rknn")
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

    def inference(self, depth_image: np.ndarray, proprioceptive: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run inference to get action and confidence.
        Uses RKNN runtime when enabled; otherwise PyTorch."""
        if self.use_rknn_inference and self.rknn_runtime is not None:
            try:
                processed = self.preprocess_depth_for_model(depth_image)  # (C,H,W)
                depth_input = processed[np.newaxis, ...].astype(np.float32)
                sensor_input = proprioceptive.astype(np.float32)[np.newaxis, ...]
                # RKNN expects list of inputs in same order as export
                outputs = self.rknn_runtime.inference(inputs=[depth_input, sensor_input])
                if not outputs:
                    raise RuntimeError("RKNN inference returned no outputs")
                out = outputs[0]  # (1,3)
                if out is None:
                    raise RuntimeError("RKNN output None")
                raw = out[0]
                # Apply same post-processing as PyTorch path
                action = np.tanh(raw[:2])
                confidence = 1.0 / (1.0 + np.exp(-raw[2]))  # sigmoid
                return action.astype(np.float32), float(confidence)
            except Exception as e:
                if self.enable_debug:
                    print(f"[RKNN] Inference failed, falling back to PyTorch: {e}")
                # Fallback to PyTorch
        # Default PyTorch path
        self.model.eval()
        with torch.no_grad():
            processed = self.preprocess_depth_for_model(depth_image)
            depth_tensor = torch.from_numpy(processed).unsqueeze(0).to(self.device)
            sensor_tensor = torch.FloatTensor(proprioceptive).unsqueeze(0).to(self.device)
            output = self.model(depth_tensor, sensor_tensor)
            action = torch.tanh(output[0, :2]).cpu().numpy()
            confidence = torch.sigmoid(output[0, 2]).item()
        self.model.train()
        return action, float(confidence)
        
    def save_model(self):
        """Save PyTorch model and training state"""
        try:
            # Ensure model directory exists
            os.makedirs(self.model_dir, exist_ok=True)
            
            timestamp = int(time.time())
            
            # Save PyTorch model
            model_path = os.path.join(self.model_dir, f"exploration_model_depth_es_{timestamp}.pth")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'generation': self.generation,
                'best_fitness': self.best_fitness,
                'fitness_history': list(self.fitness_history)
            }, model_path)
            
            # Save latest symlink
            latest_path = os.path.join(self.model_dir, "exploration_model_depth_es_latest.pth")
            if os.path.exists(latest_path):
                os.remove(latest_path)
            os.symlink(os.path.basename(model_path), latest_path)
            
            print(f"ES Model saved: {model_path}")
        except Exception as e:
            print(f"Failed to save ES model: {e}")
        
    def load_latest_model(self):
        """Load the latest saved model"""
        try:
            latest_path = os.path.join(self.model_dir, "exploration_model_depth_es_latest.pth")
            if os.path.exists(latest_path) and os.path.islink(latest_path):
                actual_model_path = os.path.join(self.model_dir, os.readlink(latest_path))
            elif os.path.exists(latest_path):
                actual_model_path = latest_path
            else:
                # Try to load RL model as starting point
                latest_path_rl = os.path.join(self.model_dir, "exploration_model_depth_latest.pth")
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
            print(f"Loaded ES model from {actual_model_path}")
            print(f"Generation: {self.generation}, Best Fitness: {self.best_fitness}")
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
                onnx_path = os.path.join(self.model_dir, "exploration_model_depth_es.onnx")
                dummy_depth = torch.randn(1, self.stacked_frames, self.target_h, self.target_w).to(self.device)
                dummy_sensor = torch.randn(1, self._proprio_feature_size()).to(self.device)
                torch.onnx.export(
                    self.model,
                    (dummy_depth, dummy_sensor),
                    onnx_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['depth_image', 'sensor'],
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
            dummy_depth = torch.randn(1, self.stacked_frames, self.target_h, self.target_w).to(self.device)
            proprio_size = self._proprio_feature_size()
            dummy_sensor = torch.randn(1, proprio_size).to(self.device)
            if self.enable_debug:
                print(f"[RKNN] Dummy tensors created depth_shape={tuple(dummy_depth.shape)} sensor_shape={tuple(dummy_sensor.shape)}")
            # Sanity check flatten size
            with torch.no_grad():
                self.model.eval()
                depth_features = self.model.depth_conv(dummy_depth)
                # depth_conv ends with Flatten(), so depth_features is (B, F)
                flat_dim = depth_features.shape[1]
                expected = self.model.depth_fc.in_features
                if self.enable_debug:
                    print(f"[RKNN] Depth feature flatten size={flat_dim} expected={expected}")
                if flat_dim != expected:
                    print(f"[RKNN Export] WARNING: depth flatten size {flat_dim} != expected {expected}. Aborting export.")
                    self.model.train()
                    return
            onnx_path = os.path.join(self.model_dir, "exploration_model_depth_es.onnx")
            if self.enable_debug:
                print(f"[RKNN] Exporting ONNX to {onnx_path}")
            torch.onnx.export(
                self.model,
                (dummy_depth, dummy_sensor),
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['depth_image', 'sensor'],
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
                print(f"[RKNN] Simple config failed: {e}. Trying depth-only mean/std.")
                try:
                    mean_values = [[0.0] * self.stacked_frames]
                    std_values = [[1.0] * self.stacked_frames]
                    rknn.config(mean_values=mean_values, std_values=std_values, target_platform='rk3588')
                except Exception as e2:
                    print(f"[RKNN] Depth-only config failed: {e2}. Aborting.")
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
            rknn_path = os.path.join(self.model_dir, "exploration_model_depth_es.rknn")
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
            "avg_fitness": np.mean(self.fitness_history) if self.fitness_history else 0.0,
            "best_fitness": self.best_fitness,
            "buffer_full": self.buffer_size / self.buffer_capacity
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

    def save_quantization_sample(self, depth_image: np.ndarray, proprioceptive: np.ndarray):
        """Save a sample to dataset.txt for RKNN quantization"""
        try:
            # Get the dataset path (same as used in convert_to_rknn)
            dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'dataset.txt')
            dataset_path = os.path.abspath(dataset_path)
            
            # Preprocess the sample to match inference format
            processed_depth = self.preprocess_depth_for_model(depth_image)  # (C,H,W)
            depth_input = processed_depth[np.newaxis, ...].astype(np.float32)  # (1,C,H,W)
            
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
            temp_depth_file = tempfile.mktemp(suffix='.npy')
            temp_sensor_file = tempfile.mktemp(suffix='.npy')
            
            # Save inputs to temporary files
            np.save(temp_depth_file, depth_input)
            np.save(temp_sensor_file, sensor_input)
            
            # Write paths to dataset.txt (one line per sample with paths to input files)
            with open(dataset_path, 'a') as f:
                f.write(f"{temp_depth_file} {temp_sensor_file}\n")
            
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
