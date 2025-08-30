#!/usr/bin/env python3
"""
RKNN Training System for Autonomous Rover Exploration (BEV Version)
Implements real-time learning with multi-modal sensor fusion using Bird's Eye View maps
"""

import numpy as np
import time
import os
import pickle
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
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
    from .bayesian_es_optimizer import BayesianTrainingOptimizer, TrainingHyperparameters
    BAYESIAN_TRAINING_AVAILABLE = True
except ImportError:
    BAYESIAN_TRAINING_AVAILABLE = False
    print("Bayesian training optimization not available - using fixed parameters")

class BEVExplorationNet(nn.Module):
    """
    Neural network for BEV-based rover exploration
    Inputs: BEV image (multi-channel), IMU data, proprioceptive data
    Outputs: Linear velocity, angular velocity, exploration confidence
    """
    
    def __init__(self, bev_channels: int = 4, extra_proprio: int = 0):
        super().__init__()
        self.bev_channels = bev_channels
        # BEV image branch (CNN)
        self.bev_conv = nn.Sequential(
            nn.Conv2d(bev_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(3, 3), stride=(3, 3)),
            nn.Flatten()
        )
        # After conv stack with input 200x200 -> sizes: /2=100x100 /2=50x50 /2=25x25 /2=12x12 then pool /3 -> 4x4
        self.bev_fc = nn.Linear(256 * 4 * 4, 512)
        proprio_inputs = 3 + extra_proprio  # base + added features
        self.sensor_fc = nn.Sequential(
            nn.Linear(proprio_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.fusion = nn.Sequential(
            nn.Linear(512 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )
        
    def forward(self, bev_image, sensor_data):
        # BEV image processing
        bev_features = self.bev_conv(bev_image)
        bev_out = self.bev_fc(bev_features)
        
        # Sensor processing
        sensor_out = self.sensor_fc(sensor_data)
        
        # Fusion
        fused = torch.cat([bev_out, sensor_out], dim=1)
        output = self.fusion(fused)
        
        return output

# NEW: lightweight proxy so external len(trainer.experience_buffer) keeps working
class _BufferLenProxy:
    def __init__(self, trainer):
        self._trainer = trainer
    def __len__(self):
        return self._trainer.buffer_size

class RKNNTrainerBEV:
    """
    Handles model training, data collection, and RKNN conversion for BEV images
    """
    
    def __init__(self, model_dir="models", bev_channels: int = 4, enable_debug: bool = False, 
                 enable_bayesian_training_optimization: bool = True):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.enable_debug = enable_debug
        self.bev_channels = bev_channels
        self.bev_height = 200
        self.bev_width = 200
        # Extended proprio feature count now: base(3) + extras(13) = 16 total features
        self.extra_proprio = 13  # updated to match 16-element proprio vector (3 base + 13 extras)
        
        # Bayesian training optimization setup
        self.enable_bayesian_training_optimization = enable_bayesian_training_optimization and BAYESIAN_TRAINING_AVAILABLE
        
        # Training parameters to encourage forward movement and reduce spinning
        self.forward_movement_weight = 2.0  # Increase weight for forward movement actions (increased from 1.5)
        self.angular_movement_penalty = 0.5  # Reduce weight for angular movement actions (decreased from 0.7)
        self.backward_movement_penalty = 0.2  # Strongly penalize backward movement actions
        # Neural network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BEVExplorationNet(bev_channels=bev_channels, extra_proprio=self.extra_proprio).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # Experience replay (replaced deque with ring buffer + priorities for prioritized replay)
        self.buffer_capacity = 50000  # Increased for RK3588's abundant RAM
        self.buffer_size = 0
        self.insert_ptr = 0
        self.proprio_dim = 3 + self.extra_proprio
        self.bev_store = np.zeros((self.buffer_capacity, self.bev_channels, self.bev_height, self.bev_width), dtype=np.float32)
        self.proprio_store = np.zeros((self.buffer_capacity, self.proprio_dim), dtype=np.float32)
        self.action_store = np.zeros((self.buffer_capacity, 2), dtype=np.float32)  # linear, angular
        self.reward_store = np.zeros((self.buffer_capacity,), dtype=np.float32)
        self.done_store = np.zeros((self.buffer_capacity,), dtype=np.uint8)
        # We do not currently use next_bev_image in training -> omit to save memory
        # Prioritized replay parameters
        self.priorities = np.zeros((self.buffer_capacity,), dtype=np.float32)
        self.pr_alpha = 0.6
        self.pr_beta = 0.4
        self.pr_beta_inc = (1.0 - 0.4) / 50000.0  # anneal over ~50k steps
        self.priority_epsilon = 0.01
        self.collision_priority_bonus = 5.0
        self.recovery_priority_bonus = 2.0
        # Backwards compatibility proxy (external code only uses len())
        self.experience_buffer = _BufferLenProxy(self)
        self.batch_size = 32
        # Adaptive batch sizing (warmup with larger batches for vectorization)
        self.max_warmup_batch = 128  # can increase if memory allows
        self.post_warmup_batch = 32
        self.warmup_batch_steps = 2500  # steps to keep large batch
        self.min_buffer_for_max_batch = 2000  # need enough samples before using max batch
        
        # Training state
        self.training_step = 0
        self.last_save_time = time.time()
        self.save_interval = 300  # Save every 5 minutes (time-based)
        self.step_save_interval = 250  # NEW: fallback checkpoint every N steps
        self.last_rknn_conversion = 0
        self.rknn_conversion_interval = 300  # hard max interval (seconds) (was 1800)
        # Metric-gated export parameters
        self.rknn_min_interval = 600  # minimum seconds between exports if improvement seen
        self.rknn_loss_improve_ratio = 0.90  # export if median loss improved by >=10%
        self.rknn_reward_improve_delta = 0.5  # or avg reward improved by this amount
        self.loss_history = deque(maxlen=200)
        self.last_export_loss = None
        self.last_export_avg_reward = None
        
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
        
        # Initialize Bayesian training optimization if available
        if self.enable_bayesian_training_optimization:
            # Extract current training parameters
            current_training_params = TrainingHyperparameters(
                learning_rate=0.001,  # Current Adam LR
                batch_size=self.batch_size,
                prioritized_replay_alpha=self.pr_alpha,
                prioritized_replay_beta=self.pr_beta,
                reward_clip_min=-30.0,  # Current reward clipping
                reward_clip_max=30.0,
                dropout_rate=0.3,  # Current dropout in network
                weight_decay=1e-5,  # Current weight decay 
                gradient_clip_norm=1.0  # Current gradient clipping
            )
            
            self.bayesian_training_optimizer = BayesianTrainingOptimizer(
                initial_params=current_training_params,
                optimization_steps=20,
                enable_debug=self.enable_debug
            )
            
            # Training optimization state
            self.training_optimization_interval = 250  # Optimize every N training steps
            self.last_training_optimization_step = 0
            self.current_training_params = current_training_params
            self.training_fitness_history = deque(maxlen=50)  # Track recent performance
            
            print("✓ Bayesian training optimization enabled")
        else:
            self.bayesian_training_optimizer = None
            print("✗ Bayesian training optimization disabled")
        
        # Initialize dataset.txt for quantization
        self.init_dataset_file()
        
        # Load existing model if available
        self.load_latest_model()
        # Runtime inference flags
        self.use_rknn_inference = False
        self.rknn_runtime = None
        print(f"[TrainerInit] Using trainer file: {__file__} expected_proprio={3 + self.extra_proprio}")

    def add_experience(self,
                      bev_image: np.ndarray,
                      proprioceptive: np.ndarray,
                      action: np.ndarray,
                      reward: float,
                      next_bev_image: np.ndarray = None,
                      done: bool = False,
                      collision: bool = False,
                      in_recovery: bool = False):
        """Add experience to prioritized replay buffer.
        Optional flags 'collision' and 'in_recovery' increase sampling priority.
        """
        # Preprocess BEV image to (C,H,W)
        # Always preprocess to ensure correct (C,H,W) format
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
        self.bev_store[i] = processed.astype(np.float32)
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
        # Priority calculation
        p = abs(reward) + self.priority_epsilon
        if collision:
            p += self.collision_priority_bonus
        if in_recovery:
            p += self.recovery_priority_bonus
        self.priorities[i] = p
        # Advance pointer / size
        self.insert_ptr = (i + 1) % self.buffer_capacity
        if self.buffer_size < self.buffer_capacity:
            self.buffer_size += 1
        self.reward_history.append(reward)
        
        # Collect samples for RKNN quantization dataset
        if (self.dataset_samples_collected < self.target_dataset_samples and 
            self.training_step % self.dataset_collection_interval == 0):
            self.save_quantization_sample(bev_image, proprioceptive)

    def calculate_reward(self, 
                        action: np.ndarray,
                        collision: bool,
                        progress: float,
                        exploration_bonus: float,
                        position: Optional[np.ndarray] = None,
                        bev_data: Optional[np.ndarray] = None,
                        wheel_velocities: Optional[Tuple[float, float]] = None) -> float:
        """Calculate reward for reinforcement learning"""
        
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
            if self.training_step % 100 == 0:
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
        
    def train_step(self) -> Dict[str, float]:
        """Perform one prioritized replay training step if enough data available"""
        # Adaptive batch size update
        if self.training_step < self.warmup_batch_steps:
            target_batch = self.max_warmup_batch
        else:
            target_batch = self.post_warmup_batch
        if self.buffer_size < self.min_buffer_for_max_batch:
            target_batch = min(target_batch, 64)
        if target_batch != self.batch_size:
            self.batch_size = target_batch
            if self.enable_debug:
                print(f"[BatchAdapt] batch_size -> {self.batch_size} (step={self.training_step} buffer={self.buffer_size})")
        if self.buffer_size < self.batch_size:
            return {"loss": 0.0, "samples": self.buffer_size}
        # Prioritized sampling
        valid = self.buffer_size
        raw = self.priorities[:valid]
        if not np.any(raw):
            probs = np.full(valid, 1.0 / valid, dtype=np.float32)
        else:
            probs = raw ** self.pr_alpha
            probs /= probs.sum()
        try:
            indices = np.random.choice(valid, self.batch_size, replace=False, p=probs)
        except ValueError:
            # Fallback uniform
            indices = np.random.choice(valid, self.batch_size, replace=False)
            probs = np.full(valid, 1.0 / valid, dtype=np.float32)
        # Importance sampling weights
        sample_probs = probs[indices]
        weights = (valid * sample_probs) ** (-self.pr_beta)
        weights /= weights.max()
        self.pr_beta = min(1.0, self.pr_beta + self.pr_beta_inc)
        # Assemble batch tensors
        # Ensure indices is a proper numpy array of integers
        indices = np.asarray(indices, dtype=np.int32)
        bev_batch = torch.from_numpy(self.bev_store[indices]).float().to(self.device)
        # Debug: check batch tensor shape
        if bev_batch.ndim != 4:
            print(f"DEBUG: Unexpected bev_batch shape: {bev_batch.shape}, indices shape: {indices.shape}")
            # Reshape if needed
            bev_batch = bev_batch.view(-1, self.bev_channels, self.bev_height, self.bev_width)
        sensor_batch = torch.from_numpy(self.proprio_store[indices]).float().to(self.device)
        action_batch = torch.from_numpy(self.action_store[indices]).float().to(self.device)
        reward_batch = torch.from_numpy(self.reward_store[indices]).float().unsqueeze(1).to(self.device)
        weights_t = torch.from_numpy(weights).float().to(self.device)
        # Forward
        predicted_actions = self.model(bev_batch, sensor_batch)
        target_actions = action_batch.clone()
        confidence_weight = torch.sigmoid(reward_batch)
        target_actions = target_actions * confidence_weight  # (B,2)
        
        # Apply action weighting to encourage forward movement and reduce spinning
        # Increase weight for forward movement actions
        forward_weight = torch.ones_like(target_actions[:, 0]) * self.forward_movement_weight
        # Reduce weight for angular movement actions
        angular_weight = torch.ones_like(target_actions[:, 1]) * self.angular_movement_penalty
        
        # Apply weights to target actions
        target_actions[:, 0] = target_actions[:, 0] * forward_weight
        target_actions[:, 1] = target_actions[:, 1] * angular_weight
        
        # Per-sample losses
        action_diff = predicted_actions[:, :2] - target_actions
        per_sample_action_loss = (action_diff ** 2).mean(dim=1)
        conf_target = torch.sigmoid(reward_batch).squeeze(1)
        conf_diff = predicted_actions[:, 2] - conf_target
        per_sample_conf_loss = conf_diff.pow(2)
        per_sample_loss = per_sample_action_loss + 0.1 * per_sample_conf_loss
        loss = (weights_t * per_sample_loss).mean()
        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        # Update priorities (blend for stability)
        new_p = per_sample_loss.detach().cpu().numpy() + self.priority_epsilon
        old_p = self.priorities[indices]
        self.priorities[indices] = 0.9 * old_p + 0.1 * new_p
        # Tracking
        self.loss_history.append(float(loss.item()))
        self.training_step += 1
        # Checkpoint logic (unchanged except for buffer reference)
        if self.step_save_interval > 0 and (self.training_step % self.step_save_interval == 0):
            self.save_model()
            self.last_save_time = time.time()
        current_time = time.time()
        if current_time - self.last_save_time > self.save_interval:
            self.save_model()
            self.last_save_time = current_time
        force_due_interval = (current_time - self.last_rknn_conversion) > self.rknn_conversion_interval
        can_check_improvement = (current_time - self.last_rknn_conversion) > self.rknn_min_interval
        median_loss = float(np.median(self.loss_history)) if self.loss_history else None
        avg_reward = float(np.mean(self.reward_history)) if self.reward_history else 0.0
        improved = False
        if can_check_improvement and median_loss is not None:
            loss_improved = (self.last_export_loss is None) or (median_loss < self.last_export_loss * self.rknn_loss_improve_ratio)
            reward_improved = (self.last_export_avg_reward is None) or (avg_reward > self.last_export_avg_reward + self.rknn_reward_improve_delta)
            if loss_improved or reward_improved:
                improved = True
        if (improved or force_due_interval) and median_loss is not None:
            try:
                self.convert_to_rknn()
                self.last_rknn_conversion = current_time
                self.last_export_loss = median_loss
                self.last_export_avg_reward = avg_reward
                if self.enable_debug:
                    print(f"RKNN export triggered (improved={improved}, force={force_due_interval}) median_loss={median_loss:.4f} avg_reward={avg_reward:.2f}")
            except Exception as e:
                print(f"RKNN export attempt failed: {e}")
        
        # Bayesian training optimization
        if self.bayesian_training_optimizer is not None:
            # Calculate fitness score for current training parameters
            fitness_score = self._calculate_training_fitness(loss.item(), avg_reward, median_loss)
            self.training_fitness_history.append(fitness_score)
            
            # Check if it's time to optimize training parameters
            if (self.training_step - self.last_training_optimization_step >= self.training_optimization_interval and 
                len(self.training_fitness_history) >= 10):
                
                # Update Bayesian optimizer with recent performance
                recent_fitness = np.mean(list(self.training_fitness_history)[-10:])
                self.bayesian_training_optimizer.update_fitness(self.current_training_params, recent_fitness)
                
                # Get new training parameter suggestions
                new_training_params = self.bayesian_training_optimizer.suggest_next_params()
                self._apply_training_params(new_training_params)
                
                self.current_training_params = new_training_params
                self.last_training_optimization_step = self.training_step
                
                if self.enable_debug:
                    print(f"[BayesianTraining] Applied new training parameters at step {self.training_step}")
                    print(f"[BayesianTraining] LR: {new_training_params.learning_rate:.6f}, "
                          f"Batch: {new_training_params.batch_size}, "
                          f"Dropout: {new_training_params.dropout_rate:.3f}")
        
        return {
            "loss": loss.item(),
            "action_loss": float(per_sample_action_loss.mean().item()),
            "confidence_loss": float(per_sample_conf_loss.mean().item()),
            "samples": self.buffer_size,
            "training_steps": self.training_step,
            "avg_reward": avg_reward,
            "median_loss": median_loss if median_loss is not None else 0.0
        }
        
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
                
            return bev_processed.astype(np.float32)
        except Exception:
            # Fallback zero tensor
            return np.zeros((self.bev_channels, self.bev_height, self.bev_width), dtype=np.float32)
    
    def preprocess_bev_for_storage(self, bev_image: np.ndarray) -> np.ndarray:
        """Preprocess BEV image for storage in replay buffer."""
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
                
            return bev_processed.astype(np.float32)
        except Exception:
            return np.zeros((self.bev_channels, self.bev_height, self.bev_width), dtype=np.float32)
        
    def enable_rknn_inference(self):
        """Enable RKNN runtime inference if model file exists."""
        if not RKNN_AVAILABLE:
            if self.enable_debug:
                print("[RKNN] Toolkit not available - cannot enable RKNN inference")
            return False
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
            # Debug: print tensor shapes
            if processed.ndim != 3:
                print(f"DEBUG: Unexpected processed shape: {processed.shape}")
                processed = np.zeros((self.bev_channels, self.bev_height, self.bev_width), dtype=np.float32)
            bev_tensor = torch.from_numpy(processed).unsqueeze(0).to(self.device)
            # Ensure tensor has correct 4D shape [N, C, H, W]
            if bev_tensor.ndim != 4:
                print(f"DEBUG: Unexpected tensor shape: {bev_tensor.shape}")
                bev_tensor = bev_tensor.view(1, self.bev_channels, self.bev_height, self.bev_width)
            sensor_tensor = torch.FloatTensor(proprioceptive).unsqueeze(0).to(self.device)
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
        
    def save_model(self):
        """Save PyTorch model and training state"""
        try:
            # Ensure model directory exists
            os.makedirs(self.model_dir, exist_ok=True)
            
            timestamp = int(time.time())
            
            # Save PyTorch model
            model_path = os.path.join(self.model_dir, f"exploration_model_bev_{timestamp}.pth")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_step': self.training_step,
                'buffer_size': self.buffer_size,
                'bev_channels': self.bev_channels
            }, model_path)
            
            # Save latest symlink
            latest_path = os.path.join(self.model_dir, "exploration_model_bev_latest.pth")
            if os.path.exists(latest_path):
                os.remove(latest_path)
            os.symlink(os.path.basename(model_path), latest_path)
            
            print(f"Model saved: {model_path}")
        except Exception as e:
            print(f"Failed to save model: {e}")
        
    def load_latest_model(self):
        """Load the latest saved model with partial transplant if proprio feature size expanded."""
        try:
            latest_path = os.path.join(self.model_dir, "exploration_model_bev_latest.pth")
            if os.path.exists(latest_path) and os.path.islink(latest_path):
                actual_model_path = os.path.join(self.model_dir, os.readlink(latest_path))
            elif os.path.exists(latest_path):
                actual_model_path = latest_path
            else:
                print("No saved model found, starting with fresh model")
                return
            if not os.path.exists(actual_model_path):
                print(f"Model file {actual_model_path} not found")
                return
            checkpoint = torch.load(actual_model_path, map_location=self.device)
            state_dict = checkpoint['model_state_dict']
            # Handle possible sensor_fc input expansion
            sensor_key_w = 'sensor_fc.0.weight'
            sensor_key_b = 'sensor_fc.0.bias'
            transplanted = False
            if sensor_key_w in state_dict:
                old_w = state_dict[sensor_key_w]
                new_w = self.model.state_dict()[sensor_key_w]
                if old_w.shape != new_w.shape:
                    # Transplant overlapping columns
                    cols = min(old_w.shape[1], new_w.shape[1])
                    new_w[:, :cols] = old_w[:, :cols]
                    if cols < new_w.shape[1]:
                        # Initialize new feature columns with small noise
                        with torch.no_grad():
                            new_w[:, cols:] = 0.01 * torch.randn_like(new_w[:, cols:])
                    state_dict[sensor_key_w] = new_w
                    # Bias shape should match; if not, skip (unlikely)
                    transplanted = True
            # Load model with strict=False to allow missing/extra keys (e.g., optimizer-specific buffers)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if transplanted:
                print(f"Partial transplant: adjusted sensor_fc input from old feature size to {self.model.sensor_fc[0].in_features}")
            if missing:
                if self.enable_debug:
                    print(f"[Load] Missing keys: {missing}")
            if unexpected:
                if self.enable_debug:
                    print(f"[Load] Unexpected keys: {unexpected}")
            # Optimizer: only load if shapes unchanged
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"Optimizer state load skipped (shape mismatch): {e}")
            self.training_step = checkpoint.get('training_step', 0)
            print(f"Loaded model from {actual_model_path}")
            print(f"Training steps: {self.training_step}")
        except Exception as e:
            print(f"Failed to load model (partial load logic): {e}")
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
                onnx_path = os.path.join(self.model_dir, "exploration_model_bev.onnx")
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
            onnx_path = os.path.join(self.model_dir, "exploration_model_bev.onnx")
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
            rknn_path = os.path.join(self.model_dir, "exploration_model_bev.rknn")
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
            "training_steps": self.training_step,
            "buffer_size": self.buffer_size,
            "avg_reward": np.mean(self.reward_history) if self.reward_history else 0.0,
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

    def _calculate_training_fitness(self, loss: float, avg_reward: float, median_loss: Optional[float]) -> float:
        """Calculate a fitness score for the current training parameters based on performance metrics"""
        
        # Normalize components to similar scales
        # Lower loss is better (invert)
        loss_component = 1.0 / (1.0 + loss)  # Range: [0, 1], higher is better
        
        # Higher average reward is better  
        reward_component = max(0.0, avg_reward / 50.0)  # Normalize assuming max reward ~50
        
        # Lower median loss is better (if available)
        if median_loss is not None:
            median_loss_component = 1.0 / (1.0 + median_loss)
        else:
            median_loss_component = loss_component
        
        # Weighted combination
        fitness = (0.4 * loss_component + 
                  0.4 * reward_component + 
                  0.2 * median_loss_component)
        
        return fitness

    def _apply_training_params(self, params: TrainingHyperparameters):
        """Apply optimized training parameters to the model and training process"""
        
        try:
            # Update optimizer learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = params.learning_rate
                param_group['weight_decay'] = params.weight_decay
            
            # Update batch size (will take effect on next training call)
            if params.batch_size != self.batch_size:
                self.batch_size = params.batch_size
                if self.enable_debug:
                    print(f"[BayesianTraining] Updated batch size to {self.batch_size}")
            
            # Update prioritized replay parameters
            self.pr_alpha = params.prioritized_replay_alpha
            self.pr_beta = params.prioritized_replay_beta
            
            # Update reward clipping (will affect future reward calculations)
            # Note: This would require integration with the reward system
            
            # Update neural network dropout rate (requires model modification)
            # This would need to be applied when creating a new model or during forward pass
            
            if self.enable_debug:
                print(f"[BayesianTraining] Applied training parameters:")
                print(f"  Learning rate: {params.learning_rate:.6f}")
                print(f"  Batch size: {params.batch_size}")  
                print(f"  PR alpha: {params.prioritized_replay_alpha:.3f}")
                print(f"  PR beta: {params.prioritized_replay_beta:.3f}")
                print(f"  Weight decay: {params.weight_decay:.6f}")
                
        except Exception as e:
            if self.enable_debug:
                print(f"[BayesianTraining] Failed to apply training parameters: {e}")

    def get_training_optimization_stats(self) -> Dict[str, float]:
        """Get statistics about the Bayesian training optimization process"""
        
        if self.bayesian_training_optimizer is None:
            return {"bayesian_training_optimization": "disabled"}
        
        stats = self.bayesian_training_optimizer.get_optimization_stats()
        best_params, best_fitness = self.bayesian_training_optimizer.get_best_params()
        
        return {
            **stats,
            "best_learning_rate": best_params.learning_rate,
            "best_batch_size": float(best_params.batch_size),
            "best_dropout_rate": best_params.dropout_rate,
            "steps_since_optimization": self.training_step - self.last_training_optimization_step,
            "current_fitness": self.training_fitness_history[-1] if self.training_fitness_history else 0.0
        }

    def safe_save(self):
        try:
            self.save_model()
        except Exception as e:
            print(f"Model save failed during shutdown: {e}")
