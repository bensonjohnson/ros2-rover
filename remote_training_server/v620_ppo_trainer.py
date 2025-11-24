#!/usr/bin/env python3
"""V620 ROCm PPO Training Server for Remote Rover Training.

This server receives RGB-D observations from the rover via ZeroMQ,
trains a PPO policy using PyTorch with ROCm acceleration,
and exports trained models in multiple formats (PyTorch, ONNX, RKNN).

Features:
- Asynchronous training (trains while receiving data)
- Experience Replay Buffer (off-policy correction)
- Curriculum Learning (adaptive difficulty)
- Dense Reward Shaping (smoothness, oscillation penalties)
- Multi-client support
"""

import os
import sys
import time
import json
import argparse
import threading
import queue
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from collections import deque

import numpy as np
import cv2
import zmq
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import model architectures
from model_architectures import RGBDEncoder, PolicyHead, ValueHead

class PPOBuffer:
    """Experience buffer for PPO training."""
    
    def __init__(self, capacity: int, rgb_shape: Tuple, depth_shape: Tuple, proprio_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate tensors on CPU (move to GPU in batches during training)
        self.rgb = torch.zeros((capacity, *rgb_shape), dtype=torch.uint8)
        self.depth = torch.zeros((capacity, *depth_shape), dtype=torch.float32)  # Changed from float16 for stability
        self.proprio = torch.zeros((capacity, proprio_dim), dtype=torch.float32)
        self.actions = torch.zeros((capacity, 2), dtype=torch.float32)
        self.rewards = torch.zeros((capacity,), dtype=torch.float32)
        self.dones = torch.zeros((capacity,), dtype=torch.bool)
        self.log_probs = torch.zeros((capacity,), dtype=torch.float32)
        self.values = torch.zeros((capacity,), dtype=torch.float32)
        
    def add_batch(self, batch_data: Dict):
        """Add a batch of transitions from rover."""
        batch_size = len(batch_data['rewards'])
        
        # Handle wrap-around
        if self.ptr + batch_size > self.capacity:
            # Split into two parts
            first_part = self.capacity - self.ptr
            second_part = batch_size - first_part
            
            # Add first part
            self._add_slice(batch_data, 0, first_part, self.ptr)
            # Add second part
            self._add_slice(batch_data, first_part, batch_size, 0)
            
            self.ptr = second_part
            self.size = self.capacity
        else:
            # Add all at once
            self._add_slice(batch_data, 0, batch_size, self.ptr)
            self.ptr += batch_size
            self.size = min(self.size + batch_size, self.capacity)
            
    def _add_slice(self, data, start_idx, end_idx, buffer_idx):
        """Helper to add a slice of data to buffer."""
        length = end_idx - start_idx
        
        # Convert numpy arrays to tensors
        self.rgb[buffer_idx:buffer_idx+length] = torch.from_numpy(data['rgb'][start_idx:end_idx])
        self.depth[buffer_idx:buffer_idx+length] = torch.from_numpy(data['depth'][start_idx:end_idx])
        self.proprio[buffer_idx:buffer_idx+length] = torch.from_numpy(data['proprio'][start_idx:end_idx])
        self.actions[buffer_idx:buffer_idx+length] = torch.from_numpy(data['actions'][start_idx:end_idx])
        self.rewards[buffer_idx:buffer_idx+length] = torch.from_numpy(data['rewards'][start_idx:end_idx])
        self.dones[buffer_idx:buffer_idx+length] = torch.from_numpy(data['dones'][start_idx:end_idx])
        self.log_probs[buffer_idx:buffer_idx+length] = torch.from_numpy(data['log_probs'][start_idx:end_idx])
        self.values[buffer_idx:buffer_idx+length] = torch.from_numpy(data['values'][start_idx:end_idx])

    def get_batch(self, indices):
        """Get batch by indices."""
        # RGB needs to be permuted from (B, H, W, C) to (B, C, H, W) for PyTorch Conv2d
        rgb_batch = self.rgb[indices].to(self.device).float() / 255.0
        rgb_batch = rgb_batch.permute(0, 3, 1, 2)
        
        return {
            'rgb': rgb_batch,
            'depth': self.depth[indices].to(self.device).float().unsqueeze(1),
            'proprio': self.proprio[indices].to(self.device),
            'actions': self.actions[indices].to(self.device),
            'rewards': self.rewards[indices].to(self.device),
            'dones': self.dones[indices].to(self.device),
            'log_probs': self.log_probs[indices].to(self.device),
            'values': self.values[indices].to(self.device),
        }

    def clear(self):
        """Clear buffer after update (PPO is on-policy)."""
        self.ptr = 0
        self.size = 0

class V620PPOTrainer:
    """PPO Trainer optimized for V620 ROCm."""
    
    def __init__(self, args):
        self.args = args
        
        # Device setup
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
            # Enable TF32 for Ampere/RDNA3
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*TF32.*")
                try:
                    torch.set_float32_matmul_precision('high')
                except AttributeError:
                    # Fallback for older PyTorch versions
                    torch.backends.cuda.matmul.allow_tf32 = True
                
                try:
                    torch.backends.cudnn.allow_tf32 = True
                except AttributeError:
                    pass
        else:
            self.device = torch.device('cpu')
            print("⚠ Using CPU (slow)")
            
        # Model setup
        self.rgb_shape = (3, 240, 424)  # C, H, W
        self.depth_shape = (240, 424)   # H, W
        self.proprio_dim = 9
        
        self.encoder = RGBDEncoder().to(self.device)
        # Disable LSTM for stateless PPO (fixes MIOpen backward error and ONNX export)
        self.policy_head = PolicyHead(self.encoder.output_dim, self.proprio_dim, use_lstm=False).to(self.device)
        self.value_head = ValueHead(self.encoder.output_dim, self.proprio_dim).to(self.device)
        
        # Trainable log_std for continuous action space
        self.log_std = nn.Parameter(torch.zeros(2, device=self.device))
        
        # Optimizer
        self.optimizer = optim.Adam([
            {'params': self.encoder.parameters(), 'lr': args.lr},
            {'params': self.policy_head.parameters(), 'lr': args.lr},
            {'params': self.value_head.parameters(), 'lr': args.lr},
            {'params': self.log_std, 'lr': args.lr}
        ])

        # DISABLED: Mixed Precision causes NaN on ROCm (FP16 instability)
        # self.scaler = torch.amp.GradScaler('cuda')
        
        # Replay Buffer
        self.buffer = PPOBuffer(
            capacity=args.buffer_size,
            rgb_shape=(240, 424, 3), # Stored as HWC uint8
            depth_shape=(240, 424),
            proprio_dim=self.proprio_dim,
            device=self.device
        )
        
        # Training state
        self.total_steps = 0
        self.update_count = 0
        self.best_reward = -float('inf')
        self.model_version = -1  # Track model updates (start at -1 until first save)
        self.is_training = False

        # KL warm-up: start with higher target_kl for first few updates (cold start)
        self.kl_warmup_updates = 10  # Warm up for first 10 updates
        self.kl_warmup_start = 0.1   # Start with 10x higher tolerance
        self.kl_warmup_end = args.target_kl  # End at configured target
        
        # ZMQ Setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{args.port}")
        
        # TensorBoard
        self.writer = SummaryWriter(args.log_dir)
        
        # Checkpoint dir
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        
        # Curriculum state
        self.curriculum_level = 0
        self.collision_dist = 0.5  # Start safe
        self.max_speed = 0.1       # Start slow
        
        # Connection tracking
        self.last_data_time = time.time()
        
        # Training Thread
        self.training_lock = threading.Lock()
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()

    def get_target_kl(self):
        """Get current target KL with warm-up schedule."""
        if self.update_count < self.kl_warmup_updates:
            # Linear interpolation from warmup_start to warmup_end
            alpha = self.update_count / self.kl_warmup_updates
            return self.kl_warmup_start * (1 - alpha) + self.kl_warmup_end * alpha
        else:
            return self.kl_warmup_end

    def update_curriculum(self):
        """Update difficulty based on performance."""
        # Simple curriculum: every 100k steps, make it harder
        level = self.total_steps // 100000
        
        if level > self.curriculum_level:
            self.curriculum_level = level
            
            # Decrease collision distance (get closer to objects)
            self.collision_dist = max(0.12, 0.5 - (level * 0.05))
            
            # Increase max speed
            self.max_speed = min(0.18, 0.1 + (level * 0.02))
            
            print(f"🎓 Curriculum Level Up! Level {level}")
            print(f"   Collision Dist: {self.collision_dist:.2f}m")
            print(f"   Max Speed: {self.max_speed:.2f}m/s")

    def _training_loop(self):
        """Background training loop."""
        print("🧵 Training thread started")
        while True:
            # Wait for enough data (rollout_steps)
            if self.buffer.size >= self.args.rollout_steps:
                self.is_training = True
                # Run training step
                with self.training_lock:
                    metrics = self.train_step()

                if metrics:
                    self.update_count += 1

                    # Log metrics
                    if self.update_count % 10 == 0:
                        current_target_kl = self.get_target_kl()
                        warmup_str = f" [KL target: {current_target_kl:.4f}]" if self.update_count < self.kl_warmup_updates else ""
                        print(f"Step {self.total_steps} | Loss: P={metrics['policy_loss']:.3f} V={metrics['value_loss']:.3f} | KL={metrics['approx_kl']:.4f} Clip={metrics['clip_fraction']:.2f}{warmup_str}")
                        for k, v in metrics.items():
                            self.writer.add_scalar(f'train/{k}', v, self.total_steps)
                        self.writer.add_scalar(f'train/target_kl', current_target_kl, self.total_steps)

                    # Save checkpoint every batch (as requested)
                    self.save_checkpoint(f"ppo_step_{self.total_steps}.pt")

                    # Always export ONNX for immediate rover update
                    self.export_onnx()

                    # Clear buffer after update (PPO is on-policy)
                    self.buffer.clear()
                else:
                    # Training failed due to NaN - clear buffer but don't save model
                    print("⚠️  Skipping checkpoint save due to failed training update")
                    self.buffer.clear()  # Still clear buffer to avoid reusing bad data
                
                self.is_training = False
            
            time.sleep(0.1)

    def train_step(self):
        """Perform PPO update with mini-batches and early stopping."""
        metrics = {'policy_loss': [], 'value_loss': [], 'entropy': [], 'approx_kl': [], 'clip_fraction': []}

        # Flag to track if NaN occurred during training
        nan_detected = False

        # Optimization: Move entire active buffer to GPU once
        print("  ⚡ Moving buffer to GPU...")
        with torch.no_grad():
            # Slice valid data
            valid_slice = slice(0, self.buffer.size)
            
            # RGB: (N, H, W, C) -> (N, C, H, W) normalized
            gpu_rgb = self.buffer.rgb[valid_slice].to(self.device).float() / 255.0
            gpu_rgb = gpu_rgb.permute(0, 3, 1, 2)
            
            # Depth: (N, H, W) -> (N, 1, H, W)
            gpu_depth = self.buffer.depth[valid_slice].to(self.device).float().unsqueeze(1)
            
            gpu_proprio = self.buffer.proprio[valid_slice].to(self.device)
            gpu_actions = self.buffer.actions[valid_slice].to(self.device)
            gpu_rewards = self.buffer.rewards[valid_slice].to(self.device)
            gpu_log_probs = self.buffer.log_probs[valid_slice].to(self.device)
        
        # Use tqdm for progress bar
        pbar = tqdm(range(self.args.update_epochs), desc="Training", leave=False, file=sys.stdout)

        early_stop = False
        for epoch in pbar:
            # Shuffle data for each epoch
            indices = torch.randperm(self.buffer.size, device=self.device)
            
            # Iterate in mini-batches
            for start_idx in range(0, self.buffer.size, self.args.mini_batch_size):
                try:
                    batch_indices = indices[start_idx : start_idx + self.args.mini_batch_size]
                    
                    # Slice from GPU tensors
                    batch = {
                        'rgb': gpu_rgb[batch_indices],
                        'depth': gpu_depth[batch_indices],
                        'proprio': gpu_proprio[batch_indices],
                        'actions': gpu_actions[batch_indices],
                        'rewards': gpu_rewards[batch_indices],
                        'log_probs': gpu_log_probs[batch_indices]
                    }
                    
                    # Forward pass (FP32 for stability on ROCm)
                    features = self.encoder(batch['rgb'], batch['depth'])
                    action_mean, _ = self.policy_head(features, batch['proprio'], hidden_state=None) # No LSTM training for now
                    values = self.value_head(features, batch['proprio'])

                    # Action distribution
                    std = self.log_std.exp().clamp(min=1e-6, max=2.0)  # Clamp std to prevent NaN

                    # Check for NaN in forward pass outputs
                    if torch.isnan(action_mean).any() or torch.isnan(values).any():
                        print(f"⚠️  NaN detected in forward pass! Aborting training update.")
                        nan_detected = True
                        break

                    dist = torch.distributions.Normal(action_mean, std)

                    # Compute log probs
                    current_log_probs = dist.log_prob(batch['actions']).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()

                    # Ratios
                    ratios = torch.exp(current_log_probs - batch['log_probs'])

                    # Approximate KL divergence for early stopping
                    with torch.no_grad():
                        log_ratio = current_log_probs - batch['log_probs']
                        approx_kl = ((ratios - 1) - log_ratio).mean()

                        # Clip fraction (percentage of samples being clipped)
                        clip_fraction = ((ratios - 1.0).abs() > self.args.clip_eps).float().mean()

                    # Advantages (with more stable normalization)
                    advantages = batch['rewards'] - values.squeeze().detach()
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

                    # Policy Loss
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1 - self.args.clip_eps, 1 + self.args.clip_eps) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value Loss (simple MSE with gradient scaling to prevent divergence)
                    values_pred = values.squeeze()
                    value_loss = 0.5 * (values_pred - batch['rewards']).pow(2).mean()

                    # Detach and rescale if loss is too high (soft prevention)
                    if value_loss.item() > 100.0:
                        value_loss = value_loss * 0.1  # Scale down contribution

                    # Safety check: abort if value loss explodes
                    if value_loss > 1000.0:
                        print(f"⚠️  Value loss too high ({value_loss:.1f})! Aborting training update.")
                        nan_detected = True
                        break

                    # Total Loss
                    loss = policy_loss + self.args.value_coef * value_loss - self.args.entropy_coef * entropy

                    # Check for NaN in loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"⚠️  NaN/Inf detected in loss! Aborting training update.")
                        nan_detected = True
                        break

                    # Standard backprop (no mixed precision)
                    self.optimizer.zero_grad()
                    loss.backward()

                    # Check for NaN in gradients
                    has_nan_grad = False
                    for name, param in self.encoder.named_parameters():
                        if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                            has_nan_grad = True
                            break
                    if not has_nan_grad:
                        for name, param in self.policy_head.named_parameters():
                            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                                has_nan_grad = True
                                break

                    if has_nan_grad:
                        print(f"⚠️  NaN detected in gradients! Aborting training update.")
                        nan_detected = True
                        break

                    # Gradient clipping (including log_std)
                    nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                    nn.utils.clip_grad_norm_(self.policy_head.parameters(), 1.0)
                    nn.utils.clip_grad_norm_(self.value_head.parameters(), 1.0)
                    nn.utils.clip_grad_norm_([self.log_std], 0.5)

                    self.optimizer.step()

                    metrics['policy_loss'].append(policy_loss.item())
                    metrics['value_loss'].append(value_loss.item())
                    metrics['entropy'].append(entropy.item())
                    metrics['approx_kl'].append(approx_kl.item())
                    metrics['clip_fraction'].append(clip_fraction.item())
                    
                except Exception as e:
                    print(f"\n❌ Error in training step: {e}")
                    import traceback
                    traceback.print_exc()
                    nan_detected = True
                    break

            # If NaN detected, abort entire training update
            if nan_detected:
                print(f"❌ NaN detected during training! Skipping this update entirely.")
                break

            # Update progress bar description (avg of current epoch)
            if metrics['policy_loss']:
                avg_kl = np.mean(metrics['approx_kl'][-10:]) if metrics['approx_kl'] else 0.0
                pbar.set_postfix({
                    'ploss': f"{np.mean(metrics['policy_loss'][-10:]):.3f}",
                    'vloss': f"{np.mean(metrics['value_loss'][-10:]):.3f}",
                    'kl': f"{avg_kl:.4f}"
                })

                # Early stopping: check if KL divergence exceeds target (with warm-up)
                current_target_kl = self.get_target_kl()
                if avg_kl > current_target_kl:
                    warmup_status = f" [warm-up {self.update_count}/{self.kl_warmup_updates}]" if self.update_count < self.kl_warmup_updates else ""
                    print(f"\n⚠️  Early stopping at epoch {epoch + 1}/{self.args.update_epochs}: KL divergence ({avg_kl:.4f}) exceeded target ({current_target_kl:.4f}){warmup_status}")
                    early_stop = True
                    break

        # If NaN was detected, return None to signal failure
        if nan_detected:
            return None

        # Only return metrics if we have valid data
        if not metrics['policy_loss']:
            print("⚠️  No valid training data collected!")
            return None

        return {k: np.mean(v) for k, v in metrics.items()}

    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        path = os.path.join(self.args.checkpoint_dir, filename)
        torch.save({
            'encoder': self.encoder.state_dict(),
            'policy': self.policy_head.state_dict(),
            'value': self.value_head.state_dict(),
            'log_std': self.log_std,
            'optimizer': self.optimizer.state_dict(),
            'steps': self.total_steps
        }, path)
        print(f"💾 Saved checkpoint: {path}")
        
        # Also export ONNX whenever we save a checkpoint
        self.export_onnx()

    def export_onnx(self):
        """Export current policy to ONNX for Rover."""
        try:
            onnx_path = os.path.join(self.args.checkpoint_dir, "latest_actor.onnx")
            
            # Create dummy inputs matching RKNN expectation
            dummy_rgb = torch.randn(1, 3, 240, 424, device=self.device)
            dummy_depth = torch.randn(1, 1, 240, 424, device=self.device)
            dummy_proprio = torch.randn(1, self.proprio_dim, device=self.device)
            
            # Wrap model to handle forward pass signature
            class ActorWrapper(nn.Module):
                def __init__(self, encoder, policy):
                    super().__init__()
                    self.encoder = encoder
                    self.policy = policy
                    
                def forward(self, rgb, depth, proprio):
                    features = self.encoder(rgb, depth)
                    # Pass None for hidden state to bypass LSTM
                    action, _ = self.policy(features, proprio, None)
                    return action

            actor = ActorWrapper(self.encoder, self.policy_head)
            actor.eval()
            
            torch.onnx.export(
                actor,
                (dummy_rgb, dummy_depth, dummy_proprio),
                onnx_path,
                opset_version=18,  # Changed from 17 to avoid version conversion warnings
                input_names=['rgb', 'depth', 'proprio'],
                output_names=['action']
            )
            print(f"📦 Exported ONNX: {onnx_path}")
            self.model_version += 1  # Signal that a new model is ready
            
        except Exception as e:
            print(f"❌ ONNX Export failed: {e}")

    def run(self):
        """Main training loop."""
        print(f"🚀 PPO Training Server started on port {self.args.port}")
        print(f"   Buffer Size: {self.args.buffer_size}")
        print(f"   Rollout Steps: {self.args.rollout_steps}")
        print(f"   Mini-Batch Size: {self.args.mini_batch_size}")
        
        while True:
            try:
                # Wait for request
                message = self.socket.recv_pyobj()
                
                response = {'type': 'ack'}
                
                if message['type'] == 'data_batch':
                    # Log connection status
                    current_time = time.time()
                    dt = current_time - self.last_data_time
                    self.last_data_time = current_time
                    batch_len = len(message['data']['rewards'])

                    # Try to acquire lock without blocking
                    if self.training_lock.acquire(blocking=False):
                        try:
                            print(f"📥 Received batch: {batch_len} steps (Rover active, {dt:.1f}s since last)")
                            # Add data to buffer (thread-safe)
                            self.buffer.add_batch(message['data'])
                            
                            self.total_steps += batch_len
                            
                            # Update curriculum
                            self.update_curriculum()
                            
                            # Send back curriculum info and latest model version
                            response['curriculum'] = {
                                'collision_dist': self.collision_dist,
                                'max_speed': self.max_speed
                            }
                            response['model_version'] = self.model_version
                            
                            # Tell rover to wait if we are about to train
                            if self.buffer.size >= self.args.rollout_steps:
                                response['wait_for_training'] = True
                        finally:
                            self.training_lock.release()
                    else:
                        # Lock is busy -> Training in progress
                        print(f"🔒 Training in progress, discarding batch of {batch_len} steps")
                        response['wait_for_training'] = True
                        # Still send curriculum/model info just in case
                        response['curriculum'] = {
                            'collision_dist': self.collision_dist,
                            'max_speed': self.max_speed
                        }
                        response['model_version'] = self.model_version
                    
                elif message['type'] == 'check_status':
                    # Rover polling for training completion
                    response['status'] = 'training' if self.is_training else 'ready'
                    response['model_version'] = self.model_version

                elif message['type'] == 'get_model':
                    # Send latest ONNX model
                    onnx_path = os.path.join(self.args.checkpoint_dir, "latest_actor.onnx")
                    if os.path.exists(onnx_path):
                        print(f"📤 Rover requested model update")
                        with open(onnx_path, 'rb') as f:
                            model_bytes = f.read()
                        response['model_bytes'] = model_bytes
                        response['model_version'] = self.model_version
                        print(f"📤 Sent ONNX model v{self.model_version} ({len(model_bytes)} bytes)")
                    else:
                        # Silent fail if not ready yet (to avoid log spam)
                        # print("⚠ No ONNX model found yet")
                        response['error'] = 'No model available'
                
                self.socket.send_pyobj(response)
                
            except Exception as e:
                print(f"❌ Error: {e}")
                # Send error response if possible, else continue
                try:
                    self.socket.send_pyobj({'type': 'error', 'msg': str(e)})
                except:
                    pass

import io

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5556)
    parser.add_argument('--buffer_size', type=int, default=25000)
    parser.add_argument('--rollout_steps', type=int, default=2048, help="Steps to collect before training")
    parser.add_argument('--mini_batch_size', type=int, default=512, help="Mini-batch size for PPO update")
    parser.add_argument('--lr', type=float, default=1e-4)  # Lowered from 3e-4 to prevent value head divergence
    parser.add_argument('--update_epochs', type=int, default=5)
    parser.add_argument('--clip_eps', type=float, default=0.2)
    parser.add_argument('--value_coef', type=float, default=0.5)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--target_kl', type=float, default=0.015, help="Target KL divergence for early stopping")
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_ppo')
    parser.add_argument('--log_dir', type=str, default='./logs_ppo')
    
    args = parser.parse_args()
    
    trainer = V620PPOTrainer(args)
    trainer.run()
