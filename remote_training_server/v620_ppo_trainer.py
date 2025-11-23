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
        self.depth = torch.zeros((capacity, *depth_shape), dtype=torch.float16)
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

    def get_batch(self, batch_size: int):
        """Sample a random batch for training."""
        indices = torch.randint(0, self.size, (batch_size,))
        
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

class V620PPOTrainer:
    """PPO Trainer optimized for V620 ROCm."""
    
    def __init__(self, args):
        self.args = args
        
        # Device setup
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
            # Enable TF32 for Ampere/RDNA3
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            self.device = torch.device('cpu')
            print("⚠ Using CPU (slow)")
            
        # Model setup
        self.rgb_shape = (3, 240, 424)  # C, H, W
        self.depth_shape = (240, 424)   # H, W
        self.proprio_dim = 9
        
        self.encoder = RGBDEncoder().to(self.device)
        self.policy_head = PolicyHead(self.encoder.output_dim, self.proprio_dim).to(self.device)
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
        
        # Mixed Precision Scaler
        self.scaler = torch.cuda.amp.GradScaler()
        
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
            if self.buffer.size > self.args.batch_size:
                self.is_training = True
                # Run training step
                with self.training_lock:
                    metrics = self.train_step()
                self.is_training = False
                
                if metrics:
                    self.update_count += 1
                    
                    # Log metrics
                    if self.update_count % 10 == 0:
                        print(f"Step {self.total_steps} | Loss: P={metrics['policy_loss']:.3f} V={metrics['value_loss']:.3f} E={metrics['entropy']:.3f}")
                        for k, v in metrics.items():
                            self.writer.add_scalar(f'train/{k}', v, self.total_steps)
                            
                    # Save checkpoint
                    if self.update_count % 100 == 0:
                        self.save_checkpoint(f"ppo_step_{self.total_steps}.pt")
            
            time.sleep(0.1)

    def train_step(self):
        """Perform one PPO update step."""
        metrics = {'policy_loss': [], 'value_loss': [], 'entropy': []}
        
        # Use tqdm for progress bar
        pbar = tqdm(range(self.args.update_epochs), desc="Training", leave=False, file=sys.stdout)
        for _ in pbar:
            try:
                batch = self.buffer.get_batch(self.args.batch_size)
                
                # Mixed Precision Context
                with torch.amp.autocast('cuda'):
                    # Forward pass
                    features = self.encoder(batch['rgb'], batch['depth'])
                    action_mean, _ = self.policy_head(features, batch['proprio'], hidden_state=None) # No LSTM training for now
                    values = self.value_head(features, batch['proprio'])
                    
                    # Action distribution
                    std = self.log_std.exp()
                    dist = torch.distributions.Normal(action_mean, std)
                    
                    # Compute log probs
                    current_log_probs = dist.log_prob(batch['actions']).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()
                    
                    # Ratios
                    ratios = torch.exp(current_log_probs - batch['log_probs'])
                    
                    # Advantages
                    advantages = batch['rewards'] - values.detach()
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    
                    # Policy Loss
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1 - self.args.clip_eps, 1 + self.args.clip_eps) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value Loss
                    value_loss = 0.5 * (values - batch['rewards']).pow(2).mean()
                    
                    # Total Loss
                    loss = policy_loss + self.args.value_coef * value_loss - self.args.entropy_coef * entropy
                
                # Update with Scaler
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Unscale before clipping
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.policy_head.parameters(), 0.5)
                nn.utils.clip_grad_norm_(self.value_head.parameters(), 0.5)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                metrics['policy_loss'].append(policy_loss.item())
                metrics['value_loss'].append(value_loss.item())
                metrics['entropy'].append(entropy.item())
                
                # Update progress bar description
                pbar.set_postfix({
                    'ploss': f"{policy_loss.item():.3f}",
                    'vloss': f"{value_loss.item():.3f}"
                })
            except Exception as e:
                print(f"\n❌ Error in training step: {e}")
                import traceback
                traceback.print_exc()
                break
            
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
        
        # Export to ONNX for Rover (RKNN)
        try:
            onnx_path = os.path.join(self.args.checkpoint_dir, "latest_actor.onnx")
            
            # Create dummy inputs matching RKNN expectation
            dummy_rgb = torch.randn(1, 3, 240, 424, device=self.device)
            dummy_depth = torch.randn(1, 1, 240, 424, device=self.device)
            dummy_proprio = torch.randn(1, self.proprio_dim, device=self.device)
            dummy_lstm_h = torch.zeros(1, 1, 128, device=self.device)
            dummy_lstm_c = torch.zeros(1, 1, 128, device=self.device)
            
            # Wrap model to handle forward pass signature
            class ActorWrapper(nn.Module):
                def __init__(self, encoder, policy):
                    super().__init__()
                    self.encoder = encoder
                    self.policy = policy
                    
                def forward(self, rgb, depth, proprio, lstm_h, lstm_c):
                    features = self.encoder(rgb, depth)
                    # Pass LSTM state if policy uses it, otherwise ignore
                    # Note: PolicyHead handles hidden_state tuple
                    action, (new_h, new_c) = self.policy(features, proprio, (lstm_h, lstm_c))
                    return action, new_h, new_c

            actor = ActorWrapper(self.encoder, self.policy_head)
            actor.eval()
            
            torch.onnx.export(
                actor,
                (dummy_rgb, dummy_depth, dummy_proprio, dummy_lstm_h, dummy_lstm_c),
                onnx_path,
                opset_version=12,
                input_names=['rgb', 'depth', 'proprio', 'lstm_h', 'lstm_c'],
                output_names=['action', 'lstm_h_out', 'lstm_c_out']
            )
            print(f"📦 Exported ONNX: {onnx_path}")
            self.model_version += 1  # Signal that a new model is ready
            
        except Exception as e:
            print(f"❌ ONNX Export failed: {e}")

    def run(self):
        """Main training loop."""
        print(f"🚀 PPO Training Server started on port {self.args.port}")
        print(f"   Buffer Size: {self.args.buffer_size}")
        print(f"   Batch Size: {self.args.batch_size}")
        
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
                    print(f"📥 Received batch: {batch_len} steps (Rover active, {dt:.1f}s since last)")

                    # Add data to buffer (thread-safe)
                    with self.training_lock:
                        self.buffer.add_batch(message['data'])
                    
                    self.total_steps += batch_len
                    
                    # Update curriculum
                    self.update_curriculum()
                    
                    # Training is handled by background thread now
                    
                    # Send back curriculum info and latest model version
                    response['curriculum'] = {
                        'collision_dist': self.collision_dist,
                        'max_speed': self.max_speed
                    }
                    response['model_version'] = self.model_version
                    
                    # Tell rover to wait if we are about to train (or are training)
                    if self.buffer.size > self.args.batch_size:
                        response['wait_for_training'] = True
                    
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
    parser.add_argument('--batch_size', type=int, default=1536)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--update_epochs', type=int, default=5)
    parser.add_argument('--clip_eps', type=float, default=0.2)
    parser.add_argument('--value_coef', type=float, default=0.5)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_ppo')
    parser.add_argument('--log_dir', type=str, default='./logs_ppo')
    
    args = parser.parse_args()
    
    trainer = V620PPOTrainer(args)
    trainer.run()
