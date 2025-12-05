#!/usr/bin/env python3
"""V620 ROCm SAC Training Server for Remote Rover Training.

This server receives RGB-D observations from the rover via NATS JetStream,
trains a SAC (Soft Actor-Critic) policy using PyTorch with ROCm acceleration,
and exports trained models in ONNX format for the rover.

Features:
- Off-policy learning (Replay Buffer)
- Entropy maximization (Exploration)
- Asynchronous training with NATS persistence
- Automatic Entropy Tuning (Alpha)
"""

import os
import sys
import time
import json
import argparse
import threading
import queue
import asyncio
import traceback
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import copy

import numpy as np
import cv2
import nats
from nats.js.api import StreamConfig, ConsumerConfig
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import model architectures
from model_architectures import OccupancyGridEncoder, GaussianPolicyHead, QNetwork

# Import serialization utilities
from serialization_utils import (
    serialize_batch, deserialize_batch,
    serialize_model_update, deserialize_model_update,
    serialize_metadata, deserialize_metadata,
    serialize_status, deserialize_status
)

# Import dashboard
from dashboard_app import TrainingDashboard

class ReplayBuffer:
    """Experience Replay Buffer for SAC."""

    def __init__(self, capacity: int, grid_shape: Tuple, proprio_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.full = False

        # Storage (CPU RAM to save GPU memory, move to GPU during sampling)
        # Grid: 64x64, uint8
        self.grid = torch.zeros((capacity, *grid_shape), dtype=torch.uint8)
        self.proprio = torch.zeros((capacity, proprio_dim), dtype=torch.float32)
        self.actions = torch.zeros((capacity, 2), dtype=torch.float32)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32)
        
    def add_batch(self, batch_data: Dict):
        """Add a batch of sequential data and construct transitions."""
        # batch_data contains lists/arrays of s, a, r, d
        # We need to construct (s, a, r, s', d)
        # Since data is sequential, s'[t] = s[t+1]

        grid = batch_data['grid']
        proprio = batch_data['proprio']
        actions = batch_data['actions']
        rewards = batch_data['rewards']
        dones = batch_data['dones']

        num_steps = len(rewards)
        print(f"DEBUG: add_batch called with {num_steps} steps")
        if num_steps < 2:
            print(f"DEBUG: Batch too small ({num_steps} < 2), ignoring.")
            return # Need at least 2 steps to form a transition

        # We can form (num_steps - 1) transitions from a single batch
        # unless we cache the last observation from the previous batch.
        # For simplicity, we'll just use the transitions within this batch.
        # This loses 1 transition per batch (negligible).

        # Current states: 0 to N-1
        # Next states: 1 to N

        # Indices for current step
        curr_slice = slice(0, num_steps - 1)
        # Indices for next step
        next_slice = slice(1, num_steps)

        # Count valid transitions
        # If done[t] is True, then s[t+1] is start of new episode, NOT next state of s[t]
        # But here we are storing (s, a, r, s', d).
        # If d[t] is True, s'[t] doesn't matter (masked by 1-d).
        # So we can just take s[t+1] as s'[t] blindly.

        batch_size = num_steps - 1

        # Handle wrap-around
        if self.ptr + batch_size > self.capacity:
            # Split
            first_part = self.capacity - self.ptr
            second_part = batch_size - first_part

            self._add_slice(grid, proprio, actions, rewards, dones, 0, first_part, self.ptr)
            self._add_slice(grid, proprio, actions, rewards, dones, first_part, second_part, 0)

            self.ptr = second_part
            self.full = True
        else:
            self._add_slice(grid, proprio, actions, rewards, dones, 0, batch_size, self.ptr)
            self.ptr += batch_size
            if self.ptr >= self.capacity:
                self.full = True
                self.ptr = self.ptr % self.capacity

        self.size = self.capacity if self.full else self.ptr

    def _add_slice(self, grid, proprio, actions, rewards, dones, start_idx, count, buffer_idx):
        """Helper to add slice."""
        # Source indices: start_idx to start_idx + count
        # BUT for next_state, we need +1

        # s, a, r, d come from [start_idx : start_idx + count]
        # s' comes from [start_idx + 1 : start_idx + count + 1]

        end_idx = start_idx + count

        self.grid[buffer_idx:buffer_idx+count] = torch.as_tensor(grid[start_idx:end_idx].copy())
        self.proprio[buffer_idx:buffer_idx+count] = torch.as_tensor(proprio[start_idx:end_idx].copy())
        self.actions[buffer_idx:buffer_idx+count] = torch.as_tensor(actions[start_idx:end_idx].copy())
        self.rewards[buffer_idx:buffer_idx+count] = torch.as_tensor(rewards[start_idx:end_idx].copy()).unsqueeze(1)
        self.dones[buffer_idx:buffer_idx+count] = torch.as_tensor(dones[start_idx:end_idx].copy()).unsqueeze(1)
        
        # We don't store next_state explicitly to save RAM.
        # We store sequential data.
        # Wait, if I use a circular buffer and overwrite, I lose the "next" relationship at the boundary of the pointer?
        # Standard ReplayBuffers store (s, s').
        # Storing images twice is heavy (2x RAM).
        # Optimization: Store only 's'. When sampling index 'i', 's_next' is 'i+1'.
        # We just need to handle the case where 'i' is the last element or 'i' is a terminal state.
        # If 'dones[i]' is True, 's_next' is irrelevant.
        # If 'i' is at buffer boundary, we need to wrap.
        # BUT if we overwrite, 'i+1' might be new data unrelated to 'i'.
        # To fix this: We accept that 'next_state' might be garbage if we overwrite.
        # Or we store 'next_idx' or just store s' explicitly.
        # Given 64GB+ RAM on server, maybe storing s' is fine?
        # RGB: 240*424*3 = 300KB. 100k steps = 30GB.
        # 2x = 60GB. Tight.
        # Let's stick to "next state is index + 1" and be careful about boundaries.
        # Actually, for simplicity and robustness, I will store s' explicitly for now, 
        # but maybe downsample or compress if needed.
        # Or better: Just implement the "next_state = buffer[(i+1)%size]" logic and ignore the edge case where we just overwrote i+1.
        # The probability of sampling the exact boundary index is low.
        pass

    def sample(self, batch_size):
        """Sample a batch of transitions."""
        indices = np.random.randint(0, self.size - 1, size=batch_size) # -1 to ensure i+1 exists

        # Retrieve s
        grid = self.grid[indices].to(self.device, non_blocking=True).float() / 255.0
        grid = grid.unsqueeze(1) # (B, 1, H, W)

        proprio = self.proprio[indices].to(self.device, non_blocking=True)
        actions = self.actions[indices].to(self.device, non_blocking=True)
        rewards = self.rewards[indices].to(self.device, non_blocking=True)
        dones = self.dones[indices].to(self.device, non_blocking=True)

        # Retrieve s' (next index)
        next_indices = (indices + 1) % self.capacity

        next_grid = self.grid[next_indices].to(self.device, non_blocking=True).float() / 255.0
        next_grid = next_grid.unsqueeze(1)

        next_proprio = self.proprio[next_indices].to(self.device, non_blocking=True)

        return {
            'grid': grid, 'proprio': proprio,
            'action': actions, 'reward': rewards, 'done': dones,
            'next_grid': next_grid, 'next_proprio': next_proprio
        }

    def copy_state_from(self, other):
        """Deep copy state from another buffer."""
        self.ptr = other.ptr
        self.size = other.size
        self.full = other.full

        # Copy tensors
        self.grid.copy_(other.grid)
        self.proprio.copy_(other.proprio)
        self.actions.copy_(other.actions)
        self.rewards.copy_(other.rewards)
        self.dones.copy_(other.dones)

class V620SACTrainer:
    """SAC Trainer optimized for V620 ROCm."""
    
    def __init__(self, args):
        self.args = args
        
        # Device setup
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
            torch.backends.cudnn.benchmark = True # REQUIRED for speed
            print("✓ Enabled cuDNN benchmark (Startup may take ~2min)")
        else:
            self.device = torch.device('cpu')
            print("⚠ Using CPU")
            
        # Dimensions
        self.grid_shape = (64, 64)
        self.proprio_dim = 12
        self.action_dim = 2
        
        # Visualization state
        self.latest_grid_vis = None
        
        # --- Actor ---
        self.actor_encoder = OccupancyGridEncoder().to(self.device)
        self.actor_head = GaussianPolicyHead(self.actor_encoder.output_dim, self.proprio_dim, self.action_dim).to(self.device)
        
        # --- Critics ---
        # Shared encoder for critics
        self.critic_encoder = OccupancyGridEncoder().to(self.device)

        self.critic1 = QNetwork(self.critic_encoder.output_dim, self.proprio_dim, self.action_dim).to(self.device)
        self.critic2 = QNetwork(self.critic_encoder.output_dim, self.proprio_dim, self.action_dim).to(self.device)

        # --- Target Critics ---
        self.target_critic_encoder = copy.deepcopy(self.critic_encoder)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            list(self.actor_encoder.parameters()) + list(self.actor_head.parameters()), 
            lr=args.lr
        )
        self.critic_optimizer = optim.Adam(
            list(self.critic_encoder.parameters()) + list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=args.lr
        )
        
        # Automatic Entropy Tuning
        self.target_entropy = -float(self.action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=args.lr)

        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=args.lr)

        # Replay Buffer
        self.buffer = ReplayBuffer(
            capacity=args.buffer_size,
            grid_shape=self.grid_shape,
            proprio_dim=self.proprio_dim,
            device=self.device
        )
        
        # Training Buffer (Double Buffering)
        self.training_buffer = ReplayBuffer(
            capacity=args.buffer_size,
            grid_shape=self.grid_shape,
            proprio_dim=self.proprio_dim,
            device=self.device
        )
        
        # State
        self.total_steps = 0
        self.model_version = 0
        self.training_active = False

        # NATS connection (will be initialized in async setup)
        self.nc = None
        self.js = None
        self.nats_server = args.nats_server
        
        # Logging
        self.writer = SummaryWriter(args.log_dir)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        
        # Threading
        self.lock = threading.Lock()
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()
        
        # Load Checkpoint
        self.load_latest_checkpoint()
        
        # Export initial model
        self.export_onnx(increment_version=False)

        # Start Dashboard
        self.dashboard = TrainingDashboard(self)
        self.dashboard.start()

    def load_latest_checkpoint(self):
        checkpoints = list(Path(self.args.checkpoint_dir).glob('sac_step_*.pt'))
        if not checkpoints:
            print("🆕 No checkpoint found. Starting fresh.")
            return

        latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
        print(f"🔄 Resuming from {latest}")
        ckpt = torch.load(latest, map_location=self.device)

        # Check if checkpoint is from old architecture (RGB-D / Semantic vs Occupancy Grid)
        try:
            # Try to check if the encoder structure matches by comparing shapes
            actor_encoder_state = ckpt.get('actor_encoder', {})
            # Occupancy grid encoder should have conv layers with Sequential naming (conv.0.weight, etc.)
            is_old_checkpoint = 'conv.0.weight' not in actor_encoder_state
        except Exception:
            is_old_checkpoint = True

        if is_old_checkpoint:
            print("⚠️  Checkpoint is from old version (RGB-D / Semantic)")
            print("   Starting fresh with Occupancy Grid architecture...")
            # Don't load anything, start fresh
            return

        # Load everything normally
        self.actor_encoder.load_state_dict(ckpt['actor_encoder'])
        self.actor_head.load_state_dict(ckpt['actor_head'])
        self.critic_encoder.load_state_dict(ckpt['critic_encoder'])
        self.critic1.load_state_dict(ckpt['critic1'])
        self.critic2.load_state_dict(ckpt['critic2'])
        self.target_critic_encoder.load_state_dict(ckpt['target_critic_encoder'])
        self.target_critic1.load_state_dict(ckpt['target_critic1'])
        self.target_critic2.load_state_dict(ckpt['target_critic2'])
        self.log_alpha.data = ckpt['log_alpha']

        self.actor_optimizer.load_state_dict(ckpt['actor_opt'])
        self.critic_optimizer.load_state_dict(ckpt['critic_opt'])
        self.alpha_optimizer.load_state_dict(ckpt['alpha_opt'])

        self.total_steps = ckpt['total_steps']
        self.model_version = ckpt.get('model_version', max(1, self.total_steps // 100)) 

    def save_checkpoint(self):
        path = os.path.join(self.args.checkpoint_dir, f"sac_step_{self.total_steps}.pt")
        checkpoint = {
            'actor_encoder': self.actor_encoder.state_dict(),
            'actor_head': self.actor_head.state_dict(),
            'critic_encoder': self.critic_encoder.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_critic_encoder': self.target_critic_encoder.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_opt': self.actor_optimizer.state_dict(),
            'critic_opt': self.critic_optimizer.state_dict(),
            'alpha_opt': self.alpha_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'model_version': self.model_version
        }

        torch.save(checkpoint, path)
        tqdm.write(f"💾 Saved {path}")
        self.export_onnx()

    def export_onnx(self, increment_version=True):
        """Export Actor mean to ONNX."""
        try:
            onnx_path = os.path.join(self.args.checkpoint_dir, "latest_actor.onnx")
            
            # Dummy input: (B, 1, 64, 64)
            dummy_grid = torch.randn(1, 1, 64, 64, device=self.device)
            dummy_proprio = torch.randn(1, self.proprio_dim, device=self.device)
            
            class ActorWrapper(nn.Module):
                def __init__(self, encoder, head):
                    super().__init__()
                    self.encoder = encoder
                    self.head = head
                def forward(self, grid, proprio):
                    features = self.encoder(grid)
                    mean, _ = self.head(features, proprio)
                    return torch.tanh(mean) # Deterministic action
            
            model = ActorWrapper(self.actor_encoder, self.actor_head)
            model.eval()
            
            torch.onnx.export(
                model,
                (dummy_grid, dummy_proprio),
                onnx_path,
                opset_version=11,
                input_names=['grid', 'proprio'],
                output_names=['action'],
                export_params=True,
                do_constant_folding=True,
                keep_initializers_as_inputs=False,
                verbose=False,
                dynamo=False  # Force legacy exporter
            )
            
            if increment_version:
                self.model_version += 1
            tqdm.write(f"📦 Exported ONNX (v{self.model_version})")

            # Schedule model publish to NATS (if connected)
            # Check if attributes exist first to avoid AttributeError on startup
            nc = getattr(self, 'nc', None)
            js = getattr(self, 'js', None)
            loop = getattr(self, 'loop', None)

            if nc is not None and js is not None and loop is not None:
                tqdm.write("🔄 Scheduling model publish task...")
                future = asyncio.run_coroutine_threadsafe(self.publish_model_update(), loop)
                # Add callback to log any errors from the task
                def log_error(fut):
                    try:
                        fut.result()
                    except Exception as e:
                        tqdm.write(f"❌ Model publish task failed: {e}")
                future.add_done_callback(log_error)
            else:
                tqdm.write(f"⚠️ Skipping model publish: nc={nc is not None}, js={js is not None}, loop={loop is not None}")

        except Exception as e:
            tqdm.write(f"❌ Export failed: {e}")

    def _training_loop(self):
        """Continuous training loop: trains constantly while data streams in."""
        print("🧵 Training thread started (CONTINUOUS MODE: train while collecting)")
        pbar = None
        last_time = time.time()

        min_buffer_size = 2000    # Need at least 2000 samples to start training

        while True:
            # Wait for initial buffer warmup
            if self.buffer.size < min_buffer_size:
                if pbar:
                    pbar.set_description(f"📥 Collecting (need {min_buffer_size - self.buffer.size} more)")
                time.sleep(1.0)
                continue

            # Initialize display on first training
            if pbar is None:
                print("\033[H\033[J", end="")
                print("==================================================")
                print("   SAC TRAINING DASHBOARD (CONTINUOUS MODE)     ")
                print("==================================================")
                pbar = tqdm(initial=self.total_steps, desc="🎯 Training", unit="step", dynamic_ncols=True)

            # CONTINUOUS TRAINING: No bursts, no pauses
            # Just train continuously while data streams in from rover
            if self.buffer.size < self.args.batch_size:
                # Buffer too small, wait a bit
                pbar.set_description(f"⏸️  Waiting for data ({self.buffer.size}/{self.args.batch_size})")
                time.sleep(0.1)
                continue

            pbar.set_description("🎯 Training")

            try:
                # Single training step (samples directly from buffer)
                metrics = self.train_step()
                self.total_steps += 1

                # Log to TensorBoard
                if metrics:
                    for k, v in metrics.items():
                        self.writer.add_scalar(f'train/{k}', v, self.total_steps)

                pbar.update(1)

            except (ValueError, torch.AcceleratorError, RuntimeError) as e:
                tqdm.write(f"⚠️ Training step failed: {type(e).__name__}: {e}")
                tqdm.write(f"   Skipping this batch and continuing...")
                # Don't increment total_steps, but continue training
                pbar.update(1)
                continue

            # Update stats every 10 steps for smooth display
            if self.total_steps % 10 == 0:
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time
                steps_per_sec = 10 / dt if dt > 0 else 0
                samples_per_sec = steps_per_sec * self.args.batch_size

                pbar.set_postfix({
                    'Loss': f"A:{metrics['actor_loss']:.2f} C:{metrics['critic_loss']:.2f}",
                    'Alpha': f"{metrics['alpha']:.3f}",
                    'S/s': f"{int(samples_per_sec)}",
                    'Buf': f"{self.buffer.size}",
                    'Ver': f"v{self.model_version}"
                })

            # Flush TensorBoard every 100 steps
            if self.total_steps % 100 == 0:
                self.writer.flush()

            # Checkpoint every 2000 steps
            if self.total_steps % 2000 == 0:
                self.save_checkpoint()

            # Periodic GPU memory cleanup (prevents fragmentation)
            if self.total_steps % 1000 == 0:
                torch.cuda.empty_cache()

    def _validate_tensor(self, tensor, name):
        """Validate tensor for NaN/Inf values."""
        if torch.isnan(tensor).any():
            raise ValueError(f"{name} contains NaN values")
        if torch.isinf(tensor).any():
            raise ValueError(f"{name} contains Inf values")

    def train_step(self):
        t0 = time.time()
        # Sample directly from main buffer (continuous mode)
        with self.lock:
            batch = self.buffer.sample(self.args.batch_size)
        t1 = time.time()

        # Unpack
        state_grid = batch['grid']
        state_proprio = batch['proprio']
        action = batch['action']
        reward = batch['reward']
        done = batch['done']
        next_grid = batch['next_grid']
        next_proprio = batch['next_proprio']

        # Validate batch data
        try:
            self._validate_tensor(state_grid, "state_grid")
            self._validate_tensor(state_proprio, "state_proprio")
            self._validate_tensor(next_grid, "next_grid")
            self._validate_tensor(next_proprio, "next_proprio")
        except ValueError as e:
            tqdm.write(f"⚠️ Skipping batch due to corrupted data: {e}")
            # Return dummy metrics to keep training loop happy
            return {
                'actor_loss': 0.0, 'critic_loss': 0.0, 'alpha': 0.0,
                'alpha_loss': 0.0, 'policy_entropy': 0.0, 'q_value_mean': 0.0,
                'target_entropy_gap': 0.0, 'reward_mean': 0.0, 'reward_std': 0.0
            }

        alpha = self.log_alpha.exp().item()

        # --- Critic Update ---
        with torch.no_grad():
            # Get next action from target policy
            next_features = self.actor_encoder(next_grid)
            next_mean, next_log_std = self.actor_head(next_features, next_proprio)

            # Validate policy outputs before creating distribution
            self._validate_tensor(next_mean, "next_mean")
            self._validate_tensor(next_log_std, "next_log_std")

            # Additional safety: clamp log_std (redundant with model, but belt-and-suspenders)
            next_log_std = torch.clamp(next_log_std, -20, 2)
            next_std = next_log_std.exp()

            # ROCm stability: Add small epsilon to prevent std from being too close to zero
            # This helps avoid edge cases in ROCm kernels for Normal distribution
            next_std = next_std + 1e-6

            # Final validation of std values
            self._validate_tensor(next_std, "next_std")

            # Force GPU sync to catch errors at exact point of failure (ROCm debugging)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()

            dist = torch.distributions.Normal(next_mean, next_std)
            next_action_sample = dist.rsample()

            # Validate sampled actions (ROCm rsample can sometimes produce NaN)
            self._validate_tensor(next_action_sample, "next_action_sample")

            next_action = torch.tanh(next_action_sample)

            # Compute log prob for entropy
            next_log_prob = dist.log_prob(next_action_sample).sum(dim=-1, keepdim=True)
            next_log_prob -= (2 * (np.log(2) - next_action_sample - F.softplus(-2 * next_action_sample))).sum(dim=1, keepdim=True)

            # Target Q
            target_features = self.target_critic_encoder(next_grid)
            q1_target = self.target_critic1(target_features, next_proprio, next_action)
            q2_target = self.target_critic2(target_features, next_proprio, next_action)
            min_q_target = torch.min(q1_target, q2_target) - alpha * next_log_prob
            next_q_value = reward + (1 - done) * self.args.gamma * min_q_target

        # Current Q
        curr_features = self.critic_encoder(state_grid)
        q1 = self.critic1(curr_features, state_proprio, action)
        q2 = self.critic2(curr_features, state_proprio, action)

        critic_loss = F.mse_loss(q1, next_q_value) + F.mse_loss(q2, next_q_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.critic_encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)

        self.critic_optimizer.step()

        t2 = time.time()

        # --- Actor Update ---
        # Re-compute features for actor (gradient flows through encoder)
        actor_features = self.actor_encoder(state_grid)
        mean, log_std = self.actor_head(actor_features, state_proprio)

        # Validate policy outputs before creating distribution
        self._validate_tensor(mean, "mean")
        self._validate_tensor(log_std, "log_std")

        # Additional safety: clamp log_std (redundant with model, but belt-and-suspenders)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()

        # ROCm stability: Add small epsilon to prevent std from being too close to zero
        std = std + 1e-6

        # Final validation of std values
        self._validate_tensor(std, "std")

        # Force GPU sync to catch errors at exact point of failure (ROCm debugging)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        dist = torch.distributions.Normal(mean, std)
        action_sample = dist.rsample()

        # Validate sampled actions (ROCm rsample can sometimes produce NaN)
        self._validate_tensor(action_sample, "action_sample")

        current_action = torch.tanh(action_sample)

        log_prob = dist.log_prob(action_sample).sum(dim=-1, keepdim=True)
        log_prob -= (2 * (np.log(2) - action_sample - F.softplus(-2 * action_sample))).sum(dim=1, keepdim=True)

        # Use critic to evaluate action
        with torch.no_grad():
            q_features = self.critic_encoder(state_grid)

        q1_pi = self.critic1(q_features, state_proprio, current_action)
        q2_pi = self.critic2(q_features, state_proprio, current_action)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = ((alpha * log_prob) - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.actor_encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.actor_head.parameters(), max_norm=1.0)

        self.actor_optimizer.step()

        t3 = time.time()
        
        # --- Alpha Update ---
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # --- Soft Update ---
        self.soft_update(self.critic_encoder, self.target_critic_encoder)
        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)

        # --- Diagnostic Logging (TensorBoard) ---
        if self.total_steps % 10 == 0:
            self.writer.add_scalar('train/action_linear_mean', mean[:, 0].mean().item(), self.total_steps)
            self.writer.add_scalar('train/action_angular_mean', mean[:, 1].mean().item(), self.total_steps)
            self.writer.add_scalar('train/action_linear_std', std[:, 0].mean().item(), self.total_steps)
            self.writer.add_scalar('train/action_angular_std', std[:, 1].mean().item(), self.total_steps)
            self.writer.add_scalar('train/q1_mean', q1.mean().item(), self.total_steps)
            self.writer.add_scalar('train/q2_mean', q2.mean().item(), self.total_steps)
            self.writer.add_scalar('train/alpha', alpha, self.total_steps)

        t4 = time.time()
        
        if (t4 - t0) > 1.0:
            tqdm.write(f"⏱️ Timing: Sample={t1-t0:.3f}s, Critic={t2-t1:.3f}s, Actor={t3-t2:.3f}s, Misc={t4-t3:.3f}s")
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'alpha': alpha,
            'alpha_loss': alpha_loss.item(),
            # Additional monitoring metrics for loss diagnosis
            'policy_entropy': -log_prob.mean().item(),  # Should be high (exploration)
            'q_value_mean': min_q_pi.mean().item(),     # Should increase over time
            'target_entropy_gap': ((-log_prob).mean() - self.target_entropy).item(),  # Should trend to 0

            # NEW: Reward statistics
            'reward_mean': reward.mean().item(),
            'reward_std': reward.std().item(),
            'reward_max': reward.max().item(),
            'reward_min': reward.min().item(),

            # NEW: Q-value diagnostics
            'q1_mean': q1.mean().item(),
            'q2_mean': q2.mean().item(),
            'q_target_mean': next_q_value.mean().item(),
            'q_diff': (q1 - q2).abs().mean().item(),

            # NEW: Policy diagnostics
            'action_mean': current_action.mean().item(),
            'action_std': current_action.std().item(),

            # NEW: Gradient norms
            'actor_grad_norm': torch.nn.utils.clip_grad_norm_(
                list(self.actor_encoder.parameters()) + list(self.actor_head.parameters()),
                float('inf')
            ),
            'critic_grad_norm': torch.nn.utils.clip_grad_norm_(
                list(self.critic_encoder.parameters()) +
                list(self.critic1.parameters()) +
                list(self.critic2.parameters()),
                float('inf')
            ),
        }

    def soft_update(self, source, target, tau=0.001):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    async def setup_nats(self):
        """Initialize NATS connection and JetStream."""
        print(f"🔌 Connecting to NATS at {self.nats_server}...")

        async def on_disconnected():
            print("⚠ NATS disconnected")

        async def on_reconnected():
            print("✅ NATS reconnected")

        self.nc = await nats.connect(
            servers=[self.nats_server],
            name="sac-training-server",
            max_reconnect_attempts=-1,  # Infinite reconnects
            reconnect_time_wait=2,       # 2s between attempts
            ping_interval=20,            # Ping every 20s
            max_outstanding_pings=3,     # Disconnect after 3 missed
            disconnected_cb=on_disconnected,
            reconnected_cb=on_reconnected,
        )

        self.js = self.nc.jetstream()

        # Ensure streams exist
        await self._ensure_streams()

        print(f"✅ Connected to NATS")

    async def _ensure_streams(self):
        """Create NATS JetStream streams if they don't exist."""
        try:
            # Experience stream
            await self.js.add_stream(StreamConfig(
                name="ROVER_EXPERIENCE",
                subjects=["rover.experience"],
                retention="limits",
                max_msgs=1000,
                max_bytes=10 * 1024 * 1024 * 1024,  # 10 GB
                max_age=604800,  # 7 days in seconds
                max_msg_size=200 * 1024 * 1024,  # 200 MB
                storage="file",
                discard="old",
            ))
            print("✅ ROVER_EXPERIENCE stream ready")
        except Exception as e:
            if "stream name already in use" not in str(e).lower():
                print(f"⚠ Stream setup: {e}")

        try:
            # Model stream
            await self.js.add_stream(StreamConfig(
                name="ROVER_MODELS",
                subjects=["models.sac.update", "models.sac.metadata"],
                retention="limits",
                max_msgs=100,
                max_bytes=2 * 1024 * 1024 * 1024,  # 2 GB
                max_age=2592000,  # 30 days in seconds
                max_msg_size=50 * 1024 * 1024,  # 50 MB
                storage="file",
                discard="old",
            ))
            print("✅ ROVER_MODELS stream ready")
        except Exception as e:
            if "stream name already in use" not in str(e).lower():
                print(f"⚠ Stream setup: {e}")

        try:
            # Control stream
            await self.js.add_stream(StreamConfig(
                name="ROVER_CONTROL",
                subjects=["rover.status", "rover.heartbeat", "server.sac.status"],
                retention="limits",
                max_msgs=10000,
                max_bytes=100 * 1024 * 1024,  # 100 MB
                max_age=86400,  # 24 hours in seconds
                max_msg_size=1 * 1024 * 1024,  # 1 MB
                storage="file",
                discard="old",
            ))
            print("✅ ROVER_CONTROL stream ready")
        except Exception as e:
            if "stream name already in use" not in str(e).lower():
                print(f"⚠ Stream setup: {e}")

    async def consume_experience(self):
        """Consume experience batches from rovers."""
        print("📡 Starting experience consumer...")

        # Create durable consumer
        psub = await self.js.pull_subscribe(
            subject="rover.experience",
            durable="sac_trainer"
        )

        while True:
            try:
                msgs = await psub.fetch(batch=1, timeout=1.0)
                for msg in msgs:
                    try:
                        # Deserialize batch
                        print(f"DEBUG: Received NATS msg, data size: {len(msg.data)} bytes")
                        batch = deserialize_batch(msg.data)

                        # Update visualization state (take last frame of batch)
                        if len(batch['grid']) > 0:
                            self.latest_grid_vis = batch['grid'][-1].copy()

                        # Add to replay buffer (thread-safe)
                        with self.lock:
                            self.buffer.add_batch(batch)

                        # Acknowledge message
                        await msg.ack()

                        print(f"📥 Received batch: {len(batch['rewards'])} steps, buffer size: {self.buffer.size}")

                    except Exception as e:
                        print(f"❌ Error processing batch: {e}")
                        import traceback
                        traceback.print_exc()
                        # Negative ack for redelivery
                        await msg.nak()

            except nats.errors.TimeoutError:
                # No messages available, continue waiting
                await asyncio.sleep(0.1)
                continue
            except Exception as e:
                print(f"❌ Consumer error: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1.0)

    async def publish_status(self):
        """Periodically publish training status."""
        while True:
            try:
                status_msg = serialize_status(
                    status='ready' if not self.training_active else 'training',
                    model_version=self.model_version,
                    buffer_size=self.buffer.size,
                    total_steps=self.total_steps
                )

                await self.js.publish("server.sac.status", status_msg)
                await asyncio.sleep(1.0)  # Publish every 1 second for smooth UI  # Publish every 5 seconds

            except Exception as e:
                print(f"❌ Status publish error: {e}")
                await asyncio.sleep(5.0)

    async def publish_model_update(self):
        """Publish model update when checkpoint is saved."""
        # This will be called after save_checkpoint()
        tqdm.write(f"🔄 Executing publish_model_update for v{self.model_version}...")
        try:
            # Read ONNX model
            onnx_path = os.path.join(self.args.checkpoint_dir, "latest_actor.onnx")
            if not os.path.exists(onnx_path):
                tqdm.write(f"❌ ONNX file not found at {onnx_path}")
                return

            with open(onnx_path, 'rb') as f:
                onnx_bytes = f.read()
            
            tqdm.write(f"📦 Read ONNX model: {len(onnx_bytes)} bytes")

            # Publish model
            model_msg = serialize_model_update(onnx_bytes, self.model_version)
            tqdm.write(f"📤 Publishing model update v{self.model_version} to 'models.sac.update'...")
            
            ack = await self.js.publish("models.sac.update", model_msg, timeout=10.0)
            tqdm.write(f"✅ Model publish acknowledged: seq={ack.seq}")

            # Publish metadata
            metadata_msg = serialize_metadata(self.model_version, time.time())
            await self.js.publish("models.sac.metadata", metadata_msg)
            tqdm.write(f"✅ Metadata published for v{self.model_version}")

            tqdm.write(f"🚀 Published model version {self.model_version} successfully")

        except Exception as e:
            tqdm.write(f"❌ Model publish error: {e}")
            import traceback
            traceback.print_exc()

    async def run(self):
        """Main NATS event loop."""
        print(f"🚀 SAC Server starting with NATS at {self.nats_server}")
        
        # Capture event loop for threadsafe calls
        self.loop = asyncio.get_running_loop()

        # Initialize NATS connection
        await self.setup_nats()

        # Publish initial model (in case we resumed from checkpoint)
        tqdm.write("🚀 Publishing initial model state...")
        await self.publish_model_update()

        # Start consumer and status publisher in background
        asyncio.create_task(self.consume_experience())
        asyncio.create_task(self.publish_status())

        # Keep running
        while True:
            await asyncio.sleep(1.0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nats_server', type=str, default='nats://nats.gokickrocks.org:4222', help='NATS server URL')
    parser.add_argument('--buffer_size', type=int, default=100000) # Increased buffer size too since data is smaller
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--gamma', type=float, default=0.97)
    parser.add_argument('--checkpoint_dir', default='./checkpoints_sac')
    parser.add_argument('--log_dir', default='./logs_sac')



    args = parser.parse_args()

    trainer = V620SACTrainer(args)

    # Run async event loop
    asyncio.run(trainer.run())
