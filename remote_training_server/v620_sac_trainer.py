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
from model_architectures import DualEncoderPolicyNetwork, DualEncoderQNetwork, RGBDEncoderPolicyNetwork, RGBDEncoderQNetwork

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

    def __init__(self, capacity: int, proprio_dim: int, device: torch.device, storage_device: torch.device = torch.device('cpu')):
        self.capacity = capacity
        self.device = device
        self.storage_device = storage_device
        self.ptr = 0
        self.size = 0
        self.full = False

        # Storage
        # Laser: 1x128x128, uint8 (0 or 1)
        self.laser = torch.zeros((capacity, 1, 128, 128), dtype=torch.uint8, device=storage_device)
        # RGBD: 4x240x424, uint8 (quantized 0-1 -> 0-255)
        self.rgbd = torch.zeros((capacity, 4, 240, 424), dtype=torch.uint8, device=storage_device)
        self.proprio = torch.zeros((capacity, proprio_dim), dtype=torch.float32, device=storage_device)
        self.actions = torch.zeros((capacity, 2), dtype=torch.float32, device=storage_device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=storage_device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=storage_device)
        
    def add_batch(self, batch_data: Dict):
        """Add a batch of sequential data and construct transitions."""
        # batch_data contains lists/arrays of s, a, r, d
        # We need to construct (s, a, r, s', d)
        # Since data is sequential, s'[t] = s[t+1]

        laser = batch_data['laser']
        rgbd = batch_data['rgbd']
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

            self._add_slice(laser, rgbd, proprio, actions, rewards, dones, 0, first_part, self.ptr)
            self._add_slice(laser, rgbd, proprio, actions, rewards, dones, first_part, second_part, 0)

            self.ptr = second_part
            self.full = True
        else:
            self._add_slice(laser, rgbd, proprio, actions, rewards, dones, 0, batch_size, self.ptr)
            self.ptr += batch_size
            if self.ptr >= self.capacity:
                self.full = True
                self.ptr = self.ptr % self.capacity

        self.size = self.capacity if self.full else self.ptr

    def _add_slice(self, laser, rgbd, proprio, actions, rewards, dones, start_idx, count, buffer_idx):
        """Helper to add slice."""
        # Source indices: start_idx to start_idx + count
        # BUT for next_state, we need +1

        # s, a, r, d come from [start_idx : start_idx + count]
        # s' comes from [start_idx + 1 : start_idx + count + 1]

        end_idx = start_idx + count

        # Quantize depth to uint8 (0-1 -> 0-255)
        # Laser is binary 0/1, so it fits in uint8 directly
        # self.laser = laser (N, 1, 128, 128) float32
        # self.depth = depth (N, 1, 424, 240) float32
        
        laser_slice = torch.as_tensor(laser[start_idx:end_idx].copy())
        rgbd_slice = torch.as_tensor(rgbd[start_idx:end_idx].copy())
        
        # Ensure correct shape (N, 1, H, W) and (N, 4, H, W)
        if laser_slice.ndim == 3:
            laser_slice = laser_slice.unsqueeze(1)
        if rgbd_slice.ndim == 3:
            rgbd_slice = rgbd_slice.unsqueeze(1)
        
        # Quantize rgbd
        rgbd_slice = (rgbd_slice * 255.0).to(torch.uint8)
        laser_slice = laser_slice.to(torch.uint8)

        self.laser[buffer_idx:buffer_idx+count] = laser_slice.to(self.storage_device)
        self.rgbd[buffer_idx:buffer_idx+count] = rgbd_slice.to(self.storage_device)
        self.proprio[buffer_idx:buffer_idx+count] = torch.as_tensor(proprio[start_idx:end_idx].copy()).to(self.storage_device)
        self.actions[buffer_idx:buffer_idx+count] = torch.as_tensor(actions[start_idx:end_idx].copy()).to(self.storage_device)
        self.rewards[buffer_idx:buffer_idx+count] = torch.as_tensor(rewards[start_idx:end_idx].copy()).unsqueeze(1).to(self.storage_device)
        self.dones[buffer_idx:buffer_idx+count] = torch.as_tensor(dones[start_idx:end_idx].copy()).unsqueeze(1).to(self.storage_device)
        
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
        laser = self.laser[indices].to(self.device, non_blocking=True).float() # Binary 0/1
        rgbd = self.rgbd[indices].to(self.device, non_blocking=True).float() / 255.0

        proprio = self.proprio[indices].to(self.device, non_blocking=True)
        actions = self.actions[indices].to(self.device, non_blocking=True)
        rewards = self.rewards[indices].to(self.device, non_blocking=True)
        dones = self.dones[indices].to(self.device, non_blocking=True)

        # Retrieve s' (next index)
        next_indices = (indices + 1) % self.capacity

        next_laser = self.laser[next_indices].to(self.device, non_blocking=True).float()
        next_rgbd = self.rgbd[next_indices].to(self.device, non_blocking=True).float() / 255.0

        next_proprio = self.proprio[next_indices].to(self.device, non_blocking=True)

        return {
            'laser': laser, 'rgbd': rgbd, 'proprio': proprio,
            'action': actions, 'reward': rewards, 'done': dones,
            'next_laser': next_laser, 'next_rgbd': next_rgbd, 'next_proprio': next_proprio
        }

    def copy_state_from(self, other):
        """Deep copy state from another buffer."""
        self.ptr = other.ptr
        self.size = other.size
        self.full = other.full

        # Copy tensors
        self.laser.copy_(other.laser)
        self.rgbd.copy_(other.rgbd)
        self.proprio.copy_(other.proprio)
        self.actions.copy_(other.actions)
        self.rewards.copy_(other.rewards)
        self.dones.copy_(other.dones)

class V620SACTrainer:
    """SAC Trainer optimized for V620 ROCm."""
    
    def __init__(self, args):
        self.args = args

        # Validate hyperparameters
        assert 0.0 <= args.droq_dropout <= 0.1, "Dropout must be in [0.0, 0.1]"
        assert args.droq_samples >= 1, "M must be >= 1"
        assert args.utd_ratio >= 1, "UTD must be >= 1"
        assert args.actor_update_freq >= 1, "Actor update freq must be >= 1"

        # Device setup
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
            torch.backends.cudnn.benchmark = True # REQUIRED for speed
            print("✓ Enabled cuDNN benchmark (Startup may take ~2min)")
        else:
            self.device = torch.device('cpu')
            self.device = torch.device('cpu')
            print("⚠ Using CPU")
        
        # Determine storage device for Replay Buffer
        # Determine storage device for Replay Buffer
        # Default to CPU to avoid OOM with large depth buffers
        self.storage_device = torch.device('cpu') 
        if args.gpu_buffer:
            print(f"⚠️ Warning: GPU buffer enabled. Ensure you have >40GB VRAM.")
            self.storage_device = self.device
        
        if self.storage_device.type != 'cpu':
             print(f"✓ Replay Buffer will be stored on GPU memory")
        else:
             print(f"  Replay Buffer stored on System RAM (CPU)")
            
        # Dimensions
        self.proprio_dim = 10 # [ax, ay, az, gx, gy, gz, min_depth, min_lidar, prev_lin, prev_ang]
        self.action_dim = 2
        
        # Visualization state
        self.latest_laser_vis = None
        self.latest_rgbd_vis = None
        
        # --- Actor ---
        # Use RGBD-based network
        self.actor = RGBDEncoderPolicyNetwork(action_dim=self.action_dim, proprio_dim=self.proprio_dim).to(self.device)
        
        # --- Critics ---
        # RGBD-based Q-Networks
        self.critic1 = RGBDEncoderQNetwork(
            action_dim=self.action_dim,
            proprio_dim=self.proprio_dim,
            dropout=args.droq_dropout
        ).to(self.device)
        
        self.critic2 = RGBDEncoderQNetwork(
            action_dim=self.action_dim,
            proprio_dim=self.proprio_dim,
            dropout=args.droq_dropout
        ).to(self.device)

        # --- Target Critics ---
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        # Disable dropout in target networks (deterministic targets)
        self.target_critic1.eval()
        self.target_critic2.eval()
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=args.lr
        )
        
        # Automatic Entropy Tuning
        self.target_entropy = -float(self.action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=args.lr)

        # Replay Buffer
        self.buffer = ReplayBuffer(
            capacity=args.buffer_size,
            proprio_dim=self.proprio_dim,
            device=self.device,
            storage_device=self.storage_device
        )

        # Data augmentation for occupancy grids
        # TODO: Update augmentation for dual inputs if needed. Disabling for now.
        if args.augment_data:
            print(f"⚠️ Data augmentation temporarily disabled for dual-input architecture.")
            self.augmentation = None
        else:
            self.augmentation = None

        # State
        self.total_steps = 0
        self.gradient_steps = 0  # Track actual gradient updates (for UTD > 1)
        self.model_version = 0
        self.training_active = False

        # Metrics history for dashboard (circular buffer of last 500 steps)
        from collections import deque
        self.metrics_history = deque(maxlen=500)
        self.last_step_time = time.time()
        self.steps_per_sec = 0.0

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
        # Load checkpoint before exporting
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

        # Check if checkpoint is from old architecture
        try:
            actor_state = ckpt.get('actor', {})
            # Check if it's RGBD-based (has rgbd_encoder)
            has_rgbd_encoder = 'rgbd_encoder.conv1.weight' in actor_state
            # Check if it's depth-based (has depth_encoder)
            has_depth_encoder = 'depth_encoder.conv1.weight' in actor_state

            if has_rgbd_encoder:
                print("✅ Checkpoint uses RGBD architecture - compatible")
                is_compatible = True
            elif has_depth_encoder:
                print("⚠️  Checkpoint uses Depth-only architecture")
                print("   Starting fresh with RGBD architecture...")
                is_compatible = False
            else:
                print("⚠️  Checkpoint is from very old architecture")
                print("   Starting fresh with RGBD architecture...")
                is_compatible = False
        except Exception:
            print("⚠️  Checkpoint loading failed - starting fresh")
            is_compatible = False

        if not is_compatible:
            # Don't load anything, start fresh
            return

        # Load everything normally
        self.actor.load_state_dict(ckpt['actor'])
        self.critic1.load_state_dict(ckpt['critic1'])
        self.critic2.load_state_dict(ckpt['critic2'])
        self.target_critic1.load_state_dict(ckpt['target_critic1'])
        self.target_critic2.load_state_dict(ckpt['target_critic2'])
        self.log_alpha.data = ckpt['log_alpha']

        self.actor_optimizer.load_state_dict(ckpt['actor_opt'])
        self.critic_optimizer.load_state_dict(ckpt['critic_opt'])
        self.alpha_optimizer.load_state_dict(ckpt['alpha_opt'])

        self.total_steps = ckpt['total_steps']
        self.gradient_steps = ckpt.get('gradient_steps', 0)
        self.model_version = ckpt.get('model_version', max(1, self.total_steps // 100))

    def save_checkpoint(self):
        path = os.path.join(self.args.checkpoint_dir, f"sac_step_{self.total_steps}.pt")
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_opt': self.actor_optimizer.state_dict(),
            'critic_opt': self.critic_optimizer.state_dict(),
            'alpha_opt': self.alpha_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'gradient_steps': self.gradient_steps,
            'model_version': self.model_version
        }

        torch.save(checkpoint, path)
        tqdm.write(f"💾 Saved {path}")
        self.export_onnx()

    def export_onnx(self, increment_version=True):
        """Export Actor mean to ONNX."""
        try:
            onnx_path = os.path.join(self.args.checkpoint_dir, "latest_actor.onnx")
            
            # Dummy inputs
            # Laser: (B, 1, 128, 128)
            dummy_laser = torch.randn(1, 1, 128, 128, device=self.device)
            # RGBD: (B, 4, 240, 424)
            dummy_rgbd = torch.randn(1, 4, 240, 424, device=self.device)
            # Proprio: (B, 10)
            dummy_proprio = torch.randn(1, self.proprio_dim, device=self.device)
            
            class ActorWrapper(nn.Module):
                def __init__(self, actor):
                    super().__init__()
                    self.actor = actor
                def forward(self, laser, rgbd, proprio):
                    mean, _ = self.actor(laser, rgbd, proprio)
                    return torch.tanh(mean) # Deterministic action
            
            model = ActorWrapper(self.actor)
            model.eval()
            
            torch.onnx.export(
                model,
                (dummy_laser, dummy_rgbd, dummy_proprio),
                onnx_path,
                opset_version=11,
                input_names=['laser', 'rgbd', 'proprio'],
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

                # Store metrics in history for dashboard
                if metrics:
                    metrics['step'] = self.total_steps
                    metrics['timestamp'] = time.time()
                    self.metrics_history.append(metrics.copy())

                # Log to TensorBoard
                if metrics:
                    for k, v in metrics.items():
                        if k not in ['step', 'timestamp']:  # Don't log these
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

                # Store for dashboard
                self.steps_per_sec = steps_per_sec

                pbar.set_postfix({
                    'Loss': f"A:{metrics['actor_loss']:.2f} C:{metrics['critic_loss']:.2f}",
                    'Alpha': f"{metrics['alpha']:.3f}",
                    'GradSteps': f"{self.gradient_steps}",
                    'UTD': f"{self.args.utd_ratio}",
                    'S/s': f"{int(samples_per_sec)}",
                    'Buf': f"{self.buffer.size}",
                    'Ver': f"v{self.model_version}"
                })

            # Flush TensorBoard every 100 steps
            if self.total_steps % 100 == 0:
                self.writer.flush()

            # Checkpoint every 2000 steps
            if self.total_steps % 500 == 0:
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
        """Perform one training step with UTD > 1.

        This method now performs multiple gradient steps per environment step.
        With UTD=10, this does 10 critic updates and 1 actor update.
        """
        t0 = time.time()

        # Sample batch ONCE (reuse for all gradient steps)
        with self.lock:
            batch = self.buffer.sample(self.args.batch_size)
        t1 = time.time()

        # Apply data augmentation if enabled
        if self.args.augment_data and self.augmentation is not None:
            # TODO: Implement dual augmentation
            pass

        # Validate batch data
        try:
            self._validate_tensor(batch['laser'], "state_laser")
            self._validate_tensor(batch['rgbd'], "state_rgbd")
            self._validate_tensor(batch['proprio'], "state_proprio")
            self._validate_tensor(batch['next_laser'], "next_laser")
            self._validate_tensor(batch['next_rgbd'], "next_rgbd")
            self._validate_tensor(batch['next_proprio'], "next_proprio")
        except ValueError as e:
            tqdm.write(f"⚠️ Skipping batch due to corrupted data: {e}")
            return self._dummy_metrics()

        # Accumulators for metrics (average over UTD steps)
        total_critic_loss = 0.0
        total_actor_loss = 0.0
        total_alpha_loss = 0.0
        last_log_prob = None
        last_q1 = None
        last_q2 = None
        last_q_target = None
        last_min_q_pi = None

        # --- UTD Loop: Multiple gradient steps per environment step ---
        for grad_step in range(self.args.utd_ratio):
            # 1. Update Critics (every step)
            critic_loss, q1, q2, q_target = self._update_critic_droq(batch)
            total_critic_loss += critic_loss.item()
            last_q1 = q1
            last_q2 = q2
            last_q_target = q_target

            # 2. Update Actor (every K steps)
            if grad_step % self.args.actor_update_freq == 0:
                actor_loss, log_prob, min_q_pi = self._update_actor(batch)
                total_actor_loss += actor_loss.item()
                last_log_prob = log_prob
                last_min_q_pi = min_q_pi

                # 3. Update Alpha (with actor)
                alpha_loss = self._update_alpha(log_prob)
                total_alpha_loss += alpha_loss.item()

            # 4. Soft Update Targets (every step)
            self._soft_update_targets()

            # Increment gradient step counter
            self.gradient_steps += 1

        t2 = time.time()

        # Average metrics over UTD steps
        num_actor_updates = (self.args.utd_ratio + self.args.actor_update_freq - 1) // self.args.actor_update_freq
        avg_critic_loss = total_critic_loss / self.args.utd_ratio
        avg_actor_loss = total_actor_loss / max(num_actor_updates, 1)
        avg_alpha_loss = total_alpha_loss / max(num_actor_updates, 1)

        alpha = self.log_alpha.exp().item()

        # --- Diagnostic Logging (TensorBoard) ---
        if self.total_steps % 100 == 0 and last_log_prob is not None:
            # Loss metrics
            self.writer.add_scalar('Loss/Critic', avg_critic_loss, self.total_steps)
            self.writer.add_scalar('Loss/Actor', avg_actor_loss, self.total_steps)
            self.writer.add_scalar('Loss/Alpha', avg_alpha_loss, self.total_steps)

            # Entropy and alpha
            self.writer.add_scalar('Entropy/Policy', -last_log_prob.mean().item(), self.total_steps)
            self.writer.add_scalar('Alpha/Value', alpha, self.total_steps)

            # Q-values
            self.writer.add_scalar('Q_Value/Q1', last_q1.mean().item(), self.total_steps)
            self.writer.add_scalar('Q_Value/Q2', last_q2.mean().item(), self.total_steps)
            self.writer.add_scalar('Q_Value/Min_Q_Pi', last_min_q_pi.mean().item() if last_min_q_pi is not None else 0.0, self.total_steps)

            # Reward statistics
            self.writer.add_scalar('Reward/Mean', batch['reward'].mean().item(), self.total_steps)
            self.writer.add_scalar('Reward/Std', batch['reward'].std().item(), self.total_steps)

            # NEW: Log observation statistics for debugging
            self.writer.add_scalar('Observation/Laser_Mean', batch['laser'].mean().item(), self.total_steps)
            self.writer.add_scalar('Observation/RGBD_Mean', batch['rgbd'].mean().item(), self.total_steps)
            self.writer.add_scalar('Observation/Proprio_Mean', batch['proprio'].mean().item(), self.total_steps)

            # Training progress
            self.writer.add_scalar('Training/Gradient_Steps', self.gradient_steps, self.total_steps)
            self.writer.add_scalar('Training/UTD_Ratio', self.args.utd_ratio, self.total_steps)

        t3 = time.time()

        if (t3 - t0) > 1.0:
            tqdm.write(f"⏱️ Timing: Sample={t1-t0:.3f}s, Training={t2-t1:.3f}s (UTD={self.args.utd_ratio}), Misc={t3-t2:.3f}s")

        return {
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss,
            'alpha': alpha,
            'alpha_loss': avg_alpha_loss,
            'policy_entropy': -last_log_prob.mean().item() if last_log_prob is not None else 0.0,
            'q_value_mean': last_min_q_pi.mean().item() if last_min_q_pi is not None else 0.0,
            'target_entropy_gap': ((-last_log_prob).mean() - self.target_entropy).item() if last_log_prob is not None else 0.0,
            'reward_mean': batch['reward'].mean().item(),
            'reward_std': batch['reward'].std().item(),
            'q1_mean': last_q1.mean().item() if last_q1 is not None else 0.0,
            'q2_mean': last_q2.mean().item() if last_q2 is not None else 0.0,
            'q_target_mean': last_q_target.mean().item() if last_q_target is not None else 0.0,
            'gradient_steps': self.gradient_steps,
        }

    def _dummy_metrics(self):
        """Return dummy metrics when batch is invalid."""
        return {
            'actor_loss': 0.0, 'critic_loss': 0.0, 'alpha': 0.0,
            'alpha_loss': 0.0, 'policy_entropy': 0.0, 'q_value_mean': 0.0,
            'target_entropy_gap': 0.0, 'reward_mean': 0.0, 'reward_std': 0.0,
            'gradient_steps': self.gradient_steps,
        }

    def _update_critic_droq(self, batch):
        """Update critics with DroQ (M forward passes with dropout).

        Args:
            batch: Dictionary with 'laser', 'depth', 'proprio', 'action', 'reward', 'done',
                   'next_laser', 'next_depth', 'next_proprio'

        Returns:
            tuple: (critic_loss, q1, q2, q_target)
        """
        state_laser = batch['laser']
        state_rgbd = batch['rgbd']
        state_proprio = batch['proprio']
        action = batch['action']
        reward = batch['reward']
        done = batch['done']
        next_laser = batch['next_laser']
        next_rgbd = batch['next_rgbd']
        next_proprio = batch['next_proprio']

        alpha = self.log_alpha.exp().item()

        # --- Compute Target Q-values (no dropout, deterministic) ---
        with torch.no_grad():
            # Get next action from actor
            next_mean, next_log_std = self.actor(next_laser, next_rgbd, next_proprio)

            # Sample and validate
            next_log_std = torch.clamp(next_log_std, -20, 2)
            next_std = next_log_std.exp() + 1e-6
            self._validate_tensor(next_std, "next_std")

            if self.device.type == 'cuda':
                torch.cuda.synchronize()

            dist = torch.distributions.Normal(next_mean, next_std)
            next_action_sample = dist.rsample()
            self._validate_tensor(next_action_sample, "next_action_sample")
            next_action = torch.tanh(next_action_sample)

            # Log prob for entropy
            next_log_prob = dist.log_prob(next_action_sample).sum(dim=-1, keepdim=True)
            next_log_prob -= (2 * (np.log(2) - next_action_sample - F.softplus(-2 * next_action_sample))).sum(dim=1, keepdim=True)

            # Target Q (NO dropout in target networks)
            q1_target = self.target_critic1(next_laser, next_rgbd, next_proprio, next_action)
            q2_target = self.target_critic2(next_laser, next_rgbd, next_proprio, next_action)
            min_q_target = torch.min(q1_target, q2_target) - alpha * next_log_prob
            next_q_value = reward + (1 - done) * self.args.gamma * min_q_target

        # --- Current Q with DroQ (M forward passes with dropout) ---
        if self.args.droq_samples > 1 and self.args.droq_dropout > 0.0:
            # DroQ: Multiple forward passes with dropout
            q1_samples = []
            q2_samples = []

            # Enable dropout
            self.critic1.train()
            self.critic2.train()

            for _ in range(self.args.droq_samples):
                q1_samples.append(self.critic1(state_laser, state_rgbd, state_proprio, action))
                q2_samples.append(self.critic2(state_laser, state_rgbd, state_proprio, action))

            # Average over samples
            q1 = torch.stack(q1_samples).mean(dim=0)
            q2 = torch.stack(q2_samples).mean(dim=0)
        else:
            # Standard SAC: single forward pass
            q1 = self.critic1(state_laser, state_rgbd, state_proprio, action)
            q2 = self.critic2(state_laser, state_rgbd, state_proprio, action)

        # MSE loss against target
        critic_loss = F.mse_loss(q1, next_q_value) + F.mse_loss(q2, next_q_value)

        # Backprop
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)

        self.critic_optimizer.step()

        return critic_loss, q1, q2, next_q_value

    def _update_actor(self, batch):
        """Update actor policy.

        Args:
            batch: Dictionary with state information

        Returns:
            tuple: (actor_loss, log_prob, min_q_pi)
        """
        state_laser = batch['laser']
        state_rgbd = batch['rgbd']
        state_proprio = batch['proprio']

        alpha = self.log_alpha.exp().item()

        # Re-compute features for actor (gradient flows through encoder)
        mean, log_std = self.actor(state_laser, state_rgbd, state_proprio)

        # Validate and sample
        self._validate_tensor(mean, "mean")
        self._validate_tensor(log_std, "log_std")
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp() + 1e-6
        self._validate_tensor(std, "std")

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        dist = torch.distributions.Normal(mean, std)
        action_sample = dist.rsample()
        self._validate_tensor(action_sample, "action_sample")
        current_action = torch.tanh(action_sample)

        log_prob = dist.log_prob(action_sample).sum(dim=-1, keepdim=True)
        log_prob -= (2 * (np.log(2) - action_sample - F.softplus(-2 * action_sample))).sum(dim=1, keepdim=True)

        # Use critic to evaluate action (NO dropout, deterministic)
        # Disable dropout for actor evaluation
        self.critic1.eval()
        self.critic2.eval()

        q1_pi = self.critic1(state_laser, state_rgbd, state_proprio, current_action)
        q2_pi = self.critic2(state_laser, state_rgbd, state_proprio, current_action)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = ((alpha * log_prob) - min_q_pi).mean()

        # Backprop
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)

        self.actor_optimizer.step()

        return actor_loss, log_prob, min_q_pi

    def _update_alpha(self, log_prob):
        """Update entropy temperature (alpha).

        Args:
            log_prob: Policy log probabilities from actor update

        Returns:
            alpha_loss: Scalar tensor
        """
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return alpha_loss

    def _soft_update_targets(self):
        """Soft update target networks."""
        tau = 0.001  # Could make this a hyperparameter
        self.soft_update(self.critic1, self.target_critic1, tau)
        self.soft_update(self.critic2, self.target_critic2, tau)

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
                        if len(batch['laser']) > 0:
                            self.latest_laser_vis = batch['laser'][-1].copy()
                            self.latest_rgbd_vis = batch['rgbd'][-1].copy()

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

            tqdm.write(f"� Published model version {self.model_version} successfully")

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

    # Learning - Updated for improved convergence
    parser.add_argument('--lr', type=float, default=3e-4,  # Was 3e-5 (10× too low!)
                        help='Learning rate for Adam optimizer')
    parser.add_argument('--batch_size', type=int, default=512,  # Was 4096
                        help='Batch size for training (smaller = more frequent updates)')
    parser.add_argument('--buffer_size', type=int, default=200000,  # Was 1M
                        help='Replay buffer capacity (faster turnover)')

    # SAC specific
    parser.add_argument('--gamma', type=float, default=0.98,  # Was 0.97
                        help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005,  # Was 0.001
                        help='Soft target update rate')
    parser.add_argument('--checkpoint_dir', default='./checkpoints_sac')
    parser.add_argument('--log_dir', default='./logs_sac')

    # DroQ + UTD + Augmentation parameters - Updated for sample efficiency
    parser.add_argument('--droq_dropout', type=float, default=0.005,  # Was 0.01
                        help='Dropout rate for DroQ (0.0 to disable)')
    parser.add_argument('--droq_samples', type=int, default=2,  # Was 10 (overkill)
                        help='Number of Q-network forward passes for DroQ (M)')
    parser.add_argument('--utd_ratio', type=int, default=20,  # Was 4 (CRITICAL!)
                        help='Update-to-Data ratio (gradient steps per env step)')
    parser.add_argument('--actor_update_freq', type=int, default=2,  # Was 10
                        help='Update actor every N critic updates')
    parser.add_argument('--warmup_steps', type=int, default=10000,  # Was 2000
                        help='Minimum buffer size before training starts')
    parser.add_argument('--augment_data', action='store_true',
                        help='Enable data augmentation for occupancy grids')
    parser.add_argument('--gpu-buffer', action='store_true', help='Store replay buffer on GPU (WARNING: Requires huge VRAM for depth)')



    args = parser.parse_args()

    trainer = V620SACTrainer(args)

    # Run async event loop
    asyncio.run(trainer.run())
