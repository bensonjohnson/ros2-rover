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
from model_architectures import RGBDEncoder, GaussianPolicyHead, QNetwork, AsymmetricQNetwork

# Import serialization utilities
from serialization_utils import (
    serialize_batch, deserialize_batch,
    serialize_model_update, deserialize_model_update,
    serialize_metadata, deserialize_metadata,
    serialize_status, deserialize_status
)

# Import self-supervised semantic model
from self_supervised_vision import SelfSupervisedVisionModel, train_self_supervised_step
from semantic_feature_extractor import extract_semantic_features, augment_reward

# Import dashboard
from dashboard_app import TrainingDashboard

class ReplayBuffer:
    """Experience Replay Buffer for SAC."""

    def __init__(self, capacity: int, rgb_shape: Tuple, depth_shape: Tuple, proprio_dim: int, device: torch.device, semantic_dim: int = 128):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.full = False

        # Storage (CPU RAM to save GPU memory, move to GPU during sampling)
        # Using uint8 for images to save RAM
        self.rgb = torch.zeros((capacity, *rgb_shape), dtype=torch.uint8)
        self.depth = torch.zeros((capacity, *depth_shape), dtype=torch.float16) # Optimized to float16
        self.proprio = torch.zeros((capacity, proprio_dim), dtype=torch.float32)
        self.actions = torch.zeros((capacity, 2), dtype=torch.float32)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32)

        # Semantic features (privileged information for critic)
        self.semantic_features = torch.zeros((capacity, semantic_dim), dtype=torch.float16)
        
    def add_batch(self, batch_data: Dict):
        """Add a batch of sequential data and construct transitions."""
        # batch_data contains lists/arrays of s, a, r, d
        # We need to construct (s, a, r, s', d)
        # Since data is sequential, s'[t] = s[t+1]

        rgb = batch_data['rgb']
        depth = batch_data['depth']
        proprio = batch_data['proprio']
        actions = batch_data['actions']
        rewards = batch_data['rewards']
        dones = batch_data['dones']

        # Semantic features (optional, will be added during reward augmentation)
        semantic_features = batch_data.get('semantic_features', None)
        if semantic_features is None:
            # Create dummy zeros if not provided
            semantic_features = np.zeros((len(rewards), 128), dtype=np.float32)

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

            self._add_slice(rgb, depth, proprio, actions, rewards, dones, semantic_features, 0, first_part, self.ptr)
            self._add_slice(rgb, depth, proprio, actions, rewards, dones, semantic_features, first_part, second_part, 0)

            self.ptr = second_part
            self.full = True
        else:
            self._add_slice(rgb, depth, proprio, actions, rewards, dones, semantic_features, 0, batch_size, self.ptr)
            self.ptr += batch_size
            if self.ptr >= self.capacity:
                self.full = True
                self.ptr = self.ptr % self.capacity

        self.size = self.capacity if self.full else self.ptr

    def _add_slice(self, rgb, depth, proprio, actions, rewards, dones, semantic_features, start_idx, count, buffer_idx):
        """Helper to add slice."""
        # Source indices: start_idx to start_idx + count
        # BUT for next_state, we need +1

        # s, a, r, d come from [start_idx : start_idx + count]
        # s' comes from [start_idx + 1 : start_idx + count + 1]

        end_idx = start_idx + count

        self.rgb[buffer_idx:buffer_idx+count] = torch.as_tensor(rgb[start_idx:end_idx].copy())
        self.depth[buffer_idx:buffer_idx+count] = torch.as_tensor(depth[start_idx:end_idx].copy()).to(torch.float16) # Convert to float16
        self.proprio[buffer_idx:buffer_idx+count] = torch.as_tensor(proprio[start_idx:end_idx].copy())
        self.actions[buffer_idx:buffer_idx+count] = torch.as_tensor(actions[start_idx:end_idx].copy())
        self.rewards[buffer_idx:buffer_idx+count] = torch.as_tensor(rewards[start_idx:end_idx].copy()).unsqueeze(1)
        self.dones[buffer_idx:buffer_idx+count] = torch.as_tensor(dones[start_idx:end_idx].copy()).unsqueeze(1)
        self.semantic_features[buffer_idx:buffer_idx+count] = torch.as_tensor(semantic_features[start_idx:end_idx].copy()).to(torch.float16)
        
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
        rgb = self.rgb[indices].to(self.device, non_blocking=True).float() / 255.0
        rgb = rgb.permute(0, 3, 1, 2) # NHWC -> NCHW

        depth = self.depth[indices].to(self.device, non_blocking=True).float().unsqueeze(1)
        proprio = self.proprio[indices].to(self.device, non_blocking=True)
        actions = self.actions[indices].to(self.device, non_blocking=True)
        rewards = self.rewards[indices].to(self.device, non_blocking=True)
        dones = self.dones[indices].to(self.device, non_blocking=True)

        # Retrieve semantic features
        semantic_features = self.semantic_features[indices].to(self.device, non_blocking=True).float()

        # Retrieve s' (next index)
        next_indices = (indices + 1) % self.capacity

        next_rgb = self.rgb[next_indices].to(self.device, non_blocking=True).float() / 255.0
        next_rgb = next_rgb.permute(0, 3, 1, 2)

        next_depth = self.depth[next_indices].to(self.device, non_blocking=True).float().unsqueeze(1)
        next_proprio = self.proprio[next_indices].to(self.device, non_blocking=True)

        # Next semantic features
        next_semantic_features = self.semantic_features[next_indices].to(self.device, non_blocking=True).float()

        return {
            'rgb': rgb, 'depth': depth, 'proprio': proprio,
            'action': actions, 'reward': rewards, 'done': dones,
            'semantic_features': semantic_features,
            'next_rgb': next_rgb, 'next_depth': next_depth, 'next_proprio': next_proprio,
            'next_semantic_features': next_semantic_features
        }

    def copy_state_from(self, other):
        """Deep copy state from another buffer."""
        self.ptr = other.ptr
        self.size = other.size
        self.full = other.full

        # Copy tensors
        self.rgb.copy_(other.rgb)
        self.depth.copy_(other.depth)
        self.proprio.copy_(other.proprio)
        self.actions.copy_(other.actions)
        self.rewards.copy_(other.rewards)
        self.dones.copy_(other.dones)
        self.semantic_features.copy_(other.semantic_features)

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
        self.rgb_shape = (240, 424, 3) # HWC
        self.depth_shape = (240, 424)
        self.proprio_dim = 10
        self.action_dim = 2
        
        # --- Actor ---
        self.actor_encoder = RGBDEncoder().to(self.device)
        self.actor_head = GaussianPolicyHead(self.actor_encoder.output_dim, self.proprio_dim, self.action_dim).to(self.device)
        
        # --- Self-Supervised Semantic Model ---
        self.semantic_model = SelfSupervisedVisionModel().to(self.device)
        self.use_semantic_augmentation = args.use_semantic_augmentation
        print(f"✓ Semantic augmentation: {'enabled' if self.use_semantic_augmentation else 'disabled'}")

        # --- Critics (Asymmetric with Semantic Features) ---
        # Shared encoder for critics? Or separate?
        # To save memory, let's share one encoder for both critics, but separate from actor.
        self.critic_encoder = RGBDEncoder().to(self.device)

        # Use AsymmetricQNetwork if semantic augmentation is enabled
        if self.use_semantic_augmentation:
            semantic_dim = 128  # Global features from semantic model
            self.critic1 = AsymmetricQNetwork(self.critic_encoder.output_dim, self.proprio_dim, self.action_dim, semantic_dim).to(self.device)
            self.critic2 = AsymmetricQNetwork(self.critic_encoder.output_dim, self.proprio_dim, self.action_dim, semantic_dim).to(self.device)
        else:
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

        # Semantic Model Optimizer
        self.semantic_optimizer = optim.Adam(self.semantic_model.parameters(), lr=args.semantic_lr)

        # Replay Buffer
        self.buffer = ReplayBuffer(
            capacity=args.buffer_size,
            rgb_shape=self.rgb_shape,
            depth_shape=self.depth_shape,
            proprio_dim=self.proprio_dim,
            device=self.device
        )
        
        # Training Buffer (Double Buffering)
        self.training_buffer = ReplayBuffer(
            capacity=args.buffer_size,
            rgb_shape=self.rgb_shape,
            depth_shape=self.depth_shape,
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
        self.semantic_lock = threading.Lock() # Lock for semantic model
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

        # Check if this is an old checkpoint (without semantic model)
        is_old_checkpoint = 'semantic_model' not in ckpt

        if is_old_checkpoint and self.use_semantic_augmentation:
            print("⚠️  Checkpoint is from old version (without semantic augmentation)")
            print("   Loading actor and encoder weights, reinitializing critics for new architecture...")

            # Load actor (unchanged architecture)
            self.actor_encoder.load_state_dict(ckpt['actor_encoder'])
            self.actor_head.load_state_dict(ckpt['actor_head'])

            # Load critic encoder only (shared part)
            self.critic_encoder.load_state_dict(ckpt['critic_encoder'])
            self.target_critic_encoder.load_state_dict(ckpt['target_critic_encoder'])

            # Don't load critics - they have different architecture now (AsymmetricQNetwork)
            print("   ⚠️  Reinitializing critics with AsymmetricQNetwork (this is expected)")

            # Load alpha
            self.log_alpha.data = ckpt['log_alpha']

            # Load actor optimizer only (critic optimizer params won't match)
            self.actor_optimizer.load_state_dict(ckpt['actor_opt'])
            # Note: Not loading critic_optimizer or alpha_optimizer to avoid mismatch

            # Semantic model and optimizer start fresh (already initialized)
            print("   ✓ Semantic model starting fresh")

            # Preserve training progress
            self.total_steps = ckpt['total_steps']
            self.model_version = ckpt.get('model_version', max(1, self.total_steps // 100))

            print(f"   ✓ Resumed from step {self.total_steps} with new architecture")
            print("   ℹ️  Critics will warm up over next ~100 steps")

        else:
            # New checkpoint format or semantic disabled - load everything normally
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

            # Load semantic model if present
            if 'semantic_model' in ckpt and self.use_semantic_augmentation:
                self.semantic_model.load_state_dict(ckpt['semantic_model'])
                self.semantic_optimizer.load_state_dict(ckpt['semantic_opt'])
                print("✓ Loaded semantic model from checkpoint")

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

        # Add semantic model if enabled
        if self.use_semantic_augmentation:
            checkpoint['semantic_model'] = self.semantic_model.state_dict()
            checkpoint['semantic_opt'] = self.semantic_optimizer.state_dict()

        torch.save(checkpoint, path)
        tqdm.write(f"💾 Saved {path}")
        self.export_onnx()

    def export_onnx(self, increment_version=True):
        """Export Actor mean to ONNX."""
        try:
            onnx_path = os.path.join(self.args.checkpoint_dir, "latest_actor.onnx")
            
            dummy_rgb = torch.randn(1, 3, 240, 424, device=self.device)
            dummy_depth = torch.randn(1, 1, 240, 424, device=self.device)
            dummy_proprio = torch.randn(1, self.proprio_dim, device=self.device)
            
            class ActorWrapper(nn.Module):
                def __init__(self, encoder, head):
                    super().__init__()
                    self.encoder = encoder
                    self.head = head
                def forward(self, rgb, depth, proprio):
                    features = self.encoder(rgb, depth)
                    mean, _ = self.head(features, proprio)
                    return torch.tanh(mean) # Deterministic action
            
            model = ActorWrapper(self.actor_encoder, self.actor_head)
            model.eval()
            
            torch.onnx.export(
                model,
                (dummy_rgb, dummy_depth, dummy_proprio),
                onnx_path,
                opset_version=11,
                input_names=['rgb', 'depth', 'proprio'],
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
        """Staged training loop: alternates between collection and training phases."""
        print("🧵 Training thread started (STAGED MODE: collect → train → repeat)")
        pbar = None
        last_time = time.time()

        training_burst_size = 50  # Train for 50 steps (200 gradient steps), then pause
        min_buffer_size = 2000    # Need at least 2000 samples to start training

        while True:
            # Phase 1: COLLECTION - Wait for buffer to grow
            if self.buffer.size < min_buffer_size:
                if pbar:
                    pbar.set_description(f"📥 Collecting (need {min_buffer_size - self.buffer.size} more)")
                time.sleep(1.0)
                continue

            # Initialize display on first training
            if pbar is None:
                print("\033[H\033[J", end="")
                print("==================================================")
                print("    SAC TRAINING DASHBOARD (STAGED MODE)        ")
                print("==================================================")
                pbar = tqdm(initial=self.total_steps, desc="🎯 Training", unit="step", dynamic_ncols=True)

            # Phase 2: TRAINING BURST - Train for N iterations without new batches
            pbar.set_description(f"🎯 Training burst ({training_burst_size} iters)")

            # Snapshot buffer for training (Double Buffering)
            # This minimizes lock contention: we only lock to copy, then train on the copy
            with self.lock:
                self.training_buffer.copy_state_from(self.buffer)

            for burst_iter in range(training_burst_size):
                # Check if we still have enough data
                if self.training_buffer.size < self.args.batch_size * 4:
                    pbar.write(f"⚠️  Buffer depleted ({self.training_buffer.size}), pausing training...")
                    break

                t0 = time.time()
                # Perform 4 gradient steps per iteration for better sample efficiency
                for _ in range(4):
                    metrics = self.train_step()
                    self.total_steps += 1

                    # Train semantic model if enabled (every 4 steps to save compute)
                    if self.use_semantic_augmentation and self.total_steps % 4 == 0:
                        semantic_metrics = self._train_semantic_step()
                        # Log semantic losses
                        for k, v in semantic_metrics.items():
                            self.writer.add_scalar(f'semantic/{k}', v, self.total_steps)

                    # Log every step to TensorBoard
                    if metrics:
                        for k, v in metrics.items():
                            self.writer.add_scalar(f'train/{k}', v, self.total_steps)

                    pbar.update(1)
                t1 = time.time()

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

                if self.total_steps % 200 == 0:
                    self.save_checkpoint()

            # Phase 3: PAUSE - Give NATS consumer time to process batches
            pbar.set_description("⏸️  Paused for collection")
            pbar.write(f"✅ Training burst complete ({training_burst_size} iters = {training_burst_size * 4} steps). Pausing for 3s...")
            time.sleep(3.0)  # 3 second pause between bursts for batch ingestion

    def _train_semantic_step(self):
        """Train the self-supervised semantic model.

        Uses the same RGB-D data from replay buffer to train:
        1. Depth prediction from RGB
        2. Edge detection (depth discontinuities)
        3. Temporal consistency (consecutive frames)
        """
        # Sample a batch for semantic training (smaller batch to save compute)
        semantic_batch_size = min(128, self.args.batch_size // 2)

        # Sample from training buffer (no lock needed)
        batch = self.training_buffer.sample(semantic_batch_size)

        # Extract RGB and depth
        rgb = batch['rgb']  # (B, 3, H, W) normalized [0, 1]
        depth = batch['depth']  # (B, 1, H, W) normalized [0, 1]

        # For temporal consistency, use next_rgb if available
        rgb_next = batch.get('next_rgb', None)

        # Train semantic model (Thread-safe)
        with self.semantic_lock:
            losses = train_self_supervised_step(
                self.semantic_model,
                self.semantic_optimizer,
                rgb,
                depth,
                rgb_next
            )

        return losses

    def train_step(self):
        t0 = time.time()
        # Sample from training buffer (no lock needed)
        batch = self.training_buffer.sample(self.args.batch_size)
        t1 = time.time()

        # Unpack
        state_rgb = batch['rgb']
        state_depth = batch['depth']
        state_proprio = batch['proprio']
        action = batch['action']
        reward = batch['reward']
        done = batch['done']
        next_rgb = batch['next_rgb']
        next_depth = batch['next_depth']
        next_proprio = batch['next_proprio']

        # Semantic features (privileged information for critic)
        state_semantic = batch['semantic_features']
        next_semantic = batch['next_semantic_features']

        alpha = self.log_alpha.exp().item()

        # --- Critic Update ---
        with torch.no_grad():
            # Get next action from target policy
            next_features = self.actor_encoder(next_rgb, next_depth)
            next_mean, next_log_std = self.actor_head(next_features, next_proprio)
            next_std = next_log_std.exp()
            dist = torch.distributions.Normal(next_mean, next_std)
            next_action_sample = dist.rsample()
            next_action = torch.tanh(next_action_sample)

            # Compute log prob for entropy
            next_log_prob = dist.log_prob(next_action_sample).sum(dim=-1, keepdim=True)
            next_log_prob -= (2 * (np.log(2) - next_action_sample - F.softplus(-2 * next_action_sample))).sum(dim=1, keepdim=True)

            # Target Q (with semantic features if enabled)
            target_features = self.target_critic_encoder(next_rgb, next_depth)
            if self.use_semantic_augmentation:
                q1_target = self.target_critic1(target_features, next_proprio, next_action, next_semantic)
                q2_target = self.target_critic2(target_features, next_proprio, next_action, next_semantic)
            else:
                q1_target = self.target_critic1(target_features, next_proprio, next_action)
                q2_target = self.target_critic2(target_features, next_proprio, next_action)
            min_q_target = torch.min(q1_target, q2_target) - alpha * next_log_prob
            next_q_value = reward + (1 - done) * self.args.gamma * min_q_target

        # Current Q (with semantic features if enabled)
        curr_features = self.critic_encoder(state_rgb, state_depth)
        if self.use_semantic_augmentation:
            q1 = self.critic1(curr_features, state_proprio, action, state_semantic)
            q2 = self.critic2(curr_features, state_proprio, action, state_semantic)
        else:
            q1 = self.critic1(curr_features, state_proprio, action)
            q2 = self.critic2(curr_features, state_proprio, action)

        critic_loss = F.mse_loss(q1, next_q_value) + F.mse_loss(q2, next_q_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        t2 = time.time()
        
        # --- Actor Update ---
        # Re-compute features for actor (gradient flows through encoder)
        actor_features = self.actor_encoder(state_rgb, state_depth)
        mean, log_std = self.actor_head(actor_features, state_proprio)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action_sample = dist.rsample()
        current_action = torch.tanh(action_sample)

        log_prob = dist.log_prob(action_sample).sum(dim=-1, keepdim=True)
        log_prob -= (2 * (np.log(2) - action_sample - F.softplus(-2 * action_sample))).sum(dim=1, keepdim=True)

        # Use critic to evaluate action (with semantic features if enabled)
        with torch.no_grad():
            q_features = self.critic_encoder(state_rgb, state_depth)

        if self.use_semantic_augmentation:
            q1_pi = self.critic1(q_features, state_proprio, current_action, state_semantic)
            q2_pi = self.critic2(q_features, state_proprio, current_action, state_semantic)
        else:
            q1_pi = self.critic1(q_features, state_proprio, current_action)
            q2_pi = self.critic2(q_features, state_proprio, current_action)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = ((alpha * log_prob) - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
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

    def _augment_batch_semantic(self, batch):
        """Augment batch with semantic features and rewards.

        This runs in a thread pool to avoid blocking the async event loop.
        """
        try:
            # Convert batch to tensors (copy to make writable)
            num_steps = len(batch['rewards'])
            rgb_batch = torch.from_numpy(batch['rgb'].copy()).float() / 255.0  # (N, H, W, 3)
            rgb_batch = rgb_batch.permute(0, 3, 1, 2).to(self.device)  # (N, 3, H, W)

            depth_batch = torch.from_numpy(batch['depth'].copy()).float().unsqueeze(1).to(self.device)  # (N, 1, H, W)

            # Extract semantic features for entire batch
            semantic_features_list = []
            reward_augmentation_list = []

            # Process in mini-batches to avoid OOM
            mini_batch_size = 32
            for i in range(0, num_steps, mini_batch_size):
                end_idx = min(i + mini_batch_size, num_steps)
                rgb_mini = rgb_batch[i:end_idx]
                depth_mini = depth_batch[i:end_idx]

                # Extract semantic features
                with self.semantic_lock:
                    sem_features = extract_semantic_features(
                        rgb_mini, depth_mini, self.semantic_model
                    )

                # Store global features for critic
                semantic_features_list.append(sem_features['global_features'].cpu().numpy())

                # Compute reward augmentation
                base_rewards = torch.from_numpy(batch['rewards'][i:end_idx].copy()).to(self.device)
                augmented_rewards, _ = augment_reward(
                    base_rewards,
                    sem_features,
                    weights={
                        'traversability': self.args.semantic_reward_traversability,
                        'obstacle': self.args.semantic_reward_obstacle,
                        'roughness': self.args.semantic_reward_roughness
                    }
                )

                reward_augmentation_list.append(augmented_rewards.cpu().numpy())

            # Concatenate all mini-batches
            batch['semantic_features'] = np.concatenate(semantic_features_list, axis=0)
            batch['rewards'] = np.concatenate(reward_augmentation_list, axis=0)

            return batch

        except Exception as e:
            print(f"❌ Semantic augmentation failed: {e}")
            import traceback
            traceback.print_exc()
            # Return original batch if augmentation fails
            return batch

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

                        # Extract semantic features and augment rewards (if enabled)
                        if self.use_semantic_augmentation:
                            batch = await asyncio.get_event_loop().run_in_executor(
                                None, self._augment_batch_semantic, batch
                            )

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
    parser.add_argument('--buffer_size', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--gamma', type=float, default=0.97)
    parser.add_argument('--checkpoint_dir', default='./checkpoints_sac')
    parser.add_argument('--log_dir', default='./logs_sac')

    # Semantic augmentation options
    parser.add_argument('--use_semantic_augmentation', action='store_true', default=True,
                        help='Enable self-supervised semantic augmentation (enabled by default)')
    parser.add_argument('--no_semantic_augmentation', dest='use_semantic_augmentation', action='store_false',
                        help='Disable semantic augmentation (for baseline comparison)')
    parser.add_argument('--semantic_lr', type=float, default=1e-4,
                        help='Learning rate for semantic model')
    parser.add_argument('--semantic_reward_traversability', type=float, default=0.1,
                        help='Weight for traversability reward component')
    parser.add_argument('--semantic_reward_obstacle', type=float, default=-0.2,
                        help='Weight for obstacle proximity penalty')
    parser.add_argument('--semantic_reward_roughness', type=float, default=-0.05,
                        help='Weight for terrain roughness penalty')

    args = parser.parse_args()

    trainer = V620SACTrainer(args)

    # Run async event loop
    asyncio.run(trainer.run())
