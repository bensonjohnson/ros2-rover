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
import signal
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
from model_architectures import UnifiedBEVPolicyNetwork, UnifiedBEVQNetwork

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
        # Unified BEV: 2x256x256, uint8 (quantized 0-1 -> 0-255)
        # Channel 0: LiDAR occupancy, Channel 1: Depth occupancy
        self.bev = torch.zeros((capacity, 2, 128, 128), dtype=torch.uint8, device=storage_device)
        self.proprio = torch.zeros((capacity, proprio_dim), dtype=torch.float32, device=storage_device)
        self.actions = torch.zeros((capacity, 2), dtype=torch.float32, device=storage_device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=storage_device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=storage_device)
        
    def add_batch(self, batch_data: Dict):
        """Add a batch of sequential data and construct transitions."""
        # batch_data contains lists/arrays of s, a, r, d
        # We need to construct (s, a, r, s', d)
        # Since data is sequential, s'[t] = s[t+1]

        bev = batch_data['bev']  # Unified BEV grid (N, 2, 128, 128)
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

            self._add_slice(bev, proprio, actions, rewards, dones, 0, first_part, self.ptr)
            self._add_slice(bev, proprio, actions, rewards, dones, first_part, second_part, 0)

            self.ptr = second_part
            self.full = True
        else:
            self._add_slice(bev, proprio, actions, rewards, dones, 0, batch_size, self.ptr)
            self.ptr += batch_size
            if self.ptr >= self.capacity:
                self.full = True
                self.ptr = self.ptr % self.capacity

        self.size = self.capacity if self.full else self.ptr

    def _add_slice(self, bev, proprio, actions, rewards, dones, start_idx, count, buffer_idx):
        """Helper to add slice."""
        # Source indices: start_idx to start_idx + count
        # BUT for next_state, we need +1

        # s, a, r, d come from [start_idx : start_idx + count]
        # s' comes from [start_idx + 1 : start_idx + count + 1]

        end_idx = start_idx + count

        # Quantize BEV to uint8 (0-1 -> 0-255)
        bev_slice = torch.as_tensor(bev[start_idx:end_idx].copy())

        # Ensure correct shape (N, 2, 128, 128)
        if bev_slice.ndim == 3:
            bev_slice = bev_slice.unsqueeze(1)

        # Quantize (0-1 -> 0-255)
        bev_slice = (bev_slice * 255.0).to(torch.uint8)

        self.bev[buffer_idx:buffer_idx+count] = bev_slice.to(self.storage_device)
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
        indices = torch.randint(0, self.size - 1, (batch_size,))  # -1 to ensure i+1 exists

        # Retrieve s
        bev = self.bev[indices].to(self.device, non_blocking=True).float() / 255.0

        proprio = self.proprio[indices].to(self.device, non_blocking=True)
        actions = self.actions[indices].to(self.device, non_blocking=True)
        rewards = self.rewards[indices].to(self.device, non_blocking=True)
        dones = self.dones[indices].to(self.device, non_blocking=True)

        # Retrieve s' (next index)
        next_indices = (indices + 1) % self.capacity

        next_bev = self.bev[next_indices].to(self.device, non_blocking=True).float() / 255.0
        next_proprio = self.proprio[next_indices].to(self.device, non_blocking=True)

        return {
            'bev': bev, 'proprio': proprio,
            'action': actions, 'reward': rewards, 'done': dones,
            'next_bev': next_bev, 'next_proprio': next_proprio
        }

    def copy_state_from(self, other):
        """Deep copy state from another buffer."""
        self.ptr = other.ptr
        self.size = other.size
        self.full = other.full

        # Copy tensors
        self.bev.copy_(other.bev)
        self.proprio.copy_(other.proprio)
        self.actions.copy_(other.actions)
        self.rewards.copy_(other.rewards)
        self.dones.copy_(other.dones)

    def save(self, path: str):
        """Save buffer state to disk.
        
        Args:
            path: File path to save buffer (e.g., 'checkpoints_sac/replay_buffer.pt')
        """
        print(f"💾 Saving replay buffer ({self.size} samples) to {path}...")
        import time
        start = time.time()
        
        # Save only the filled portion of the buffer
        torch.save({
            'bev': self.bev[:self.size].cpu(),  # Move to CPU for saving
            'proprio': self.proprio[:self.size].cpu(),
            'actions': self.actions[:self.size].cpu(),
            'rewards': self.rewards[:self.size].cpu(),
            'dones': self.dones[:self.size].cpu(),
            'ptr': self.ptr,
            'size': self.size,
            'full': self.full,
            'capacity': self.capacity,
        }, path)
        
        elapsed = time.time() - start
        print(f"✅ Buffer saved in {elapsed:.1f}s")

    def load(self, path: str):
        """Load buffer state from disk.
        
        Args:
            path: File path to load buffer from
        """
        if not os.path.exists(path):
            print(f"ℹ️ No existing buffer found at {path}, starting fresh")
            return False
        
        print(f"📂 Loading replay buffer from {path}...")
        import time
        start = time.time()
        
        data = torch.load(path, map_location='cpu')
        
        # Validate capacity matches
        if data['size'] > self.capacity:
            print(f"⚠️ Warning: Saved buffer has {data['size']} samples but capacity is {self.capacity}")
            print(f"   Loading first {self.capacity} samples...")
            size_to_load = self.capacity
        else:
            size_to_load = data['size']
        
        # Load data (will be moved to storage_device when assigned)
        self.bev[:size_to_load] = data['bev'][:size_to_load].to(self.storage_device)
        self.proprio[:size_to_load] = data['proprio'][:size_to_load].to(self.storage_device)
        self.actions[:size_to_load] = data['actions'][:size_to_load].to(self.storage_device)
        self.rewards[:size_to_load] = data['rewards'][:size_to_load].to(self.storage_device)
        self.dones[:size_to_load] = data['dones'][:size_to_load].to(self.storage_device)
        
        self.size = size_to_load
        self.ptr = data['ptr'] if data['ptr'] < self.capacity else self.capacity - 1
        self.full = data.get('full', False)
        
        elapsed = time.time() - start
        print(f"✅ Loaded {self.size} samples in {elapsed:.1f}s")
        return True

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

            # Enable TF32 for faster matmul/conv on supported hardware
            torch.set_float32_matmul_precision('high')
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("✓ TF32 precision enabled (matmul + cudnn)")
            except AttributeError:
                pass

            # Enable Automatic Mixed Precision for 2x speedup
            self.use_amp = True
            self.scaler = torch.amp.GradScaler('cuda')
            print("✓ Enabled AMP (Automatic Mixed Precision)")
        else:
            self.device = torch.device('cpu')
            self.use_amp = False
            self.scaler = None
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
        self.proprio_dim = 6 # [lidar_min, prev_lin, prev_ang, lin_vel, ang_vel, gap_heading]
        self.action_dim = 2
        
        # Visualization state
        self.latest_bev_vis = None

        # --- Actor ---
        # Unified BEV network with residual encoder + LSTMCell temporal memory
        self.actor = UnifiedBEVPolicyNetwork(action_dim=self.action_dim, proprio_dim=self.proprio_dim).to(self.device)
        self.lstm_hidden_size = UnifiedBEVPolicyNetwork.LSTM_HIDDEN

        # --- Critics ---
        # Unified BEV Q-Networks
        self.critic1 = UnifiedBEVQNetwork(
            action_dim=self.action_dim,
            proprio_dim=self.proprio_dim,
            dropout=args.droq_dropout
        ).to(self.device)

        self.critic2 = UnifiedBEVQNetwork(
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

        # Learning rate schedulers (cosine annealing)
        self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer, T_max=100000, eta_min=1e-5
        )
        self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer, T_max=100000, eta_min=1e-5
        )
        print("✓ Learning rate schedulers initialized (CosineAnnealing)")

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

        # Episode reward tracking
        self.episode_rewards = deque(maxlen=100)  # Track last 100 episodes
        self.current_episode_reward = 0.0
        self.episode_count = 0

        # Evaluation Tracking
        self.eval_episode_rewards = deque(maxlen=20)
        self.current_eval_reward = 0.0
        self.best_eval_reward = -float('inf')
        self.last_eval_step = 0

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

        # torch.compile for kernel fusion and reduced launch overhead (ROCm/CUDA)
        # Must be applied AFTER checkpoint loading (compiled modules prefix keys with _orig_mod.)
        if self.device.type == 'cuda':
            try:
                self.actor = torch.compile(self.actor)
                self.critic1 = torch.compile(self.critic1)
                self.critic2 = torch.compile(self.critic2)
                self.target_critic1 = torch.compile(self.target_critic1)
                self.target_critic2 = torch.compile(self.target_critic2)
                print("✓ torch.compile enabled (default mode)")
            except Exception as e:
                print(f"⚠ torch.compile failed, continuing without: {e}")

        # Load replay buffer if exists
        buffer_path = os.path.join(args.checkpoint_dir, "replay_buffer.pt")
        self.buffer.load(buffer_path)

        # Set up signal handler for graceful shutdown
        self._shutdown_requested = False
        signal.signal(signal.SIGINT, self._save_on_shutdown)
        signal.signal(signal.SIGTERM, self._save_on_shutdown)

        # Export initial model
        self.export_onnx(increment_version=False)

        # Start Dashboard
        self.dashboard = TrainingDashboard(self)
        self.dashboard.start()

    def _save_on_shutdown(self, signum, frame):
        """Signal handler to save replay buffer on Ctrl+C or kill."""
        if self._shutdown_requested:
            return  # Prevent double handling
        self._shutdown_requested = True
        
        print("\n")
        print("=" * 50)
        print("🛑 Shutdown signal received!")
        print("=" * 50)
        
        # Save replay buffer
        buffer_path = os.path.join(self.args.checkpoint_dir, "replay_buffer.pt")
        self.buffer.save(buffer_path)
        
        print("=" * 50)
        print("✅ Shutdown complete. Buffer saved!")
        print("=" * 50)
        sys.exit(0)

    def load_latest_checkpoint(self):
        checkpoints = list(Path(self.args.checkpoint_dir).glob('sac_step_*.pt'))
        if not checkpoints:
            print("🆕 No checkpoint found. Starting fresh.")
            return

        latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
        print(f"🔄 Resuming from {latest}")
        ckpt = torch.load(latest, map_location=self.device)

        # Check if checkpoint is from current architecture (residual encoder + LSTM)
        try:
            actor_state = ckpt.get('actor', {})
            has_res_encoder = 'bev_encoder.res1.conv1.weight' in actor_state
            has_lstm = 'lstm_cell.weight_ih' in actor_state

            if has_res_encoder and has_lstm:
                print("✅ Checkpoint uses current architecture (ResBlock + LSTM) - compatible")
                is_compatible = True
            else:
                print("⚠️  Checkpoint is from older architecture")
                print("   Starting fresh with ResBlock + LSTM architecture...")
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

        # Load schedulers if available
        if 'actor_scheduler' in ckpt:
            self.actor_scheduler.load_state_dict(ckpt['actor_scheduler'])
        if 'critic_scheduler' in ckpt:
            self.critic_scheduler.load_state_dict(ckpt['critic_scheduler'])

        self.total_steps = ckpt['total_steps']
        self.gradient_steps = ckpt.get('gradient_steps', 0)
        self.model_version = ckpt.get('model_version', max(1, self.total_steps // 100))
        self.episode_count = ckpt.get('episode_count', 0)

    @staticmethod
    def _unwrap_model(model):
        """Get the underlying module from a torch.compile'd model."""
        if hasattr(model, '_orig_mod'):
            return model._orig_mod
        return model

    def save_checkpoint(self):
        path = os.path.join(self.args.checkpoint_dir, f"sac_step_{self.total_steps}.pt")
        checkpoint = {
            'actor': self._unwrap_model(self.actor).state_dict(),
            'critic1': self._unwrap_model(self.critic1).state_dict(),
            'critic2': self._unwrap_model(self.critic2).state_dict(),
            'target_critic1': self._unwrap_model(self.target_critic1).state_dict(),
            'target_critic2': self._unwrap_model(self.target_critic2).state_dict(),
            'log_alpha': self.log_alpha,
            'actor_opt': self.actor_optimizer.state_dict(),
            'critic_opt': self.critic_optimizer.state_dict(),
            'alpha_opt': self.alpha_optimizer.state_dict(),
            'actor_scheduler': self.actor_scheduler.state_dict(),
            'critic_scheduler': self.critic_scheduler.state_dict(),
            'total_steps': self.total_steps,
            'gradient_steps': self.gradient_steps,
            'model_version': self.model_version,
            'episode_count': self.episode_count
        }

        torch.save(checkpoint, path)
        tqdm.write(f"💾 Saved {path}")
        self.export_onnx()

    def save_best_model(self):
        """Save the current model as 'best_eval_actor.onnx'."""
        try:
            onnx_path = os.path.join(self.args.checkpoint_dir, "best_eval_actor.onnx")
            self._export_actor_onnx(onnx_path)
            tqdm.write(f"🏆 Saved BEST EVAL model to {onnx_path}")
        except Exception as e:
            tqdm.write(f"❌ Failed to save best model: {e}")

    def _export_actor_onnx(self, onnx_path: str):
        """Export actor to ONNX with LSTMCell hidden state inputs/outputs.

        Inputs: bev (1,2,128,128), proprio (1,6), hx (1,128), cx (1,128)
        Outputs: action (1,2), hx_out (1,128), cx_out (1,128)
        """
        class ActorWrapper(nn.Module):
            def __init__(self, actor):
                super().__init__()
                self.actor = actor
            def forward(self, bev, proprio, hx, cx):
                mean, _, hx_out, cx_out = self.actor(bev, proprio, hx, cx)
                return torch.tanh(mean), hx_out, cx_out

        model = ActorWrapper(self._unwrap_model(self.actor))
        model.eval()

        dummy_bev = torch.randn(1, 2, 128, 128, device=self.device)
        dummy_proprio = torch.randn(1, self.proprio_dim, device=self.device)
        dummy_hx = torch.zeros(1, self.lstm_hidden_size, device=self.device)
        dummy_cx = torch.zeros(1, self.lstm_hidden_size, device=self.device)

        torch.onnx.export(
            model,
            (dummy_bev, dummy_proprio, dummy_hx, dummy_cx),
            onnx_path,
            opset_version=11,
            input_names=['bev', 'proprio', 'hx', 'cx'],
            output_names=['action', 'hx_out', 'cx_out'],
            export_params=True,
            do_constant_folding=True,
            keep_initializers_as_inputs=False,
            verbose=False,
            dynamo=False
        )

    def export_onnx(self, increment_version=True):
        """Export Actor mean to ONNX with LSTM hidden states."""
        try:
            onnx_path = os.path.join(self.args.checkpoint_dir, "latest_actor.onnx")
            self._export_actor_onnx(onnx_path)
            
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

    def pretrain_bc(self, warmup_dir: str, epochs: int = 50, batch_size: int = 64):
        """
        Behavior Cloning pre-training: train actor to mimic warmup policy.
        
        Args:
            warmup_dir: Directory containing calibration .npz files from warmup
            epochs: Number of training epochs
            batch_size: Batch size for BC training
        """
        print(f"\n🎓 Starting Behavior Cloning Pre-training...")
        print(f"   Loading warmup data from: {warmup_dir}")
        
        warmup_path = Path(warmup_dir)
        if not warmup_path.exists():
            print(f"   ⚠️ Warmup directory not found: {warmup_dir}")
            print(f"   Skipping BC pre-training.")
            return
        
        # Load all .npz files from warmup directory
        npz_files = list(warmup_path.glob("*.npz"))
        if len(npz_files) < 10:
            print(f"   ⚠️ Not enough warmup data ({len(npz_files)} files). Need at least 10.")
            print(f"   Skipping BC pre-training.")
            return
        
        print(f"   Found {len(npz_files)} warmup samples")
        
        # Load data
        bevs, proprios, actions = [], [], []
        for file_path in npz_files:
            try:
                data = np.load(file_path)
                bev = data['bev']  # (2, 128, 128)
                proprio = data['proprio']  # (10,)
                
                # For warmup data, action is stored OR we can infer from warmup policy:
                # Warmup policy: linear = uniform(0.3, 1.0), angular = uniform(-0.8, 0.8)
                # If 'action' key exists, use it; otherwise generate random warmup-like action
                if 'action' in data.files:
                    action = data['action']
                else:
                    # Generate pseudo-action matching warmup distribution
                    # Forward bias with moderate turning
                    action = np.array([
                        np.random.uniform(0.3, 0.8),  # Linear (forward bias)
                        np.random.uniform(-0.3, 0.3)  # Angular (prefer straight)
                    ], dtype=np.float32)
                
                bevs.append(bev)
                proprios.append(proprio)
                actions.append(action)
            except Exception as e:
                print(f"   ⚠️ Error loading {file_path}: {e}")
                continue
        
        if len(bevs) < 10:
            print(f"   ⚠️ Not enough valid data loaded. Skipping BC pre-training.")
            return
        
        # Convert to tensors
        bevs = torch.tensor(np.stack(bevs), dtype=torch.float32, device=self.device)
        proprios = torch.tensor(np.stack(proprios), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.stack(actions), dtype=torch.float32, device=self.device)
        
        print(f"   Loaded {len(bevs)} samples")
        print(f"   BEV shape: {bevs.shape}, Proprio shape: {proprios.shape}, Actions shape: {actions.shape}")
        
        # BC training loop
        dataset_size = len(bevs)
        optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        
        self.actor.train()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            # Shuffle
            perm = torch.randperm(dataset_size)
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, dataset_size - batch_size, batch_size):
                indices = perm[i:i+batch_size]
                bev_batch = bevs[indices]
                proprio_batch = proprios[indices]
                action_batch = actions[indices]
                
                # Forward pass
                mean, _, _, _ = self.actor(bev_batch, proprio_batch)
                predicted_action = torch.tanh(mean)  # Apply tanh like in inference
                
                # MSE loss
                loss = F.mse_loss(predicted_action, action_batch)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / max(num_batches, 1)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
        
        print(f"✅ BC Pre-training complete! Best loss: {best_loss:.6f}")
        print(f"   Actor now initialized to mimic warmup behavior.")
        self.actor.eval()
        
    def pretrain_bc_from_buffer(self, epochs: int = 50, batch_size: int = 64):
        """
        Behavior Cloning pre-training using data from replay buffer.
        
        This runs AFTER warmup data has been collected via NATS and stored in buffer.
        """
        print(f"\n🎓 Starting Behavior Cloning Pre-training from Replay Buffer...")
        print(f"   Buffer size: {self.buffer.size}")
        
        if self.buffer.size < batch_size * 10:
            print(f"   ⚠️ Not enough data in buffer ({self.buffer.size}). Need at least {batch_size * 10}.")
            return
        
        # BC training loop
        optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.actor.train()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = self.buffer.size // batch_size
            
            for _ in range(num_batches):
                # Sample batch from replay buffer
                batch = self.buffer.sample(batch_size)
                bev = batch['bev']        # (B, 2, 128, 128)
                proprio = batch['proprio'] # (B, 10)
                actions = batch['action']  # (B, 2)
                
                # Forward pass
                mean, _, _, _ = self.actor(bev, proprio)
                predicted_action = torch.tanh(mean)
                
                # MSE loss
                loss = F.mse_loss(predicted_action, actions)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / max(num_batches, 1)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
        
        print(f"✅ BC Pre-training from buffer complete! Best loss: {best_loss:.6f}")
        self.actor.eval()

    def _training_loop(self):
        """Continuous training loop with integrated BC pre-training phase."""
        print("🧵 Training thread started (BC → SAC CONTINUOUS MODE)")
        pbar = None
        last_time = time.time()
        
        # BC pre-training config
        bc_warmup_samples = getattr(self.args, 'bc_warmup_samples', 0)  # Default 0 = skip BC
        bc_epochs = getattr(self.args, 'bc_epochs', 50)
        # Skip BC entirely if bc_warmup_samples <= 0
        bc_done = bc_warmup_samples <= 0
        if bc_done:
            print("ℹ️ BC pre-training disabled (bc_warmup_samples=0). Starting SAC directly.")
        
        min_buffer_size = 2000    # Minimum for SAC training

        while True:
            # ==========================================
            # PHASE 1: BC PRE-TRAINING (run once, only if enabled)
            # ==========================================
            if not bc_done and bc_warmup_samples > 0 and self.buffer.size >= bc_warmup_samples:
                print(f"\n📊 Buffer has {self.buffer.size} samples. Starting BC pre-training phase...")
                
                # Run BC pre-training on buffer data
                self.pretrain_bc_from_buffer(epochs=bc_epochs, batch_size=64)
                bc_done = True
                
                # Publish BC-initialized model as v1
                self.model_version = 1
                print(f"🚀 Publishing BC-initialized model as v{self.model_version}...")
                
                # Schedule async publish
                if hasattr(self, 'loop') and self.loop:
                    asyncio.run_coroutine_threadsafe(self.publish_model_update(), self.loop)
                
                print("✅ BC phase complete. Switching to SAC training...")
                continue
            
            # ==========================================
            # PHASE 0: WAITING FOR BC WARMUP DATA
            # ==========================================
            if not bc_done:
                if pbar is None:
                    pbar = tqdm(total=bc_warmup_samples, desc="🔥 BC Warmup", unit="sample", dynamic_ncols=True)
                pbar.n = self.buffer.size
                pbar.set_description(f"🔥 BC Warmup ({self.buffer.size}/{bc_warmup_samples})")
                pbar.refresh()
                time.sleep(0.5)
                continue
            
            # ==========================================
            # PHASE 2: SAC TRAINING
            # ==========================================
            if self.buffer.size < min_buffer_size:
                time.sleep(1.0)
                continue

            # Initialize SAC display on first SAC training
            if pbar is None or not hasattr(self, '_sac_started'):
                self._sac_started = True
                print("\033[H\033[J", end="")
                print("==================================================")
                print("   SAC TRAINING DASHBOARD (CONTINUOUS MODE)     ")
                print("==================================================")
                pbar = tqdm(initial=self.total_steps, desc="🎯 SAC Training", unit="step", dynamic_ncols=True)

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

            # Checkpoint and export model every 200 steps
            if self.total_steps % 200 == 0:
                self.save_checkpoint()

            # Step learning rate schedulers every 100 steps
            if self.total_steps % 100 == 0:
                self.actor_scheduler.step()
                self.critic_scheduler.step()
                # Log current learning rates
                self.writer.add_scalar('LR/Actor', self.actor_optimizer.param_groups[0]['lr'], self.total_steps)
                self.writer.add_scalar('LR/Critic', self.critic_optimizer.param_groups[0]['lr'], self.total_steps)


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

        # Validate batch data periodically (every 100 steps to avoid GPU sync overhead)
        if self.total_steps % 100 == 0:
            try:
                self._validate_tensor(batch['bev'], "state_bev")
                self._validate_tensor(batch['proprio'], "state_proprio")
                self._validate_tensor(batch['next_bev'], "next_bev")
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

            # Episode reward statistics
            if len(self.episode_rewards) > 0:
                self.writer.add_scalar('Episode/Mean_Reward', np.mean(self.episode_rewards), self.total_steps)
                self.writer.add_scalar('Episode/Max_Reward', np.max(self.episode_rewards), self.total_steps)
                self.writer.add_scalar('Episode/Min_Reward', np.min(self.episode_rewards), self.total_steps)
                self.writer.add_scalar('Episode/Count', self.episode_count, self.total_steps)

            # NEW: Log observation statistics for debugging
            self.writer.add_scalar('Observation/BEV_Mean', batch['bev'].mean().item(), self.total_steps)
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
            batch: Dictionary with 'bev', 'proprio', 'action', 'reward', 'done',
                   'next_bev', 'next_proprio'

        Returns:
            tuple: (critic_loss, q1, q2, q_target)
        """
        state_bev = batch['bev']
        state_proprio = batch['proprio']
        action = batch['action']
        reward = batch['reward']
        done = batch['done']
        next_bev = batch['next_bev']
        next_proprio = batch['next_proprio']

        alpha = self.log_alpha.exp().item()

        # --- Compute Target Q-values (no dropout, deterministic) ---
        with torch.no_grad():
            # Get next action from actor (zero hidden state — replay buffer samples are independent)
            next_mean, next_log_std, _, _ = self.actor(next_bev, next_proprio)

            next_log_std = torch.clamp(next_log_std, -20, 2)
            next_std = next_log_std.exp() + 1e-6

            dist = torch.distributions.Normal(next_mean, next_std)
            next_action_sample = dist.rsample()
            next_action = torch.tanh(next_action_sample)

            # Log prob for entropy
            next_log_prob = dist.log_prob(next_action_sample).sum(dim=-1, keepdim=True)
            next_log_prob -= (2 * (np.log(2) - next_action_sample - F.softplus(-2 * next_action_sample))).sum(dim=1, keepdim=True)

            # Target Q (NO dropout in target networks)
            q1_target = self.target_critic1(next_bev, next_proprio, next_action)
            q2_target = self.target_critic2(next_bev, next_proprio, next_action)
            min_q_target = torch.min(q1_target, q2_target) - alpha * next_log_prob
            next_q_value = reward + (1 - done) * self.args.gamma * min_q_target

        # --- Current Q with DroQ (M forward passes with dropout) ---
        # Use AMP for forward pass and loss computation
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            if self.args.droq_samples > 1 and self.args.droq_dropout > 0.0:
                # DroQ: Multiple forward passes with dropout
                q1_samples = []
                q2_samples = []

                # Enable dropout
                self.critic1.train()
                self.critic2.train()

                for _ in range(self.args.droq_samples):
                    q1_samples.append(self.critic1(state_bev, state_proprio, action))
                    q2_samples.append(self.critic2(state_bev, state_proprio, action))

                # Average over samples
                q1 = torch.stack(q1_samples).mean(dim=0)
                q2 = torch.stack(q2_samples).mean(dim=0)
            else:
                # Standard SAC: single forward pass
                q1 = self.critic1(state_bev, state_proprio, action)
                q2 = self.critic2(state_bev, state_proprio, action)

            # MSE loss against target
            critic_loss = F.mse_loss(q1, next_q_value) + F.mse_loss(q2, next_q_value)

        # Backprop with AMP scaling
        self.critic_optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(critic_loss).backward()
            self.scaler.unscale_(self.critic_optimizer)
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=1.0)
            self.scaler.step(self.critic_optimizer)
            self.scaler.update()
        else:
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
        state_bev = batch['bev']
        state_proprio = batch['proprio']

        alpha = self.log_alpha.exp().item()

        # Use AMP for actor forward pass
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            # Re-compute features for actor (gradient flows through encoder)
            # Zero hidden state — replay buffer samples are independent
            mean, log_std, _, _ = self.actor(state_bev, state_proprio)

            log_std = torch.clamp(log_std, -20, 2)
            std = log_std.exp() + 1e-6

            dist = torch.distributions.Normal(mean, std)
            action_sample = dist.rsample()
            current_action = torch.tanh(action_sample)

            log_prob = dist.log_prob(action_sample).sum(dim=-1, keepdim=True)
            log_prob -= (2 * (np.log(2) - action_sample - F.softplus(-2 * action_sample))).sum(dim=1, keepdim=True)

            # Use critic to evaluate action (NO dropout, deterministic)
            # Disable dropout for actor evaluation
            self.critic1.eval()
            self.critic2.eval()

            q1_pi = self.critic1(state_bev, state_proprio, current_action)
            q2_pi = self.critic2(state_bev, state_proprio, current_action)
            min_q_pi = torch.min(q1_pi, q2_pi)

            actor_loss = ((alpha * log_prob) - min_q_pi).mean()

        # Backprop with AMP scaling
        self.actor_optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(actor_loss).backward()
            self.scaler.unscale_(self.actor_optimizer)
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
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

        self.alpha_optimizer.zero_grad(set_to_none=True)
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
                        if len(batch['bev']) > 0:
                            self.latest_bev_vis = batch['bev'][-1].copy()

                        # Process rewards and track episodes
                        # Separate Eval and Training data
                        is_eval_mask = batch.get('is_eval', np.zeros(len(batch['rewards']), dtype=bool))
                        rewards = batch['rewards']
                        dones = batch['dones']

                        for i, (r, d, is_eval) in enumerate(zip(rewards, dones, is_eval_mask)):
                            if is_eval:
                                self.current_eval_reward += r
                                if d:
                                    self.eval_episode_rewards.append(self.current_eval_reward)
                                    print(f"🧪 EVAL Episode completed: Reward = {self.current_eval_reward:.2f}")
                                    
                                    # Log immediate eval result
                                    self.writer.add_scalar('Reward/Eval_Episode', self.current_eval_reward, self.total_steps)
                                    
                                    # Check for new best model
                                    if self.current_eval_reward > self.best_eval_reward:
                                        self.best_eval_reward = self.current_eval_reward
                                        print(f"🏆 New Best Eval Reward: {self.best_eval_reward:.2f}")
                                        self.save_best_model()
                                    
                                    self.current_eval_reward = 0.0
                            else:
                                self.current_episode_reward += r
                                if d:
                                    self.episode_rewards.append(self.current_episode_reward)
                                    self.episode_count += 1
                                    print(f"📊 Training Episode {self.episode_count}: Reward = {self.current_episode_reward:.2f}")
                                    self.current_episode_reward = 0.0

                        # Add ONLY training data to replay buffer (filter out eval data)
                        # We don't want deterministic eval data in the buffer as it lacks exploration noise info
                        # (SAC assumes off-policy data, but best to keep it clean)
                        not_eval_indices = np.where(~is_eval_mask)[0]
                        
                        if len(not_eval_indices) > 0:
                            # Filter batch
                            train_batch = {
                                'bev': batch['bev'][not_eval_indices],
                                'proprio': batch['proprio'][not_eval_indices],
                                'actions': batch['actions'][not_eval_indices],
                                'rewards': batch['rewards'][not_eval_indices],
                                'dones': batch['dones'][not_eval_indices]
                            }
                            
                            # Add to replay buffer (thread-safe)
                            with self.lock:
                                self.buffer.add_batch(train_batch)

                            # Acknowledge message
                            await msg.ack()

                            print(f"📥 Added {len(train_batch['rewards'])} training steps (filtered from {len(batch['rewards'])})")
                        else:
                            # Batch was all eval data
                            await msg.ack()
                            print(f"📥 Processed batch of {len(batch['rewards'])} eval steps (not added to buffer)")

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
        
        # BC pre-training is now integrated into _training_loop
        # It will run automatically after buffer reaches bc_warmup_samples

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
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='Learning rate for Adam optimizer')
    parser.add_argument('--batch_size', type=int, default=256,  # Reduced from 768 for 3x more gradient steps/sec
                        help='Batch size for training (smaller = better for high UTD)')
    parser.add_argument('--buffer_size', type=int, default=50000,  # Reduced for Depth memory usage (240x424x4 = ~20GB at 50k)
                        help='Replay buffer capacity')

    # SAC specific
    parser.add_argument('--gamma', type=float, default=0.99,  # Increased from 0.98 (standard for robotics)
                        help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Soft target update rate')
    parser.add_argument('--checkpoint_dir', default='./checkpoints_sac')
    parser.add_argument('--log_dir', default='./logs_sac')

    # DroQ + UTD + Augmentation parameters - Updated for sample efficiency
    parser.add_argument('--droq_dropout', type=float, default=0.01,  # Increased from 0.005 for better regularization
                        help='Dropout rate for DroQ (0.0 to disable)')
    parser.add_argument('--droq_samples', type=int, default=2,  # Increased from 2 for better ensemble
                        help='Number of Q-network forward passes for DroQ (M)')
    parser.add_argument('--utd_ratio', type=int, default=2,  # High UTD for sample efficiency
                        help='Update-to-Data ratio (gradient steps per env step)')
    parser.add_argument('--actor_update_freq', type=int, default=4,  # Increased from 2 (update actor less frequently)
                        help='Update actor every N critic updates')
    parser.add_argument('--warmup_steps', type=int, default=10000,  # Was 2000
                        help='Minimum buffer size before training starts')
    parser.add_argument('--augment_data', action='store_true',
                        help='Enable data augmentation for occupancy grids')
    parser.add_argument('--gpu-buffer', action='store_true', help='Store replay buffer on GPU (WARNING: Requires huge VRAM for depth)')
    
    # Behavior Cloning Pre-training (integrated with buffer)
    parser.add_argument('--bc_warmup_samples', type=int, default=0,
                        help='Number of warmup samples to collect before BC pre-training (0 = skip BC)')
    parser.add_argument('--bc_epochs', type=int, default=50,
                        help='Number of epochs for BC pre-training')
    parser.add_argument('--bc_batch_size', type=int, default=64,
                        help='Batch size for BC pre-training')



    args = parser.parse_args()

    trainer = V620SACTrainer(args)

    # Run async event loop
    asyncio.run(trainer.run())
