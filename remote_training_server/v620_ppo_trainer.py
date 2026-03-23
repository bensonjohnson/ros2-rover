#!/usr/bin/env python3
"""V620 PPO Training Server for Remote Rover Training.

Receives complete PPO rollouts from the rover via NATS JetStream,
runs PPO training on GPU, and sends updated ONNX models back.

Flow:
1. Rover collects rollout (2048 steps) using RKNN NPU inference
2. Rover ships rollout to this server via NATS
3. Server recomputes log_probs + values from its PyTorch model
4. Server runs PPO update (10 epochs, FP16 AMP on GPU)
5. Server exports ONNX and publishes model update back via NATS
6. Rover downloads ONNX, converts to RKNN, loads on NPU
7. Rover resumes driving with updated policy

On-policy correctness is preserved because:
- The rover stops after each rollout
- The server recomputes log_probs from the CURRENT PyTorch weights
- PPO ratio is always PyTorch-vs-PyTorch (no RKNN quantization in ratio)
"""

import os
import sys
import time
import json
import argparse
import threading
import asyncio
import signal
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import nats
from nats.js.api import StreamConfig

# Import model architectures
from model_architectures import UnifiedBEVPPOPolicy

# Import serialization utilities
from serialization_utils import (
    serialize_batch, deserialize_batch,
    serialize_model_update, deserialize_model_update,
    serialize_model_chunks, deserialize_model_chunk,
    serialize_metadata, deserialize_metadata,
    serialize_status, deserialize_status
)

# Import dashboard
try:
    from dashboard_app import TrainingDashboard
    HAS_DASHBOARD = True
except ImportError:
    HAS_DASHBOARD = False

# Import Phase Manager
try:
    from tractor_bringup.phase_manager import PhaseManager
    HAS_PHASE_MANAGER = True
except ImportError:
    HAS_PHASE_MANAGER = False
    print("PhaseManager not available - phase-based curriculum disabled")


@torch.jit.script
def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                last_value: torch.Tensor, gamma: float, gae_lambda: float) -> torch.Tensor:
    """Compute GAE advantages with a compiled loop on GPU (no Python overhead)."""
    n = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.tensor(0.0, device=rewards.device, dtype=rewards.dtype)
    for t in range(n - 1, -1, -1):
        next_val = last_value if t == n - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * not_done - values[t]
        last_gae = delta + gamma * gae_lambda * not_done * last_gae
        advantages[t] = last_gae
    return advantages


class V620PPOTrainer:
    """PPO Trainer optimized for V620 ROCm / CUDA GPU."""

    def __init__(self, args):
        self.args = args

        # Device setup
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Using GPU: {gpu_name} ({total_mem:.1f}GB)")

            self._is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
            self._is_blackwell = any(k in gpu_name for k in ('B200', 'B100', 'GB10', 'GB200', 'Blackwell'))

            # GPU sanity test
            try:
                _t = torch.randn(256, 512, device='cuda')
                _r = torch.mm(_t, _t.t())
                torch.cuda.synchronize()
                del _t, _r
                torch.cuda.empty_cache()
                print("GPU sanity test passed")
            except Exception as e:
                print(f"GPU SANITY TEST FAILED: {e}")
                sys.exit(1)

            torch.backends.cudnn.benchmark = True

            # AMP — backend-specific dtype and scaler
            self.use_amp = True
            if self._is_rocm:
                # ROCm V620: FP16 via HIP, conservative scaler for RDNA 2
                self.amp_dtype = torch.float16
                self.scaler = torch.amp.GradScaler('cuda', init_scale=1024.0, growth_factor=1.5)
                print("AMP enabled (FP16) - ROCm")
            elif self._is_blackwell:
                # Blackwell (DGX Spark): native BF16, no scaler needed
                # BF16 has same exponent range as FP32 → no overflow → no GradScaler
                # TF32 matmuls reduce bandwidth pressure on unified memory
                self.amp_dtype = torch.bfloat16
                self.scaler = None
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print(f"AMP enabled (BF16) - Blackwell ({gpu_name})")
                print("  TF32 matmuls enabled (reduces memory bandwidth)")
            else:
                # Generic CUDA (H100, A100, etc): FP16 with scaler
                self.amp_dtype = torch.float16
                self.scaler = torch.amp.GradScaler('cuda', init_scale=65536.0, growth_factor=2.0)
                # H100/A100 also benefit from TF32
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("AMP enabled (FP16) - CUDA")
        else:
            self.device = torch.device('cpu')
            self.use_amp = False
            self.amp_dtype = torch.float32
            self.scaler = None
            print("Using CPU (slower)")

        # Model
        self.proprio_dim = 6
        self.action_dim = 2

        self.policy = UnifiedBEVPPOPolicy(
            action_dim=self.action_dim,
            proprio_dim=self.proprio_dim
        ).to(self.device)

        # Optimizer — separate param groups so value head can learn faster
        # while policy stays stable during early value calibration
        policy_params = (
            list(self.policy.policy_net.parameters()) +
            [self.policy.log_std]
        )
        value_params = list(self.policy.value_net.parameters())
        shared_params = (
            list(self.policy.bev_encoder.parameters()) +
            list(self.policy.proprio_encoder.parameters())
        )
        self.optimizer = optim.Adam([
            {'params': shared_params, 'lr': args.lr},
            {'params': policy_params, 'lr': args.lr},
            {'params': value_params, 'lr': args.lr * 3},  # Value head learns 3x faster
        ], eps=1e-5)

        # PPO hyperparameters
        self.clip_eps = args.clip_eps
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.update_epochs = args.update_epochs
        self.mini_batch_size = args.mini_batch_size
        self.value_coef = 0.5
        self.vf_clip = 10.0  # Clip value loss to prevent huge gradients
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.3
        self.target_kl = args.target_kl

        # State
        self.total_steps = 0
        self.update_count = 0
        self.model_version = 0
        self.best_reward = -float('inf')

        # TensorBoard
        os.makedirs(args.log_dir, exist_ok=True)
        self.writer = SummaryWriter(args.log_dir)

        # Checkpoint dir
        os.makedirs(args.checkpoint_dir, exist_ok=True)

        # Episode tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_count = 0

        # Phase Manager
        if HAS_PHASE_MANAGER:
            self.phase_manager = PhaseManager(initial_phase='exploration')
            print("PhaseManager initialized")
        else:
            self.phase_manager = None

        # NATS
        self.nc = None
        self.js = None
        self.nats_server = args.nats_server
        self.lock = threading.Lock()
        self._server_start_time = time.time()

        # Load checkpoint if exists
        self._load_latest_checkpoint()

        # Export initial model (v1 so rover always picks it up on fresh start)
        self._export_onnx(increment_version=True)

        param_count = sum(p.numel() for p in self.policy.parameters())
        print(f"PPO Trainer initialized: {param_count:,} params")
        print(f"  LR: {args.lr}, Clip: {args.clip_eps}, Epochs: {args.update_epochs} (KL stop: {args.target_kl})")
        print(f"  Mini-batch: {args.mini_batch_size}, Gamma: {args.gamma}")

    # ========== Checkpoint Management ==========

    def _load_latest_checkpoint(self):
        checkpoints = list(Path(self.args.checkpoint_dir).glob('ppo_step_*.pt'))
        latest_path = Path(self.args.checkpoint_dir) / 'latest.pt'

        if latest_path.exists():
            ckpt = torch.load(latest_path, map_location=self.device, weights_only=False)
        elif checkpoints:
            latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
            ckpt = torch.load(latest, map_location=self.device, weights_only=False)
        else:
            print("No checkpoint found. Starting fresh.")
            return

        self.policy.load_state_dict(ckpt['policy_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.total_steps = ckpt.get('total_steps', 0)
        self.update_count = ckpt.get('update_count', 0)
        self.model_version = ckpt.get('model_version', self.update_count)
        self.episode_count = ckpt.get('episode_count', 0)
        print(f"Restored: {self.total_steps} steps, v{self.model_version}")

    def _save_checkpoint(self):
        state = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'update_count': self.update_count,
            'model_version': self.model_version,
            'episode_count': self.episode_count,
        }
        ckpt_path = os.path.join(self.args.checkpoint_dir, f'ppo_step_{self.total_steps}.pt')
        latest_path = os.path.join(self.args.checkpoint_dir, 'latest.pt')
        torch.save(state, ckpt_path)
        torch.save(state, latest_path)
        print(f"Checkpoint saved: {ckpt_path}")

    def _export_onnx(self, increment_version=True):
        """Export policy to ONNX format."""
        try:
            onnx_path = os.path.join(self.args.checkpoint_dir, "latest_actor.onnx")
            self.policy.eval()
            dummy_bev = torch.zeros(1, 2, 128, 128, device=self.device)
            dummy_proprio = torch.zeros(1, self.proprio_dim, device=self.device)

            torch.onnx.export(
                self.policy,
                (dummy_bev, dummy_proprio),
                onnx_path,
                input_names=['bev', 'proprio'],
                output_names=['action_mean', 'log_std', 'value'],
                opset_version=12,
                dynamic_axes={'bev': {0: 'batch'}, 'proprio': {0: 'batch'}}
            )
            self.policy.train()

            if increment_version:
                self.model_version += 1

            file_size = os.path.getsize(onnx_path)
            print(f"Exported ONNX: {onnx_path} ({file_size} bytes)")
            return onnx_path
        except Exception as e:
            print(f"ONNX export failed: {e}")
            self.policy.train()
            return None

    # ========== PPO Training ==========

    def train_on_rollout(self, rollout: dict):
        """Run PPO training on a received rollout.

        Args:
            rollout: dict with keys: bev, proprio, actions, rewards, dones
                     bev: (N, 2, 128, 128) float32 [0,1]
                     proprio: (N, 6) float32
                     actions: (N, 2) float32
                     rewards: (N,) float32
                     dones: (N,) bool
        """
        t0 = time.time()
        n = len(rollout['rewards'])
        self.total_steps += n
        _LOG2 = np.log(2.0)  # cached constant for tanh-squashed log_prob
        print(f"\n--- PPO Update #{self.update_count + 1} ({n} steps) ---")

        # Convert to tensors on GPU (.copy() needed — deserialization yields read-only arrays)
        bev = torch.from_numpy(rollout['bev'].copy()).float().pin_memory().to(self.device, non_blocking=True)
        proprio = torch.from_numpy(rollout['proprio'].copy()).float().pin_memory().to(self.device, non_blocking=True)
        actions = torch.from_numpy(rollout['actions'].copy()).float().pin_memory().to(self.device, non_blocking=True)
        rewards = torch.from_numpy(rollout['rewards'].copy()).float().pin_memory().to(self.device, non_blocking=True)
        dones = torch.from_numpy(rollout['dones'].copy()).float().pin_memory().to(self.device, non_blocking=True)

        # Track episodes from rollout (count on GPU, single sync)
        self.episode_count += int(dones.sum().item())

        # 1. Recompute log_probs and values — single forward pass (fits in VRAM)
        print("  Recomputing log_probs from PyTorch model...")
        self.policy.eval()
        with torch.no_grad():
            if self.use_amp:
                with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                    action_mean, log_std, values = self.policy(bev, proprio)
            else:
                action_mean, log_std, values = self.policy(bev, proprio)

            action_mean = action_mean.float()
            log_std = log_std.float()
            values = values.float()

            std = log_std.exp().clamp(min=1e-6, max=2.0)
            acts_clamped = actions.clamp(-0.999, 0.999)
            raw_acts = torch.atanh(acts_clamped)
            # Inline Normal log_prob: -0.5*((x-mu)/std)^2 - log(std) - 0.5*log(2*pi)
            var = std * std
            old_log_probs = (-0.5 * ((raw_acts - action_mean) ** 2) / var
                             - std.log() - 0.9189385332046727).sum(dim=-1)
            # Tanh squash correction
            old_log_probs -= (2 * (_LOG2 - raw_acts - F.softplus(-2 * raw_acts))).sum(dim=-1)

        self.policy.train()

        # 2. Compute GAE (JIT-compiled loop on GPU — no CPU transfer needed)
        print("  Computing GAE...")
        with torch.no_grad():
            last_bev = bev[-1:]
            last_pro = proprio[-1:]
            if self.use_amp:
                with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                    _, _, last_value = self.policy(last_bev, last_pro)
            else:
                _, _, last_value = self.policy(last_bev, last_pro)
            last_value = last_value.float().squeeze()

            advantages = compute_gae(rewards, values, dones, last_value,
                                     self.gamma, self.gae_lambda)
        returns = advantages + values

        # Normalize returns (keeps value targets in reasonable range)
        ret_mean = returns.mean()
        ret_std = returns.std() + 1e-8
        returns = (returns - ret_mean) / ret_std

        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # 3. PPO epochs
        print(f"  Training: {self.update_epochs} epochs, mini-batch {self.mini_batch_size}")
        # Accumulate on GPU — single sync at end instead of per-minibatch .item()
        total_policy_loss = torch.tensor(0.0, device=self.device)
        total_value_loss = torch.tensor(0.0, device=self.device)
        total_entropy = torch.tensor(0.0, device=self.device)
        n_updates = 0

        early_stopped = False
        for epoch in range(self.update_epochs):
            epoch_kl = torch.tensor(0.0, device=self.device)
            epoch_count = 0
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, self.mini_batch_size):
                end = min(start + self.mini_batch_size, n)
                idx = indices[start:end]

                bev_b = bev[idx]
                pro_b = proprio[idx]
                act_b = actions[idx]
                ret_b = returns[idx]
                adv_b = advantages[idx]
                old_lp_b = old_log_probs[idx]
                old_val_b = values[idx]

                if self.use_amp:
                    with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                        action_mean, log_std, val = self.policy(bev_b, pro_b)
                        action_mean = action_mean.float()
                        log_std = log_std.float()
                        val = val.float()
                else:
                    action_mean, log_std, val = self.policy(bev_b, pro_b)

                std = log_std.exp().clamp(min=1e-6, max=2.0)
                var = std * std

                acts_clamped = act_b.clamp(-0.999, 0.999)
                raw_actions = torch.atanh(acts_clamped)

                # Inline Normal log_prob (avoids torch.distributions Python overhead)
                new_log_probs = (-0.5 * ((raw_actions - action_mean) ** 2) / var
                                 - std.log() - 0.9189385332046727).sum(dim=-1)
                # Tanh squash correction
                new_log_probs -= (2 * (_LOG2 - raw_actions - F.softplus(-2 * raw_actions))).sum(dim=-1)

                # Approx KL for early stopping: mean((ratio - 1) - log(ratio))
                log_ratio = new_log_probs - old_lp_b
                epoch_kl += (log_ratio.exp() - 1.0 - log_ratio).mean().detach()
                epoch_count += 1

                # PPO clipped objective
                ratio = log_ratio.exp()
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # Clipped value loss (prevents value network from destabilizing policy)
                val_clipped = old_val_b + (val - old_val_b).clamp(-self.vf_clip, self.vf_clip)
                vl_unclipped = (val - ret_b).pow(2)
                vl_clipped = (val_clipped - ret_b).pow(2)
                value_loss = self.value_coef * torch.max(vl_unclipped, vl_clipped).mean()

                # Inline Normal entropy: 0.5 * log(2*pi*e*var) = 0.5 + 0.5*log(2*pi) + log(std)
                entropy = (0.5 + 0.9189385332046727 + std.log()).sum(dim=-1).mean()

                loss = policy_loss + value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)

                if self.use_amp and self.scaler is not None:
                    # FP16 path (ROCm / generic CUDA) — needs GradScaler
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # BF16 (Blackwell) or CPU — no scaler needed
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                total_policy_loss += policy_loss.detach()
                total_value_loss += value_loss.detach()
                total_entropy += entropy.detach()
                n_updates += 1

            # KL early stopping check (single sync per epoch, not per mini-batch)
            approx_kl = (epoch_kl / max(epoch_count, 1)).item()
            if approx_kl > self.target_kl:
                print(f"  Early stop at epoch {epoch + 1}/{self.update_epochs} (KL={approx_kl:.4f} > {self.target_kl})")
                early_stopped = True
                break

        dt = time.time() - t0
        self.update_count += 1

        # Single GPU→CPU sync for all three metrics
        avg_pl = (total_policy_loss / max(n_updates, 1)).item()
        avg_vl = (total_value_loss / max(n_updates, 1)).item()
        avg_ent = (total_entropy / max(n_updates, 1)).item()

        epochs_used = epoch + 1
        print(f"  Done in {dt:.1f}s | {epochs_used}/{self.update_epochs} epochs | PL: {avg_pl:.4f} | VL: {avg_vl:.4f} | Ent: {avg_ent:.4f}")

        # TensorBoard
        self.writer.add_scalar('loss/policy', avg_pl, self.update_count)
        self.writer.add_scalar('loss/value', avg_vl, self.update_count)
        self.writer.add_scalar('loss/entropy', avg_ent, self.update_count)
        self.writer.add_scalar('training/total_steps', self.total_steps, self.update_count)
        self.writer.add_scalar('training/model_version', self.model_version, self.update_count)
        self.writer.add_scalar('training/train_time_s', dt, self.update_count)
        self.writer.add_scalar('training/epochs_used', epochs_used, self.update_count)
        self.writer.add_scalar('training/approx_kl', approx_kl, self.update_count)

        # Save checkpoint
        if self.update_count % self.args.checkpoint_interval == 0:
            self._save_checkpoint()

        # Export ONNX
        onnx_path = self._export_onnx()

        return {
            'policy_loss': avg_pl,
            'value_loss': avg_vl,
            'entropy': avg_ent,
            'train_time': dt,
            'model_version': self.model_version,
            'onnx_path': onnx_path,
        }

    # ========== NATS Communication ==========

    async def setup_nats(self):
        """Connect to NATS and set up streams."""
        print(f"Connecting to NATS at {self.nats_server}...")

        async def on_disconnected():
            print("NATS disconnected")

        async def on_reconnected():
            print("NATS reconnected")

        self.nc = await nats.connect(
            servers=[self.nats_server],
            name="v620-ppo-trainer",
            max_reconnect_attempts=-1,
            reconnect_time_wait=2,
            ping_interval=20,
            max_outstanding_pings=3,
            disconnected_cb=on_disconnected,
            reconnected_cb=on_reconnected,
        )

        self.js = self.nc.jetstream()
        print("Connected to NATS")

        await self._ensure_streams()

    async def _ensure_streams(self):
        """Create NATS JetStream streams if they don't exist."""
        # PPO Rollout stream
        try:
            await self.js.add_stream(StreamConfig(
                name="ROVER_PPO_ROLLOUTS",
                subjects=["rover.ppo.rollout"],
                retention="limits",
                max_msgs=100,
                max_bytes=10 * 1024 * 1024 * 1024,  # 10 GB
                max_age=604800,  # 7 days
                max_msg_size=200 * 1024 * 1024,  # 200 MB
                storage="file",
                discard="old",
            ))
            print("ROVER_PPO_ROLLOUTS stream ready")
        except Exception as e:
            if "stream name already in use" not in str(e).lower():
                print(f"Stream setup: {e}")

        # PPO Model stream
        try:
            await self.js.add_stream(StreamConfig(
                name="ROVER_PPO_MODELS",
                subjects=["models.ppo.update", "models.ppo.update.chunk", "models.ppo.metadata"],
                retention="limits",
                max_msgs=100,
                max_bytes=2 * 1024 * 1024 * 1024,  # 2 GB
                max_age=2592000,  # 30 days
                max_msg_size=50 * 1024 * 1024,  # 50 MB
                storage="file",
                discard="old",
            ))
            print("ROVER_PPO_MODELS stream ready")
        except Exception as e:
            if "stream name already in use" not in str(e).lower():
                print(f"Stream setup: {e}")

        # PPO Control stream
        try:
            await self.js.add_stream(StreamConfig(
                name="ROVER_PPO_CONTROL",
                subjects=["server.ppo.status"],
                retention="limits",
                max_msgs=10000,
                max_bytes=100 * 1024 * 1024,
                max_age=86400,  # 24 hours
                max_msg_size=1 * 1024 * 1024,
                storage="file",
                discard="old",
            ))
            print("ROVER_PPO_CONTROL stream ready")
        except Exception as e:
            if "stream name already in use" not in str(e).lower():
                print(f"Stream setup: {e}")

    async def consume_rollouts(self):
        """Consume PPO rollouts from rover and train."""
        print("Starting rollout consumer...")

        psub = await self.js.pull_subscribe(
            subject="rover.ppo.rollout",
            durable="ppo_trainer"
        )

        while True:
            try:
                msgs = await psub.fetch(batch=1, timeout=1.0)
                for msg in msgs:
                    try:
                        print(f"Received rollout: {len(msg.data)} bytes")
                        rollout = deserialize_batch(msg.data)

                        # Skip stale rollouts from before this server started
                        rollout_ts = rollout.get('metadata', {}).get('timestamp', 0)
                        if rollout_ts and rollout_ts < self._server_start_time:
                            age = self._server_start_time - rollout_ts
                            print(f"Skipping stale rollout (sent {age:.0f}s before server start)")
                            await msg.ack()
                            continue

                        n_steps = len(rollout['rewards'])
                        print(f"Rollout: {n_steps} steps")

                        # Train PPO
                        train_result = self.train_on_rollout(rollout)

                        # Publish updated model
                        if train_result['onnx_path'] and os.path.exists(train_result['onnx_path']):
                            await self._publish_model(train_result)

                        # Acknowledge
                        await msg.ack()
                        print(f"Rollout processed, model v{self.model_version} published")

                    except Exception as e:
                        print(f"Error processing rollout: {e}")
                        import traceback
                        traceback.print_exc()
                        await msg.ack()

            except nats.errors.TimeoutError:
                await asyncio.sleep(0.1)
                continue
            except Exception as e:
                print(f"Consumer error: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1.0)

    async def _publish_model(self, train_result):
        """Publish updated ONNX model to NATS in chunks for rover to download."""
        try:
            onnx_path = train_result['onnx_path']
            with open(onnx_path, 'rb') as f:
                onnx_bytes = f.read()

            # Publish model as chunks (4MB each to stay under NATS max_payload)
            chunks = serialize_model_chunks(onnx_bytes, self.model_version)
            for i, chunk in enumerate(chunks):
                await self.js.publish(
                    subject="models.ppo.update.chunk",
                    payload=chunk,
                    timeout=30.0
                )
            print(f"Published model v{self.model_version} ({len(onnx_bytes)} bytes, {len(chunks)} chunks)")

            # Publish metadata with training stats + chunk info
            import msgpack
            metadata = {
                "latest_version": self.model_version,
                "timestamp": time.time(),
                "total_chunks": len(chunks),
                "onnx_size": len(onnx_bytes),
                "train_stats": {
                    'policy_loss': train_result['policy_loss'],
                    'value_loss': train_result['value_loss'],
                    'entropy': train_result['entropy'],
                    'train_time': train_result['train_time'],
                },
            }
            await self.js.publish(
                subject="models.ppo.metadata",
                payload=msgpack.packb(metadata),
                timeout=10.0
            )

        except Exception as e:
            print(f"Model publish failed: {e}")

    async def publish_status(self):
        """Periodically publish training status."""
        while True:
            try:
                status_msg = serialize_status(
                    status='ready',
                    model_version=self.model_version,
                    total_steps=self.total_steps,
                    update_count=self.update_count,
                )
                await self.js.publish(
                    subject="server.ppo.status",
                    payload=status_msg,
                    timeout=5.0
                )
            except Exception:
                pass
            await asyncio.sleep(5.0)

    async def _publish_initial_model(self):
        """Publish the initial (v0) ONNX model so the rover can start immediately."""
        onnx_path = os.path.join(self.args.checkpoint_dir, "latest_actor.onnx")
        if not os.path.exists(onnx_path):
            return
        train_result = {
            'onnx_path': onnx_path,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'train_time': 0.0,
        }
        await self._publish_model(train_result)
        print(f"Published initial model v{self.model_version}")

    async def run(self):
        """Main async loop."""
        await self.setup_nats()

        # Publish initial model so rover picks it up immediately
        await self._publish_initial_model()

        # Start background tasks
        asyncio.create_task(self.publish_status())

        # Main loop: consume rollouts and train
        await self.consume_rollouts()

    def start(self):
        """Entry point."""
        # Signal handler
        def _shutdown(signum, frame):
            print("\nShutdown signal received!")
            self._save_checkpoint()
            self.writer.close()
            print("Shutdown complete")
            sys.exit(0)

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        # Run async loop
        asyncio.run(self.run())


def main():
    parser = argparse.ArgumentParser(description='V620 PPO Training Server')
    parser.add_argument('--nats_server', type=str, default='nats://nats.gokickrocks.org:4222')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_ppo')
    parser.add_argument('--log_dir', type=str, default='./logs_ppo')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--clip_eps', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--update_epochs', type=int, default=20)
    parser.add_argument('--mini_batch_size', type=int, default=256)
    parser.add_argument('--target_kl', type=float, default=0.04)
    parser.add_argument('--checkpoint_interval', type=int, default=5)
    args = parser.parse_args()

    trainer = V620PPOTrainer(args)
    trainer.start()


if __name__ == '__main__':
    main()
