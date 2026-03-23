#!/usr/bin/env python3
"""PPO Training Server for Remote Rover Training.

Receives complete PPO rollouts from the rover via ZeroMQ,
runs PPO training on GPU, and sends updated ONNX models back.

Flow:
1. Rover collects rollout (2048 steps) using RKNN NPU inference
2. Rover ships rollout to this server via ZMQ PUSH/PULL
3. Server recomputes log_probs + values from its PyTorch model
4. Server runs PPO update (10 epochs, FP16 AMP on GPU)
5. Server exports ONNX and publishes model update back via ZMQ PUB/SUB
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

import zmq
import zmq.asyncio

# RKNN toolkit for server-side conversion (optional)
try:
    from rknn.api import RKNN
    HAS_RKNN_TOOLKIT = True
except ImportError:
    HAS_RKNN_TOOLKIT = False

# Import model architectures
from model_architectures import UnifiedBEVPPOPolicy

# Import serialization utilities
from serialization_utils import (
    serialize_batch, deserialize_batch,
    serialize_model_update, deserialize_model_update,
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

        # ZMQ
        self.zmq_ctx = None
        self.pull_sock = None
        self.pub_sock = None
        self.zmq_pull_port = args.zmq_pull_port
        self.zmq_pub_port = args.zmq_pub_port
        self.lock = threading.Lock()
        self._server_start_time = time.time()

        # Load checkpoint if exists
        self._load_latest_checkpoint()

        # Re-export ONNX from loaded weights (don't bump version on restart)
        self._export_onnx(increment_version=(self.model_version == 0))

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
                opset_version=18,
                dynamic_axes={'bev': {0: 'batch'}, 'proprio': {0: 'batch'}}
            )

            # Ensure all weights are embedded inline (not in external .data file)
            # PyTorch's dynamo exporter sometimes creates external data files,
            # but we need a single self-contained .onnx for ZMQ transfer
            import onnx
            data_file = onnx_path + ".data"
            if os.path.exists(data_file):
                model = onnx.load(onnx_path, load_external_data=True)
                onnx.save(model, onnx_path, save_as_external_data=False)
                os.remove(data_file)

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

    def _convert_onnx_to_rknn(self, onnx_path: str) -> str:
        """Convert ONNX to RKNN on the server (FP16, no quantization)."""
        if not HAS_RKNN_TOOLKIT:
            return None

        rknn_path = onnx_path.replace('.onnx', '.rknn')
        try:
            rknn = RKNN(verbose=False)

            # Auto-detect inputs from ONNX
            import onnx as onnx_lib
            model = onnx_lib.load(onnx_path)
            input_names = []
            input_sizes = []
            mean_values = []
            std_values = []
            for inp in model.graph.input:
                name = inp.name
                shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
                shape = [1 if d == 0 else d for d in shape]
                input_names.append(name)
                input_sizes.append(shape)
                n_channels = shape[1] if len(shape) == 4 else shape[-1]
                mean_values.append([0] * n_channels)
                std_values.append([1] * n_channels)

            ret = rknn.config(
                mean_values=mean_values,
                std_values=std_values,
                target_platform='rk3588',
                optimization_level=3,
            )
            if ret != 0:
                print(f"RKNN config failed: {ret}")
                return None

            ret = rknn.load_onnx(
                model=onnx_path,
                inputs=input_names,
                input_size_list=input_sizes,
            )
            if ret != 0:
                print(f"RKNN load_onnx failed: {ret}")
                return None

            ret = rknn.build(do_quantization=False)
            if ret != 0:
                print(f"RKNN build failed: {ret}")
                return None

            ret = rknn.export_rknn(rknn_path)
            if ret != 0:
                print(f"RKNN export failed: {ret}")
                return None

            rknn_size = os.path.getsize(rknn_path)
            print(f"RKNN converted: {rknn_path} ({rknn_size} bytes)")
            return rknn_path

        except Exception as e:
            print(f"RKNN conversion failed: {e}")
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
        # pin_memory() registers CPU pages with CUDA for fast DMA transfer
        if self.device.type == 'cuda':
            bev = torch.from_numpy(rollout['bev'].copy()).float().pin_memory().to(self.device, non_blocking=True)
            proprio = torch.from_numpy(rollout['proprio'].copy()).float().pin_memory().to(self.device, non_blocking=True)
            actions = torch.from_numpy(rollout['actions'].copy()).float().pin_memory().to(self.device, non_blocking=True)
            rewards = torch.from_numpy(rollout['rewards'].copy()).float().pin_memory().to(self.device, non_blocking=True)
            dones = torch.from_numpy(rollout['dones'].copy()).float().pin_memory().to(self.device, non_blocking=True)
            torch.cuda.synchronize()  # ensure transfers complete before training
        else:
            bev = torch.from_numpy(rollout['bev'].copy()).float()
            proprio = torch.from_numpy(rollout['proprio'].copy()).float()
            actions = torch.from_numpy(rollout['actions'].copy()).float()
            rewards = torch.from_numpy(rollout['rewards'].copy()).float()
            dones = torch.from_numpy(rollout['dones'].copy()).float()

        # Track episodes from rollout (count on GPU, single sync)
        self.episode_count += int(dones.sum().item())

        # 1. Recompute log_probs and values — single forward pass
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

    # ========== ZMQ Communication ==========

    def setup_zmq(self):
        """Bind ZMQ sockets for rover communication."""
        self.zmq_ctx = zmq.asyncio.Context()

        # PULL socket — receives rollouts from rover
        self.pull_sock = self.zmq_ctx.socket(zmq.PULL)
        self.pull_sock.bind(f"tcp://*:{self.zmq_pull_port}")
        print(f"ZMQ PULL bound on tcp://*:{self.zmq_pull_port}")

        # XPUB socket — publishes models + status, notifies on new subscriptions
        self.pub_sock = self.zmq_ctx.socket(zmq.XPUB)
        self.pub_sock.setsockopt(zmq.XPUB_VERBOSE, 1)  # notify on every subscribe, not just first
        self.pub_sock.bind(f"tcp://*:{self.zmq_pub_port}")
        print(f"ZMQ XPUB bound on tcp://*:{self.zmq_pub_port}")

    async def consume_rollouts(self):
        """Consume PPO rollouts from rover and train."""
        print("Waiting for rollouts...")

        while True:
            try:
                data = await self.pull_sock.recv()
                print(f"Received rollout: {len(data)} bytes")
                rollout = deserialize_batch(data)

                n_steps = len(rollout['rewards'])
                print(f"Rollout: {n_steps} steps")

                # Train PPO
                train_result = self.train_on_rollout(rollout)

                # Publish updated model
                if train_result['onnx_path'] and os.path.exists(train_result['onnx_path']):
                    await self._publish_model(train_result)

                print(f"Rollout processed, model v{self.model_version} published")

            except Exception as e:
                print(f"Error processing rollout: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1.0)

    async def _publish_model(self, train_result):
        """Publish model to rover via ZMQ XPUB. Includes RKNN if toolkit available."""
        try:
            onnx_path = train_result['onnx_path']
            with open(onnx_path, 'rb') as f:
                onnx_bytes = f.read()

            # Convert to RKNN on server if toolkit available
            rknn_bytes = None
            rknn_path = self._convert_onnx_to_rknn(onnx_path)
            if rknn_path and os.path.exists(rknn_path):
                with open(rknn_path, 'rb') as f:
                    rknn_bytes = f.read()

            import msgpack
            msg = {
                "version": self.model_version,
                "onnx_bytes": onnx_bytes,
                "timestamp": time.time(),
                "train_stats": {
                    'policy_loss': train_result['policy_loss'],
                    'value_loss': train_result['value_loss'],
                    'entropy': train_result['entropy'],
                    'train_time': train_result['train_time'],
                },
            }
            if rknn_bytes:
                msg["rknn_bytes"] = rknn_bytes

            model_msg = msgpack.packb(msg)
            await self.pub_sock.send_multipart([b"model", model_msg])

            if rknn_bytes:
                print(f"Published model v{self.model_version} (ONNX: {len(onnx_bytes)}B, RKNN: {len(rknn_bytes)}B)")
            else:
                print(f"Published model v{self.model_version} (ONNX: {len(onnx_bytes)}B, no RKNN)")

        except Exception as e:
            print(f"Model publish failed: {e}")

    async def publish_status(self):
        """Periodically publish training status via ZMQ XPUB."""
        while True:
            try:
                status_msg = serialize_status(
                    status='ready',
                    model_version=self.model_version,
                    total_steps=self.total_steps,
                    update_count=self.update_count,
                )
                await self.pub_sock.send_multipart([b"status", status_msg])
            except Exception:
                pass
            await asyncio.sleep(5.0)

    async def watch_subscriptions(self):
        """Listen for new SUB connections on XPUB and send them the current model."""
        while True:
            try:
                event = await self.pub_sock.recv()
                # XPUB subscription events: first byte 1=subscribe, 0=unsubscribe
                # remaining bytes = topic
                if len(event) > 0 and event[0] == 1:
                    topic = event[1:].decode('utf-8', errors='replace')
                    print(f"New subscriber for topic: '{topic}'")
                    if topic == "model":
                        print(f"Rover subscribed — sending current model v{self.model_version}")
                        await asyncio.sleep(0.2)  # brief settle
                        onnx_path = os.path.join(self.args.checkpoint_dir, "latest_actor.onnx")
                        if os.path.exists(onnx_path):
                            await self._publish_model({
                                'onnx_path': onnx_path,
                                'policy_loss': 0.0,
                                'value_loss': 0.0,
                                'entropy': 0.0,
                                'train_time': 0.0,
                            })
            except Exception as e:
                print(f"Subscription watch error: {e}")
                await asyncio.sleep(1.0)

    async def _publish_initial_model(self):
        """Publish the initial ONNX model so the rover can start immediately."""
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
        self.setup_zmq()

        # Small delay to let PUB socket settle before publishing
        await asyncio.sleep(0.5)

        # Publish initial model so rover picks it up immediately
        await self._publish_initial_model()

        # Start background tasks
        asyncio.create_task(self.publish_status())
        asyncio.create_task(self.watch_subscriptions())

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
    parser.add_argument('--zmq_pull_port', type=int, default=5555)
    parser.add_argument('--zmq_pub_port', type=int, default=5556)
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
