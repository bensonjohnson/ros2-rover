#!/usr/bin/env python3
"""PPO BEV Training Server for Local Rover Training.

This trainer uses the Unified BEV architecture (LiDAR + Depth fusion)
for PPO policy learning on the rover. Supports:
- Local training with checkpoint saving every 200 steps
- Automatic ONNX and RKNN export
- NATS communication for receiving BEV observations
"""

import os
import sys
import time
import json
import argparse
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque

# Import model architectures
from model_architectures import UnifiedBEVPPOPolicy

# Import NATS for communication
import nats
from nats.js.api import StreamConfig, ConsumerConfig


class PPOBuffer:
    """PPO Experience Buffer for on-policy learning."""
    
    def __init__(self, capacity: int, proprio_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Unified BEV: 2x128x128 uint8 (quantized 0-255)
        self.bev = torch.zeros((capacity, 2, 128, 128), dtype=torch.uint8)
        self.proprio = torch.zeros((capacity, proprio_dim), dtype=torch.float32)
        self.actions = torch.zeros((capacity, 2), dtype=torch.float32)
        self.rewards = torch.zeros((capacity,), dtype=torch.float32)
        self.dones = torch.zeros((capacity,), dtype=torch.bool)
        self.log_probs = torch.zeros((capacity,), dtype=torch.float32)
        self.values = torch.zeros((capacity,), dtype=torch.float32)
        
    def add_batch(self, batch_data: Dict):
        """Add a batch of transitions."""
        batch_size = len(batch_data['rewards'])
        
        if self.ptr + batch_size > self.capacity:
            first_part = self.capacity - self.ptr
            second_part = batch_size - first_part
            
            self._add_slice(batch_data, 0, first_part, self.ptr)
            self._add_slice(batch_data, first_part, batch_size, 0)
            
            self.ptr = second_part
            self.size = self.capacity
        else:
            self._add_slice(batch_data, 0, batch_size, self.ptr)
            self.ptr += batch_size
            self.size = min(self.size + batch_size, self.capacity)
            
    def _add_slice(self, data, start_idx, end_idx, buffer_idx):
        """Helper to add a slice of data to buffer."""
        length = end_idx - start_idx
        
        # Quantize BEV to uint8 (0-1 -> 0-255)
        bev_slice = torch.as_tensor(data['bev'][start_idx:end_idx].copy())
        if bev_slice.ndim == 3:
            bev_slice = bev_slice.unsqueeze(1)
        bev_slice = (bev_slice * 255.0).to(torch.uint8)
        
        self.bev[buffer_idx:buffer_idx+length] = bev_slice
        self.proprio[buffer_idx:buffer_idx+length] = torch.as_tensor(data['proprio'][start_idx:end_idx].copy())
        self.actions[buffer_idx:buffer_idx+length] = torch.as_tensor(data['actions'][start_idx:end_idx].copy())
        self.rewards[buffer_idx:buffer_idx+length] = torch.as_tensor(data['rewards'][start_idx:end_idx].copy())
        self.dones[buffer_idx:buffer_idx+length] = torch.as_tensor(data['dones'][start_idx:end_idx].copy())
        self.log_probs[buffer_idx:buffer_idx+length] = torch.as_tensor(data['log_probs'][start_idx:end_idx].copy())
        self.values[buffer_idx:buffer_idx+length] = torch.as_tensor(data['values'][start_idx:end_idx].copy())
        
    def get_batch(self, indices):
        """Get batch by indices."""
        bev_batch = self.bev[indices].to(self.device).float() / 255.0
        
        return {
            'bev': bev_batch,
            'proprio': self.proprio[indices].to(self.device),
            'actions': self.actions[indices].to(self.device),
            'rewards': self.rewards[indices].to(self.device),
            'dones': self.dones[indices].to(self.device),
            'log_probs': self.log_probs[indices].to(self.device),
            'values': self.values[indices].to(self.device),
        }

    def compute_gae(self, policy, gamma=0.99, lam=0.95):
        """Compute Generalized Advantage Estimation (GAE)."""
        # Compute values for all states
        with torch.no_grad():
            bev = self.bev[:self.size].to(self.device).float() / 255.0
            proprio = self.proprio[:self.size].to(self.device)
            _, _, values = policy(bev, proprio)
        
        # Compute GAE
        advantages = torch.zeros(self.size, device=self.device)
        last_gae_lam = 0
        
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_value = 0.0
            else:
                next_value = values[t+1]
                
            delta = self.rewards[t] + gamma * next_value * (1.0 - self.dones[t].float()) - values[t]
            last_gae_lam = delta + gamma * lam * (1.0 - self.dones[t].float()) * last_gae_lam
            advantages[t] = last_gae_lam
            
        returns = advantages + values
        return advantages, returns

    def clear(self):
        """Clear buffer after update (PPO is on-policy)."""
        self.ptr = 0
        self.size = 0


class PPOTrainer:
    """PPO Trainer with BEV architecture."""
    
    def __init__(self, args):
        self.args = args
        
        # Device setup
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✓ Using GPU: {gpu_name} ({total_mem:.1f}GB)")
        else:
            self.device = torch.device('cpu')
            print("⚠ Using CPU (slow)")
        
        # Model setup
        self.proprio_dim = 6  # [lidar_min, prev_lin, prev_ang, lin_vel, ang_vel, gap_heading]
        self.action_dim = 2
        
        self.policy = UnifiedBEVPPOPolicy(
            action_dim=self.action_dim,
            proprio_dim=self.proprio_dim
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=args.lr,
            eps=1e-5
        )
        
        # Replay Buffer
        self.buffer = PPOBuffer(
            capacity=args.buffer_size,
            proprio_dim=self.proprio_dim,
            device=self.device
        )
        
        # Training state
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
        self.current_episode_reward = 0.0
        self.episode_count = 0
        
        # NATS connection
        self.nc = None
        self.js = None
        self.nats_server = args.nats_server
        self.loop = None
        
        # Signal handler for graceful shutdown
        import signal
        signal.signal(signal.SIGINT, self._save_on_shutdown)
        signal.signal(signal.SIGTERM, self._save_on_shutdown)
        
        # Load checkpoint if exists
        self.load_latest_checkpoint()
        
        # Export initial model
        self.export_onnx(increment_version=False)
        
        print(f"✅ PPO BEV Trainer initialized")
        print(f"   Checkpoint interval: {args.checkpoint_interval} steps")
        print(f"   Buffer size: {args.buffer_size}")
        
    def _save_on_shutdown(self, signum, frame):
        """Save checkpoint on shutdown."""
        print("\n🛑 Shutdown signal received!")
        self.save_checkpoint(f"ppo_step_{self.total_steps}.pt")
        print("✅ Shutdown complete")
        sys.exit(0)

    def load_latest_checkpoint(self):
        """Load latest checkpoint if exists."""
        checkpoints = list(Path(self.args.checkpoint_dir).glob('ppo_step_*.pt'))
        if not checkpoints:
            print("🆕 No checkpoint found. Starting fresh.")
            return

        latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
        print(f"🔄 Resuming from {latest}")
        ckpt = torch.load(latest, map_location=self.device)
        
        self.policy.load_state_dict(ckpt['policy_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.total_steps = ckpt['total_steps']
        self.update_count = ckpt.get('update_count', 0)
        self.model_version = self.update_count
        self.episode_count = ckpt.get('episode_count', 0)
        
        print(f"  ✓ Restored: {self.total_steps} steps, {self.update_count} updates")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.args.checkpoint_dir, filename)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'update_count': self.update_count,
            'model_version': self.model_version,
            'episode_count': self.episode_count,
        }, path)
        print(f"💾 Saved checkpoint: {path}")
        
        # Export ONNX
        self.export_onnx()
        
        # Export RKNN
        self.export_rknn()

    def export_onnx(self, increment_version: bool = True):
        """Export policy to ONNX format."""
        try:
            onnx_path = os.path.join(self.args.checkpoint_dir, "latest_actor.onnx")
            
            # Create wrapper for ONNX export
            class PolicyWrapper(nn.Module):
                def __init__(self, policy):
                    super().__init__()
                    self.policy = policy
                    
                def forward(self, bev, proprio):
                    action_mean, log_std, value = self.policy(bev, proprio)
                    return action_mean, log_std, value

            model = PolicyWrapper(self.policy)
            model.eval()
            
            # Dummy inputs
            dummy_bev = torch.randn(1, 2, 128, 128)
            dummy_proprio = torch.randn(1, self.proprio_dim)
            
            torch.onnx.export(
                model,
                (dummy_bev, dummy_proprio),
                onnx_path,
                opset_version=11,
                input_names=['bev', 'proprio'],
                output_names=['action_mean', 'log_std', 'value'],
                export_params=True,
                do_constant_folding=True,
                keep_initializers_as_inputs=False,
                verbose=False,
                dynamo=False
            )
            
            if increment_version:
                self.model_version += 1
                
            file_size = os.path.getsize(onnx_path)
            print(f"📦 Exported ONNX: {onnx_path} ({file_size} bytes)")
            
        except Exception as e:
            print(f"❌ ONNX export failed: {e}")

    def export_rknn(self):
        """Export model to RKNN format for NPU inference."""
        try:
            onnx_path = os.path.join(self.args.checkpoint_dir, "latest_actor.onnx")
            rknn_path = os.path.join(self.args.checkpoint_dir, "latest_actor.rknn")
            
            if not os.path.exists(onnx_path):
                print("⚠ No ONNX model found for RKNN export")
                return
            
            # Import RKNN toolkit
            try:
                from rknn.api import RKNN
            except ImportError:
                print("⚠ RKNN toolkit not available - skipping RKNN export")
                print("  Install with: pip install rknn-toolkit2")
                return
            
            print("🔄 Converting to RKNN...")
            rknn = RKNN()
            
            # Configure
            rknn.config(
                target_platform='rk3588',
                quant_dq=True,
                quantized_activation_per_channel=False,
            )
            
            # Load ONNX
            rknn.load_onnx(model=onnx_path)
            
            # Build
            rknn.build(
                do_quantization=True,
                dataset=None,  # Add calibration dataset for better quantization
            )
            
            # Export RKNN
            rknn.export_rknn(rknn_path)
            rknn.release()
            
            file_size = os.path.getsize(rknn_path)
            print(f"🔧 Exported RKNN: {rknn_path} ({file_size} bytes)")
            
        except Exception as e:
            print(f"❌ RKNN export failed: {e}")
            import traceback
            traceback.print_exc()

    def train_step(self) -> Dict:
        """Perform one PPO update."""
        metrics = {'policy_loss': [], 'value_loss': [], 'entropy': [], 'approx_kl': [], 'clip_fraction': []}
        
        # Move buffer to GPU
        bev = self.buffer.bev[:self.buffer.size].to(self.device).float() / 255.0
        proprio = self.buffer.proprio[:self.buffer.size].to(self.device)
        
        # Compute GAE
        advantages, returns = self.buffer.compute_gae(self.policy)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training loop
        for epoch in range(self.args.update_epochs):
            # Shuffle
            indices = torch.randperm(self.buffer.size, device=self.device)
            
            for start_idx in range(0, self.buffer.size, self.args.mini_batch_size):
                batch_indices = indices[start_idx : start_idx + self.args.mini_batch_size]
                
                batch = {
                    'bev': bev[batch_indices],
                    'proprio': proprio[batch_indices],
                    'actions': self.buffer.actions[batch_indices].to(self.device),
                    'returns': returns[batch_indices],
                    'advantages': advantages[batch_indices],
                    'log_probs': self.buffer.log_probs[batch_indices].to(self.device),
                }
                
                # Forward pass
                action_mean, log_std, values = self.policy(batch['bev'], batch['proprio'])
                
                # Action distribution
                std = log_std.exp().clamp(min=1e-6, max=2.0)
                dist = torch.distributions.Normal(action_mean, std)
                
                # Log probs
                current_log_probs = dist.log_prob(batch['actions']).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                # Ratios
                ratios = torch.exp(current_log_probs - batch['log_probs'])
                
                # Approx KL
                log_ratio = current_log_probs - batch['log_probs']
                approx_kl = ((ratios - 1) - log_ratio).mean()
                
                # Clip fraction
                clip_fraction = ((ratios - 1.0).abs() > self.args.clip_eps).float().mean()
                
                # Policy loss
                surr1 = ratios * batch['advantages']
                surr2 = torch.clamp(ratios, 1 - self.args.clip_eps, 1 + self.args.clip_eps) * batch['advantages']
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = 0.5 * (values - batch['returns']).pow(2).mean()
                
                # Total loss
                loss = policy_loss + self.args.value_coef * value_loss - self.args.entropy_coef * entropy
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                metrics['policy_loss'].append(policy_loss.item())
                metrics['value_loss'].append(value_loss.item())
                metrics['entropy'].append(entropy.item())
                metrics['approx_kl'].append(approx_kl.item())
                metrics['clip_fraction'].append(clip_fraction.item())
        
        return {k: np.mean(v) for k, v in metrics.items()}

    async def setup_nats(self):
        """Initialize NATS connection."""
        print(f"🔌 Connecting to NATS at {self.nats_server}...")
        
        self.nc = await nats.connect(
            servers=[self.nats_server],
            name="ppo-bev-trainer",
            max_reconnect_attempts=-1,
            reconnect_time_wait=2,
        )
        
        self.js = self.nc.jetstream()
        
        # Create streams
        try:
            await self.js.add_stream(StreamConfig(
                name="ROVER_EXPERIENCE",
                subjects=["rover.experience"],
                retention="limits",
                max_msgs=1000,
                max_bytes=10 * 1024 * 1024 * 1024,
                max_age=604800,
                max_msg_size=200 * 1024 * 1024,
                storage="file",
                discard="old",
            ))
        except Exception as e:
            if "stream name already in use" not in str(e).lower():
                print(f"⚠ Stream setup: {e}")
        
        print(f"✅ Connected to NATS")

    async def consume_experience(self):
        """Consume experience batches from rover."""
        print("📡 Starting experience consumer...")
        
        psub = await self.js.pull_subscribe(
            subject="rover.experience",
            durable="ppo_bev_trainer"
        )
        
        while True:
            try:
                msgs = await psub.fetch(batch=1, timeout=1.0)
                for msg in msgs:
                    try:
                        # Deserialize batch (simplified - assume JSON for now)
                        import json
                        batch_data = json.loads(msg.data.decode())
                        
                        # Convert to numpy arrays
                        batch = {
                            'bev': np.array(batch_data['bev']),
                            'proprio': np.array(batch_data['proprio']),
                            'actions': np.array(batch_data['actions']),
                            'rewards': np.array(batch_data['rewards']),
                            'dones': np.array(batch_data['dones']),
                            'log_probs': np.array(batch_data['log_probs']),
                            'values': np.array(batch_data['values']),
                        }
                        
                        # Add to buffer
                        self.buffer.add_batch(batch)
                        
                        # Track episode rewards
                        for r, d in zip(batch['rewards'], batch['dones']):
                            self.current_episode_reward += r
                            if d:
                                self.episode_rewards.append(self.current_episode_reward)
                                self.episode_count += 1
                                print(f"📊 Episode {self.episode_count}: Reward = {self.current_episode_reward:.2f}")
                                self.current_episode_reward = 0.0
                        
                        await msg.ack()
                        print(f"📥 Added {len(batch['rewards'])} steps to buffer")
                        
                    except Exception as e:
                        print(f"⚠ Error processing batch: {e}")
                        await msg.ack()
                        
            except nats.errors.TimeoutError:
                await asyncio.sleep(0.1)
                continue
            except Exception as e:
                print(f"❌ Consumer error: {e}")
                await asyncio.sleep(1.0)

    async def run(self):
        """Main training loop."""
        print(f"🚀 PPO BEV Trainer starting...")
        
        # Connect to NATS
        await self.setup_nats()
        
        # Start consumer
        asyncio.create_task(self.consume_experience())
        
        # Training loop
        checkpoint_interval = self.args.checkpoint_interval
        last_checkpoint = 0
        
        while True:
            # Check if buffer has enough data
            if self.buffer.size >= self.args.rollout_steps:
                print(f"🔔 Buffer full ({self.buffer.size} steps). Starting PPO update...")
                
                # Train
                metrics = self.train_step()
                self.update_count += 1
                self.total_steps += self.buffer.size
                
                # Log metrics
                if self.update_count % 10 == 0:
                    print(f"Step {self.total_steps} | P-loss: {metrics['policy_loss']:.3f} V-loss: {metrics['value_loss']:.3f} KL: {metrics['approx_kl']:.4f}")
                    
                    for k, v in metrics.items():
                        self.writer.add_scalar(f'train/{k}', v, self.total_steps)
                    
                    # Log episode stats
                    if len(self.episode_rewards) > 0:
                        self.writer.add_scalar('episode/mean_reward', np.mean(self.episode_rewards), self.total_steps)
                        self.writer.add_scalar('episode/episode_count', self.episode_count, self.total_steps)
                
                # Save checkpoint every checkpoint_interval steps
                if self.total_steps - last_checkpoint >= checkpoint_interval:
                    self.save_checkpoint(f"ppo_step_{self.total_steps}.pt")
                    last_checkpoint = self.total_steps
                
                # Clear buffer (on-policy)
                self.buffer.clear()
            
            await asyncio.sleep(0.1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nats_server', type=str, default='nats://nats.gokickrocks.org:4222')
    parser.add_argument('--checkpoint_dir', default='./checkpoints_ppo')
    parser.add_argument('--log_dir', default='./logs_ppo')
    parser.add_argument('--buffer_size', type=int, default=25000)
    parser.add_argument('--rollout_steps', type=int, default=2048)
    parser.add_argument('--mini_batch_size', type=int, default=512)
    parser.add_argument('--update_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--clip_eps', type=float, default=0.2)
    parser.add_argument('--value_coef', type=float, default=0.5)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--checkpoint_interval', type=int, default=200, help='Save checkpoint every N steps')
    
    args = parser.parse_args()
    
    trainer = PPOTrainer(args)
    asyncio.run(trainer.run())