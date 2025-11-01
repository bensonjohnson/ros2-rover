#!/usr/bin/env python3
"""V620 ROCm PPO Training Server for Remote Rover Training.

This server receives RGB-D observations from the rover via ZeroMQ,
trains a PPO policy using PyTorch with ROCm acceleration,
and exports trained models in multiple formats (PyTorch, ONNX, RKNN).

Hardware: AMD Radeon Pro V620 with ROCm
Training: PPO with RGB-D vision encoder
Export: PyTorch → ONNX → RKNN for rover NPU deployment
"""

import os
import time
import json
import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict
from collections import deque

import numpy as np
import cv2

try:
    import zmq
    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False
    print("ERROR: ZeroMQ not installed. Install with: pip install pyzmq")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class RGBDEncoder(nn.Module):
    """Vision encoder for RGB-D inputs.

    Takes RGB (3 channels) and Depth (1 channel) and fuses them through CNN.
    """

    def __init__(self, rgb_channels: int = 3, depth_channels: int = 1):
        super().__init__()

        # Separate encoders for RGB and depth
        self.rgb_conv = nn.Sequential(
            nn.Conv2d(rgb_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.depth_conv = nn.Sequential(
            nn.Conv2d(depth_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # Fused feature encoder
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),  # 64 + 32 = 96
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_dim = 128

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: (B, 3, H, W) normalized to [0, 1]
            depth: (B, 1, H, W) normalized to [0, 1]
        Returns:
            features: (B, 128)
        """
        rgb_feat = self.rgb_conv(rgb)
        depth_feat = self.depth_conv(depth)
        fused = torch.cat([rgb_feat, depth_feat], dim=1)
        features = self.fusion_conv(fused)
        features = self.pool(features)
        return features.view(features.size(0), -1)


class PolicyHead(nn.Module):
    """Policy network head with proprioception fusion."""

    def __init__(self, feature_dim: int, proprio_dim: int, action_dim: int = 2):
        super().__init__()
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.policy = nn.Sequential(
            nn.Linear(feature_dim + 64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_dim),
        )

    def forward(self, features: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        proprio_feat = self.proprio_encoder(proprio)
        combined = torch.cat([features, proprio_feat], dim=1)
        return self.policy(combined)


class ValueHead(nn.Module):
    """Value network head with proprioception fusion."""

    def __init__(self, feature_dim: int, proprio_dim: int):
        super().__init__()
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.value = nn.Sequential(
            nn.Linear(feature_dim + 64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        proprio_feat = self.proprio_encoder(proprio)
        combined = torch.cat([features, proprio_feat], dim=1)
        return self.value(combined).squeeze(-1)


class RolloutBuffer:
    """Experience replay buffer for PPO."""

    def __init__(self, capacity: int, rgb_shape: Tuple, depth_shape: Tuple, proprio_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        # Store as uint8/float16 to save memory
        self.rgb = np.zeros((capacity, *rgb_shape), dtype=np.uint8)
        self.depth = np.zeros((capacity, *depth_shape), dtype=np.float16)
        self.proprio = np.zeros((capacity, proprio_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 2), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=bool)
        self.log_probs = np.zeros((capacity,), dtype=np.float32)
        self.values = np.zeros((capacity,), dtype=np.float32)

    def add(self, rgb, depth, proprio, action, reward, done, log_prob, value):
        idx = self.ptr
        self.rgb[idx] = rgb
        self.depth[idx] = depth.astype(np.float16)
        self.proprio[idx] = proprio
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.log_probs[idx] = log_prob
        self.values[idx] = value

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get(self, device):
        """Get all data as tensors on specified device."""
        idx = slice(0, self.size)
        return {
            'rgb': torch.from_numpy(self.rgb[idx]).to(device).float() / 255.0,
            'depth': torch.from_numpy(self.depth[idx].astype(np.float32)).to(device).unsqueeze(1),
            'proprio': torch.from_numpy(self.proprio[idx]).to(device),
            'actions': torch.from_numpy(self.actions[idx]).to(device),
            'rewards': torch.from_numpy(self.rewards[idx]),
            'dones': torch.from_numpy(self.dones[idx]),
            'log_probs': torch.from_numpy(self.log_probs[idx]).to(device),
            'values': torch.from_numpy(self.values[idx]),
        }

    def clear(self):
        self.ptr = 0
        self.size = 0


class V620PPOTrainer:
    """PPO trainer optimized for AMD ROCm."""

    def __init__(
        self,
        rgb_shape: Tuple[int, int, int] = (240, 424, 3),
        depth_shape: Tuple[int, int] = (240, 424),
        proprio_dim: int = 6,
        rollout_capacity: int = 8192,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        update_epochs: int = 4,
        minibatch_size: int = 256,
        device: Optional[str] = None,
    ):
        # ROCm device detection
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"ROCm GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                device = 'cpu'
                print("WARNING: No GPU detected, using CPU (training will be slow)")

        self.device = torch.device(device)
        self.rgb_shape = rgb_shape
        self.depth_shape = depth_shape
        self.proprio_dim = proprio_dim

        # Networks
        self.encoder = RGBDEncoder().to(self.device)
        self.policy_head = PolicyHead(self.encoder.output_dim, proprio_dim).to(self.device)
        self.value_head = ValueHead(self.encoder.output_dim, proprio_dim).to(self.device)
        self.log_std = nn.Parameter(torch.zeros(2, device=self.device))

        # Optimizer
        params = (
            list(self.encoder.parameters()) +
            list(self.policy_head.parameters()) +
            list(self.value_head.parameters()) +
            [self.log_std]
        )
        self.optimizer = optim.Adam(params, lr=lr)

        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size

        # Buffer
        self.buffer = RolloutBuffer(rollout_capacity, rgb_shape, depth_shape, proprio_dim)

        # Metrics
        self.update_count = 0
        self.total_steps = 0

    def select_action(self, rgb: np.ndarray, depth: np.ndarray, proprio: np.ndarray, deterministic: bool = False):
        """Select action using current policy."""
        with torch.no_grad():
            # Prepare inputs
            rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
            depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float().to(self.device)
            proprio_t = torch.from_numpy(proprio).unsqueeze(0).float().to(self.device)

            # Forward pass
            features = self.encoder(rgb_t, depth_t)
            action_mean = self.policy_head(features, proprio_t)
            value = self.value_head(features, proprio_t)

            if deterministic:
                action = torch.tanh(action_mean)
            else:
                std = self.log_std.exp()
                dist = torch.distributions.Normal(action_mean, std)
                action_sample = dist.sample()
                action = torch.tanh(action_sample)
                log_prob = dist.log_prob(action_sample).sum(dim=1)

            return (
                action.cpu().numpy()[0],
                log_prob.cpu().numpy()[0] if not deterministic else 0.0,
                value.cpu().numpy()[0]
            )

    def add_transition(self, rgb, depth, proprio, action, reward, done, log_prob, value):
        """Add transition to buffer."""
        self.buffer.add(rgb, depth, proprio, action, reward, done, log_prob, value)
        self.total_steps += 1

    def update(self) -> Dict:
        """Perform PPO update."""
        if self.buffer.size < self.minibatch_size:
            return {'updated': False, 'reason': 'insufficient_data'}

        print(f"[UPDATE] Starting update with buffer size: {self.buffer.size}", flush=True)

        # Get all data
        print(f"[UPDATE] Getting data from buffer...", flush=True)
        data = self.buffer.get(self.device)
        print(f"[UPDATE] Data retrieved, preparing tensors...", flush=True)
        rgb = data['rgb'].permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)
        depth = data['depth']
        proprio = data['proprio']
        actions = data['actions']
        old_log_probs = data['log_probs']
        rewards_np = data['rewards'].numpy()
        dones_np = data['dones'].numpy()
        values_np = data['values'].numpy()

        # Compute advantages with GAE
        advantages_np, returns_np = self._compute_gae(rewards_np, dones_np, values_np)
        advantages = torch.from_numpy(advantages_np).to(self.device)
        returns = torch.from_numpy(returns_np).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update epochs
        dataset_size = self.buffer.size
        indices = np.arange(dataset_size)

        metrics = {'policy_loss': [], 'value_loss': [], 'entropy': []}

        print(f"[UPDATE] Starting {self.update_epochs} epochs of training...", flush=True)

        for epoch in range(self.update_epochs):
            np.random.shuffle(indices)

            if epoch == 0:
                print(f"[UPDATE] Epoch {epoch+1}/{self.update_epochs}, processing {dataset_size} samples in batches of {self.minibatch_size}...", flush=True)

            for start in range(0, dataset_size, self.minibatch_size):
                end = min(start + self.minibatch_size, dataset_size)
                batch_idx = indices[start:end]

                # Batch data
                batch_rgb = rgb[batch_idx]
                batch_depth = depth[batch_idx]
                batch_proprio = proprio[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # Forward pass
                if epoch == 0 and start == 0:
                    print(f"[UPDATE] Running first forward pass through encoder (this may take a while for ROCm kernel compilation)...", flush=True)
                features = self.encoder(batch_rgb, batch_depth)
                if epoch == 0 and start == 0:
                    print(f"[UPDATE] Encoder forward pass complete!", flush=True)
                action_mean = self.policy_head(features, batch_proprio)
                values_pred = self.value_head(features, batch_proprio)

                # Compute policy loss
                std = self.log_std.exp()
                dist = torch.distributions.Normal(action_mean, std)
                action_tanh = batch_actions
                action_pre_tanh = torch.atanh(torch.clamp(action_tanh, -0.999, 0.999))
                log_probs = dist.log_prob(action_pre_tanh).sum(dim=1)

                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Compute value loss
                value_loss = nn.functional.mse_loss(values_pred, batch_returns)

                # Compute entropy
                entropy = dist.entropy().sum(dim=1).mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) +
                    list(self.policy_head.parameters()) +
                    list(self.value_head.parameters()),
                    max_norm=0.5
                )
                self.optimizer.step()

                # Track metrics
                metrics['policy_loss'].append(policy_loss.item())
                metrics['value_loss'].append(value_loss.item())
                metrics['entropy'].append(entropy.item())

        self.buffer.clear()
        self.update_count += 1

        return {
            'updated': True,
            'update_count': self.update_count,
            'policy_loss': np.mean(metrics['policy_loss']),
            'value_loss': np.mean(metrics['value_loss']),
            'entropy': np.mean(metrics['entropy']),
        }

    def _compute_gae(self, rewards, dones, values):
        """Compute Generalized Advantage Estimation."""
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'encoder': self.encoder.state_dict(),
            'policy_head': self.policy_head.state_dict(),
            'value_head': self.value_head.state_dict(),
            'log_std': self.log_std,
            'optimizer': self.optimizer.state_dict(),
            'update_count': self.update_count,
            'total_steps': self.total_steps,
        }, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.policy_head.load_state_dict(checkpoint['policy_head'])
        self.value_head.load_state_dict(checkpoint['value_head'])
        self.log_std.data = checkpoint['log_std'].to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.update_count = checkpoint['update_count']
        self.total_steps = checkpoint['total_steps']
        print(f"Checkpoint loaded: {path}")

    def export_onnx(self, path: str):
        """Export policy to ONNX format."""
        # Create a wrapper for ONNX export
        class ONNXWrapper(nn.Module):
            def __init__(self, encoder, policy_head):
                super().__init__()
                self.encoder = encoder
                self.policy_head = policy_head

            def forward(self, rgb, depth, proprio):
                features = self.encoder(rgb, depth)
                action_mean = self.policy_head(features, proprio)
                return torch.tanh(action_mean)

        wrapper = ONNXWrapper(self.encoder, self.policy_head)
        wrapper.eval()

        # Dummy inputs
        dummy_rgb = torch.randn(1, 3, *self.rgb_shape[:2], device=self.device)
        dummy_depth = torch.randn(1, 1, *self.depth_shape, device=self.device)
        dummy_proprio = torch.randn(1, self.proprio_dim, device=self.device)

        torch.onnx.export(
            wrapper,
            (dummy_rgb, dummy_depth, dummy_proprio),
            path,
            input_names=['rgb', 'depth', 'proprio'],
            output_names=['action'],
            dynamic_axes={
                'rgb': {0: 'batch'},
                'depth': {0: 'batch'},
                'proprio': {0: 'batch'},
                'action': {0: 'batch'}
            },
            opset_version=11
        )
        print(f"ONNX model exported: {path}")


def parse_packet(data: bytes) -> Optional[Dict]:
    """Parse binary packet from rover."""
    try:
        # Read metadata length
        metadata_len = int.from_bytes(data[:4], byteorder='little')
        metadata_json = data[4:4+metadata_len].decode('utf-8')
        metadata = json.loads(metadata_json)

        # Parse data
        offset = 4 + metadata_len

        # RGB
        rgb_shape = tuple(metadata['rgb_shape'])
        if metadata.get('rgb_compressed', False):
            # Decompress JPEG
            rgb_size = len(data) - offset - (np.prod(metadata['depth_shape']) * 4) - (metadata['proprio_shape'][0] * 4) - (metadata['action_shape'][0] * 4)
            rgb_bytes = data[offset:offset+rgb_size]
            rgb = cv2.imdecode(np.frombuffer(rgb_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            offset += rgb_size
        else:
            rgb_size = np.prod(rgb_shape)
            rgb = np.frombuffer(data[offset:offset+rgb_size], dtype=np.uint8).reshape(rgb_shape)
            offset += rgb_size

        # Depth
        depth_shape = tuple(metadata['depth_shape'])
        depth_size = np.prod(depth_shape) * 4
        depth = np.frombuffer(data[offset:offset+depth_size], dtype=np.float32).reshape(depth_shape)
        offset += depth_size

        # Proprio
        proprio_size = metadata['proprio_shape'][0] * 4
        proprio = np.frombuffer(data[offset:offset+proprio_size], dtype=np.float32)
        offset += proprio_size

        # Action
        action_size = metadata['action_shape'][0] * 4
        action = np.frombuffer(data[offset:offset+action_size], dtype=np.float32)

        return {
            'rgb': rgb,
            'depth': depth,
            'proprio': proprio,
            'action': action,
            'reward': metadata['reward'],
            'done': metadata['done'],
            'timestamp': metadata['timestamp'],
        }
    except Exception as exc:
        print(f"Failed to parse packet: {exc}")
        return None


def main():
    parser = argparse.ArgumentParser(description='V620 PPO Training Server')
    parser.add_argument('--port', type=int, default=5555, help='ZMQ port')
    parser.add_argument('--update-interval', type=int, default=8192, help='Steps between updates')
    parser.add_argument('--checkpoint-interval', type=int, default=10, help='Updates between checkpoints')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--tensorboard-dir', type=str, default='./runs', help='TensorBoard directory')
    args = parser.parse_args()

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)

    # Initialize trainer
    print("Initializing V620 PPO Trainer...")
    trainer = V620PPOTrainer()

    # TensorBoard
    writer = SummaryWriter(args.tensorboard_dir)

    # ZeroMQ server
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(f"tcp://*:{args.port}")
    print(f"Training server listening on port {args.port}")

    # Training loop
    episode_rewards = deque(maxlen=100)
    episode_reward = 0.0

    try:
        while True:
            # Receive data
            data = socket.recv()
            packet = parse_packet(data)

            if packet is None:
                continue

            # Add to buffer (using ground truth action from rover for now)
            # In full implementation, you'd compute action with current policy
            trainer.add_transition(
                packet['rgb'],
                packet['depth'],
                packet['proprio'],
                packet['action'],
                packet['reward'],
                packet['done'],
                log_prob=0.0,  # Placeholder - compute from policy
                value=0.0      # Placeholder - compute from value network
            )

            episode_reward += packet['reward']

            # Debug: Print every 1000 steps
            if trainer.total_steps % 1000 == 0 and trainer.total_steps > 0:
                import sys
                print(f"DEBUG: Step {trainer.total_steps}, Buffer size: {trainer.buffer.size}, Minibatch size: {trainer.minibatch_size}")
                sys.stdout.flush()

            if packet['done']:
                episode_rewards.append(episode_reward)
                print(f"Episode finished | Reward: {episode_reward:.2f} | Avg (100): {np.mean(episode_rewards):.2f} | Total steps: {trainer.total_steps}", flush=True)
                writer.add_scalar('train/episode_reward', episode_reward, trainer.total_steps)
                episode_reward = 0.0

            # Update policy
            if trainer.total_steps % args.update_interval == 0 and trainer.buffer.size >= trainer.minibatch_size:
                print(f"Performing PPO update at step {trainer.total_steps}...", flush=True)
                print(f"Buffer size: {trainer.buffer.size}, Starting update...", flush=True)
                metrics = trainer.update()
                print(f"Update complete! Metrics: {metrics}", flush=True)

                if metrics['updated']:
                    print(f"Update {metrics['update_count']}: "
                          f"Policy Loss={metrics['policy_loss']:.4f}, "
                          f"Value Loss={metrics['value_loss']:.4f}, "
                          f"Entropy={metrics['entropy']:.4f}")

                    writer.add_scalar('train/policy_loss', metrics['policy_loss'], trainer.update_count)
                    writer.add_scalar('train/value_loss', metrics['value_loss'], trainer.update_count)
                    writer.add_scalar('train/entropy', metrics['entropy'], trainer.update_count)

                    # Save checkpoint
                    if trainer.update_count % args.checkpoint_interval == 0:
                        ckpt_path = os.path.join(args.checkpoint_dir, f'ppo_v620_update_{trainer.update_count}.pt')
                        trainer.save_checkpoint(ckpt_path)

                        # Export ONNX
                        onnx_path = os.path.join(args.checkpoint_dir, f'ppo_v620_update_{trainer.update_count}.onnx')
                        trainer.export_onnx(onnx_path)

    except KeyboardInterrupt:
        print("\nShutting down training server...")
        writer.close()
        socket.close()
        context.term()


if __name__ == '__main__':
    main()
