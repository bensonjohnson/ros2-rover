#!/usr/bin/env python3
"""PPO trainer tailored for RTAB observation tensors (occupancy + depth + extras)."""

from typing import Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = torch.clamp(x, -1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class RTABEncoder(nn.Module):
    """Lightweight CNN encoder that reduces CxHxW observations to a feature vector."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_dim = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)


class PolicyHead(nn.Module):
    def __init__(self, feature_dim: int, proprio_dim: int):
        super().__init__()
        self.proprio = nn.Sequential(
            nn.Linear(proprio_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim + 128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(self, feat: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        p = self.proprio(proprio)
        x = torch.cat([feat, p], dim=1)
        return self.fusion(x)


class ValueHead(nn.Module):
    def __init__(self, feature_dim: int, proprio_dim: int):
        super().__init__()
        self.proprio = nn.Sequential(
            nn.Linear(proprio_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim + 128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, feat: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        p = self.proprio(proprio)
        x = torch.cat([feat, p], dim=1)
        return self.fusion(x).squeeze(1)


class RolloutBuffer:
    def __init__(self, capacity: int, obs_shape: Tuple[int, int, int], proprio_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        C, H, W = obs_shape
        self.obs = np.zeros((capacity, C, H, W), dtype=np.float32)
        self.proprio = np.zeros((capacity, proprio_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 2), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.log_probs = np.zeros((capacity,), dtype=np.float32)
        self.values = np.zeros((capacity,), dtype=np.float32)

    def add(self, obs, proprio, action, reward, done, log_prob, value):
        idx = self.ptr
        self.obs[idx] = obs
        self.proprio[idx] = proprio
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = float(done)
        self.log_probs[idx] = log_prob
        self.values[idx] = value
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def clear(self):
        self.ptr = 0
        self.size = 0

    def get(self) -> Dict[str, torch.Tensor]:
        idx = slice(0, self.size)
        return {
            'obs': torch.from_numpy(self.obs[idx]),
            'proprio': torch.from_numpy(self.proprio[idx]),
            'actions': torch.from_numpy(self.actions[idx]),
            'rewards': torch.from_numpy(self.rewards[idx]),
            'dones': torch.from_numpy(self.dones[idx]),
            'log_probs': torch.from_numpy(self.log_probs[idx]),
            'values': torch.from_numpy(self.values[idx]),
        }


class PPOTrainerRTAB:
    def __init__(
        self,
        obs_shape: Tuple[int, int, int],
        proprio_dim: int,
        rollout_capacity: int = 4096,
        lr: float = 3e-4,
        ppo_clip: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        update_epochs: int = 3,
        minibatch_size: int = 128,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.obs_shape = obs_shape
        self.proprio_dim = proprio_dim
        self.encoder = RTABEncoder(in_channels=obs_shape[0]).to(self.device)
        self.policy = PolicyHead(self.encoder.output_dim, proprio_dim).to(self.device)
        self.value_head = ValueHead(self.encoder.output_dim, proprio_dim).to(self.device)
        self.log_std = nn.Parameter(torch.zeros(2, dtype=torch.float32, device=self.device))
        params = list(self.encoder.parameters()) + list(self.policy.parameters()) + list(self.value_head.parameters()) + [self.log_std]
        self.optimizer = optim.Adam(params, lr=lr)
        self.clip = ppo_clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.buffer = RolloutBuffer(rollout_capacity, obs_shape, proprio_dim)
        self.update_count = 0

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device)

    def _policy_forward(self, obs: torch.Tensor, proprio: torch.Tensor):
        features = self.encoder(obs)
        mean_raw = self.policy(features, proprio)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mean_raw, std)
        return dist, features

    def add_transition(self, obs: np.ndarray, proprio: np.ndarray, action_tanh: np.ndarray, reward: float, done: bool) -> None:
        obs = obs.astype(np.float32)
        proprio = proprio.astype(np.float32)
        action_tanh = action_tanh.astype(np.float32)
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        proprio_t = torch.from_numpy(proprio).unsqueeze(0).to(self.device)
        action_t = torch.from_numpy(action_tanh).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist, _ = self._policy_forward(obs_t, proprio_t)
            log_prob = dist.log_prob(atanh(action_t)).sum(dim=1)
            value = self.value_head(self.encoder(obs_t), proprio_t)
        self.buffer.add(obs, proprio, action_tanh, float(reward), done, float(log_prob.item()), float(value.item()))

    def _compute_gae(self, rewards: np.ndarray, dones: np.ndarray, values: np.ndarray, next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            lastgaelam = delta + self.gamma * self.gae_lambda * mask * lastgaelam
            adv[t] = lastgaelam
            next_value = values[t]
        returns = values + adv
        return adv, returns

    def update(self) -> Dict[str, float]:
        if self.buffer.size < self.minibatch_size:
            return {"updated": False, "size": self.buffer.size}
        data = self.buffer.get()
        obs = self._to_device(data['obs'])
        proprio = self._to_device(data['proprio'])
        actions = self._to_device(data['actions'])
        old_log_probs = self._to_device(data['log_probs'])
        values = data['values'].numpy()
        rewards = data['rewards'].numpy()
        dones = data['dones'].numpy()

        with torch.no_grad():
            next_obs = obs[-1:, ...]
            next_proprio = proprio[-1:, ...]
            next_value = self.value_head(self.encoder(next_obs), next_proprio).cpu().numpy()[0]
        advantages, returns = self._compute_gae(rewards, dones, values, next_value)
        advantages_t = self._to_device(torch.from_numpy((advantages - advantages.mean()) / (advantages.std() + 1e-8)))
        returns_t = self._to_device(torch.from_numpy(returns))

        dataset_size = self.buffer.size
        indices = np.arange(dataset_size)
        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.minibatch_size):
                end = start + self.minibatch_size
                batch_idx = indices[start:end]
                batch_obs = obs[batch_idx]
                batch_proprio = proprio[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_adv = advantages_t[batch_idx]
                batch_returns = returns_t[batch_idx]

                dist, feat = self._policy_forward(batch_obs, batch_proprio)
                pre_squash = atanh(batch_actions)
                log_probs = dist.log_prob(pre_squash).sum(dim=1) - torch.log(1 - batch_actions.pow(2) + 1e-6).sum(dim=1)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                values_pred = self.value_head(feat, batch_proprio)
                value_loss = nn.functional.mse_loss(values_pred, batch_returns)
                entropy = dist.entropy().sum(dim=1).mean()

                loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.policy.parameters()) + list(self.value_head.parameters()),
                    max_norm=0.5,
                )
                self.optimizer.step()

        self.buffer.clear()
        self.update_count += 1
        return {
            "updated": True,
            "updates": self.update_count,
            "buffer_size": dataset_size,
            "loss_actor": float(actor_loss.item()),
            "loss_value": float(value_loss.item()),
        }

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            'encoder': self.encoder.state_dict(),
            'policy': self.policy.state_dict(),
            'value': self.value_head.state_dict(),
            'log_std': self.log_std.detach().cpu(),
        }

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.encoder.load_state_dict(state['encoder'])
        self.policy.load_state_dict(state['policy'])
        self.value_head.load_state_dict(state['value'])
        with torch.no_grad():
            self.log_std.copy_(state['log_std'].to(self.device))
