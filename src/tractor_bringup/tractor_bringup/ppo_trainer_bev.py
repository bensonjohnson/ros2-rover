#!/usr/bin/env python3
"""
Lightweight PPO trainer for BEV inputs.
 - Uses BEVExplorationNet (from rknn_trainer_bev) as the actor (3 outputs; first two are actions pre-tanh)
 - Adds a separate critic network
 - Maintains a small rollout FIFO (on-policy-ish)
 - Provides update() that runs bounded PPO epochs/minibatches
 - Exposes actor state_dict() for export to RKNN via existing pipeline
"""

from typing import Tuple, Optional, Deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from .rknn_trainer_bev import BEVExplorationNet


def atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = torch.clamp(x, -1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class CriticNet(nn.Module):
    def __init__(self, bev_channels: int, proprio_inputs: int):
        super().__init__()
        # Mirror actor trunk (separate weights)
        self.bev_conv = nn.Sequential(
            nn.Conv2d(bev_channels, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.AvgPool2d(kernel_size=(3, 3), stride=(3, 3)), nn.Flatten()
        )
        self.bev_fc = nn.Linear(256 * 4 * 4, 512)
        self.sensor_fc = nn.Sequential(
            nn.Linear(proprio_inputs, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.fusion = nn.Sequential(
            nn.Linear(512 + 128, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, bev_image: torch.Tensor, sensor: torch.Tensor) -> torch.Tensor:
        bev_feat = self.bev_conv(bev_image)
        bev_out = self.bev_fc(bev_feat)
        sens_out = self.sensor_fc(sensor)
        fused = torch.cat([bev_out, sens_out], dim=1)
        value = self.fusion(fused)
        return value.squeeze(1)


class RolloutBuffer:
    def __init__(self, capacity: int, bev_shape: Tuple[int, int, int], proprio_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        C, H, W = bev_shape
        self.bev = np.zeros((capacity, C, H, W), dtype=np.float32)
        self.sensor = np.zeros((capacity, proprio_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 2), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.log_probs = np.zeros((capacity,), dtype=np.float32)
        self.values = np.zeros((capacity,), dtype=np.float32)

    def add(self, bev_chw, sensor_vec, action, reward, done, log_prob, value):
        i = self.ptr
        self.bev[i] = bev_chw
        self.sensor[i] = sensor_vec
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.log_probs[i] = log_prob
        self.values[i] = value
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get(self):
        idx = np.arange(self.size)
        data = dict(
            bev=torch.from_numpy(self.bev[idx]),
            sensor=torch.from_numpy(self.sensor[idx]),
            actions=torch.from_numpy(self.actions[idx]),
            rewards=torch.from_numpy(self.rewards[idx]),
            dones=torch.from_numpy(self.dones[idx]),
            log_probs=torch.from_numpy(self.log_probs[idx]),
            values=torch.from_numpy(self.values[idx]),
        )
        return data

    def clear(self):
        self.ptr = 0
        self.size = 0


class PPOTrainerBEV:
    def __init__(self,
                 bev_channels: int = 4,
                 bev_size: Tuple[int, int] = (200, 200),
                 proprio_dim: int = 3,
                 device: Optional[str] = None,
                 rollout_capacity: int = 4096,
                 lr: float = 3e-4,
                 ppo_clip: float = 0.2,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 update_epochs: int = 2,
                 minibatch_size: int = 128):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        C = bev_channels
        H, W = bev_size
        self.actor = BEVExplorationNet(bev_channels=C, extra_proprio=proprio_dim - 3).to(self.device)
        self.critic = CriticNet(bev_channels=C, proprio_inputs=proprio_dim).to(self.device)
        # Learnable log_std
        self.log_std = nn.Parameter(torch.zeros(2, dtype=torch.float32, device=self.device))
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()) + [self.log_std], lr=lr)
        self.clip = ppo_clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.buffer = RolloutBuffer(rollout_capacity, (C, H, W), proprio_dim)
        self.bev_channels = C
        self.bev_size = (H, W)
        self.proprio_dim = proprio_dim

    def preprocess_bev(self, bev_hwc: np.ndarray) -> np.ndarray:
        # Input HxWxC float32; transpose to CxHxW
        if bev_hwc.ndim == 3:
            return np.transpose(bev_hwc, (2, 0, 1)).astype(np.float32)
        return bev_hwc.astype(np.float32)

    def act_eval(self, bev_chw: np.ndarray, sensor_vec: np.ndarray) -> Tuple[np.ndarray, float, float]:
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            bev_t = torch.from_numpy(bev_chw).unsqueeze(0).to(self.device)
            sens_t = torch.from_numpy(sensor_vec.astype(np.float32)).unsqueeze(0).to(self.device)
            out = self.actor(bev_t, sens_t)[0]  # shape (3)
            mean_raw = out[:2]
            value = self.critic(bev_t, sens_t)[0]
            std = self.log_std.exp()
            dist = torch.distributions.Normal(mean_raw, std)
            # Runtime action is tanh(mean); we evaluate log_prob for observed action later via atanh
            action_mean = torch.tanh(mean_raw)
        return action_mean.cpu().numpy(), float(value.item()), float(std.mean().item())

    def log_prob_and_value(self, bev_batch: torch.Tensor, sens_batch: torch.Tensor, actions_tanh: torch.Tensor):
        # Compute log_prob with tanh-squash correction
        out = self.actor(bev_batch, sens_batch)
        mean_raw = out[:, :2]
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mean_raw, std)
        pre_squash = atanh(actions_tanh)
        log_prob = dist.log_prob(pre_squash).sum(dim=1)
        # Change of variables: log|det J| = sum log(1 - a^2)
        log_prob -= torch.log(1 - actions_tanh.pow(2) + 1e-6).sum(dim=1)
        value = self.critic(bev_batch, sens_batch)
        entropy = dist.entropy().sum(dim=1)
        return log_prob, value, entropy

    def add_transition(self, bev_hwc: np.ndarray, sensor_vec: np.ndarray, action_tanh: np.ndarray, reward: float, done: bool):
        bev_chw = self.preprocess_bev(bev_hwc)
        # Compute log_prob and value snapshot for storage
        bev_t = torch.from_numpy(bev_chw).unsqueeze(0).to(self.device)
        sens_t = torch.from_numpy(sensor_vec.astype(np.float32)).unsqueeze(0).to(self.device)
        act_t = torch.from_numpy(action_tanh.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logp, val, _ = self.log_prob_and_value(bev_t, sens_t, act_t)
        self.buffer.add(bev_chw, sensor_vec.astype(np.float32), action_tanh.astype(np.float32), float(reward), done, float(logp.item()), float(val.item()))

    def _gae(self, rewards, dones, values, next_value):
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * nonterminal - values[t]
            lastgaelam = delta + self.gamma * self.gae_lambda * nonterminal * lastgaelam
            adv[t] = lastgaelam
            next_value = values[t]
        returns = values + adv
        return adv, returns

    def update(self) -> dict:
        if self.buffer.size < self.minibatch_size:
            return {"updated": False, "size": self.buffer.size}
        data = self.buffer.get()
        bev = data['bev'].to(self.device)
        sens = data['sensor'].to(self.device)
        actions = data['actions'].to(self.device)
        rewards = data['rewards'].cpu().numpy()
        dones = data['dones'].cpu().numpy()
        old_logp = data['log_probs'].to(self.device)
        values_np = data['values'].cpu().numpy()
        # Bootstrap next_value as last value
        with torch.no_grad():
            next_value = self.critic(bev[-1:], sens[-1:])[0].item()
        adv, ret = self._gae(rewards, dones, values_np, next_value)
        adv_t = torch.from_numpy((adv - adv.mean()) / (adv.std() + 1e-8)).to(self.device)
        ret_t = torch.from_numpy(ret).to(self.device)

        N = bev.shape[0]
        idx = np.arange(N)
        total_loss = 0.0
        total_vloss = 0.0
        total_closs = 0.0
        for _ in range(self.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, N, self.minibatch_size):
                mb = idx[start:start + self.minibatch_size]
                b_bev = bev[mb]
                b_sens = sens[mb]
                b_act = actions[mb]
                b_adv = adv_t[mb]
                b_ret = ret_t[mb]
                b_old = old_logp[mb]
                new_logp, v_pred, ent = self.log_prob_and_value(b_bev, b_sens, b_act)
                ratio = torch.exp(new_logp - b_old)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                v_loss = (v_pred - b_ret).pow(2).mean()
                loss = policy_loss + self.value_coef * v_loss - self.entropy_coef * ent.mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 1.0)
                self.optimizer.step()
                total_loss += loss.item()
                total_vloss += v_loss.item()
                total_closs += policy_loss.item()

        avg_loss = total_loss / max(1, (self.update_epochs * (N // self.minibatch_size + 1)))
        self.buffer.clear()
        return {
            "updated": True,
            "avg_loss": avg_loss,
            "size": N,
            "policy_loss": total_closs,
            "value_loss": total_vloss,
        }

    def actor_state_dict(self):
        return self.actor.state_dict()

