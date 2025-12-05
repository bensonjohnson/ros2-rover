#!/usr/bin/env python3
"""Model architectures for MAP-Elites rover training.

This file contains only the neural network architectures without any
training dependencies, making it safe to import on the rover for model conversion.
"""

import torch
import torch.nn as nn
from typing import Tuple


class OccupancyGridEncoder(nn.Module):
    """Vision encoder for Top-Down Occupancy Grid.

    Takes 64x64 Occupancy Grid (1 channel) and encodes it.
    """

    def __init__(self, input_channels: int = 1):
        super().__init__()

        self.conv = nn.Sequential(
            # Input: (B, 1, 64, 64)
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1), # -> (32, 32, 32)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (64, 16, 16)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> (128, 8, 8)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # -> (256, 4, 4)
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output_dim = 256

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid: (B, 1, 64, 64) normalized to [0, 1]
        Returns:
            features: (B, 256)
        """
        features = self.conv(grid)
        features = self.pool(features)
        return features.view(features.size(0), -1)


class PolicyHead(nn.Module):
    """Policy network head with proprioception fusion and optional LSTM memory."""

    def __init__(self, feature_dim: int, proprio_dim: int, action_dim: int = 2, use_lstm: bool = True, hidden_size: int = 128):
        super().__init__()
        self.use_lstm = use_lstm
        self.hidden_size = hidden_size
        
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        
        # LSTM for temporal memory
        if self.use_lstm:
            lstm_input_dim = feature_dim + 64  # Visual features + proprioception
            self.lstm = nn.LSTM(
                input_size=lstm_input_dim,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True
            )
            policy_input_dim = hidden_size
        else:
            policy_input_dim = feature_dim + 64
        
        self.policy = nn.Sequential(
            nn.Linear(policy_input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_dim),
        )

        # Initialize final layer to output small actions (near-zero mean)
        # Use gain=0.1 instead of 0.01 to avoid quantization issues in RKNN
        nn.init.orthogonal_(self.policy[-1].weight, gain=0.1)
        nn.init.constant_(self.policy[-1].bias, 0.0)

    def forward(self, features: torch.Tensor, proprio: torch.Tensor, hidden_state=None) -> tuple:
        """Forward pass with optional hidden state.
        
        Args:
            features: (B, feature_dim) visual features
            proprio: (B, proprio_dim) proprioception
            hidden_state: Optional (h, c) tuple for LSTM, each (1, B, hidden_size)
            
        Returns:
            If use_lstm: (actions, (h_new, c_new))
            Else: (actions, None)
        """
        proprio_feat = self.proprio_encoder(proprio)
        combined = torch.cat([features, proprio_feat], dim=1)  # (B, feature_dim + 64)
        
        if self.use_lstm:
            # LSTM expects (B, seq_len, input_dim), we have single timestep
            lstm_input = combined.unsqueeze(1)  # (B, 1, feature_dim + 64)
            
            if hidden_state is None:
                # Initialize hidden state to zeros
                batch_size = features.size(0)
                h0 = torch.zeros(1, batch_size, self.hidden_size, device=features.device)
                c0 = torch.zeros(1, batch_size, self.hidden_size, device=features.device)
                hidden_state = (h0, c0)
            
            lstm_out, hidden_state_new = self.lstm(lstm_input, hidden_state)
            lstm_out = lstm_out.squeeze(1)  # (B, hidden_size)
            actions = self.policy(lstm_out)
            return actions, hidden_state_new
        else:
            actions = self.policy(combined)
            return actions, None


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

        # Initialize final layer to output near-zero values
        # This prevents huge value loss on first training step when model is random
        nn.init.orthogonal_(self.value[-1].weight, gain=0.01)
        nn.init.constant_(self.value[-1].bias, 0.0)

    def forward(self, features: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        proprio_feat = self.proprio_encoder(proprio)
        combined = torch.cat([features, proprio_feat], dim=1)
        return self.value(combined).squeeze(-1)


class QNetwork(nn.Module):
    """Critic network for SAC (Q-function).

    Estimates Q(s, a) - the expected return for taking action a in state s.
    """
    def __init__(self, feature_dim: int, proprio_dim: int = 6, action_dim: int = 2, dropout: float = 0.0):
        super().__init__()

        # Proprioception encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        # Q-network: takes (features + proprio + action)
        # Build with conditional dropout for DroQ
        layers = [
            nn.Linear(feature_dim + 64 + action_dim, 256),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))

        layers.extend([
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        ])
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))

        layers.append(nn.Linear(128, 1))

        self.q_net = nn.Sequential(*layers)

    def forward(self, features, proprio, action):
        proprio_feat = self.proprio_encoder(proprio)
        combined = torch.cat([features, proprio_feat, action], dim=1)
        return self.q_net(combined)


class GaussianPolicyHead(nn.Module):
    """SAC Policy head that outputs mean and log_std for continuous actions."""

    def __init__(self, feature_dim: int, proprio_dim: int, action_dim: int = 2, hidden_size: int = 256):
        super().__init__()

        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        self.net = nn.Sequential(
            nn.Linear(feature_dim + 64, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
        )

        self.mean_layer = nn.Linear(hidden_size, action_dim)
        self.log_std_layer = nn.Linear(hidden_size, action_dim)

    def forward(self, features: torch.Tensor, proprio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        proprio_feat = self.proprio_encoder(proprio)
        combined = torch.cat([features, proprio_feat], dim=1)

        x = self.net(combined)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)

        # Clamp log_std for stability
        log_std = torch.clamp(log_std, -20, 2)

        return mean, log_std
