#!/usr/bin/env python3
"""Model architectures for MAP-Elites rover training.

This file contains only the neural network architectures without any
training dependencies, making it safe to import on the rover for model conversion.
"""

import torch
import torch.nn as nn


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
        
        # Spatial Attention
        # Compute attention map from features
        attention = torch.sigmoid(torch.mean(features, dim=1, keepdim=True))
        features = features * attention
        
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

    def forward(self, features: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        proprio_feat = self.proprio_encoder(proprio)
        combined = torch.cat([features, proprio_feat], dim=1)
        return self.value(combined).squeeze(-1)


class CriticNetwork(nn.Module):
    """Critic network for PGA-MAP-Elites (Q-function).
    
    Estimates Q(s, a) - the expected return for taking action a in state s.
    """
    def __init__(self, proprio_dim: int = 6, action_dim: int = 2):
        super().__init__()
        self.encoder = RGBDEncoder()
        
        # Proprioception encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        
        # Q-network: takes (features + proprio + action)
        self.q_net = nn.Sequential(
            nn.Linear(self.encoder.output_dim + 64 + action_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, rgb, depth, proprio, action):
        features = self.encoder(rgb, depth)
        proprio_feat = self.proprio_encoder(proprio)
        
        combined = torch.cat([features, proprio_feat, action], dim=1)
        return self.q_net(combined)
