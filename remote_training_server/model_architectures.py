#!/usr/bin/env python3
"""Model architectures for MAP-Elites rover training.

This file contains only the neural network architectures without any
training dependencies, making it safe to import on the rover for model conversion.
"""

import torch
import torch.nn as nn
from typing import Tuple


class SpatialAttention(nn.Module):
    """
    Learns to focus on important spatial regions (gaps, obstacles).
    Applies channel-wise attention to emphasize relevant features.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x):
        attention = torch.sigmoid(self.conv(x))  # (B, 1, H, W)
        return x * attention  # Element-wise weighting


class OccupancyGridEncoder(nn.Module):
    """
    Encoder for 4-channel 128×128 occupancy grid.
    Preserves spatial structure without global pooling.

    Input channels:
    - 0: Distance to nearest obstacle [0.0, 1.0]
    - 1: Exploration history [0.0, 1.0]
    - 2: Obstacle confidence [0.0, 1.0]
    - 3: Terrain height [0.0, 1.0]
    """

    def __init__(self, input_channels: int = 4):
        super().__init__()

        # Input: (B, 4, 128, 128) - Matched to rover resolution
        self.conv = nn.Sequential(
            # Stage 1: 128 → 64
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # Stage 2: 64 → 32
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # Stage 3: 32 → 16
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # Stage 4: 16 → 8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # Output: 128 * 8 * 8 = 8192 features (preserves more spatial detail)
        self.output_dim = 128 * 8 * 8

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid: (B, 4, 128, 128) normalized to [0, 1]
        Returns:
            features: (B, 4096) - preserves spatial info!
        """
        x = self.conv(grid)  # (B, 256, 4, 4)
        x = x.flatten(start_dim=1)  # (B, 4096) - preserves spatial structure
        return x


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
    def __init__(self, feature_dim: int = 2048, proprio_dim: int = 10, action_dim: int = 2, dropout: float = 0.0):
        super().__init__()

        # Proprioception encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 32),
            nn.ReLU(inplace=True),
        )

        # Q-function: visual (2048) + proprio (32) + action (2) = 2082
        # Build with conditional dropout for DroQ
        layers = [
            nn.Linear(feature_dim + 32 + action_dim, 256),
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

    def __init__(self, feature_dim: int = 2048, proprio_dim: int = 10, action_dim: int = 2, hidden_size: int = 128):
        super().__init__()

        # Proprioception encoder (unchanged)
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        # Fusion: 2048 (visual) + 32 (proprio) = 2080
        self.net = nn.Sequential(
            nn.Linear(feature_dim + 32, 256),  # Smaller first layer
            nn.ReLU(inplace=True),
            nn.Linear(256, hidden_size),
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


class LaserEncoder(nn.Module):
    """
    Encoder for 128×128 laser occupancy grid.
    Input: (B, 1, 128, 128)
    Output: (B, 4096) features
    """
    def __init__(self):
        super().__init__()
        # 128 -> 64 -> 32 -> 16 -> 8
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)   # 128->64
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 64->32
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 32->16
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)  # 16->8

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Input: (B, 1, 128, 128)
        x = self.relu(self.conv1(x))  # (B, 16, 64, 64)
        x = self.relu(self.conv2(x))  # (B, 32, 32, 32)
        x = self.relu(self.conv3(x))  # (B, 64, 16, 16)
        x = self.relu(self.conv4(x))  # (B, 64, 8, 8)

        x = x.flatten(start_dim=1)    # (B, 64*8*8) = (B, 4096)
        return x

    @property
    def output_dim(self):
        return 4096


class RGBDEncoder(nn.Module):
    """
    Encoder for 4-channel 240×424 RGB-D image.
    Input: (B, 4, 240, 424) - RGB + Depth
    Output: (B, 11648) features
    """
    def __init__(self):
        super().__init__()
        # 240x424 (H, W) -> 120x212 -> 60x106 -> 30x53 -> 15x26 -> 7x13
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1)   # 240x424 -> 120x212
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 120x212 -> 60x106
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 60x106 -> 30x53
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # 30x53 -> 15x26
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)# 15x26 -> 7x13

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Input: (B, 4, 240, 424)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))

        x = x.flatten(start_dim=1)    # (B, 128*7*13) = (B, 11648)
        return x

    @property
    def output_dim(self):
        return 11648  # 128 * 7 * 13 = 11648 features

class DepthEncoder(nn.Module):
    """
    Encoder for 240x424 raw depth image.
    Input: (B, 1, 240, 424)
    Output: (B, 11648) features
    """
    def __init__(self):
        super().__init__()
        # 240x424 (H, W) -> 120x212 -> 60x106 -> 30x53 -> 15x26 -> 7x13
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)   # 240x424 -> 120x212
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 212x120 -> 106x60
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 106x60 -> 53x30
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # 53x30 -> 26x15
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)# 26x15 -> 13x7

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Input: (B, 1, 424, 240)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))

        x = x.flatten(start_dim=1)    # (B, 128*14*8) = (B, 14336). Note: 13*7 was incorrect.
        return x

    @property
    def output_dim(self):
        return 14336 # 128 * 7 * 14 = 12544? No wait.
        # 240 -s2-> 120 -s2-> 60 -s2-> 30 -s2-> 15 -s2-> 8 (padding=1 keeps it ceil(H/2))
        # 15 / 2 = 7.5 -> 8 (if padding=1, kernel=3, stride=2: out = floor((in + 2*1 - 3)/2 + 1)
        # formula: floor((W + 2*P - K)/S + 1)
        # 240 -> (240+2-3)/2 + 1 = 119.5 -> 120
        # 120 -> 60
        # 60 -> 30
        # 30 -> 15
        # 15 -> (15+2-3)/2 + 1 = 7.5 -> 7
        
        # 424 -> 212 -> 106 -> 53 -> 26 -> 13
        
        # Last layer: 128 channels * 7 * 13 = 11648
        return 12544  # 128 * 7 * 14 = 12544 features


class DualEncoderPolicyNetwork(nn.Module):
    """Policy network using dual encoders (Laser + Depth)."""
    def __init__(self, action_dim=2, proprio_dim=10, hidden_size=128):
        super().__init__()

        # Separate encoders
        self.laser_encoder = LaserEncoder()    # -> 4096 features
        self.depth_encoder = DepthEncoder()    # -> 11648 features

        # Proprioception encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        # Fusion: laser + depth + proprio
        fusion_dim = self.laser_encoder.output_dim + self.depth_encoder.output_dim + 32

        self.net = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, hidden_size),
            nn.ReLU(inplace=True),
        )

        self.mean_layer = nn.Linear(hidden_size, action_dim)
        self.log_std_layer = nn.Linear(hidden_size, action_dim)

    def forward(self, laser_grid, depth_img, proprio):
        # Encode each modality
        laser_feats = self.laser_encoder(laser_grid)
        depth_feats = self.depth_encoder(depth_img)
        proprio_feats = self.proprio_encoder(proprio)

        # Concatenate features
        fused = torch.cat([laser_feats, depth_feats, proprio_feats], dim=1)

        x = self.net(fused)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)

        # Clamp log_std
        log_std = torch.clamp(log_std, -20, 2)

        return mean, log_std


class DualEncoderQNetwork(nn.Module):
    """Q-network using dual encoders (Laser + Depth)."""
    def __init__(self, action_dim=2, proprio_dim=10, dropout=0.0):
        super().__init__()

        self.laser_encoder = LaserEncoder()
        self.depth_encoder = DepthEncoder()
        
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 32),
            nn.ReLU(inplace=True),
        )

        # Q-network: fusion + action
        fusion_dim = self.laser_encoder.output_dim + self.depth_encoder.output_dim + 32 + action_dim

        layers = [
            nn.Linear(fusion_dim, 256),
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

    def forward(self, laser_grid, depth_img, proprio, action):
        laser_feats = self.laser_encoder(laser_grid)
        depth_feats = self.depth_encoder(depth_img)
        proprio_feats = self.proprio_encoder(proprio)

        fused = torch.cat([laser_feats, depth_feats, proprio_feats, action], dim=1)

        q_value = self.q_net(fused)
        return q_value

class RGBDEncoderPolicyNetwork(nn.Module):
    """Policy network using RGBD + Laser encoders."""
    def __init__(self, action_dim=2, proprio_dim=10, hidden_size=128):
        super().__init__()

        # Separate encoders
        self.laser_encoder = LaserEncoder()    # -> 4096 features
        self.rgbd_encoder = RGBDEncoder()      # -> 11648 features

        # Proprioception encoder (unchanged)
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        # Fusion: laser + rgbd + proprio
        fusion_dim = self.laser_encoder.output_dim + self.rgbd_encoder.output_dim + 32

        self.net = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, hidden_size),
            nn.ReLU(inplace=True),
        )

        self.mean_layer = nn.Linear(hidden_size, action_dim)
        self.log_std_layer = nn.Linear(hidden_size, action_dim)

    def forward(self, laser_grid, rgbd_img, proprio):
        # Encode each modality
        laser_feats = self.laser_encoder(laser_grid)
        rgbd_feats = self.rgbd_encoder(rgbd_img)
        proprio_feats = self.proprio_encoder(proprio)

        # Concatenate features
        fused = torch.cat([laser_feats, rgbd_feats, proprio_feats], dim=1)

        x = self.net(fused)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)

        # Clamp log_std
        log_std = torch.clamp(log_std, -20, 2)

        return mean, log_std

class RGBDEncoderQNetwork(nn.Module):
    """Q-network using RGBD + Laser encoders."""
    def __init__(self, action_dim=2, proprio_dim=10, dropout=0.0):
        super().__init__()

        self.laser_encoder = LaserEncoder()
        self.rgbd_encoder = RGBDEncoder()
        
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 32),
            nn.ReLU(inplace=True),
        )

        # Q-network: fusion + action
        fusion_dim = self.laser_encoder.output_dim + self.rgbd_encoder.output_dim + 32 + action_dim

        layers = [
            nn.Linear(fusion_dim, 256),
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

    def forward(self, laser_grid, rgbd_img, proprio, action):
        laser_feats = self.laser_encoder(laser_grid)
        rgbd_feats = self.rgbd_encoder(rgbd_img)
        proprio_feats = self.proprio_encoder(proprio)

        fused = torch.cat([laser_feats, rgbd_feats, proprio_feats, action], dim=1)

        q_value = self.q_net(fused)
        return q_value
