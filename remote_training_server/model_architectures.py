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
