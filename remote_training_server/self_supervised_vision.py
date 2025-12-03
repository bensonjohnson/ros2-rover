#!/usr/bin/env python3
"""Self-Supervised Vision Model for Semantic Understanding.

This model learns semantic features from RGB-D data without manual labeling.
It uses multiple self-supervised tasks:
1. Depth prediction from RGB (main task)
2. Edge/discontinuity detection
3. Temporal consistency encoding

The learned features are used for:
- Reward augmentation (traversability, obstacle proximity)
- Privileged information for asymmetric critic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class DepthPredictionHead(nn.Module):
    """Predicts depth map from RGB features."""

    def __init__(self, feature_channels: int = 128):
        super().__init__()

        # Decoder to upsample features back to depth resolution
        self.decoder = nn.Sequential(
            # Input: feature_channels x H/16 x W/16
            nn.ConvTranspose2d(feature_channels, 64, kernel_size=4, stride=2, padding=1),  # H/8 x W/8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # H/4 x W/4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # H/2 x W/2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),   # H x W
            nn.Sigmoid()  # Normalize to [0, 1]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, H/16, W/16) from encoder
        Returns:
            depth_pred: (B, 1, H, W) predicted depth
        """
        return self.decoder(features)


class EdgeDetectionHead(nn.Module):
    """Detects depth discontinuities and edges (obstacle boundaries)."""

    def __init__(self, feature_channels: int = 128):
        super().__init__()

        # Edge detection decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Edge probability [0, 1]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, H/16, W/16)
        Returns:
            edges: (B, 1, H, W) edge probability map
        """
        return self.decoder(features)


class TemporalConsistencyHead(nn.Module):
    """Learns temporal consistency for identifying static vs dynamic elements."""

    def __init__(self, feature_dim: int = 128):
        super().__init__()

        # Embedding network for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, feature_dim) global features
        Returns:
            embeddings: (B, 64) normalized embeddings for contrastive loss
        """
        embeddings = self.projector(features)
        return F.normalize(embeddings, dim=-1)


class SelfSupervisedEncoder(nn.Module):
    """Lightweight RGB encoder for self-supervised learning.

    Similar to RGBDEncoder but only processes RGB (depth is prediction target).
    """

    def __init__(self, rgb_channels: int = 3):
        super().__init__()

        # RGB encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(rgb_channels, 32, kernel_size=5, stride=2, padding=2),  # H/2 x W/2
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # H/4 x W/4
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # H/8 x W/8
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # H/16 x W/16
            nn.ReLU(inplace=True),
        )

        # Global pooling for temporal consistency task
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = 128

    def forward(self, rgb: torch.Tensor, return_spatial: bool = True) -> Dict[str, torch.Tensor]:
        """
        Args:
            rgb: (B, 3, H, W) normalized RGB image
            return_spatial: If True, return spatial features for dense prediction tasks
        Returns:
            Dict with 'spatial_features' (B, 128, H/16, W/16) and 'global_features' (B, 128)
        """
        x = self.conv1(rgb)
        x = self.conv2(x)
        x = self.conv3(x)
        spatial_features = self.conv4(x)  # (B, 128, H/16, W/16)

        # Global features for temporal consistency
        global_features = self.pool(spatial_features).view(spatial_features.size(0), -1)

        return {
            'spatial_features': spatial_features,
            'global_features': global_features
        }


class SelfSupervisedVisionModel(nn.Module):
    """Multi-task self-supervised model for semantic understanding.

    Learns from RGB-D data without labels using:
    1. Depth prediction (RGB → Depth)
    2. Edge detection (depth discontinuities)
    3. Temporal consistency (consecutive frames should be similar)
    """

    def __init__(self):
        super().__init__()

        # Shared encoder
        self.encoder = SelfSupervisedEncoder(rgb_channels=3)

        # Task-specific heads
        self.depth_head = DepthPredictionHead(feature_channels=128)
        self.edge_head = EdgeDetectionHead(feature_channels=128)
        self.temporal_head = TemporalConsistencyHead(feature_dim=128)

    def forward(self, rgb: torch.Tensor, return_all: bool = True) -> Dict[str, torch.Tensor]:
        """
        Args:
            rgb: (B, 3, H, W) normalized RGB image [0, 1]
            return_all: If True, compute all tasks. If False, only encoder features.
        Returns:
            Dict containing:
            - 'spatial_features': (B, 128, H/16, W/16) for reward extraction
            - 'global_features': (B, 128) for critic
            - 'depth_pred': (B, 1, H, W) predicted depth
            - 'edge_pred': (B, 1, H, W) predicted edges
            - 'temporal_embedding': (B, 64) for temporal consistency
        """
        # Encode RGB
        encoder_out = self.encoder(rgb)
        spatial_features = encoder_out['spatial_features']
        global_features = encoder_out['global_features']

        if not return_all:
            return {
                'spatial_features': spatial_features,
                'global_features': global_features
            }

        # Predict depth
        depth_pred = self.depth_head(spatial_features)

        # Detect edges
        edge_pred = self.edge_head(spatial_features)

        # Temporal embedding
        temporal_embedding = self.temporal_head(global_features)

        return {
            'spatial_features': spatial_features,
            'global_features': global_features,
            'depth_pred': depth_pred,
            'edge_pred': edge_pred,
            'temporal_embedding': temporal_embedding
        }

    def compute_loss(self,
                     rgb: torch.Tensor,
                     depth_gt: torch.Tensor,
                     rgb_next: torch.Tensor = None,
                     weights: Dict[str, float] = None) -> Dict[str, torch.Tensor]:
        """Compute self-supervised losses.

        Args:
            rgb: (B, 3, H, W) current RGB
            depth_gt: (B, 1, H, W) ground truth depth
            rgb_next: (B, 3, H, W) next frame RGB (for temporal consistency)
            weights: Loss weights dict (default: {depth: 1.0, edge: 0.1, temporal: 0.1})
        Returns:
            Dict of losses and total loss
        """
        if weights is None:
            weights = {'depth': 1.0, 'edge': 0.1, 'temporal': 0.1}

        # Forward pass
        outputs = self.forward(rgb)

        # 1. Depth prediction loss (L1)
        depth_pred = outputs['depth_pred']

        # Resize depth_pred to match depth_gt dimensions if needed
        # (encoder/decoder may not preserve exact dimensions for non-16-divisible sizes)
        if depth_pred.shape != depth_gt.shape:
            depth_pred = F.interpolate(
                depth_pred,
                size=(depth_gt.shape[2], depth_gt.shape[3]),
                mode='bilinear',
                align_corners=False
            )

        depth_loss = F.l1_loss(depth_pred, depth_gt)

        # 2. Edge detection loss (derived from depth gradients)
        # Compute ground truth edges from depth using Sobel filter
        edge_gt = self._compute_depth_edges(depth_gt)
        edge_pred = outputs['edge_pred']

        # Resize edge_pred to match edge_gt dimensions if needed
        if edge_pred.shape != edge_gt.shape:
            edge_pred = F.interpolate(
                edge_pred,
                size=(edge_gt.shape[2], edge_gt.shape[3]),
                mode='bilinear',
                align_corners=False
            )

        edge_loss = F.binary_cross_entropy(edge_pred, edge_gt)

        # 3. Temporal consistency loss (if next frame provided)
        temporal_loss = torch.tensor(0.0, device=rgb.device)
        if rgb_next is not None:
            outputs_next = self.forward(rgb_next)
            embedding_curr = outputs['temporal_embedding']
            embedding_next = outputs_next['temporal_embedding']

            # Cosine similarity (should be high for consecutive frames)
            similarity = F.cosine_similarity(embedding_curr, embedding_next, dim=-1)
            temporal_loss = 1.0 - similarity.mean()  # Loss = 1 - similarity

        # Total weighted loss
        total_loss = (
            weights['depth'] * depth_loss +
            weights['edge'] * edge_loss +
            weights['temporal'] * temporal_loss
        )

        return {
            'total': total_loss,
            'depth': depth_loss,
            'edge': edge_loss,
            'temporal': temporal_loss
        }

    def _compute_depth_edges(self, depth: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
        """Compute edge map from depth using Sobel filter.

        Args:
            depth: (B, 1, H, W)
            threshold: Edge threshold
        Returns:
            edges: (B, 1, H, W) binary edge map
        """
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=depth.dtype, device=depth.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=depth.dtype, device=depth.device).view(1, 1, 3, 3)

        # Compute gradients
        grad_x = F.conv2d(depth, sobel_x, padding=1)
        grad_y = F.conv2d(depth, sobel_y, padding=1)

        # Gradient magnitude
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        # Threshold to get binary edges
        edges = (gradient_magnitude > threshold).float()

        return edges


# Utility function for training
def train_self_supervised_step(model: SelfSupervisedVisionModel,
                               optimizer: torch.optim.Optimizer,
                               rgb: torch.Tensor,
                               depth: torch.Tensor,
                               rgb_next: torch.Tensor = None) -> Dict[str, float]:
    """Single training step for self-supervised model.

    Args:
        model: SelfSupervisedVisionModel
        optimizer: Optimizer
        rgb: (B, 3, H, W) current RGB
        depth: (B, 1, H, W) ground truth depth
        rgb_next: (B, 3, H, W) next frame (optional)
    Returns:
        Dict of scalar losses
    """
    model.train()
    optimizer.zero_grad()

    # Compute losses
    losses = model.compute_loss(rgb, depth, rgb_next)

    # Backward pass
    losses['total'].backward()
    optimizer.step()

    # Return scalar losses
    return {k: v.item() for k, v in losses.items()}
