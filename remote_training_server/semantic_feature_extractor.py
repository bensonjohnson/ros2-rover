#!/usr/bin/env python3
"""Semantic Feature Extraction for Reward Augmentation.

Extracts high-level semantic features from RGB-D data using the self-supervised model.
These features are used for:
1. Reward augmentation (traversability, obstacle proximity, terrain smoothness)
2. Privileged information for asymmetric critic

All features are derived from depth statistics and learned representations (no labels required).
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np


def compute_local_variance(tensor: torch.Tensor, kernel_size: int = 15) -> torch.Tensor:
    """Compute local variance using sliding window.

    Args:
        tensor: (B, 1, H, W) input tensor
        kernel_size: Size of sliding window
    Returns:
        variance_map: (B, 1, H, W) local variance
    """
    # Compute local mean
    padding = kernel_size // 2
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=tensor.device) / (kernel_size ** 2)
    mean = F.conv2d(tensor, kernel, padding=padding)

    # Compute local variance
    squared_diff = (tensor - mean) ** 2
    variance = F.conv2d(squared_diff, kernel, padding=padding)

    return variance


def compute_depth_gradient_magnitude(depth: torch.Tensor) -> torch.Tensor:
    """Compute depth gradient magnitude (edge strength).

    Args:
        depth: (B, 1, H, W)
    Returns:
        gradient_magnitude: (B, 1, H, W)
    """
    # Sobel filters
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=depth.dtype, device=depth.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=depth.dtype, device=depth.device).view(1, 1, 3, 3)

    grad_x = F.conv2d(depth, sobel_x, padding=1)
    grad_y = F.conv2d(depth, sobel_y, padding=1)

    gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    return gradient_magnitude


def extract_forward_roi(tensor: torch.Tensor, roi_params: Dict[str, float] = None) -> torch.Tensor:
    """Extract forward-facing region of interest (where rover will drive).

    Args:
        tensor: (B, 1, H, W) input tensor
        roi_params: Dict with 'top', 'bottom', 'left', 'right' as fractions of image size
    Returns:
        roi: (B, 1, roi_H, roi_W) forward ROI
    """
    if roi_params is None:
        # Default: lower-center region (forward path)
        # H: 240, W: 424
        # Focus on bottom 40% (height 144-240) and center 60% (width 85-339)
        roi_params = {
            'top': 0.6,      # Start at 60% down (row 144)
            'bottom': 1.0,   # End at bottom (row 240)
            'left': 0.2,     # Start at 20% from left (col 85)
            'right': 0.8     # End at 80% from left (col 339)
        }

    B, C, H, W = tensor.shape
    top_idx = int(H * roi_params['top'])
    bottom_idx = int(H * roi_params['bottom'])
    left_idx = int(W * roi_params['left'])
    right_idx = int(W * roi_params['right'])

    roi = tensor[:, :, top_idx:bottom_idx, left_idx:right_idx]
    return roi


def compute_traversability_score(depth: torch.Tensor,
                                 depth_variance: torch.Tensor,
                                 edges: torch.Tensor,
                                 roi_params: Dict[str, float] = None) -> torch.Tensor:
    """Compute traversability score for forward path.

    High traversability = smooth, flat terrain ahead.
    Low traversability = rough, uneven, or obstacle-filled terrain.

    Args:
        depth: (B, 1, H, W) depth map [0, 1]
        depth_variance: (B, 1, H, W) local depth variance
        edges: (B, 1, H, W) edge prediction [0, 1]
        roi_params: ROI parameters
    Returns:
        traversability: (B,) scalar traversability score [0, 1]
    """
    # Extract forward ROI
    depth_roi = extract_forward_roi(depth, roi_params)
    variance_roi = extract_forward_roi(depth_variance, roi_params)
    edges_roi = extract_forward_roi(edges, roi_params)

    # Traversability components:
    # 1. Low variance (smooth terrain)
    smoothness_score = 1.0 - torch.clamp(variance_roi.mean(dim=[1, 2, 3]), 0, 1)

    # 2. Few edges (no obstacles)
    clearance_score = 1.0 - edges_roi.mean(dim=[1, 2, 3])

    # 3. Moderate depth (not too close, not too far)
    # Ideal depth range: 0.3-0.7 (normalized)
    depth_mean = depth_roi.mean(dim=[1, 2, 3])
    depth_score = 1.0 - torch.abs(depth_mean - 0.5) * 2.0  # Penalty for extreme depths
    depth_score = torch.clamp(depth_score, 0, 1)

    # Combine scores (weighted average)
    traversability = (
        0.4 * smoothness_score +
        0.4 * clearance_score +
        0.2 * depth_score
    )

    return traversability


def compute_obstacle_proximity(depth: torch.Tensor,
                               edges: torch.Tensor,
                               roi_params: Dict[str, float] = None) -> torch.Tensor:
    """Compute obstacle proximity in forward path.

    High proximity = close obstacles detected.
    Low proximity = clear path.

    Args:
        depth: (B, 1, H, W)
        edges: (B, 1, H, W) edge map
        roi_params: ROI parameters
    Returns:
        proximity: (B,) obstacle proximity score [0, 1]
    """
    # Extract forward ROI
    depth_roi = extract_forward_roi(depth, roi_params)
    edges_roi = extract_forward_roi(edges, roi_params)

    # Obstacles are characterized by:
    # 1. Strong edges (depth discontinuities)
    edge_strength = edges_roi.mean(dim=[1, 2, 3])

    # 2. Close depth values (high depth in normalized [0,1] = close in real world)
    # Note: Depth is typically inverted (1 = close, 0 = far)
    closeness = depth_roi.mean(dim=[1, 2, 3])

    # Combine: high edge strength + close depth = obstacle nearby
    proximity = 0.6 * edge_strength + 0.4 * closeness

    return proximity


def compute_terrain_roughness(depth_variance: torch.Tensor,
                              roi_params: Dict[str, float] = None) -> torch.Tensor:
    """Compute terrain roughness (variance in depth).

    Args:
        depth_variance: (B, 1, H, W)
        roi_params: ROI parameters
    Returns:
        roughness: (B,) terrain roughness score [0, 1]
    """
    variance_roi = extract_forward_roi(depth_variance, roi_params)
    roughness = torch.clamp(variance_roi.mean(dim=[1, 2, 3]), 0, 1)
    return roughness


def extract_semantic_features(rgb: torch.Tensor,
                              depth: torch.Tensor,
                              semantic_model,
                              roi_params: Dict[str, float] = None) -> Dict[str, torch.Tensor]:
    """Extract all semantic features for reward augmentation and critic.

    Args:
        rgb: (B, 3, H, W) RGB image [0, 1]
        depth: (B, 1, H, W) depth map [0, 1]
        semantic_model: SelfSupervisedVisionModel
        roi_params: ROI parameters
    Returns:
        Dict containing:
        - 'global_features': (B, 128) for asymmetric critic
        - 'spatial_features': (B, 128, H/16, W/16) spatial features
        - 'traversability_score': (B,) [0, 1]
        - 'obstacle_proximity': (B,) [0, 1]
        - 'terrain_roughness': (B,) [0, 1]
        - 'depth_error': (B,) prediction error (anomaly detection)
        - 'edge_map': (B, 1, H, W) predicted edges
    """
    with torch.no_grad():
        semantic_model.eval()

        # Run self-supervised model
        outputs = semantic_model(rgb, return_all=True)

        # Extract outputs
        global_features = outputs['global_features']  # (B, 128)
        spatial_features = outputs['spatial_features']  # (B, 128, H/16, W/16)
        depth_pred = outputs['depth_pred']  # (B, 1, H_pred, W_pred)
        edge_pred = outputs['edge_pred']  # (B, 1, H_pred, W_pred)

        # Resize predictions to match input depth size (in case of slight size mismatch)
        if depth_pred.shape != depth.shape:
            depth_pred = F.interpolate(depth_pred, size=depth.shape[2:], mode='bilinear', align_corners=False)
            edge_pred = F.interpolate(edge_pred, size=depth.shape[2:], mode='bilinear', align_corners=False)

        # Compute depth statistics
        depth_variance = compute_local_variance(depth, kernel_size=15)
        depth_gradients = compute_depth_gradient_magnitude(depth)

        # Depth prediction error (anomaly detection)
        # High error = unusual/unexpected scene
        depth_error = torch.abs(depth_pred - depth).mean(dim=[1, 2, 3])

        # Compute semantic scores
        traversability_score = compute_traversability_score(
            depth, depth_variance, edge_pred, roi_params
        )

        obstacle_proximity = compute_obstacle_proximity(
            depth, edge_pred, roi_params
        )

        terrain_roughness = compute_terrain_roughness(
            depth_variance, roi_params
        )

    return {
        # For asymmetric critic (privileged info)
        'global_features': global_features,
        'spatial_features': spatial_features,

        # For reward augmentation (scalar scores)
        'traversability_score': traversability_score,
        'obstacle_proximity': obstacle_proximity,
        'terrain_roughness': terrain_roughness,
        'depth_error': depth_error,

        # Dense maps (for visualization or advanced rewards)
        'edge_map': edge_pred,
        'depth_variance_map': depth_variance,
        'depth_pred': depth_pred
    }


def augment_reward(base_reward: torch.Tensor,
                  semantic_features: Dict[str, torch.Tensor],
                  weights: Dict[str, float] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Augment base reward with semantic understanding.

    Args:
        base_reward: (B,) base reward from rover
        semantic_features: Dict from extract_semantic_features()
        weights: Reward component weights
    Returns:
        augmented_reward: (B,) total reward
        reward_components: Dict of individual reward components (for logging)
    """
    if weights is None:
        weights = {
            'traversability': 0.1,
            'obstacle': -0.2,
            'roughness': -0.05
        }

    # Extract scores
    traversability = semantic_features['traversability_score']
    obstacle_proximity = semantic_features['obstacle_proximity']
    terrain_roughness = semantic_features['terrain_roughness']

    # Compute semantic reward components
    r_traversability = weights['traversability'] * traversability
    r_obstacle = weights['obstacle'] * obstacle_proximity
    r_roughness = weights['roughness'] * terrain_roughness

    # Total augmented reward
    augmented_reward = base_reward + r_traversability + r_obstacle + r_roughness

    reward_components = {
        'base': base_reward,
        'traversability': r_traversability,
        'obstacle': r_obstacle,
        'roughness': r_roughness,
        'total': augmented_reward
    }

    return augmented_reward, reward_components


# Batch processing utilities
def extract_batch_semantic_features(rgb_batch: torch.Tensor,
                                    depth_batch: torch.Tensor,
                                    semantic_model,
                                    device: torch.device,
                                    batch_size: int = 32) -> Dict[str, torch.Tensor]:
    """Extract semantic features for a large batch (with mini-batching to avoid OOM).

    Args:
        rgb_batch: (N, 3, H, W) large batch of RGB images
        depth_batch: (N, 1, H, W) large batch of depth maps
        semantic_model: SelfSupervisedVisionModel
        device: Device to run on
        batch_size: Mini-batch size
    Returns:
        Dict of semantic features for entire batch
    """
    N = rgb_batch.shape[0]
    num_batches = (N + batch_size - 1) // batch_size

    all_features = {
        'global_features': [],
        'traversability_score': [],
        'obstacle_proximity': [],
        'terrain_roughness': [],
        'depth_error': []
    }

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, N)

        rgb_mini = rgb_batch[start_idx:end_idx].to(device)
        depth_mini = depth_batch[start_idx:end_idx].to(device)

        features = extract_semantic_features(rgb_mini, depth_mini, semantic_model)

        # Collect results
        for key in all_features.keys():
            all_features[key].append(features[key].cpu())

    # Concatenate all mini-batches
    for key in all_features.keys():
        all_features[key] = torch.cat(all_features[key], dim=0)

    return all_features
