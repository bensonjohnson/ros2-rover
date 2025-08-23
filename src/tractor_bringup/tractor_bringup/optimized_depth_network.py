#!/usr/bin/env python3
"""
Optimized Neural Network Architecture for RK3588 NPU
Lightweight MobileNet-inspired design with efficient attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class DepthwiseSeparableConv2d(nn.Module):
    """Efficient depthwise separable convolution for NPU optimization"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu6(x)  # ReLU6 is more NPU-friendly

class ChannelAttention(nn.Module):
    """Lightweight channel attention mechanism"""
    
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        hidden_channels = max(channels // reduction, 8)  # Minimum 8 channels
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, bias=False)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = torch.sigmoid(avg_out + max_out)
        return x * attention

class OptimizedDepthBackbone(nn.Module):
    """Optimized depth processing backbone for RK3588 NPU"""
    
    def __init__(self, in_channels: int = 1, width_multiplier: float = 1.0):
        super().__init__()
        
        # Calculate channel dimensions based on width multiplier
        def make_divisible(v, divisor=8):
            return max(divisor, int(v + divisor / 2) // divisor * divisor)
        
        # Progressive channel expansion optimized for NPU
        channels = [
            make_divisible(32 * width_multiplier),   # 32
            make_divisible(64 * width_multiplier),   # 64
            make_divisible(128 * width_multiplier),  # 128
            make_divisible(256 * width_multiplier),  # 256
        ]
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU6(inplace=True)
        )
        
        # Efficient depthwise separable blocks
        self.conv2 = DepthwiseSeparableConv2d(channels[0], channels[1], stride=2)
        self.attention1 = ChannelAttention(channels[1])
        
        self.conv3 = DepthwiseSeparableConv2d(channels[1], channels[2], stride=2)
        self.attention2 = ChannelAttention(channels[2])
        
        self.conv4 = DepthwiseSeparableConv2d(channels[2], channels[3], stride=2)
        self.attention3 = ChannelAttention(channels[3])
        
        # Global pooling and feature compression
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_compress = nn.Sequential(
            nn.Conv2d(channels[3], 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True),
            nn.Dropout2d(0.2)
        )
        
        # Calculate output size after all operations
        # Input: 160x288 -> /2=80x144 -> /2=40x72 -> /2=20x36 -> /2=10x18 -> global_pool=1x1
        self.output_features = 256
    
    def forward(self, x):
        # Progressive feature extraction with attention
        x = self.conv1(x)                    # [B, 32, 80, 144]
        
        x = self.conv2(x)                    # [B, 64, 40, 72]
        x = self.attention1(x)
        
        x = self.conv3(x)                    # [B, 128, 20, 36]
        x = self.attention2(x)
        
        x = self.conv4(x)                    # [B, 256, 10, 18]
        x = self.attention3(x)
        
        x = self.global_pool(x)              # [B, 256, 1, 1]
        x = self.feature_compress(x)         # [B, 256, 1, 1]
        
        return x.flatten(1)                  # [B, 256]

class AdaptiveProprioceptiveProcessor(nn.Module):
    """Adaptive proprioceptive data processor with feature importance learning"""
    
    def __init__(self, input_dim: int, output_dim: int = 128):
        super().__init__()
        
        # Feature importance learning
        self.feature_importance = nn.Parameter(torch.ones(input_dim))
        
        # Efficient MLP with residual connections
        hidden_dim = max(input_dim * 2, 64)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU6(inplace=True)
        )
        
        # Residual connection for input features
        if input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
        else:
            self.residual_proj = nn.Identity()
    
    def forward(self, x):
        # Apply learned feature importance
        weighted_input = x * torch.sigmoid(self.feature_importance)
        
        # Process through MLP
        processed = self.layers(weighted_input)
        
        # Add residual connection
        residual = self.residual_proj(x)
        return processed + residual * 0.2  # Scaled residual

class OptimizedDepthExplorationNet(nn.Module):
    """
    Optimized neural network for depth-based exploration
    Designed for efficient inference on RK3588 NPU
    """
    
    def __init__(self, stacked_frames: int = 1, extra_proprio: int = 13, 
                 width_multiplier: float = 1.0, enable_temporal: bool = False):
        super().__init__()
        
        self.stacked_frames = stacked_frames
        self.enable_temporal = enable_temporal
        
        # Optimized depth processing backbone
        self.depth_backbone = OptimizedDepthBackbone(
            in_channels=stacked_frames, 
            width_multiplier=width_multiplier
        )
        
        # Proprioceptive processing
        proprio_input_dim = 3 + extra_proprio
        self.proprio_processor = AdaptiveProprioceptiveProcessor(
            input_dim=proprio_input_dim,
            output_dim=128
        )
        
        # Temporal modeling (optional for advanced scenarios)
        if enable_temporal:
            self.temporal_lstm = nn.LSTM(
                input_size=self.depth_backbone.output_features + 128,
                hidden_size=256,
                num_layers=1,
                batch_first=True,
                dropout=0.1
            )
            fusion_input_dim = 256
        else:
            fusion_input_dim = self.depth_backbone.output_features + 128
        
        # Multi-head action prediction
        self.action_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Separate heads for different outputs
        self.linear_velocity_head = nn.Linear(128, 1)
        self.angular_velocity_head = nn.Linear(128, 1) 
        self.confidence_head = nn.Linear(128, 1)
        
        # Value head for advanced RL techniques (optional)
        self.value_head = nn.Linear(128, 1)
        
        # Initialize weights for better convergence
        self._initialize_weights()
        
        # Hidden state for temporal modeling
        self.hidden_state = None
    
    def _initialize_weights(self):
        """Initialize weights for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize action heads with smaller weights
        nn.init.normal_(self.linear_velocity_head.weight, 0, 0.01)
        nn.init.normal_(self.angular_velocity_head.weight, 0, 0.01)
        nn.init.normal_(self.confidence_head.weight, 0, 0.01)
    
    def forward(self, depth_image, sensor_data, return_value: bool = False):
        # Process depth image
        depth_features = self.depth_backbone(depth_image)
        
        # Process proprioceptive data
        proprio_features = self.proprio_processor(sensor_data)
        
        # Combine features
        combined_features = torch.cat([depth_features, proprio_features], dim=1)
        
        # Temporal modeling (if enabled)
        if self.enable_temporal:
            # Reshape for LSTM: [batch, seq_len=1, features]
            lstm_input = combined_features.unsqueeze(1)
            lstm_output, self.hidden_state = self.temporal_lstm(lstm_input, self.hidden_state)
            fused_features = lstm_output.squeeze(1)
        else:
            fused_features = combined_features
        
        # Action prediction
        action_features = self.action_fusion(fused_features)
        
        # Multi-head outputs
        linear_vel = self.linear_velocity_head(action_features)
        angular_vel = self.angular_velocity_head(action_features)
        confidence = self.confidence_head(action_features)
        
        # Combine action outputs
        actions = torch.cat([linear_vel, angular_vel, confidence], dim=1)
        
        if return_value:
            value = self.value_head(action_features)
            return actions, value
        
        return actions
    
    def reset_temporal_state(self):
        """Reset temporal state for new episodes"""
        self.hidden_state = None
    
    def get_feature_importance(self) -> torch.Tensor:
        """Get learned feature importance weights"""
        return torch.sigmoid(self.proprio_processor.feature_importance).detach()

class DynamicInferenceController:
    """
    Controls inference frequency and model complexity based on scene complexity
    and performance requirements
    """
    
    def __init__(self, target_fps: float = 30.0, complexity_window: int = 10):
        self.target_fps = target_fps
        self.target_latency = 1.0 / target_fps
        self.complexity_window = complexity_window
        
        # Performance tracking
        self.recent_latencies = []
        self.recent_complexities = []
        
        # Adaptive parameters
        self.current_width_multiplier = 1.0
        self.min_width_multiplier = 0.5
        self.max_width_multiplier = 1.5
        
        # Complexity thresholds
        self.low_complexity_threshold = 0.3
        self.high_complexity_threshold = 0.7
    
    def update_performance(self, latency: float, scene_complexity: float):
        """Update performance metrics and adjust inference parameters"""
        self.recent_latencies.append(latency)
        self.recent_complexities.append(scene_complexity)
        
        # Keep only recent history
        if len(self.recent_latencies) > self.complexity_window:
            self.recent_latencies.pop(0)
            self.recent_complexities.pop(0)
        
        # Adjust model complexity based on performance
        if len(self.recent_latencies) >= 3:
            avg_latency = np.mean(self.recent_latencies[-3:])
            avg_complexity = np.mean(self.recent_complexities[-3:])
            
            if avg_latency > self.target_latency * 1.2:  # Too slow
                self.current_width_multiplier *= 0.9
                self.current_width_multiplier = max(self.current_width_multiplier, 
                                                  self.min_width_multiplier)
            elif avg_latency < self.target_latency * 0.8 and avg_complexity > self.high_complexity_threshold:
                # Fast enough and complex scene - can afford larger model
                self.current_width_multiplier *= 1.05
                self.current_width_multiplier = min(self.current_width_multiplier, 
                                                  self.max_width_multiplier)
    
    def should_skip_inference(self, scene_complexity: float) -> bool:
        """Decide whether to skip inference based on scene complexity"""
        if len(self.recent_latencies) < 3:
            return False
        
        avg_latency = np.mean(self.recent_latencies[-3:])
        
        # Skip inference if scene is simple and we're running behind
        if (scene_complexity < self.low_complexity_threshold and 
            avg_latency > self.target_latency):
            return True
        
        return False
    
    def get_optimal_width_multiplier(self) -> float:
        """Get current optimal width multiplier"""
        return self.current_width_multiplier
    
    def calculate_scene_complexity(self, depth_image: np.ndarray) -> float:
        """Calculate scene complexity from depth image"""
        try:
            # Simple complexity metrics
            depth_variance = np.var(depth_image[depth_image > 0])
            edge_density = self._calculate_edge_density(depth_image)
            obstacle_density = self._calculate_obstacle_density(depth_image)
            
            # Normalize and combine metrics
            complexity = (
                min(depth_variance / 2.0, 1.0) * 0.4 +
                edge_density * 0.3 +
                obstacle_density * 0.3
            )
            
            return np.clip(complexity, 0.0, 1.0)
        except:
            return 0.5  # Default complexity
    
    def _calculate_edge_density(self, depth_image: np.ndarray) -> float:
        """Calculate edge density in depth image"""
        try:
            import cv2
            # Simple edge detection
            edges = cv2.Canny((depth_image * 255).astype(np.uint8), 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            return min(edge_density * 10, 1.0)
        except:
            return 0.5
    
    def _calculate_obstacle_density(self, depth_image: np.ndarray) -> float:
        """Calculate obstacle density (close objects)"""
        try:
            close_pixels = np.sum((depth_image > 0) & (depth_image < 1.5))
            total_valid_pixels = np.sum(depth_image > 0)
            if total_valid_pixels > 0:
                return close_pixels / total_valid_pixels
            return 0.0
        except:
            return 0.5

# Factory function for creating optimized models
def create_optimized_model(stacked_frames: int = 1, extra_proprio: int = 13,
                          performance_mode: str = "balanced",
                          enable_temporal: bool = False) -> OptimizedDepthExplorationNet:
    """
    Create optimized model based on performance requirements
    
    Args:
        stacked_frames: Number of stacked depth frames
        extra_proprio: Number of extra proprioceptive features
        performance_mode: "fast", "balanced", or "accurate"
        enable_temporal: Whether to enable temporal modeling
    
    Returns:
        Optimized neural network model
    """
    
    # Performance mode configurations
    mode_configs = {
        "fast": {"width_multiplier": 0.75, "enable_temporal": False},
        "balanced": {"width_multiplier": 1.0, "enable_temporal": enable_temporal},
        "accurate": {"width_multiplier": 1.25, "enable_temporal": True}
    }
    
    config = mode_configs.get(performance_mode, mode_configs["balanced"])
    
    model = OptimizedDepthExplorationNet(
        stacked_frames=stacked_frames,
        extra_proprio=extra_proprio,
        width_multiplier=config["width_multiplier"],
        enable_temporal=config["enable_temporal"]
    )
    
    return model