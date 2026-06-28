"""Deep Exploration Network (DEN) — core neural architecture.

Designed from scratch for RK3588 NPU deployment (ONNX → RKNN) with all rover
sensor inputs fused into a single forward pass. Two output heads:
  - actor: continuous track commands [left, right] ∈ [-1, 1]
  - value: scalar state value for TD-learning

Architecture rationale:
  - The NPU excels at static-graph CNN inference with INT8 quantization.
  - Every op used here maps to NPU hardware ops (Conv2D, Gemm, Relu, Tanh,
    Add, Mul, Concat, Reshape). No dynamic control flow, no loops, no
    autograd — pure feedthrough.
  - LiDAR is treated as a 1D signal (72 bins), refined by a small 1D CNN that
    learns local range structure (openings, obstacles) before the MLP fusion.
  - Occupancy grid is a local 64×64 crop from RTAB-Map, downsampled by a
    tiny 3-layer 2D CNN.
  - Depth image (from D435i) is optionally downsampled to 32×32 and passed
    through a separate tiny CNN stream — but this is optional because NPU
    latency scales with input pixel count.
  - Proprioception (IMU yaw rate + wheel velocities) is a tiny MLP.

Total params: ~210K (actor + value combined), well under RKNN limits.

Usage:
  model = DeepExplorerNetwork()
  model.eval()
  # Export:
  torch.onnx.export(model, dummy_input, "explorer.onnx", ...)
"""

from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ExplorerConfig:
    """All hyperparameters for the Deep Exploration Network."""
    # Input dimensions
    lidar_bins: int = 72
    occ_grid_size: int = 64          # 64×64 crop of occupancy grid
    depth_size: int = 0              # 0 = disable depth stream (save NPU cycles)
    use_depth: bool = False          # set True to enable depth camera input

    # Proprioception: [vx_wheels, vyaw_wheels, imu_yaw_rate, novelty, safety_hold]
    #   (5 dims — wheel odometry gives us vx + vyaw, IMU gives yaw rate, and
    #    we add interoceptive novelty + safety hold from the start script's
    #    pc_active_inference_runner convention)
    proprio_dim: int = 5

    # Latent sizes
    lidar_latent: int = 32           # after 1D conv + MLP
    occ_latent: int = 64             # after 2D conv
    depth_latent: int = 32           # after 2D conv (if enabled)
    proprio_latent: int = 16
    fusion_latent: int = 128

    # Actor
    action_dim: int = 2
    actor_hidden: int = 64

    # Critic (value)
    value_hidden: int = 64

    # Normalization constants (clamp inputs to [0,1] or [-1,1] before model)
    max_lidar_range: float = 5.0
    max_wheel_vel: float = 8.0       # rad/s
    max_yaw_rate: float = 2.5        # rad/s

    # Training
    discount: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_lr: float = 3e-4
    actor_lr: float = 3e-4
    max_grad_norm: float = 1.0
    n_envs: int = 4                  # parallel sim envs during training


class _LidarEncoder(nn.Module):
    """1D CNN over LiDAR bins, then MLP to latent.

    A 72-bin lidar scan is a 1D signal: adjacent bins are spatially related
    (an obstacle at bin 30 is next to the opening at bin 31). A 1D conv
    learns local range-structure features before the MLP.
    """
    def __init__(self, n_bins: int, latent: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),  # 72 -> 72
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),                      # 8
        )
        self.mlp = nn.Sequential(
            nn.Linear(8 + 4, 32),  # 8 from conv + 4 scan statistics
            nn.ReLU(),
            nn.Linear(32, latent),
            nn.Tanh(),
        )

    def forward(self, lidar: torch.Tensor) -> torch.Tensor:
        """lidar: [B, N] normalized [0,1] float32. Returns [B, latent]."""
        B = lidar.shape[0]
        # 1D conv expects [B, C, L]
        feat = self.conv(lidar.unsqueeze(1)).squeeze(-1)  # [B, 8]
        # Scan statistics: mean openness, min distance (obstacle proximity),
        # std dev (roughness), # of bins < 0.3 (free-space ratio)
        stats = torch.stack([
            lidar.mean(dim=-1),
            lidar.min(dim=-1).values,
            lidar.std(dim=-1),
            (lidar < 0.3).float().mean(dim=-1),
        ], dim=-1)  # [B, 4]
        return self.mlp(torch.cat([feat, stats], dim=-1))


class _OccMapEncoder(nn.Module):
    """2D CNN over a local occupancy grid crop (64×64).

    3 conv layers with stride 2: 64 -> 32 -> 16 -> 8. The RK3588 NPU handles
    small feature maps efficiently; 8×8×16 = 1024 -> MLP -> 64 is cheap.
    """
    def __init__(self, grid_size: int, latent: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),   # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(8, 12, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(12, 16, kernel_size=3, stride=2, padding=1), # 16 -> 8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 96),
            nn.ReLU(),
            nn.Linear(96, latent),
            nn.Tanh(),
        )

    def forward(self, occ: torch.Tensor) -> torch.Tensor:
        """occ: [B, H, W] float32 in [0,1] (1=occupied, 0=free, 0.5=unknown).
        Returns [B, latent]."""
        B = occ.shape[0]
        # Add channel dim for Conv2d
        return self.cnn(occ.unsqueeze(1))


class _DepthEncoder(nn.Module):
    """Optional tiny 2D CNN for downsampled depth image (32×32)."""
    def __init__(self, latent: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1),   # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),   # 16 -> 8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, latent),
            nn.Tanh(),
        )

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """depth: [B, H, W] float32 normalized [0,1]. Returns [B, latent]."""
        return self.cnn(depth.unsqueeze(1))


class _ProprioEncoder(nn.Module):
    """Tiny MLP for proprioception: wheel vx, vyaw, IMU yaw rate, novelty, safety hold."""
    def __init__(self, dim: int, latent: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 24),
            nn.ReLU(),
            nn.Linear(24, latent),
            nn.Tanh(),
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        """proprio: [B, D] normalized. Returns [B, latent]."""
        return self.net(proprio)


class _ActorHead(nn.Module):
    """Continuous action: [left_track, right_track] ∈ [-1, 1] with tanh."""
    def __init__(self, fusion_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fusion_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
            nn.Tanh(),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Returns [B, 2] in [-1, 1]."""
        return self.net(feat)


class _ValueHead(nn.Module):
    """Scalar state value for TD learning / PPO critic."""
    def __init__(self, fusion_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fusion_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Returns [B, 1]."""
        return self.net(feat)


class DeepExplorerNetwork(nn.Module):
    """Main network — all sensor streams → fused latent → actor + value.

    Forward pass is fully static and ONNX-exportable. All inputs should be
    pre-normalized before calling:
      - lidar:     [B, 72]   in [0,1]   (1 = open, 0 = obstacle)
      - occ_grid:  [B, 64,64] in [0,1]  (0=free, 1=occupied, 0.5=unknown)
      - depth:     [B, 32,32] in [0,1]  (1 = far, 0 = near) — if enabled
      - proprio:   [B, 5]    normalized to roughly [-1,1] or [0,1]
          [0]: wheel_vx     — forward velocity from wheel odometry (normalized)
          [1]: wheel_vyaw   — yaw rate from wheel odometry (normalized)
          [2]: imu_yaw_rate — from IMU gyro (normalized)
          [3]: novelty      — place novelty in [0,1] (1=novel)
          [4]: safety_hold  — 1.0 if lidar gate is clamping, else 0.0
    """
    def __init__(self, cfg: Optional[ExplorerConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = ExplorerConfig()
        self.cfg = cfg
        self._use_depth = cfg.use_depth and cfg.depth_size > 0

        # Encoders
        self.lidar_enc = _LidarEncoder(cfg.lidar_bins, cfg.lidar_latent)
        self.occ_enc = _OccMapEncoder(cfg.occ_grid_size, cfg.occ_latent)
        if self._use_depth:
            self.depth_enc = _DepthEncoder(cfg.depth_latent)
        self.proprio_enc = _ProprioEncoder(cfg.proprio_dim, cfg.proprio_latent)

        # Fusion MLP
        fuse_in = cfg.lidar_latent + cfg.occ_latent + cfg.proprio_latent
        if self._use_depth:
            fuse_in += cfg.depth_latent
        self.fusion = nn.Sequential(
            nn.Linear(fuse_in, cfg.fusion_latent),
            nn.ReLU(),
            nn.Linear(cfg.fusion_latent, cfg.fusion_latent),
            nn.ReLU(),
        )

        # Output heads
        self.actor = _ActorHead(cfg.fusion_latent, cfg.actor_hidden)
        self.value = _ValueHead(cfg.fusion_latent, cfg.value_hidden)

        # Weight init
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, lidar, occ_grid, proprio,
                depth: Optional[torch.Tensor] = None):
        """Full forward pass. Returns (action, value).

        action: [B, 2] in [-1, 1]  — [left_track, right_track]
        value:  [B, 1]             — scalar state value
        """
        l = self.lidar_enc(lidar)
        o = self.occ_enc(occ_grid)
        p = self.proprio_enc(proprio)

        if self._use_depth and depth is not None:
            d = self.depth_enc(depth)
            feat = self.fusion(torch.cat([l, o, d, p], dim=-1))
        else:
            feat = self.fusion(torch.cat([l, o, p], dim=-1))

        action = self.actor(feat)
        value = self.value(feat)
        return action, value

    @torch.no_grad()
    def step(self, lidar, occ_grid, proprio,
             depth: Optional[torch.Tensor] = None,
             deterministic: bool = False):
        """Inference convenience: returns (action, value).

        Adds exploration noise (Gaussian) when deterministic=False.
        """
        action, value = self.forward(lidar, occ_grid, proprio, depth)
        if not deterministic:
            action = action + torch.randn_like(action) * 0.05
            action = torch.clamp(action, -1.0, 1.0)
        return action, value


# ============================================================================
# Normalization helpers (used by the ROS2 node; also ONNX-exportable as a
# preprocessing sub-graph if needed)
# ============================================================================

def normalize_lidar(ranges: torch.Tensor, max_range: float = 5.0,
                    min_range: float = 0.05) -> torch.Tensor:
    """Raw lidar ranges [m] -> [0,1] where 1 = open."""
    v = torch.clamp(ranges, min_range, max_range)
    return (v - min_range) / (max_range - min_range)


def normalize_proprio(wheel_l: torch.Tensor, wheel_r: torch.Tensor,
                      yaw_rate: torch.Tensor, max_wv: float = 8.0,
                      max_yr: float = 2.5,
                      novelty: torch.Tensor = None,
                      safety_hold: torch.Tensor = None) -> torch.Tensor:
    """Normalize proprioception to 5-D vector."""
    # Wheel forward velocity (average) -> [-1, 1]
    vx = (wheel_l + wheel_r) / (2.0 * max_wv)
    # Wheel yaw (difference) -> [-1, 1]
    vyaw = (wheel_r - wheel_l) / (2.0 * max_wv)
    # IMU yaw rate -> [-1, 1]
    imu = yaw_rate / max_yr

    components = [vx, vyaw, imu]
    if novelty is not None:
        components.append(novelty)        # already [0, 1]
    else:
        components.append(torch.zeros_like(vx))
    if safety_hold is not None:
        components.append(safety_hold)    # already [0, 1]
    else:
        components.append(torch.zeros_like(vx))

    return torch.stack(components, dim=-1).clamp(-1.0, 1.0)
