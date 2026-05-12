"""RLPD + HIL-SERL networks for the V620 remote trainer.

Model-free SAC architecture with:
  - Frozen ImageNet-pretrained ResNet18 RGB encoder (sample-efficiency lever).
  - Small BEV CNN over stacked occupancy frames.
  - Proprio MLP over stacked proprioception vectors.
  - LayerNorm-equipped critic ensemble (N critics; REDQ-style target subset).
  - Tanh-squashed Gaussian actor.
  - Single-blob ONNX wrapper (no recurrent state — model-free is much simpler
    than Dreamer's encoder + RSSM step + actor graph).

Reward channels carried on the wire:
    (coverage, frontier, collision, episodic, intervention)
where `intervention = -1.0` when the human overrides via the RB deadman during
autonomy. This is the HIL-SERL "policy learns to avoid needing correction"
signal.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


REWARD_CHANNELS_RLPD = ('coverage', 'frontier', 'collision', 'episodic', 'intervention')

# ImageNet normalization (constants — registered as buffers inside the encoder so
# they travel with the module and end up baked into the exported ONNX graph).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Vision encoder — frozen ResNet18 + BEV CNN + proprio MLP
# ---------------------------------------------------------------------------


class _BEVStackCNN(nn.Module):
    """Small CNN over `2 * frame_stack` BEV channels → bev_feat_dim.

    The input is the concatenation along channel dim of `frame_stack` BEV
    grids (each 2×128×128). Output is a flat feature vector.
    """

    def __init__(self, in_channels: int, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # 64
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),           # 32
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),          # 16
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),         # 8
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, bev_stack: torch.Tensor) -> torch.Tensor:
        return self.net(bev_stack)


class RLPDVisionEncoder(nn.Module):
    """Stacked-frame multimodal encoder.

    Inputs:
        rgb_stack:     (B, 3*K, 84, 84) float32 in [0, 1] (rover sends raw uint8
                       divided by 255 inside the ONNX wrapper before this is called).
        bev_stack:     (B, 2*K, 128, 128) float32 in [0, 1].
        proprio_stack: (B, 6*K) float32, already normalized by the rover's
                       PROPRIO_MEAN / PROPRIO_STD.

    Output:
        state: (B, state_dim) float32. ImageNet normalization is performed
        internally so the wire format and ONNX inputs stay simple (raw [0,1] RGB).

    The ResNet18 trunk is frozen (`requires_grad_(False)` + `.eval()`) so BN
    stats never drift. Only the projection head and BEV / proprio branches train.
    """

    def __init__(
        self,
        proprio_dim: int = 6,
        frame_stack: int = 4,
        state_dim: int = 512,
    ):
        super().__init__()
        self.frame_stack = frame_stack
        self.proprio_dim = proprio_dim
        self.state_dim = state_dim

        # Frozen ResNet18 trunk → 512-D per frame
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        backbone = torchvision.models.resnet18(weights=weights)
        # Replace the classification head with identity so we get 512-D features
        backbone.fc = nn.Identity()
        backbone.requires_grad_(False)
        backbone.eval()
        self.rgb_backbone = backbone
        self._rgb_feat_per_frame = 512

        # ImageNet normalization buffers (baked into ONNX graph at export)
        self.register_buffer(
            'rgb_mean',
            torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            'rgb_std',
            torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )

        # Trainable projection over the concatenated per-frame features
        self.rgb_proj = nn.Sequential(
            nn.Linear(self._rgb_feat_per_frame * frame_stack, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
        )

        # BEV branch over stacked occupancy frames
        self.bev_cnn = _BEVStackCNN(in_channels=2 * frame_stack, out_dim=256)

        # Proprio branch
        self.proprio_mlp = nn.Sequential(
            nn.Linear(proprio_dim * frame_stack, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

        # Fusion
        fuse_in = 256 + 256 + 64
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, state_dim),
            nn.LayerNorm(state_dim),
            nn.ReLU(inplace=True),
        )

    def train(self, mode: bool = True):
        # Keep the ResNet backbone frozen in eval() so BN running stats don't
        # drift no matter which training/eval mode the rest of the network is in.
        super().train(mode)
        self.rgb_backbone.eval()
        return self

    def _encode_rgb_stack(self, rgb_stack: torch.Tensor) -> torch.Tensor:
        """rgb_stack: (B, 3*K, 84, 84) in [0, 1] → (B, 512*K)."""
        B = rgb_stack.shape[0]
        K = self.frame_stack
        # Reshape to (B*K, 3, 84, 84), apply ImageNet normalization, run trunk
        x = rgb_stack.reshape(B, K, 3, 84, 84).reshape(B * K, 3, 84, 84)
        x = (x - self.rgb_mean) / self.rgb_std
        # Upsample 84→224 for ResNet18 (it was trained on 224×224 ImageNet).
        # This makes the frozen features dramatically better than feeding 84×84
        # directly, and the cost is negligible (one bilinear resize per frame).
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        with torch.no_grad():
            feats = self.rgb_backbone(x)  # (B*K, 512)
        feats = feats.reshape(B, K * self._rgb_feat_per_frame)
        return feats

    def forward(
        self,
        rgb_stack: torch.Tensor,
        bev_stack: torch.Tensor,
        proprio_stack: torch.Tensor,
    ) -> torch.Tensor:
        rgb_feat = self.rgb_proj(self._encode_rgb_stack(rgb_stack))
        bev_feat = self.bev_cnn(bev_stack)
        pro_feat = self.proprio_mlp(proprio_stack)
        state = self.fuse(torch.cat([rgb_feat, bev_feat, pro_feat], dim=-1))
        return state


# ---------------------------------------------------------------------------
# Actor — tanh-squashed Gaussian
# ---------------------------------------------------------------------------


LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class RLPDActor(nn.Module):
    """SAC actor on state → tanh-squashed Gaussian over a 2-D action.

    The `forward` returns `(mean_logstd)` so the ONNX wrapper has a single
    output. Training-time helpers (`sample`, `log_prob`) operate on the
    underlying Gaussian with the tanh Jacobian correction.
    """

    def __init__(self, state_dim: int = 512, action_dim: int = 2, hidden: int = 512):
        super().__init__()
        self.action_dim = action_dim
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return `(B, 2*action_dim)` = [mean, log_std] concatenated."""
        h = self.trunk(state)
        mean = self.mean_head(h)
        log_std = torch.clamp(self.log_std_head(h), LOG_STD_MIN, LOG_STD_MAX)
        return torch.cat([mean, log_std], dim=-1)

    def sample(self, state: torch.Tensor, deterministic: bool = False):
        """Returns (action, log_prob, pre_tanh).

        Includes the tanh-squash Jacobian correction in `log_prob` — necessary
        for the SAC actor objective `α·log_prob - Q`.
        """
        h = self.trunk(state)
        mean = self.mean_head(h)
        log_std = torch.clamp(self.log_std_head(h), LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        if deterministic:
            pre_tanh = mean
            log_prob = torch.zeros(state.shape[0], device=state.device)
        else:
            eps = torch.randn_like(mean)
            pre_tanh = mean + std * eps
            # log N(pre_tanh; mean, std)
            log_prob = -0.5 * (((pre_tanh - mean) / std) ** 2 + 2 * log_std + math.log(2 * math.pi))
            log_prob = log_prob.sum(dim=-1)
            # tanh Jacobian: log(1 - tanh^2(pre)) — numerically stable form
            log_prob = log_prob - (2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))).sum(dim=-1)
        action = torch.tanh(pre_tanh)
        return action, log_prob, pre_tanh


# ---------------------------------------------------------------------------
# Critic ensemble — N critics each with LayerNorm; REDQ subset for target Q
# ---------------------------------------------------------------------------


class _CriticHead(nn.Module):
    """Single LayerNorm-MLP critic: (state, action) → Q.

    LayerNorm is the RLPD trick that enables aggressive UTD ratios and offline
    data utilization without value collapse.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class RLPDCriticEnsemble(nn.Module):
    """Stack of N critics. Returns `(N, B)` Q-values."""

    def __init__(
        self,
        state_dim: int = 512,
        action_dim: int = 2,
        n_critics: int = 10,
        hidden: int = 512,
    ):
        super().__init__()
        self.n_critics = n_critics
        self.heads = nn.ModuleList([
            _CriticHead(state_dim, action_dim, hidden) for _ in range(n_critics)
        ])

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.stack([h(state, action) for h in self.heads], dim=0)

    def q_subset(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """REDQ-style: evaluate only `indices` heads. Returns `(M, B)`."""
        return torch.stack([self.heads[int(i)](state, action) for i in indices], dim=0)


# ---------------------------------------------------------------------------
# ONNX wrapper — state-less, single output `mean_logstd`
# ---------------------------------------------------------------------------


class RLPDActorOnnxWrapper(nn.Module):
    """Wraps encoder + actor for a single-blob ONNX export.

    Inputs (named, in this order):
        rgb_stack     : (1, 3*K, 84, 84) float32 in [0, 1]  — rover sends uint8/255
        bev_stack     : (1, 2*K, 128, 128) float32 in [0, 1]
        proprio_stack : (1, 6*K) float32 (already normalized on the rover)

    Output:
        mean_logstd : (1, 2*action_dim) — [mean_l, mean_r, log_std_l, log_std_r]

    No recurrent state inputs/outputs — model-free SAC.
    """

    def __init__(self, encoder: RLPDVisionEncoder, actor: RLPDActor):
        super().__init__()
        self.encoder = encoder
        self.actor = actor

    def forward(
        self,
        rgb_stack: torch.Tensor,
        bev_stack: torch.Tensor,
        proprio_stack: torch.Tensor,
    ) -> torch.Tensor:
        state = self.encoder(rgb_stack, bev_stack, proprio_stack)
        return self.actor(state)
