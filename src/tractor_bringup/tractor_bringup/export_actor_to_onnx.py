#!/usr/bin/env python3
"""Export ActorNetwork from PyTorch to ONNX.

This script is called by the MAP-Elites episode runner to convert models.
"""

import sys
import torch
import torch.nn as nn

# Add remote_training_server to path
sys.path.append('/home/ubuntu/ros2-rover/remote_training_server')

from model_architectures import RGBDEncoder, PolicyHead


class ActorNetwork(nn.Module):
    """Actor-only network for MAP-Elites."""

    def __init__(self, proprio_dim: int = 6):
        super().__init__()
        self.encoder = RGBDEncoder()
        self.policy_head = PolicyHead(self.encoder.output_dim, proprio_dim)

    def forward(self, rgb, depth, proprio):
        features = self.encoder(rgb, depth)
        action = self.policy_head(features, proprio)
        return action


def export_to_onnx(pt_path: str, onnx_path: str):
    """Export PyTorch model to ONNX.

    Args:
        pt_path: Path to .pt file
        onnx_path: Output ONNX path
    """
    # Load model
    model = ActorNetwork()
    model.load_state_dict(torch.load(pt_path, map_location='cpu'))
    model.eval()

    # Dummy inputs
    rgb = torch.randn(1, 3, 240, 424)
    depth = torch.randn(1, 1, 240, 424)
    proprio = torch.randn(1, 6)

    # Export to ONNX
    torch.onnx.export(
        model,
        (rgb, depth, proprio),
        onnx_path,
        input_names=['rgb', 'depth', 'proprio'],
        output_names=['action'],
        opset_version=11,
        do_constant_folding=True,
        export_params=True
    )

    print(f'✓ ONNX export complete: {onnx_path}')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} <pt_path> <onnx_path>')
        sys.exit(1)

    pt_path = sys.argv[1]
    onnx_path = sys.argv[2]

    export_to_onnx(pt_path, onnx_path)
