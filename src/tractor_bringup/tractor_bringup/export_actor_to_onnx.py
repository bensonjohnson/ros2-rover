#!/usr/bin/env python3
"""Export ActorNetwork from PyTorch to ONNX with LSTM support.

This script is called by the MAP-Elites episode runner to convert models.
"""

import sys
import torch
import torch.nn as nn

# Add remote_training_server to path
sys.path.append('/home/ubuntu/ros2-rover/remote_training_server')

from model_architectures import RGBDEncoder, PolicyHead


class ActorNetwork(nn.Module):
    """Actor-only network for MAP-Elites with LSTM memory."""

    def __init__(self, proprio_dim: int = 9, use_lstm: bool = True):
        super().__init__()
        self.encoder = RGBDEncoder()
        self.policy_head = PolicyHead(self.encoder.output_dim, proprio_dim, use_lstm=use_lstm)
        self.use_lstm = use_lstm

    def forward(self, rgb, depth, proprio, lstm_h, lstm_c):
        """Forward pass with LSTM hidden states.
        
        Args:
            rgb: (1, 3, H, W) RGB image
            depth: (1, 1, H, W) Depth image
            proprio: (1, 9) Proprioception
            lstm_h: (1, 1, 128) LSTM hidden state
            lstm_c: (1, 1, 128) LSTM cell state
            
        Returns:
            action: (1, 2) Actions in [-1, 1]
            new_lstm_h: (1, 1, 128) Updated hidden state
            new_lstm_c: (1, 1, 128) Updated cell state
        """
        features = self.encoder(rgb, depth)
        
        if self.use_lstm:
            # Pass hidden state to policy head
            action, (new_lstm_h, new_lstm_c) = self.policy_head(features, proprio, (lstm_h, lstm_c))
            action = torch.tanh(action)
            return action, new_lstm_h, new_lstm_c
        else:
            # No LSTM, ignore hidden states
            action, _ = self.policy_head(features, proprio, None)
            action = torch.tanh(action)
            # Return dummy hidden states for compatibility
            return action, lstm_h, lstm_c


def export_to_onnx(pt_path: str, onnx_path: str):
    """Export PyTorch model to ONNX with LSTM support.

    Args:
        pt_path: Path to .pt file
        onnx_path: Output ONNX path
    """
    # Load model
    model = ActorNetwork(proprio_dim=9, use_lstm=True)
    model.load_state_dict(torch.load(pt_path, map_location='cpu'))
    model.eval()

    # Dummy inputs (640x480 resolution + LSTM hidden states)
    rgb = torch.randn(1, 3, 480, 640)
    depth = torch.randn(1, 1, 480, 640)
    proprio = torch.randn(1, 9)
    lstm_h = torch.zeros(1, 1, 128)  # LSTM hidden state
    lstm_c = torch.zeros(1, 1, 128)  # LSTM cell state

    # Export to ONNX with LSTM inputs/outputs
    torch.onnx.export(
        model,
        (rgb, depth, proprio, lstm_h, lstm_c),
        onnx_path,
        input_names=['rgb', 'depth', 'proprio', 'lstm_h', 'lstm_c'],
        output_names=['action', 'new_lstm_h', 'new_lstm_c'],
        opset_version=11,
        do_constant_folding=True,
        export_params=True,
        dynamic_axes={
            # Allow batch size to vary (though we always use 1)
            'rgb': {0: 'batch'},
            'depth': {0: 'batch'},
            'proprio': {0: 'batch'},
            'lstm_h': {1: 'batch'},
            'lstm_c': {1: 'batch'},
            'action': {0: 'batch'},
            'new_lstm_h': {1: 'batch'},
            'new_lstm_c': {1: 'batch'},
        }
    )

    print(f'✓ ONNX export complete: {onnx_path}')
    print(f'  Inputs: rgb, depth, proprio, lstm_h (1,1,128), lstm_c (1,1,128)')
    print(f'  Outputs: action, new_lstm_h (1,1,128), new_lstm_c (1,1,128)')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} <pt_path> <onnx_path>')
        sys.exit(1)

    pt_path = sys.argv[1]
    onnx_path = sys.argv[2]

    export_to_onnx(pt_path, onnx_path)
