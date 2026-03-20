#!/usr/bin/env python3
"""Export PPO BEV model to RKNN format for NPU inference.

This script converts a trained PyTorch model to ONNX and then to RKNN
format for deployment on Rockchip NPU (RK3588).

Usage:
    python export_bev_to_rknn.py --checkpoint checkpoints_ppo/ppo_step_200.pt
    python export_bev_to_rknn.py --onnx checkpoints_ppo/latest_actor.onnx
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Import model architecture
from model_architectures import UnifiedBEVPPOPolicy


def export_to_onnx(checkpoint_path: str, output_path: str, device: str = 'cpu'):
    """Export PyTorch checkpoint to ONNX format."""
    print(f"🔄 Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy_state_dict = checkpoint.get('policy_state_dict', checkpoint)
    
    # Create policy and load weights
    policy = UnifiedBEVPPOPolicy(action_dim=2, proprio_dim=6)
    policy.load_state_dict(policy_state_dict)
    policy.eval()
    
    # Create wrapper for ONNX export
    class PolicyWrapper(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy
            
        def forward(self, bev, proprio):
            action_mean, log_std, value = self.policy(bev, proprio)
            return action_mean, log_std, value
    
    model = PolicyWrapper(policy)
    model.eval()
    
    # Create dummy inputs
    dummy_bev = torch.randn(1, 2, 128, 128)
    dummy_proprio = torch.randn(1, 6)
    
    # Export to ONNX
    print(f"📦 Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        (dummy_bev, dummy_proprio),
        output_path,
        opset_version=11,
        input_names=['bev', 'proprio'],
        output_names=['action_mean', 'log_std', 'value'],
        export_params=True,
        do_constant_folding=True,
        keep_initialators_as_inputs=False,
        verbose=False,
        dynamo=False
    )
    
    file_size = os.path.getsize(output_path)
    print(f"✅ ONNX exported: {output_path} ({file_size} bytes)")
    
    return output_path


def export_to_rknn(onnx_path: str, output_path: str, target_platform: str = 'rk3588'):
    """Convert ONNX model to RKNN format."""
    print(f"🔄 Converting to RKNN: {output_path}")
    
    try:
        from rknn.api import RKNN
    except ImportError:
        print("❌ RKNN toolkit not available!")
        print("   Install with: pip install rknn-toolkit2")
        print("   Or on Rockchip device: apt install rknn-toolkit2")
        return None
    
    rknn = RKNN()
    
    # Configure
    print(f"   Configuring for {target_platform}...")
    rknn.config(
        target_platform=target_platform,
        quant_dq=True,
        quantized_activation_per_channel=False,
        enable_relu_fusion=True,
    )
    
    # Load ONNX
    print(f"   Loading ONNX model: {onnx_path}")
    rknn.load_onnx(model=onnx_path)
    
    # Build with quantization
    print("   Building RKNN model (this may take a few minutes)...")
    rknn.build(
        do_quantization=True,
        dataset=None,  # For better results, provide a calibration dataset
        batch_size=1,
    )
    
    # Export RKNN
    rknn.export_rknn(output_path)
    rknn.release()
    
    file_size = os.path.getsize(output_path)
    print(f"✅ RKNN exported: {output_path} ({file_size} bytes)")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Export PPO BEV model to RKNN')
    parser.add_argument('--checkpoint', type=str, help='Path to PyTorch checkpoint (.pt)')
    parser.add_argument('--onnx', type=str, help='Path to ONNX model (skip PyTorch→ONNX if provided)')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory for RKNN model')
    parser.add_argument('--target_platform', type=str, default='rk3588', 
                        choices=['rk356x', 'rk3588', 'rv1126', 'rv1109'],
                        help='Target Rockchip platform')
    parser.add_argument('--device', type=str, default='cpu', help='Device for loading checkpoint')
    
    args = parser.parse_args()
    
    # Determine checkpoint step from filename
    checkpoint_name = "latest"
    if args.checkpoint:
        checkpoint_name = Path(args.checkpoint).stem.replace('ppo_step_', 'step_')
    
    # Step 1: Export to ONNX (if checkpoint provided)
    if args.checkpoint:
        onnx_path = os.path.join(args.output_dir, f"{checkpoint_name}.onnx")
        export_to_onnx(args.checkpoint, onnx_path, args.device)
    else:
        onnx_path = args.onnx
        if not os.path.exists(onnx_path):
            print(f"❌ ONNX file not found: {onnx_path}")
            sys.exit(1)
    
    # Step 2: Export to RKNN
    rknn_path = os.path.join(args.output_dir, f"{checkpoint_name}.rknn")
    export_to_rknn(onnx_path, rknn_path, args.target_platform)
    
    print("\n" + "="*50)
    print("✅ Export complete!")
    print(f"   ONNX: {onnx_path}")
    print(f"   RKNN: {rknn_path}")
    print("="*50)


if __name__ == '__main__':
    main()