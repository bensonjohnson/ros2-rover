#!/usr/bin/env python3
"""Export DeepExplorerNetwork to ONNX and convert to RKNN for NPU inference.

Usage:
  # Export PyTorch -> ONNX (can run on the rover or training server)
  python3 convert_to_rknn.py export --checkpoint model.pt --output model.onnx

  # Convert ONNX -> RKNN (needs rknn-toolkit2, runs on x86_64 Linux)
  python3 convert_to_rknn.py convert --onnx model.onnx --output model.rknn

  # Full pipeline
  python3 convert_to_rknn.py full --checkpoint model.pt --output model.rknn
"""

import os
import sys
import argparse
from typing import Optional

import numpy as np

# We always need torch for export
import torch

from tractor_explorer.deep_explorer_network import (
    DeepExplorerNetwork, ExplorerConfig,
)

try:
    from rknn.api import RKNN
    HAS_RKNN_TOOLKIT = True
except ImportError:
    HAS_RKNN_TOOLKIT = False


def export_onnx(checkpoint_path: str, output_path: str,
                cfg: Optional[ExplorerConfig] = None) -> str:
    """Export PyTorch checkpoint to ONNX.

    The ONNX graph is the full observation->[action, value] forward pass.
    Static shapes (batch=1) for RKNN compatibility.
    """
    if cfg is None:
        cfg = ExplorerConfig()

    model = DeepExplorerNetwork(cfg)
    model.eval()

    if os.path.exists(checkpoint_path):
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        own = model.state_dict()
        for k, v in list(sd.items()):
            if k in own and v.shape != own[k].shape:
                print(f"  Skipping {k}: saved {v.shape} != current {own[k].shape}")
                del sd[k]
        model.load_state_dict(sd, strict=False)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"No checkpoint at {checkpoint_path} — exporting untrained model")

    # Dummy inputs
    B = 1
    lidar = torch.randn(B, cfg.lidar_bins)
    occ = torch.randn(B, cfg.occ_grid_size, cfg.occ_grid_size)
    proprio = torch.randn(B, cfg.proprio_dim)

    input_names = ["lidar", "occ_grid", "proprio"]
    output_names = ["action", "value"]

    if cfg.use_depth:
        depth = torch.randn(B, cfg.depth_size, cfg.depth_size)
        input_names.append("depth")
    else:
        depth = None

    # Export
    torch.onnx.export(
        model,
        (lidar, occ, proprio, depth) if cfg.use_depth else (lidar, occ, proprio),
        output_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        dynamic_axes={
            name: {0: "batch"} for name in input_names + output_names
        },
    )

    # Collapse external data
    import onnx
    data_file = output_path + ".data"
    if os.path.exists(data_file):
        m = onnx.load(output_path, load_external_data=True)
        onnx.save(m, output_path, save_as_external_data=False)
        os.remove(data_file)

    size = os.path.getsize(output_path)
    print(f"ONNX exported: {output_path} ({size:,} bytes)")
    return output_path


def convert_to_rknn(onnx_path: str, output_path: str,
                    target_platform: str = "rk3588",
                    do_quantize: bool = False) -> Optional[str]:
    """Convert ONNX to RKNN format for the RK3588 NPU."""
    if not HAS_RKNN_TOOLKIT:
        print("rknn-toolkit2 not installed. Install with:")
        print("  pip install rknn_toolkit2-*-cp310-cp310-linux_x86_64.whl")
        return None

    if not os.path.exists(onnx_path):
        print(f"ONNX file not found: {onnx_path}")
        return None

    print(f"Converting {onnx_path} -> {output_path} ...")

    rknn = RKNN(verbose=False)

    # Load ONNX model and infer input details
    import onnx
    model = onnx.load(onnx_path)
    input_names = []
    input_sizes = []
    for inp in model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        shape = [1 if d <= 0 else d for d in shape]
        input_names.append(inp.name)
        input_sizes.append(shape)
        print(f"  Input: {inp.name} {shape}")

    # Calculate mean/std per input
    mean_values = []
    std_values = []
    for shape in input_sizes:
        if len(shape) == 4:
            # Image format NHWC or NCHW — RKNN expects NHWC
            n_ch = shape[-1] if shape[-1] in (1, 3) else shape[1]
            mean_values.append([0] * n_ch)
            std_values.append([1] * n_ch)
        else:
            # Vector input
            mean_values.append([0])
            std_values.append([1])

    # Configure
    ret = rknn.config(
        mean_values=mean_values,
        std_values=std_values,
        target_platform=target_platform,
        optimization_level=3,
        quantize_input_node=do_quantize,
        quantized_dtype="asymmetric_quantized-8" if do_quantize else None,
    )
    if ret != 0:
        print(f"RKNN config failed: {ret}")
        return None

    # Load
    ret = rknn.load_onnx(
        model=onnx_path,
        inputs=input_names,
        input_size_list=input_sizes,
    )
    if ret != 0:
        print(f"RKNN load ONNX failed: {ret}")
        return None

    # Build
    ret = rknn.build(do_quantization=do_quantize)
    if ret != 0:
        print(f"RKNN build failed: {ret}")
        return None

    # Export
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print(f"RKNN export failed: {ret}")
        return None

    size = os.path.getsize(output_path)
    print(f"RKNN exported: {output_path} ({size:,} bytes)")
    rknn.release()
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Export DeepExplorerNetwork to ONNX/RKNN")
    sub = parser.add_subparsers(dest="command")

    # Export subcommand
    p_export = sub.add_parser("export", help="Export PyTorch -> ONNX")
    p_export.add_argument("--checkpoint", default=os.path.expanduser("~/.ros/explorer_brain.pt"))
    p_export.add_argument("--output", default="explorer_model.onnx")
    p_export.add_argument("--use-depth", action="store_true")

    # Convert subcommand
    p_convert = sub.add_parser("convert", help="Convert ONNX -> RKNN")
    p_convert.add_argument("--onnx", default="explorer_model.onnx")
    p_convert.add_argument("--output", default="explorer_model.rknn")
    p_convert.add_argument("--quantize", action="store_true")
    p_convert.add_argument("--platform", default="rk3588")

    # Full subcommand
    p_full = sub.add_parser("full", help="Full pipeline: checkpoint -> RKNN")
    p_full.add_argument("--checkpoint", default=os.path.expanduser("~/.ros/explorer_brain.pt"))
    p_full.add_argument("--output", default=os.path.expanduser("~/.ros/explorer_brain.rknn"))
    p_full.add_argument("--use-depth", action="store_true")
    p_full.add_argument("--quantize", action="store_true")
    p_full.add_argument("--platform", default="rk3588")

    args = parser.parse_args()

    if args.command == "export":
        cfg = ExplorerConfig(use_depth=args.use_depth)
        export_onnx(args.checkpoint, args.output, cfg)

    elif args.command == "convert":
        convert_to_rknn(args.onnx, args.output,
                        target_platform=args.platform,
                        do_quantize=args.quantize)

    elif args.command == "full":
        cfg = ExplorerConfig(use_depth=args.use_depth)
        onnx_path = args.output.replace(".rknn", ".onnx")
        export_onnx(args.checkpoint, onnx_path, cfg)
        convert_to_rknn(onnx_path, args.output,
                        target_platform=args.platform,
                        do_quantize=args.quantize)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
