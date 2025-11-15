#!/usr/bin/env python3
"""Convert ONNX model to RKNN on the rover (RK3588).

This script converts ONNX models exported from the V620 training server
to RKNN format for deployment on the RK3588 NPU.

Unlike RKNN-Toolkit2 (x86_64 only), this uses RKNN-Toolkit-Lite2 which
runs directly on the RK3588.
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from rknn.api import RKNN
    HAS_RKNN = True
except ImportError:
    try:
        from rknnlite.api import RKNNLite
        HAS_RKNN = False
        print("ERROR: Full RKNN toolkit not available, only RKNNLite found")
        print("RKNNLite cannot convert models, only run inference")
        sys.exit(1)
    except ImportError:
        HAS_RKNN = False
        print("ERROR: RKNN toolkit not installed!")
        sys.exit(1)

import numpy as np


def _load_calibration_dataset(calibration_dir: str, max_samples: int = 100):
    """Load calibration dataset from .npz files.

    Args:
        calibration_dir: Directory containing calibration_XXXX.npz files
        max_samples: Maximum number of samples to load

    Returns:
        Generator function that yields calibration samples
    """
    calibration_files = sorted([
        os.path.join(calibration_dir, f)
        for f in os.listdir(calibration_dir)
        if f.endswith('.npz')
    ])[:max_samples]

    print(f"Loading {len(calibration_files)} calibration samples...")

    # RKNN expects a generator function for multi-input models
    # The generator should yield tuples of (input1, input2, input3, ...)
    loaded_samples = []
    for i, file_path in enumerate(calibration_files):
        try:
            data = np.load(file_path)
            rgb = data['rgb']  # (H, W, 3) uint8
            depth = data['depth']  # (H, W) float32
            proprio = data['proprio']  # (6,) float32

            # Store samples in order: rgb, depth, proprio, lstm_h, lstm_c
            # Initialize LSTM states to zeros for calibration
            lstm_h = np.zeros((1, 1, 128), dtype=np.float32)
            lstm_c = np.zeros((1, 1, 128), dtype=np.float32)
            loaded_samples.append((rgb, depth, proprio, lstm_h, lstm_c))

            if (i + 1) % 10 == 0:
                print(f"  Loaded {i + 1}/{len(calibration_files)} samples")

        except Exception as exc:
            print(f"⚠ Warning: Failed to load {file_path}: {exc}")
            continue

    print(f"✓ Loaded {len(loaded_samples)} calibration samples")

    # Return a generator function that RKNN will call
    def dataset_generator():
        for sample in loaded_samples:
            yield sample

    return dataset_generator


def convert_onnx_to_rknn(
    onnx_path: str,
    output_path: str,
    target_platform: str = 'rk3588',
    quantize: bool = False,
    calibration_dir: str = None
):
    """Convert ONNX to RKNN using RKNNLite on the rover.

    Args:
        onnx_path: Path to ONNX model
        output_path: Output RKNN path
        target_platform: Target platform (default: rk3588)
        quantize: Enable INT8 quantization (requires calibration_dir)
        calibration_dir: Directory with calibration .npz files for quantization
    """

    if not os.path.exists(onnx_path):
        print(f"❌ ONNX file not found: {onnx_path}")
        return False

    print(f"Converting {onnx_path} to RKNN...")
    print(f"Target platform: {target_platform}")

    # Check for calibration data
    if quantize and calibration_dir:
        if not os.path.exists(calibration_dir):
            print(f"⚠ Warning: Calibration directory not found: {calibration_dir}")
            print("  Falling back to float16 mode")
            quantize = False
        else:
            calibration_files = [f for f in os.listdir(calibration_dir) if f.endswith('.npz')]
            if not calibration_files:
                print(f"⚠ Warning: No calibration files found in {calibration_dir}")
                print("  Falling back to float16 mode")
                quantize = False
            else:
                print(f"✓ Found {len(calibration_files)} calibration samples")
    elif quantize and not calibration_dir:
        print("⚠ Warning: Quantization requested but no calibration data provided")
        print("  Falling back to float16 mode")
        quantize = False

    try:
        # Initialize RKNN (full toolkit, not RKNNLite)
        rknn = RKNN(verbose=True)

        # Configure RKNN
        print("Configuring RKNN...")
        # RK3588 supports: 'asymmetric_quantized-8', 'asymmetric_quantized-16'
        # asymmetric_quantized-8 = INT8 quantization (requires calibration dataset)
        # asymmetric_quantized-16 = INT16 quantization (better accuracy, larger model)
        # Note: For LSTM hidden states, RKNN expects single scalar values, not arrays
        ret = rknn.config(
            mean_values=[
                [127.5, 127.5, 127.5],  # RGB (3 channels)
                [0],                     # Depth (1 channel)
                [0, 0, 0, 0, 0, 0],     # Proprio (6 values)
                [0],                     # LSTM hidden state (scalar for whole tensor)
                [0]                      # LSTM cell state (scalar for whole tensor)
            ],
            std_values=[
                [127.5, 127.5, 127.5],  # RGB (3 channels)
                [1],                     # Depth (1 channel)
                [1, 1, 1, 1, 1, 1],     # Proprio (6 values)
                [1],                     # LSTM hidden state (scalar - no normalization)
                [1]                      # LSTM cell state (scalar - no normalization)
            ],
            target_platform=target_platform,
            quantized_dtype='asymmetric_quantized-8' if quantize else 'asymmetric_quantized-16',
            optimization_level=3
        )
        if ret != 0:
            print(f"❌ Failed to configure RKNN: {ret}")
            return False

        # Load ONNX
        print("Loading ONNX model with LSTM inputs...")
        # Specify fixed input shapes (batch=1) since RKNN doesn't support dynamic shapes
        ret = rknn.load_onnx(
            model=onnx_path,
            inputs=['rgb', 'depth', 'proprio', 'lstm_h', 'lstm_c'],
            input_size_list=[
                [1, 3, 240, 424],   # RGB
                [1, 1, 240, 424],   # Depth
                [1, 6],             # Proprio
                [1, 1, 128],        # LSTM hidden state
                [1, 1, 128]         # LSTM cell state
            ]
        )
        if ret != 0:
            print(f"❌ Failed to load ONNX: {ret}")
            return False

        # Build
        if quantize and calibration_dir:
            print("Building RKNN model with INT8 quantization...")
            print("⚠ Note: Quantization with multi-input NPZ calibration not yet implemented")
            print("  Falling back to INT16 mode (no calibration required)")
            # TODO: Implement proper dataset.txt generation for multi-input models
            # dataset = _load_calibration_dataset(calibration_dir)
            # ret = rknn.build(do_quantization=True, dataset=dataset)
            ret = rknn.build(do_quantization=False)
        else:
            print("Building RKNN model (INT16 mode - no calibration)...")
            ret = rknn.build(do_quantization=False)

        if ret != 0:
            print(f"❌ Failed to build RKNN: {ret}")
            return False

        # Export
        print(f"Exporting to {output_path}...")
        ret = rknn.export_rknn(output_path)
        if ret != 0:
            print(f"❌ Failed to export RKNN: {ret}")
            return False

        print(f"✓ Successfully converted to RKNN: {output_path}")
        print(f"  Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

        # Cleanup
        rknn.release()

        return True

    except Exception as exc:
        print(f"❌ Conversion failed: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert ONNX to RKNN on RK3588 (on-device conversion)'
    )
    parser.add_argument('onnx_path', type=str, help='Path to ONNX model')
    parser.add_argument(
        '--output', type=str,
        help='Output RKNN path (default: same as input with .rknn extension)'
    )
    parser.add_argument(
        '--target', type=str, default='rk3588',
        help='Target platform (default: rk3588)'
    )
    parser.add_argument(
        '--quantize', action='store_true',
        help='Enable INT8 quantization (requires --calibration-dir)'
    )
    parser.add_argument(
        '--calibration-dir', type=str,
        help='Directory containing calibration .npz files for quantization'
    )
    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = args.onnx_path.replace('.onnx', '.rknn')

    print("=" * 60)
    print("ONNX to RKNN Converter (On-Device)")
    print("=" * 60)
    print()
    if args.quantize and args.calibration_dir:
        print("Mode: INT8 quantization with calibration data")
        print("      (will fall back to INT16 until dataset.txt implemented)")
    else:
        print("Mode: INT16 (no quantization)")
        if args.quantize:
            print("Note: --quantize requires --calibration-dir")
    print()

    # Convert
    success = convert_onnx_to_rknn(
        onnx_path=args.onnx_path,
        output_path=output_path,
        target_platform=args.target,
        quantize=args.quantize,
        calibration_dir=args.calibration_dir
    )

    if success:
        print()
        print("=" * 60)
        print("✅ Conversion Complete!")
        print("=" * 60)
        print()
        print(f"RKNN model: {output_path}")
        print()
        print("Deploy with:")
        print(f"  cp {output_path} models/remote_trained.rknn")
        print("  ros2 service call /reload_remote_model std_srvs/srv/Trigger")
        return 0
    else:
        print()
        print("❌ Conversion failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
