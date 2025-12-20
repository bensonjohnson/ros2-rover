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
    """Prepare calibration dataset by saving samples to .npy files and creating a dataset.txt.

    Args:
        calibration_dir: Directory containing calibration_XXXX.npz files
        max_samples: Maximum number of samples to use

    Returns:
        Path to the generated dataset.txt file
    """
    calibration_files = sorted([
        os.path.join(calibration_dir, f)
        for f in os.listdir(calibration_dir)
        if f.endswith('.npz')
    ])[:max_samples]

    print(f"Preparing calibration dataset from {len(calibration_files)} samples...")

    # Create a temporary directory for calibration artifacts
    # We use a fixed path inside calibration_dir to avoid filling /tmp
    dataset_dir = os.path.join(calibration_dir, "rknn_dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    dataset_txt_path = os.path.join(dataset_dir, "dataset.txt")

    valid_samples = 0
    with open(dataset_txt_path, 'w') as f:
        for i, file_path in enumerate(calibration_files):
            try:
                data = np.load(file_path)
                bev = data['bev']  # Unified BEV grid (2, 128, 128)
                proprio = data['proprio']

                # Validate and Fix BEV shape
                # Expected: (2, 128, 128)
                if bev.ndim == 2 and bev.shape == (128, 128):
                    # Single channel, duplicate to 2 channels
                    bev = np.stack([bev, bev], axis=0)
                elif bev.ndim == 3 and bev.shape == (2, 128, 128):
                    pass  # Already correct shape
                else:
                    print(f"⚠ Warning: Expected bev shape (2, 128, 128), got {bev.shape} in {file_path}")
                    continue

                if proprio.shape != (10,):
                    print(f"⚠ Warning: Expected proprio shape (10,), got {proprio.shape} in {file_path}")
                    continue

                # Sanitize Proprio
                proprio = np.nan_to_num(proprio, nan=0.0, posinf=100.0, neginf=-100.0)

                # Add batch dimension
                bev_batch = bev[None, ...]
                proprio_batch = proprio[None, ...]

                bev_path = os.path.abspath(os.path.join(dataset_dir, f"bev_{i}.npy"))
                proprio_path = os.path.abspath(os.path.join(dataset_dir, f"proprio_{i}.npy"))

                np.save(bev_path, bev_batch.astype(np.float32))
                np.save(proprio_path, proprio_batch.astype(np.float32))

                # Write to dataset.txt (space separated)
                f.write(f"{bev_path} {proprio_path}\n")

                valid_samples += 1

            except Exception as exc:
                print(f"⚠ Warning: Failed to process {file_path}: {exc}")
                continue

    print(f"✓ Generated dataset.txt with {valid_samples} samples")
    return dataset_txt_path


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
        # RK3588 supports: 'asymmetric_quantized-8', 'asymmetric_quantized-16', 'fp16'
        # asymmetric_quantized-8 = INT8 quantization (requires calibration dataset)
        # fp16 = Floating Point 16 (default if no quantization)
        
        config_args = {
            # Disable RKNN normalization - we'll normalize in calibration generator
            # This ensures exact match between calibration and inference preprocessing
            'mean_values': [
                [0, 0],           # BEV (2 channels)
                [0] * 10,         # Proprio
            ],
            'std_values': [
                [1, 1],           # BEV
                [1] * 10,         # Proprio
            ],
            'target_platform': target_platform,
            'optimization_level': 3
        }
        
        if quantize:
            config_args['quantized_dtype'] = 'asymmetric_quantized-8'
            
        ret = rknn.config(**config_args)
        if ret != 0:
            print(f"❌ Failed to configure RKNN: {ret}")
            return False

        # Load ONNX
        print("Loading ONNX model (stateless)...")
        # Specify fixed input shapes (batch=1) since RKNN doesn't support dynamic shapes
        ret = rknn.load_onnx(
            model=onnx_path,
            inputs=['bev', 'proprio'],
            input_size_list=[
                [1, 2, 128, 128],   # Unified BEV grid (2 channels: LiDAR + Depth)
                [1, 10],            # Proprio (10-dim)
            ]
        )
        if ret != 0:
            print(f"❌ Failed to load ONNX: {ret}")
            return False

        # Build
        if quantize and calibration_dir:
            print("Building RKNN model with INT8 quantization...")
            print("  Loading calibration dataset...")
            dataset = _load_calibration_dataset(calibration_dir)
            print("  Running quantization (this may take a few minutes)...")
            ret = rknn.build(do_quantization=True, dataset=dataset)
        else:
            print("Building RKNN model (FP16 mode - no calibration)...")
            ret = rknn.build(do_quantization=False)

        if ret != 0:
            print(f"❌ Failed to build RKNN: {ret}")
            return False

        # Test inference BEFORE export to validate model
        print("Testing RKNN model with sample inputs...")
        try:
            # Initialize runtime for testing
            ret = rknn.init_runtime()
            if ret != 0:
                print(f"⚠ Warning: Failed to init runtime for testing: {ret}")
            else:
                # Create test inputs (normalized like rover)
                test_bev = np.random.rand(1, 2, 128, 128).astype(np.float32)  # Unified BEV
                test_proprio = np.random.rand(1, 10).astype(np.float32)

                # Run inference
                outputs = rknn.inference(inputs=[test_bev, test_proprio])

                if outputs and len(outputs) > 0:
                    test_output = outputs[0]
                    print(f"  Test output: {test_output}")
                    print(f"  Range: [{test_output.min():.6f}, {test_output.max():.6f}]")

                    if np.isnan(test_output).any() or np.isinf(test_output).any():
                        print(f"  ❌ RKNN model produces NaN/Inf! Conversion may be broken.")
                        print(f"  This indicates an issue with FP16 precision or RKNN compatibility.")
                    else:
                        print(f"  ✓ RKNN test inference passed")
                else:
                    print(f"  ⚠ Warning: No outputs from test inference")
        except Exception as exc:
            print(f"⚠ Warning: Test inference failed: {exc}")

        # Export
        print(f"Exporting to {output_path}...")
        ret = rknn.export_rknn(output_path)
        if ret != 0:
            print(f"❌ Failed to export RKNN: {ret}")
            return False

        print(f"✓ Successfully converted to RKNN: {output_path}")
        print(f"  Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

        # Cleanup
        # rknn.release() # Avoid double free on exit

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
    else:
        print("Mode: FP16 (no quantization)")
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
