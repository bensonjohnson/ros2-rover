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
    from rknnlite.api import RKNNLite
    HAS_RKNN = True
except ImportError:
    HAS_RKNN = False
    print("ERROR: rknnlite not installed!")
    print("Install with: pip3 install rknn_toolkit_lite2-*-cp310-cp310-linux_aarch64.whl")
    sys.exit(1)


def convert_onnx_to_rknn(
    onnx_path: str,
    output_path: str,
    target_platform: str = 'rk3588',
    quantize: bool = False
):
    """Convert ONNX to RKNN using RKNNLite on the rover.

    Note: On-device conversion with RKNNLite has limitations:
    - Cannot do quantization (use float16 mode)
    - Slower conversion than RKNN-Toolkit2
    - But works directly on RK3588 without x86_64 machine

    Args:
        onnx_path: Path to ONNX model
        output_path: Output RKNN path
        target_platform: Target platform (default: rk3588)
        quantize: Enable quantization (NOT SUPPORTED on RKNNLite)
    """

    if not os.path.exists(onnx_path):
        print(f"❌ ONNX file not found: {onnx_path}")
        return False

    print(f"Converting {onnx_path} to RKNN...")
    print(f"Target platform: {target_platform}")

    # Note: RKNNLite on device has limited conversion capabilities
    # For full features, use RKNN-Toolkit2 on x86_64

    if quantize:
        print("⚠ Warning: Quantization not supported with RKNNLite on-device conversion")
        print("  Model will be converted to float16 mode")
        print("  For INT8 quantization, use RKNN-Toolkit2 on x86_64 Linux")

    try:
        # Initialize RKNNLite
        rknn = RKNNLite(verbose=True)

        # Load ONNX
        print("Loading ONNX model...")
        ret = rknn.load_onnx(model=onnx_path)
        if ret != 0:
            print(f"❌ Failed to load ONNX: {ret}")
            return False

        # Build (float16 mode - quantization not available on-device)
        print("Building RKNN model (float16 mode)...")
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
    print("Note: On-device conversion uses float16 mode (no quantization)")
    print("For INT8 quantization, use RKNN-Toolkit2 on x86_64 Linux")
    print()

    # Convert
    success = convert_onnx_to_rknn(
        onnx_path=args.onnx_path,
        output_path=output_path,
        target_platform=args.target,
        quantize=False  # Not supported on-device
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
