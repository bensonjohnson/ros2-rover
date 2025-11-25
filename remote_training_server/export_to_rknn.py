#!/usr/bin/env python3
"""Convert ONNX model to RKNN format for RK3588 NPU deployment.

This script takes an ONNX model exported from the V620 training server
and converts it to RKNN format optimized for the RK3588 NPU on the rover.

Requirements:
- RKNN-Toolkit2 (only works on x86_64 Linux)
- Calibration dataset (sample images from rover)
"""

import argparse
import os
from pathlib import Path
import numpy as np
import cv2

try:
    from rknn.api import RKNN
    HAS_RKNN = True
except ImportError:
    HAS_RKNN = False
    print("ERROR: RKNN-Toolkit2 not installed")
    print("Install from: https://github.com/rockchip-linux/rknn-toolkit2")


class RKNNConverter:
    """Converts ONNX models to RKNN format with quantization."""

    def __init__(self, onnx_path: str, output_path: str, calibration_data_dir: Optional[str] = None):
        if not HAS_RKNN:
            raise RuntimeError("RKNN-Toolkit2 is required for conversion")

        self.onnx_path = onnx_path
        self.output_path = output_path
        self.calibration_data_dir = calibration_data_dir

        # Initialize RKNN
        self.rknn = RKNN(verbose=True)

    def convert(
        self,
        target_platform: str = 'rk3588',
        quantize: bool = True,
        do_quantization: bool = True,
        optimization_level: int = 3
    ):
        """Convert ONNX to RKNN with optional quantization.

        Args:
            target_platform: Target NPU platform ('rk3588', 'rk3566', etc.)
            quantize: Enable quantization (reduces model size, increases speed)
            do_quantization: Perform quantization (requires calibration data)
            optimization_level: RKNN optimization level (0-3, higher = more optimized)
        """
        print(f"Converting {self.onnx_path} to RKNN...")

        # Configure RKNN
        print("Configuring RKNN...")
        ret = self.rknn.config(
            mean_values=[[0, 0, 0]],  # RGB: [0, 255] -> [0, 1] (mean=0, std=255)
            std_values=[[255, 255, 255]],
            target_platform=target_platform,
            quantized_dtype='asymmetric_quantized-8' if quantize else 'float16',
            quantized_algorithm='normal',
            quantized_method='channel',
            optimization_level=optimization_level,
        )

        if ret != 0:
            print(f"RKNN config failed: {ret}")
            return False

        # Load ONNX model
        print("Loading ONNX model...")
        ret = self.rknn.load_onnx(model=self.onnx_path)
        if ret != 0:
            print(f"RKNN load_onnx failed: {ret}")
            return False

        # Build RKNN model
        print("Building RKNN model...")
        if do_quantization and quantize and self.calibration_data_dir:
            # Load calibration dataset
            calibration_dataset = self._prepare_calibration_dataset()
            ret = self.rknn.build(do_quantization=True, dataset=calibration_dataset)
        else:
            ret = self.rknn.build(do_quantization=False)

        if ret != 0:
            print(f"RKNN build failed: {ret}")
            return False

        # Export RKNN model
        print(f"Exporting to {self.output_path}...")
        ret = self.rknn.export_rknn(self.output_path)
        if ret != 0:
            print(f"RKNN export failed: {ret}")
            return False

        print(f"✓ Successfully converted to RKNN: {self.output_path}")

        # Print model info
        self.print_model_info()

        return True

    def _prepare_calibration_dataset(self, num_samples: int = 100):
        """Prepare calibration dataset for quantization.

        The calibration dataset should contain representative RGB-D samples
        collected from the rover during operation.
        """
        if not self.calibration_data_dir or not os.path.exists(self.calibration_data_dir):
            print(f"WARNING: Calibration data directory not found: {self.calibration_data_dir}")
            return None

        print(f"Loading calibration data from {self.calibration_data_dir}...")

        # Look for .npz files saved by the observation node
        calibration_files = list(Path(self.calibration_data_dir).glob('*.npz'))

        if not calibration_files:
            print("WARNING: No calibration files found")
            return None

        # Load samples
        dataset = []
        for i, file_path in enumerate(calibration_files[:num_samples]):
            try:
                data = np.load(file_path)
                # Assume the observation includes RGB + depth channels
                # Extract RGB (first 3 channels) and depth (channel index 1)
                obs = data['observation']

                # For RGB-D model: need separate RGB and depth
                # This is a placeholder - adjust based on your actual data format
                if obs.shape[0] >= 3:  # Has RGB (3 channels) + Depth (1 channel)
                    # Extract RGB (first 3 channels) and depth (channel index 3)
                    # Assuming format is (C, H, W)
                    rgb = obs[:3].transpose(1, 2, 0)  # (H, W, 3)
                    depth = obs[3]  # Depth channel
                    dataset.append({'rgb': rgb, 'depth': depth})
                elif obs.shape[0] >= 2:  # Has at least occupancy + depth (fallback)
                    # Create dummy RGB for now (you'd extract real RGB from your data)
                    rgb = np.random.randint(0, 255, (240, 424, 3), dtype=np.uint8)
                    depth = obs[1]  # Depth channel
                    dataset.append({'rgb': rgb, 'depth': depth})

            except Exception as exc:
                print(f"Failed to load {file_path}: {exc}")
                continue

        if not dataset:
            print("WARNING: No valid calibration samples loaded")
            return None

        print(f"Loaded {len(dataset)} calibration samples")

        # Convert to RKNN dataset format
        # Note: This is a simplified version - adjust based on your model inputs
        def data_generator():
            for sample in dataset:
                yield [sample['rgb'], sample['depth']]

        return data_generator

    def print_model_info(self):
        """Print RKNN model information."""
        # Get SDK version
        sdk_version = self.rknn.get_sdk_version()
        print(f"RKNN SDK Version: {sdk_version}")

    def test_inference(self, test_rgb: np.ndarray, test_depth: np.ndarray):
        """Test inference on sample data."""
        print("Testing RKNN inference...")

        # Initialize runtime on simulator (for testing on x86)
        ret = self.rknn.init_runtime(target='rk3588', target_sub_class='RKNN_NPU_CORE_0')
        if ret != 0:
            print(f"RKNN init_runtime failed: {ret}")
            return

        # Run inference
        outputs = self.rknn.inference(inputs=[test_rgb, test_depth])
        print(f"Inference output shape: {[o.shape for o in outputs]}")
        print(f"Action: {outputs[0]}")

        # Release runtime
        self.rknn.release()

    def __del__(self):
        """Cleanup RKNN resources."""
        if hasattr(self, 'rknn'):
            self.rknn.release()


def main():
    parser = argparse.ArgumentParser(description='Convert ONNX to RKNN')
    parser.add_argument('onnx_path', type=str, help='Path to ONNX model')
    parser.add_argument('--output', type=str, help='Output RKNN path (default: same as input with .rknn extension)')
    parser.add_argument('--calibration-data', type=str, help='Path to calibration dataset directory')
    parser.add_argument('--target', type=str, default='rk3588', help='Target platform (default: rk3588)')
    parser.add_argument('--no-quantize', action='store_true', help='Disable quantization (float16 mode)')
    parser.add_argument('--optimization-level', type=int, default=3, choices=[0, 1, 2, 3],
                        help='Optimization level (default: 3)')
    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = args.onnx_path.replace('.onnx', '.rknn')

    # Convert
    converter = RKNNConverter(
        onnx_path=args.onnx_path,
        output_path=output_path,
        calibration_data_dir=args.calibration_data
    )

    success = converter.convert(
        target_platform=args.target,
        quantize=not args.no_quantize,
        do_quantization=bool(args.calibration_data and not args.no_quantize),
        optimization_level=args.optimization_level
    )

    if success:
        print("\n✓ Conversion complete!")
        print(f"Deploy this model to your rover: {output_path}")
    else:
        print("\n✗ Conversion failed")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
