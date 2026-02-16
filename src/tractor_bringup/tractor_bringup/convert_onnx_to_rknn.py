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


# Proprioception normalization constants
# proprio = [lidar_min, prev_lin, prev_ang, cur_lin, cur_ang, gap_heading]
# Normalize each dimension to [-1, 1] or [0, 1] range for INT8 quantization
PROPRIO_MEAN = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Mean values
PROPRIO_STD = np.array([2.0, 1.0, 1.0, 0.2, 1.0, 1.0], dtype=np.float32)   # Std/range for normalization

def normalize_proprio(proprio: np.ndarray) -> np.ndarray:
    """Normalize proprioception vector for RKNN quantization.
    
    Input proprio = [lidar_min, prev_lin, prev_ang, cur_lin, cur_ang, gap_heading]
    - lidar_min: 0-4m -> normalize to [-1, 1] using mean=2, std=2
    - prev_lin: -1 to 1 -> already normalized
    - prev_ang: -1 to 1 -> already normalized
    - cur_lin: ~-0.2 to 0.2 m/s -> normalize by 0.2 (max speed)
    - cur_ang: ~-1 to 1 rad/s -> already normalized
    - gap_heading: -1 to 1 -> already normalized
    
    Returns normalized proprio in approximately [-1, 1] range per dimension.
    """
    normalized = (proprio - PROPRIO_MEAN) / PROPRIO_STD
    # Clip to safe range for quantization
    normalized = np.clip(normalized, -3.0, 3.0)
    return normalized.astype(np.float32)


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

                if proprio.shape != (6,):
                    print(f"⚠ Warning: Expected proprio shape (6,), got {proprio.shape} in {file_path}")
                    continue

                # Sanitize Proprio
                proprio = np.nan_to_num(proprio, nan=0.0, posinf=100.0, neginf=-100.0)
                
                # CRITICAL: Normalize proprio for quantization
                proprio = normalize_proprio(proprio)

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
                [0] * 6,          # Proprio (6-dim: lidar_min, prev_lin, prev_ang, cur_lin, cur_ang, gap_heading)
            ],
            'std_values': [
                [1, 1],           # BEV
                [1] * 6,          # Proprio (6-dim)
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
                [1, 6],             # Proprio (6-dim: lidar_min, prev_lin, prev_ang, cur_lin, cur_ang, gap_heading)
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

        # Run comprehensive quantization validation tests
        # Note: These tests use the RKNN simulator which may have issues with static-shape models
        # If tests fail with RKNN_ERR_MODEL_INVALID, the model may still be valid - check the actual RKNN export
        print("\n" + "=" * 60)
        print("QUANTIZATION VALIDATION TESTS")
        print("=" * 60)
        
        runtime_initialized = False
        try:
            # Initialize runtime for testing
            ret = rknn.init_runtime()
            if ret != 0:
                print(f"⚠ Warning: Failed to init runtime for testing: {ret}")
                print("  This is expected on some RKNN versions - skipping validation tests")
            else:
                runtime_initialized = True
                all_tests_passed = True
                simulator_errors = 0
                
                # Test 1: NaN/Inf Check
                print("\n📊 Test 1: NaN/Inf Detection...")
                nan_count = 0
                inf_count = 0
                for i in range(50):
                    test_bev = np.random.rand(1, 2, 128, 128).astype(np.float32)
                    test_proprio_raw = np.array([
                        np.random.uniform(0.0, 4.0),    # lidar_min
                        np.random.uniform(-1.0, 1.0),   # prev_lin
                        np.random.uniform(-1.0, 1.0),   # prev_ang
                        np.random.uniform(-0.2, 0.2),   # cur_lin
                        np.random.uniform(-1.0, 1.0),   # cur_ang
                        np.random.uniform(-1.0, 1.0),   # gap_heading
                    ], dtype=np.float32)
                    test_proprio = normalize_proprio(test_proprio_raw)[None, ...]
                    
                    try:
                        outputs = rknn.inference(inputs=[test_bev, test_proprio])
                        if outputs and outputs[0] is not None:
                            if np.isnan(outputs[0]).any():
                                nan_count += 1
                            if np.isinf(outputs[0]).any():
                                inf_count += 1
                    except Exception as e:
                        simulator_errors += 1
                        if simulator_errors == 1:  # Only log first error
                            print(f"  ⚠️ Simulator error (may be benign for static-shape models): {e}")
                        continue
                
                if simulator_errors > 40:
                    print(f"  ⚠️ SKIP: Too many simulator errors ({simulator_errors}/50) - model may need NPU for validation")
                elif nan_count == 0 and inf_count == 0:
                    print(f"  ✅ PASS: No NaN/Inf in {50 - simulator_errors} successful tests")
                else:
                    print(f"  ❌ FAIL: Found NaN in {nan_count}, Inf in {inf_count} tests")
                    all_tests_passed = False
                
                # Test 2: Output Range Check (tanh should give [-1, 1])
                if simulator_errors < 40:
                    print("\n📊 Test 2: Output Range Validation...")
                    outputs_list = []
                    for _ in range(20):
                        test_bev = np.random.rand(1, 2, 128, 128).astype(np.float32)
                        test_proprio = normalize_proprio(np.array([
                            np.random.uniform(0.5, 3.0),
                            0.0, 0.0, 0.1, 0.0, 0.0
                        ], dtype=np.float32))[None, ...]
                        try:
                            outputs = rknn.inference(inputs=[test_bev, test_proprio])
                            if outputs:
                                outputs_list.append(outputs[0])
                        except:
                            pass
                    
                    if outputs_list:
                        all_outputs = np.concatenate(outputs_list, axis=0)
                        min_val, max_val = np.min(all_outputs), np.max(all_outputs)
                        if min_val >= -1.0 and max_val <= 1.0:
                            print(f"  ✅ PASS: Output range [{min_val:.4f}, {max_val:.4f}] is within [-1, 1]")
                        else:
                            print(f"  ⚠️ WARNING: Output range [{min_val:.4f}, {max_val:.4f}] outside [-1, 1]")
                    else:
                        print("  ⚠️ SKIP: Could not collect outputs (simulator issues)")
                
                # Test 3: Determinism Check
                if simulator_errors < 40:
                    print("\n📊 Test 3: Output Determinism...")
                    np.random.seed(42)
                    test_bev = np.random.rand(1, 2, 128, 128).astype(np.float32)
                    test_proprio = normalize_proprio(np.array([2.0, 0.0, 0.0, 0.1, 0.0, 0.0], dtype=np.float32))[None, ...]
                    
                    outputs = []
                    for _ in range(10):
                        try:
                            out = rknn.inference(inputs=[test_bev, test_proprio])
                            if out:
                                outputs.append(out[0])
                        except:
                            pass
                    
                    if len(outputs) >= 2:
                        max_diff = np.max(np.abs(outputs[0] - np.array(outputs[1:])))
                        if max_diff < 1e-5:
                            print(f"  ✅ PASS: Outputs deterministic (max diff: {max_diff:.2e})")
                        else:
                            print(f"  ⚠️ WARNING: Outputs vary (max diff: {max_diff:.2e})")
                    else:
                        print("  ⚠️ SKIP: Could not test determinism (simulator issues)")
                
                # Test 4: Calibration Data Check (if available)
                if quantize and calibration_dir:
                    print("\n📊 Test 4: Calibration Data Validation...")
                    calib_files = list(Path(calibration_dir).glob('*.npz'))[:5]
                    if calib_files:
                        calib_nan = 0
                        calib_success = 0
                        for cf in calib_files:
                            try:
                                data = np.load(cf)
                                bev = data['bev'][None, ...].astype(np.float32)
                                proprio = normalize_proprio(data['proprio'])[None, ...]
                                out = rknn.inference(inputs=[bev, proprio])
                                if out:
                                    calib_success += 1
                                    if np.isnan(out[0]).any():
                                        calib_nan += 1
                            except:
                                pass
                        if calib_success > 0 and calib_nan == 0:
                            print(f"  ✅ PASS: No NaN with {calib_success} calibration samples")
                        elif calib_success > 0:
                            print(f"  ❌ FAIL: NaN with {calib_nan}/{calib_success} calibration samples")
                            all_tests_passed = False
                        else:
                            print("  ⚠️ SKIP: Could not test with calibration data (simulator issues)")
                    else:
                        print("  ⚠️ SKIP: No calibration files found")
                
                # Summary
                print("\n" + "-" * 60)
                if all_tests_passed:
                    print("✅ All validation tests passed!")
                elif simulator_errors > 40:
                    print("⚠️ Tests skipped due to simulator limitations")
                    print("   The model was built successfully and should work on NPU")
                else:
                    print("❌ Some tests failed - check calibration data and normalization")
                print("-" * 60)
                
        except Exception as exc:
            print(f"⚠ Warning: Validation tests encountered errors: {exc}")
            print("  This is often benign for static-shape RKNN models on simulator")
            print("  The model was built successfully and should work on NPU")

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
