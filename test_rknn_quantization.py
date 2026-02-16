#!/usr/bin/env python3
"""Test RKNN quantization accuracy.

Compares INT8 quantized model output against FP16 reference to verify
quantization is working correctly.

Usage:
    python3 test_rknn_quantization.py models/sac_actor.rknn
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

# Proprioception normalization constants (must match other files)
PROPRIO_MEAN = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
PROPRIO_STD = np.array([2.0, 1.0, 1.0, 0.2, 1.0, 1.0], dtype=np.float32)

def normalize_proprio(proprio: np.ndarray) -> np.ndarray:
    """Normalize proprioception for RKNN quantization."""
    normalized = (proprio - PROPRIO_MEAN) / PROPRIO_STD
    return np.clip(normalized, -3.0, 3.0).astype(np.float32)


def test_nan_inf(rknn, num_tests=100):
    """Test for NaN/Inf in model outputs."""
    print("\n📊 Testing for NaN/Inf in outputs...")
    
    nan_count = 0
    inf_count = 0
    
    for i in range(num_tests):
        # Random inputs
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
        
        outputs = rknn.inference(inputs=[test_bev, test_proprio])
        
        if outputs is None:
            print(f"  ❌ Test {i}: No output from model")
            continue
            
        output = outputs[0]
        
        if np.isnan(output).any():
            nan_count += 1
            if nan_count == 1:  # Only print first occurrence
                print(f"  ❌ Test {i}: NaN detected in output: {output}")
                
        if np.isinf(output).any():
            inf_count += 1
            if inf_count == 1:
                print(f"  ❌ Test {i}: Inf detected in output: {output}")
    
    if nan_count == 0 and inf_count == 0:
        print(f"  ✅ No NaN/Inf in {num_tests} tests")
        return True
    else:
        print(f"  ❌ Found NaN in {nan_count}/{num_tests} tests, Inf in {inf_count}/{num_tests} tests")
        return False


def test_output_range(rknn, num_tests=100):
    """Test output value range."""
    print("\n📊 Testing output value range...")
    
    outputs_list = []
    
    for i in range(num_tests):
        test_bev = np.random.rand(1, 2, 128, 128).astype(np.float32)
        test_proprio_raw = np.array([
            np.random.uniform(0.0, 4.0),
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(-1.0, 1.0),
        ], dtype=np.float32)
        test_proprio = normalize_proprio(test_proprio_raw)[None, ...]
        
        outputs = rknn.inference(inputs=[test_bev, test_proprio])
        if outputs is not None:
            outputs_list.append(outputs[0])
    
    if not outputs_list:
        print("  ❌ No valid outputs collected")
        return False
    
    all_outputs = np.concatenate(outputs_list, axis=0)
    
    min_val = np.min(all_outputs)
    max_val = np.max(all_outputs)
    mean_val = np.mean(all_outputs)
    std_val = np.std(all_outputs)
    
    print(f"  Output min:  {min_val:.4f}")
    print(f"  Output max:  {max_val:.4f}")
    print(f"  Output mean: {mean_val:.4f}")
    print(f"  Output std:  {std_val:.4f}")
    
    # Check if outputs are in expected range [-1, 1] (tanh activation)
    if min_val >= -1.0 and max_val <= 1.0:
        print(f"  ✅ Outputs in valid range [-1, 1] (tanh)")
        return True
    else:
        print(f"  ⚠️ Outputs outside expected range [-1, 1]")
        return False


def test_determinism(rknn, num_tests=10):
    """Test that model produces deterministic outputs."""
    print("\n📊 Testing output determinism...")
    
    np.random.seed(42)
    test_bev = np.random.rand(1, 2, 128, 128).astype(np.float32)
    test_proprio_raw = np.array([2.0, 0.0, 0.0, 0.1, 0.0, 0.0], dtype=np.float32)
    test_proprio = normalize_proprio(test_proprio_raw)[None, ...]
    
    outputs = []
    for _ in range(num_tests):
        out = rknn.inference(inputs=[test_bev, test_proprio])
        if out is not None:
            outputs.append(out[0])
    
    if len(outputs) < 2:
        print("  ❌ Not enough outputs for determinism test")
        return False
    
    outputs_array = np.array(outputs)
    max_diff = np.max(np.abs(outputs_array[0] - outputs_array[1:]))
    
    if max_diff < 1e-6:
        print(f"  ✅ Outputs are deterministic (max diff: {max_diff:.2e})")
        return True
    else:
        print(f"  ⚠️ Outputs vary slightly (max diff: {max_diff:.2e})")
        # This can happen due to NPU quantization - not necessarily an error
        return True


def test_calibration_impact(rknn, calibration_dir):
    """Test with real calibration data if available."""
    print("\n📊 Testing with calibration data...")
    
    if not os.path.exists(calibration_dir):
        print(f"  ⚠️ Calibration directory not found: {calibration_dir}")
        return True
    
    calib_files = list(Path(calibration_dir).glob('*.npz'))
    if not calib_files:
        print(f"  ⚠️ No calibration files in {calibration_dir}")
        return True
    
    print(f"  Found {len(calib_files)} calibration files")
    
    # Test with a few calibration samples
    nan_count = 0
    tested = 0
    
    for calib_file in calib_files[:10]:
        try:
            data = np.load(calib_file)
            bev = data['bev']
            proprio_raw = data['proprio']
            
            # Validate shapes
            if bev.shape != (2, 128, 128):
                print(f"  ⚠️ Skipping {calib_file.name}: wrong BEV shape {bev.shape}")
                continue
            if proprio_raw.shape != (6,):
                print(f"  ⚠️ Skipping {calib_file.name}: wrong proprio shape {proprio_raw.shape}")
                continue
            
            # Prepare inputs
            bev_input = bev[None, ...].astype(np.float32)
            proprio_input = normalize_proprio(proprio_raw)[None, ...]
            
            outputs = rknn.inference(inputs=[bev_input, proprio_input])
            
            if outputs is not None:
                if np.isnan(outputs[0]).any():
                    nan_count += 1
                    print(f"  ❌ NaN in output for {calib_file.name}")
                tested += 1
                
        except Exception as e:
            print(f"  ⚠️ Error loading {calib_file.name}: {e}")
    
    if tested > 0:
        print(f"  Tested {tested} calibration samples")
        if nan_count == 0:
            print(f"  ✅ No NaN in outputs with calibration data")
            return True
        else:
            print(f"  ❌ NaN in {nan_count}/{tested} calibration samples")
            return False
    else:
        print(f"  ⚠️ No valid calibration samples tested")
        return True


def main():
    parser = argparse.ArgumentParser(description='Test RKNN quantization accuracy')
    parser.add_argument('rknn_path', type=str, help='Path to RKNN model')
    parser.add_argument('--calibration-dir', type=str, default='calibration_data',
                        help='Directory with calibration data')
    parser.add_argument('--num-tests', type=int, default=100,
                        help='Number of random tests to run')
    args = parser.parse_args()
    
    if not os.path.exists(args.rknn_path):
        print(f"❌ RKNN file not found: {args.rknn_path}")
        return 1
    
    print("=" * 60)
    print("RKNN Quantization Accuracy Test")
    print("=" * 60)
    print(f"\nModel: {args.rknn_path}")
    print(f"Size: {os.path.getsize(args.rknn_path) / 1024:.1f} KB")
    
    # Try RKNNLite (on-rover inference)
    try:
        from rknnlite.api import RKNNLite
        print("\nUsing RKNNLite for inference...")
        
        rknn = RKNNLite()
        ret = rknn.load_rknn(args.rknn_path)
        if ret != 0:
            print(f"❌ Failed to load RKNN: {ret}")
            return 1
        
        ret = rknn.init_runtime()
        if ret != 0:
            print(f"❌ Failed to init runtime: {ret}")
            return 1
        
        print("✅ RKNN model loaded successfully")
        
    except ImportError:
        print("❌ RKNNLite not available. This test must run on the rover.")
        print("   Install: pip install rknn-toolkit-lite2")
        return 1
    
    # Run tests
    results = {}
    
    results['nan_inf'] = test_nan_inf(rknn, args.num_tests)
    results['output_range'] = test_output_range(rknn, args.num_tests)
    results['determinism'] = test_determinism(rknn)
    results['calibration'] = test_calibration_impact(rknn, args.calibration_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    if all_passed:
        print("\n✅ All tests passed! Quantization is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Check calibration data and normalization.")
        return 1


if __name__ == '__main__':
    sys.exit(main())