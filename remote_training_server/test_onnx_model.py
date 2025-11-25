#!/usr/bin/env python3
"""Test ONNX model with realistic rover inputs to validate before RKNN conversion."""

import numpy as np
import onnx
import onnxruntime as ort

def test_onnx_model(onnx_path: str):
    """Test ONNX model with realistic inputs."""

    print(f"Testing ONNX model: {onnx_path}\n")

    # Load ONNX model
    try:
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print("✓ ONNX model is valid\n")
    except Exception as e:
        print(f"❌ ONNX model validation failed: {e}")
        return False

    # Create ONNX Runtime session
    session = ort.InferenceSession(onnx_path)

    # Print model info
    print("Model Inputs:")
    for input in session.get_inputs():
        print(f"  {input.name}: {input.shape} ({input.type})")
    print("\nModel Outputs:")
    for output in session.get_outputs():
        print(f"  {output.name}: {output.shape} ({output.type})")
    print()

    # Test 1: Random Gaussian inputs (like training validation)
    print("Test 1: Random Gaussian inputs (training-style validation)")
    rgb = np.random.randn(1, 3, 240, 424).astype(np.float32)
    depth = np.random.randn(1, 1, 240, 424).astype(np.float32)
    proprio = np.random.randn(1, 6).astype(np.float32)

    result = session.run(None, {'rgb': rgb, 'depth': depth, 'proprio': proprio})
    print(f"  Output: {result[0]}")
    print(f"  Range: [{result[0].min():.6f}, {result[0].max():.6f}]")
    if np.isnan(result[0]).any() or np.isinf(result[0]).any():
        print("  ❌ NaN/Inf detected!")
    else:
        print("  ✓ Valid output\n")

    # Test 2: Normalized inputs (like rover inference)
    print("Test 2: Normalized rover-like inputs")
    # Simulate normalized rover data
    rgb = np.random.rand(1, 3, 240, 424).astype(np.float32)  # [0, 1] like RGB/255
    depth = np.random.rand(1, 1, 240, 424).astype(np.float32)  # [0, 1] like depth/6
    # Realistic proprio values from rover logs
    proprio = np.array([[-0.21, 2.33, 1.68, -0.18, 1.68, 0.43]], dtype=np.float32)

    result = session.run(None, {'rgb': rgb, 'depth': depth, 'proprio': proprio})
    print(f"  Output: {result[0]}")
    print(f"  Range: [{result[0].min():.6f}, {result[0].max():.6f}]")
    if np.isnan(result[0]).any() or np.isinf(result[0]).any():
        print("  ❌ NaN/Inf detected!")
    else:
        print("  ✓ Valid output\n")

    # Test 3: Zero inputs (edge case)
    print("Test 3: Zero inputs (edge case)")
    rgb = np.zeros((1, 3, 240, 424), dtype=np.float32)
    depth = np.zeros((1, 1, 240, 424), dtype=np.float32)
    proprio = np.zeros((1, 6), dtype=np.float32)

    result = session.run(None, {'rgb': rgb, 'depth': depth, 'proprio': proprio})
    print(f"  Output: {result[0]}")
    print(f"  Range: [{result[0].min():.6f}, {result[0].max():.6f}]")
    if np.isnan(result[0]).any() or np.isinf(result[0]).any():
        print("  ❌ NaN/Inf detected!")
    else:
        print("  ✓ Valid output\n")

    # Test 4: Extreme proprio values
    print("Test 4: Extreme proprio values")
    rgb = np.random.rand(1, 3, 240, 424).astype(np.float32)
    depth = np.random.rand(1, 1, 240, 424).astype(np.float32)
    proprio = np.array([[5.0, -5.0, 10.0, -10.0, 5.0, 3.0]], dtype=np.float32)

    result = session.run(None, {'rgb': rgb, 'depth': depth, 'proprio': proprio})
    print(f"  Output: {result[0]}")
    print(f"  Range: [{result[0].min():.6f}, {result[0].max():.6f}]")
    if np.isnan(result[0]).any() or np.isinf(result[0]).any():
        print("  ❌ NaN/Inf detected!")
    else:
        print("  ✓ Valid output\n")

    print("=" * 60)
    print("All tests completed!")
    return True


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        onnx_path = sys.argv[1]
    else:
        onnx_path = './checkpoints_ppo/latest_actor.onnx'

    test_onnx_model(onnx_path)
