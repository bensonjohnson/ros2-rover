#!/usr/bin/env python3
"""
Test script to verify the neural network model can be exported to ONNX
without the adaptive pooling issue
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add the source directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'tractor_bringup'))

try:
    from tractor_bringup.rknn_trainer_depth import DepthImageExplorationNet
    print("✓ Successfully imported DepthImageExplorationNet")
except ImportError as e:
    print(f"✗ Failed to import DepthImageExplorationNet: {e}")
    sys.exit(1)

def test_model_architecture():
    """Test the model architecture and forward pass"""
    print("\n=== Testing Model Architecture ===")
    
    # Initialize model
    model = DepthImageExplorationNet()
    model.eval()
    
    # Test input dimensions
    batch_size = 1
    depth_height, depth_width = 240, 424
    sensor_dim = 10
    
    # Create dummy inputs
    dummy_depth = torch.randn(batch_size, 1, depth_height, depth_width)
    dummy_sensor = torch.randn(batch_size, sensor_dim)
    
    print(f"Input depth shape: {dummy_depth.shape}")
    print(f"Input sensor shape: {dummy_sensor.shape}")
    
    # Forward pass
    try:
        with torch.no_grad():
            output = model(dummy_depth, dummy_sensor)
        print(f"✓ Forward pass successful")
        print(f"Output shape: {output.shape}")
        print(f"Output: {output}")
        
        # Check output dimensions
        if output.shape == (batch_size, 3):
            print("✓ Output shape is correct [batch_size, 3]")
        else:
            print(f"✗ Expected output shape ({batch_size}, 3), got {output.shape}")
            return False
            
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    return True

def test_onnx_export():
    """Test ONNX export without adaptive pooling issues"""
    print("\n=== Testing ONNX Export ===")
    
    # Initialize model
    model = DepthImageExplorationNet()
    model.eval()
    
    # Create dummy inputs
    dummy_depth = torch.randn(1, 1, 240, 424)
    dummy_sensor = torch.randn(1, 10)
    
    # Export to ONNX
    onnx_path = "test_model.onnx"
    
    try:
        torch.onnx.export(
            model,
            (dummy_depth, dummy_sensor),
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['depth_image', 'sensor'],
            output_names=['action_confidence'],
            verbose=True
        )
        print(f"✓ ONNX export successful: {onnx_path}")
        
        # Check if file was created
        if os.path.exists(onnx_path):
            file_size = os.path.getsize(onnx_path)
            print(f"✓ ONNX file created, size: {file_size} bytes")
            
            # Clean up
            os.remove(onnx_path)
            print("✓ Cleaned up test file")
            return True
        else:
            print("✗ ONNX file was not created")
            return False
            
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        return False

def test_layer_outputs():
    """Test individual layer outputs to verify dimensions"""
    print("\n=== Testing Layer Outputs ===")
    
    model = DepthImageExplorationNet()
    model.eval()
    
    # Create test input
    dummy_depth = torch.randn(1, 1, 240, 424)
    
    print(f"Input: {dummy_depth.shape}")
    
    # Test depth convolution layers step by step
    with torch.no_grad():
        x = dummy_depth
        for i, layer in enumerate(model.depth_conv):
            x = layer(x)
            if isinstance(layer, (nn.Conv2d, nn.AvgPool2d)):
                print(f"After layer {i} ({layer.__class__.__name__}): {x.shape}")
        
        print(f"Final depth features shape: {x.shape}")
        
        # Test if the final shape matches expected Linear input
        expected_features = 256 * 9 * 5  # Based on our fix
        actual_features = x.shape[1]
        
        if actual_features == expected_features:
            print(f"✓ Feature dimensions match: {actual_features}")
        else:
            print(f"✗ Feature dimensions mismatch: expected {expected_features}, got {actual_features}")
            return False
    
    return True

def main():
    """Run all tests"""
    print("Testing NPU Depth Exploration Model")
    print("=" * 50)
    
    tests = [
        ("Model Architecture", test_model_architecture),
        ("Layer Outputs", test_layer_outputs),
        ("ONNX Export", test_onnx_export),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST RESULTS SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:20} {status}")
        if not success:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! The model should now convert to RKNN successfully.")
    else:
        print("❌ SOME TESTS FAILED. Please check the model architecture.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
