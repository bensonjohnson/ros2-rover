#!/usr/bin/env python3
"""
Test script for RKNN model conversion
"""

import numpy as np
import sys
import os

# Add the tractor_bringup package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'tractor_bringup'))

try:
    from tractor_bringup.rknn_trainer_depth import RKNNTrainerDepth, RKNN_AVAILABLE
    print("Successfully imported RKNNTrainerDepth")
    
    if not RKNN_AVAILABLE:
        print("RKNN toolkit not available - testing without it")
    
    # Create a trainer instance
    trainer = RKNNTrainerDepth(model_dir="models", enable_debug=True)
    print("Created RKNNTrainerDepth instance")
    
    # Check if dataset.txt exists and what's in it
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset.txt')
    print(f"Dataset path: {dataset_path}")
    
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r') as f:
            content = f.read()
            print(f"Dataset file content ({len(content)} characters):")
            print(repr(content[:200]))  # Show first 200 characters
            if len(content) > 200:
                print("...")
    else:
        print("Dataset file does not exist")
    
    # Try to convert to RKNN
    print("Attempting RKNN conversion...")
    try:
        trainer.convert_to_rknn()
        print("RKNN conversion completed successfully")
    except Exception as e:
        print(f"RKNN conversion failed: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"Failed to import or create RKNNTrainerDepth: {e}")
    import traceback
    traceback.print_exc()
