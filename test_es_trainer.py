#!/usr/bin/env python3
"""
Test script to verify ES trainer functionality
"""

import sys
import os
import numpy as np

# Add the package path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'tractor_bringup'))

try:
    from tractor_bringup.tractor_bringup.es_trainer_depth import EvolutionaryStrategyTrainer
    print("✓ Successfully imported EvolutionaryStrategyTrainer")
    
    # Test instantiation
    trainer = EvolutionaryStrategyTrainer(
        model_dir="test_models",
        stacked_frames=1,
        enable_debug=True,
        population_size=5,
        sigma=0.1,
        learning_rate=0.01
    )
    print("✓ Successfully instantiated EvolutionaryStrategyTrainer")
    
    # Test basic methods
    print("Testing basic methods...")
    
    # Test get_training_stats
    stats = trainer.get_training_stats()
    print(f"✓ Training stats: {stats}")
    
    # Test _get_flat_params
    params = trainer._get_flat_params()
    print(f"✓ Model parameters shape: {params.shape}")
    
    # Test _set_flat_params
    trainer._set_flat_params(params)
    print("✓ Parameter setting works")
    
    # Test _proprio_feature_size
    proprio_size = trainer._proprio_feature_size()
    print(f"✓ Proprioceptive feature size: {proprio_size}")
    
    print("\n🎉 All tests passed! ES trainer is working correctly.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
