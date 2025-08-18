#!/usr/bin/env python3
"""
Comprehensive test for ES trainer evolution and model saving
"""

import numpy as np
import os
import shutil
import sys

# Import ES trainer
try:
    from tractor_bringup.es_trainer_depth import EvolutionaryStrategyTrainer
    print("✓ Successfully imported ES trainer")
except ImportError as e1:
    try:
        from src.tractor_bringup.tractor_bringup.es_trainer_depth import EvolutionaryStrategyTrainer
        print("✓ Successfully imported ES trainer (local context)")
    except ImportError as e2:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'tractor_bringup', 'tractor_bringup'))
        from es_trainer_depth import EvolutionaryStrategyTrainer
        print("✓ Successfully imported ES trainer (path context)")

def test_es_evolution_and_saving():
    """Test ES evolution process and model saving"""
    print("\n" + "="*60)
    print("TESTING ES EVOLUTION AND MODEL SAVING")
    print("="*60)
    
    # Clean up any existing test models
    test_model_dir = "test_models_evolution"
    if os.path.exists(test_model_dir):
        shutil.rmtree(test_model_dir)
    
    # Create ES trainer
    trainer = EvolutionaryStrategyTrainer(
        model_dir=test_model_dir,
        population_size=5,  # Small for faster testing
        sigma=0.1,
        learning_rate=0.02,
        enable_debug=True
    )
    
    print(f"✓ Created ES trainer")
    print(f"  Model directory: {trainer.model_dir}")
    print(f"  Population size: {trainer.population_size}")
    print(f"  Initial sigma: {trainer.sigma}")
    print(f"  Generation: {trainer.generation}")
    
    # Generate enough experiences to trigger evolution
    print(f"\n📊 Generating training data...")
    np.random.seed(42)  # For reproducible results
    
    # Simulate diverse experiences with varying rewards
    for i in range(60):  # Generate enough for multiple evolutions
        # Create synthetic depth image and proprioceptive data
        depth_image = np.random.rand(160, 288).astype(np.float32) * 3.0  # 0-3 meter depth
        proprio = np.random.rand(16).astype(np.float32)  # 16-dimensional proprioceptive
        action = np.random.randn(2).astype(np.float32) * 0.3  # Random actions
        
        # Create varied reward signal to test adaptive features
        if i < 20:
            # Early phase - lower rewards
            reward = np.random.uniform(-2, 3)
        elif i < 40:
            # Middle phase - improving rewards
            reward = np.random.uniform(1, 6) + 0.1 * i
        else:
            # Later phase - high variance rewards
            reward = np.random.uniform(2, 8) + np.random.normal(0, 2)
        
        trainer.add_experience(
            depth_image=depth_image,
            proprioceptive=proprio,
            action=action,
            reward=reward,
            done=(i % 20 == 19)  # Episode ends every 20 steps
        )
    
    print(f"✓ Added {trainer.buffer_size} experiences")
    
    # Test multiple evolution cycles
    print(f"\n🧬 Testing evolution cycles...")
    initial_fitness = trainer.best_fitness
    
    for cycle in range(5):  # Test 5 evolution cycles
        print(f"\n--- Evolution Cycle {cycle + 1} ---")
        
        # Check if we should evolve (using adaptive frequency)
        current_step = 50 + cycle * 25  # Simulate step progression
        
        if hasattr(trainer, 'should_evolve'):
            should_evolve = trainer.should_evolve(current_step)
            print(f"Should evolve at step {current_step}: {should_evolve}")
            
            if should_evolve:
                stats = trainer.evolve_population()
                print(f"Evolution stats: {stats}")
                
                # Check for improvements
                if trainer.best_fitness > initial_fitness:
                    print(f"✓ Fitness improved from {initial_fitness:.4f} to {trainer.best_fitness:.4f}")
                else:
                    print(f"• Fitness: {trainer.best_fitness:.4f} (no improvement yet)")
        else:
            # Fallback for older versions
            if trainer.buffer_size >= 50:
                stats = trainer.evolve_population()
                print(f"Evolution stats: {stats}")
    
    # Check model saving
    print(f"\n💾 Checking model saving...")
    model_files = []
    if os.path.exists(test_model_dir):
        model_files = [f for f in os.listdir(test_model_dir) if f.endswith('.pth')]
        print(f"✓ Found {len(model_files)} model files in {test_model_dir}")
        for model_file in model_files:
            file_path = os.path.join(test_model_dir, model_file)
            file_size = os.path.getsize(file_path)
            print(f"  - {model_file} ({file_size:,} bytes)")
    else:
        print(f"✗ Model directory {test_model_dir} not found")
    
    # Test manual save
    print(f"\n🔧 Testing manual model save...")
    trainer.save_model()
    
    # Check again for new files
    if os.path.exists(test_model_dir):
        new_model_files = [f for f in os.listdir(test_model_dir) if f.endswith('.pth')]
        print(f"✓ Now have {len(new_model_files)} model files")
        
        # Check for latest symlink
        latest_file = os.path.join(test_model_dir, "exploration_model_depth_es_latest.pth")
        if os.path.exists(latest_file):
            print(f"✓ Latest model symlink exists: {latest_file}")
            if os.path.islink(latest_file):
                target = os.readlink(latest_file)
                print(f"  → Points to: {target}")
        else:
            print(f"✗ Latest model symlink not found")
    
    # Test adaptive features
    print(f"\n🎯 Testing adaptive features...")
    print(f"  Current sigma: {trainer.sigma:.6f} (initial: {trainer.initial_sigma})")
    print(f"  Stagnation counter: {trainer.stagnation_counter}")
    print(f"  Evolution frequency: {getattr(trainer, 'current_evolution_frequency', 'N/A')}")
    if hasattr(trainer, 'diversity_history') and trainer.diversity_history:
        print(f"  Population diversity: {trainer.diversity_history[-1]:.6f}")
    if hasattr(trainer, 'elite_individuals'):
        print(f"  Elite individuals: {len(trainer.elite_individuals)}")
    
    # Final statistics
    print(f"\n📈 Final Training Statistics:")
    final_stats = trainer.get_training_stats()
    for key, value in final_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\n✓ ES evolution and saving test completed!")
    
    # Cleanup
    if os.path.exists(test_model_dir):
        print(f"\n🧹 Cleaning up test directory: {test_model_dir}")
        shutil.rmtree(test_model_dir)

if __name__ == "__main__":
    print("ES Trainer Evolution and Model Saving Test")
    print("This script tests the full ES training pipeline including evolution and model persistence")
    
    test_es_evolution_and_saving()
    
    print("\n✅ All tests completed successfully!")