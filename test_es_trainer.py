#!/usr/bin/env python3
"""
Test script to demonstrate Evolutionary Strategy trainer functionality
"""

import numpy as np
import sys
import os

# Add the tractor_bringup package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'tractor_bringup', 'tractor_bringup'))

try:
    from es_trainer_depth import EvolutionaryStrategyTrainer
    print("✓ Successfully imported ES trainer")
except ImportError as e:
    print(f"✗ Failed to import ES trainer: {e}")
    sys.exit(1)

def demonstrate_es_concept():
    """Demonstrate how Evolutionary Strategies work"""
    print("\n" + "="*60)
    print("EVOLUTIONARY STRATEGY (ES) EXPLANATION")
    print("="*60)
    
    print("\n1. POPULATION-BASED OPTIMIZATION:")
    print("   - Instead of gradient descent, ES maintains a population of parameter variations")
    print("   - Each individual in the population represents a slight modification of the current model")
    print("   - Population size determines how many variations we test simultaneously")
    
    print("\n2. FITNESS EVALUATION:")
    print("   - Each individual is evaluated on how well it performs the task")
    print("   - Fitness is calculated based on rewards and action similarity")
    print("   - Higher fitness = better performing model")
    
    print("\n3. NATURAL GRADIENT ESTIMATION:")
    print("   - Instead of computing gradients through backpropagation")
    print("   - ES estimates the gradient by observing how parameter changes affect fitness")
    print("   - This makes ES more robust in noisy, non-differentiable environments")
    
    print("\n4. PARAMETER UPDATE:")
    print("   - Parameters are updated in the direction that improves fitness")
    print("   - The best performing individuals guide the evolution of the next generation")
    print("   - This process repeats for multiple generations")
    
    print("\n" + "="*60)
    print("ES VS REINFORCEMENT LEARNING (RL)")
    print("="*60)
    
    print("\nADVANTAGES OF ES:")
    print("   ✓ More robust in noisy environments")
    print("   ✓ Parallelizable - can evaluate multiple individuals simultaneously")
    print("   ✓ No local minima issues - population-based search")
    print("   ✓ Simpler implementation - no complex backpropagation")
    print("   ✓ Better for robotics - handles discontinuous action spaces well")
    
    print("\nDISADVANTAGES OF ES:")
    print("   ✗ Requires more function evaluations than RL")
    print("   ✗ May be less sample-efficient in some cases")
    print("   ✗ No explicit policy gradient estimation")

def test_es_trainer():
    """Test the ES trainer functionality"""
    print("\n" + "="*60)
    print("TESTING ES TRAINER")
    print("="*60)
    
    # Create a trainer instance
    trainer = EvolutionaryStrategyTrainer(
        model_dir="test_models",
        population_size=5,  # Small population for testing
        sigma=0.05,         # Small perturbation
        learning_rate=0.01,
        enable_debug=True
    )
    
    print(f"✓ ES Trainer created with population size: {trainer.population_size}")
    print(f"✓ Sigma (perturbation size): {trainer.sigma}")
    print(f"✓ Learning rate: {trainer.learning_rate}")
    print(f"✓ Current generation: {trainer.generation}")
    
    # Show population details
    print(f"\nPopulation details:")
    print(f"  - Number of individuals: {len(trainer.population)}")
    if trainer.population:
        print(f"  - Parameter shape: {trainer.population[0].shape}")
        print(f"  - First individual norm: {np.linalg.norm(trainer.population[0]):.4f}")
    
    # Simulate adding some experience
    print(f"\nSimulating experience collection...")
    dummy_depth = np.random.rand(160, 288).astype(np.float32)  # Simulate depth image
    dummy_proprio = np.random.rand(16).astype(np.float32)     # Simulate proprioceptive data
    dummy_action = np.array([0.5, -0.2], dtype=np.float32)    # Simulate action
    
    # Add multiple experiences
    for i in range(10):
        trainer.add_experience(
            depth_image=dummy_depth,
            proprioceptive=dummy_proprio,
            action=dummy_action,
            reward=np.random.rand() * 10 - 5,  # Random reward between -5 and 5
            done=False
        )
    
    print(f"✓ Added 10 experiences to buffer")
    print(f"✓ Buffer size: {trainer.buffer_size}")
    
    # Show fitness evaluation for one individual
    if trainer.population:
        print(f"\nEvaluating fitness for first individual...")
        fitness = trainer.evaluate_individual(trainer.population[0])
        print(f"✓ Fitness score: {fitness:.4f}")
    
    print(f"\n✓ ES trainer test completed successfully!")

if __name__ == "__main__":
    print("Evolutionary Strategy Trainer Test Script")
    print("This script demonstrates how ES works in the rover navigation system")
    
    demonstrate_es_concept()
    test_es_trainer()
    
    print("\n" + "="*60)
    print("HOW TO USE ES IN ROVER NAVIGATION")
    print("="*60)
    print("\n1. Start the rover with ES training mode:")
    print("   ./start_npu_exploration_depth.sh es_training")
    print("\n2. Monitor training progress:")
    print("   ros2 topic echo /npu_exploration_status")
    print("\n3. View fitness improvements:")
    print("   Look for 'Best Fitness' values in the status messages")
    print("\n4. Switch to inference mode once training is sufficient:")
    print("   ./start_npu_exploration_depth.sh es_inference")
    
    print("\n✓ Test script completed!")
