#!/usr/bin/env python3
"""
Example usage of the Anti-Overtraining Reward System
Demonstrates proper setup and monitoring for robust training
"""

import numpy as np
from improved_reward_system import ImprovedRewardCalculator
from anti_overtraining_config import get_safe_config, get_curriculum_config, should_progress_curriculum
from training_monitor import TrainingMonitor

def example_training_loop():
    """Example of how to use the anti-overtraining reward system"""
    
    # Initialize with safe configuration
    config = get_safe_config()
    reward_calculator = ImprovedRewardCalculator(**config)
    monitor = TrainingMonitor()
    
    # Training parameters
    max_episodes = 1000
    current_curriculum_stage = 'beginner'
    
    print("Starting anti-overtraining training...")
    print(f"Initial configuration: {config}")
    
    for episode in range(max_episodes):
        episode_reward = 0.0
        
        # Simulate an episode
        for step in range(100):  # 100 steps per episode
            # Simulate robot state (replace with actual sensor data)
            action = np.array([0.1, 0.05])  # [linear_vel, angular_vel]
            position = np.array([step * 0.01, episode * 0.001])  # Simulated position
            collision = False
            near_collision = np.random.random() < 0.05  # 5% chance of near collision
            progress = 0.01  # Simulated progress
            
            # Calculate reward
            reward, breakdown = reward_calculator.calculate_comprehensive_reward(
                action=action,
                position=position,
                collision=collision,
                near_collision=near_collision,
                progress=progress
            )
            
            episode_reward += reward
            
            # Monitor for overtraining every 50 steps
            if step % 50 == 0:
                report = monitor.update(reward_calculator, episode_reward, episode * 100 + step)
                
                # Check for early stopping recommendation
                if monitor.get_early_stopping_recommendation():
                    print(f"Early stopping recommended at episode {episode}")
                    break
        
        # Print progress every 100 episodes
        if episode % 100 == 0:
            stats = reward_calculator.get_reward_statistics()
            print(f"\nEpisode {episode}:")
            print(f"  Average reward: {episode_reward/100:.2f}")
            print(f"  Behavioral diversity: {stats.get('behavioral_diversity', 0):.3f}")
            print(f"  Overtraining risk: {stats.get('potential_overtraining_score', 0):.3f}")
            print(f"  Explored areas: {stats.get('unique_areas', 0)}")
            
            # Check if ready for curriculum progression
            if should_progress_curriculum(stats):
                if current_curriculum_stage == 'beginner':
                    current_curriculum_stage = 'intermediate'
                    new_config = get_curriculum_config('intermediate')
                    reward_calculator = ImprovedRewardCalculator(**new_config)
                    print(f"  -> Progressed to {current_curriculum_stage} curriculum")
                elif current_curriculum_stage == 'intermediate':
                    current_curriculum_stage = 'advanced'
                    new_config = get_curriculum_config('advanced')
                    reward_calculator = ImprovedRewardCalculator(**new_config)
                    print(f"  -> Progressed to {current_curriculum_stage} curriculum")
        
        # Check if we should stop training
        if reward_calculator.should_stop_training():
            print(f"Training stopped early at episode {episode} due to overtraining detection")
            break
    
    # Final report
    print("\nTraining completed!")
    final_stats = reward_calculator.get_reward_statistics()
    print(f"Final statistics:")
    print(f"  Mean reward: {final_stats.get('mean_reward', 0):.2f}")
    print(f"  Behavioral diversity: {final_stats.get('behavioral_diversity', 0):.3f}")
    print(f"  Overtraining risk: {final_stats.get('potential_overtraining_score', 0):.3f}")
    print(f"  Total explored areas: {final_stats.get('unique_areas', 0)}")
    
    # Save training log
    monitor.save_training_log('/tmp/training_log.json')
    
    return reward_calculator, monitor

def validate_on_test_data(reward_calculator, test_positions, test_actions):
    """Validate the trained model on test data"""
    print("\nRunning validation...")
    
    validation_metrics = reward_calculator.get_validation_metrics(test_positions, test_actions)
    
    print(f"Validation results:")
    print(f"  Test mean reward: {validation_metrics.get('test_mean_reward', 0):.2f}")
    print(f"  Training-test gap: {validation_metrics.get('training_test_gap', 0):.2f}")
    print(f"  Generalization score: {validation_metrics.get('generalization_score', 0):.3f}")
    
    if validation_metrics.get('training_test_gap', 0) > 5.0:
        print("  WARNING: Large training-test gap suggests overtraining!")
    
    return validation_metrics

def demonstrate_anti_gaming_detection():
    """Demonstrate the anti-gaming detection capabilities"""
    print("\nDemonstrating anti-gaming detection...")
    
    config = get_safe_config()
    reward_calculator = ImprovedRewardCalculator(**config)
    
    # Simulate gaming behavior: spinning in place
    print("Simulating spinning behavior:")
    for i in range(10):
        action = np.array([0.0, 1.0])  # No forward movement, high angular velocity
        position = np.array([0.0, 0.0])  # Not moving
        
        reward, breakdown = reward_calculator.calculate_comprehensive_reward(
            action=action,
            position=position,
            collision=False,
            near_collision=False,
            progress=0.0
        )
        
        print(f"  Step {i}: Reward = {reward:.2f}, Anti-gaming penalty = {breakdown.get('anti_gaming', 0):.2f}")
    
    # Simulate oscillatory behavior
    print("\nSimulating oscillatory behavior:")
    positions = [np.array([0.0, 0.0]), np.array([0.1, 0.0]), np.array([0.0, 0.0]), 
                np.array([0.1, 0.0]), np.array([0.0, 0.0]), np.array([0.1, 0.0])]
    
    for i, pos in enumerate(positions):
        action = np.array([0.1, 0.0])  # Moving forward and backward
        
        reward, breakdown = reward_calculator.calculate_comprehensive_reward(
            action=action,
            position=pos,
            collision=False,
            near_collision=False,
            progress=0.0
        )
        
        print(f"  Step {i}: Position = {pos}, Reward = {reward:.2f}, Anti-gaming penalty = {breakdown.get('anti_gaming', 0):.2f}")

if __name__ == "__main__":
    # Run the example
    print("=== Anti-Overtraining Reward System Demo ===")
    
    # Demonstrate anti-gaming detection
    demonstrate_anti_gaming_detection()
    
    # Run training example
    reward_calculator, monitor = example_training_loop()
    
    # Generate test data for validation
    test_positions = [np.random.randn(2) for _ in range(100)]
    test_actions = [np.random.randn(2) * 0.1 for _ in range(100)]
    
    # Validate on test data
    validation_results = validate_on_test_data(reward_calculator, test_positions, test_actions)
    
    print("\n=== Demo completed successfully! ===")
    print("Key anti-overtraining features demonstrated:")
    print("  ✓ Simplified reward structure")
    print("  ✓ Reward clipping and noise injection")
    print("  ✓ Anti-gaming behavior detection")
    print("  ✓ Behavioral diversity tracking")
    print("  ✓ Curriculum learning progression")
    print("  ✓ Early stopping detection")
    print("  ✓ Validation-based generalization testing")
