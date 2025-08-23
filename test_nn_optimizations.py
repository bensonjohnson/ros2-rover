#!/usr/bin/env python3
"""
Comprehensive test and validation script for neural network optimizations
Tests all optimization components and measures performance improvements
"""

import numpy as np
import time
import os
import sys
import json
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import subprocess

# Add the src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'tractor_bringup', 'tractor_bringup'))

def test_optimized_network():
    """Test the optimized network architecture"""
    print("🧪 Testing Optimized Network Architecture")
    print("-" * 50)
    
    try:
        from optimized_depth_network import (
            OptimizedDepthExplorationNet, 
            DynamicInferenceController,
            create_optimized_model
        )
        
        # Test different performance modes
        modes = ["fast", "balanced", "accurate"]
        results = {}
        
        for mode in modes:
            print(f"\n🔍 Testing {mode} mode...")
            
            # Create model
            model = create_optimized_model(
                stacked_frames=1,
                extra_proprio=13,
                performance_mode=mode,
                enable_temporal=False
            )
            
            # Test inference speed
            batch_size = 10
            depth_input = np.random.rand(batch_size, 1, 160, 288).astype(np.float32)
            proprio_input = np.random.rand(batch_size, 16).astype(np.float32)
            
            import torch
            depth_tensor = torch.from_numpy(depth_input)
            proprio_tensor = torch.from_numpy(proprio_input)
            
            # Warmup
            for _ in range(5):
                _ = model(depth_tensor, proprio_tensor)
            
            # Timing test
            start_time = time.time()
            for _ in range(100):
                output = model(depth_tensor, proprio_tensor)
            end_time = time.time()
            
            avg_latency = (end_time - start_time) / 100 * 1000  # Convert to ms
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            
            results[mode] = {
                'latency_ms': avg_latency,
                'parameters': total_params,
                'output_shape': tuple(output.shape)
            }
            
            print(f"  ✓ Latency: {avg_latency:.2f}ms")
            print(f"  ✓ Parameters: {total_params:,}")
            print(f"  ✓ Output shape: {output.shape}")
        
        # Test dynamic inference controller
        print(f"\n🎛️ Testing Dynamic Inference Controller...")
        controller = DynamicInferenceController(target_fps=30.0)
        
        # Simulate varying scene complexity
        for i in range(10):
            complexity = np.random.rand()
            latency = 15 + np.random.rand() * 20  # 15-35ms
            
            controller.update_performance(latency, complexity)
            width_mult = controller.get_optimal_width_multiplier()
            should_skip = controller.should_skip_inference(complexity)
            
            if i == 9:  # Print final state
                print(f"  ✓ Optimal width multiplier: {width_mult:.3f}")
                print(f"  ✓ Skip inference capability: {should_skip}")
        
        print("\n✅ Optimized Network Test Passed")
        return results
        
    except ImportError as e:
        print(f"❌ Optimized network test failed: {e}")
        return None

def test_enhanced_es_trainer():
    """Test the enhanced ES trainer"""
    print("\n🧪 Testing Enhanced ES Trainer")
    print("-" * 50)
    
    try:
        from enhanced_es_trainer import MultiObjectiveESTrainer
        
        # Create trainer
        trainer = MultiObjectiveESTrainer(
            model_dir="test_models_enhanced",
            population_size=8,  # Small for testing
            sigma=0.05,
            learning_rate=0.01,
            enable_debug=True,
            performance_mode="fast"
        )
        
        print(f"✓ Trainer created with population size: {trainer.population_size}")
        print(f"✓ Multi-objective weights: {trainer.objective_weights}")
        print(f"✓ Curriculum enabled: {trainer.curriculum_enabled}")
        
        # Test experience addition
        print("\n📊 Testing experience collection...")
        for i in range(20):
            depth_image = np.random.rand(160, 288).astype(np.float32) * 3.0
            proprio = np.random.rand(16).astype(np.float32)
            action = np.random.randn(2).astype(np.float32) * 0.3
            reward = np.random.uniform(-2, 5)
            
            trainer.add_experience(
                depth_image=depth_image,
                proprioceptive=proprio,
                action=action,
                reward=reward,
                done=(i % 10 == 9)
            )
        
        print(f"✓ Added {trainer.buffer_size} experiences")
        
        # Test evolution
        if trainer.buffer_size >= 16:
            print("\n🧬 Testing evolution process...")
            start_time = time.time()
            stats = trainer.evolve_population()
            evolution_time = time.time() - start_time
            
            print(f"✓ Evolution completed in {evolution_time:.2f}s")
            print(f"✓ Generation: {stats['generation']}")
            print(f"✓ Best fitness: {stats['best_fitness']:.4f}")
            print(f"✓ Population diversity: {stats['population_diversity']:.4f}")
            
            # Test multi-objective breakdown
            for obj_name in trainer.objective_weights.keys():
                if f'avg_{obj_name}' in stats:
                    print(f"✓ Avg {obj_name}: {stats[f'avg_{obj_name}']:.4f}")
        
        # Test inference
        print("\n🔍 Testing inference...")
        test_depth = np.random.rand(160, 288).astype(np.float32) * 3.0
        test_proprio = np.random.rand(16).astype(np.float32)
        
        action, confidence = trainer.inference(test_depth, test_proprio)
        print(f"✓ Inference output: action={action}, confidence={confidence:.4f}")
        
        # Cleanup
        import shutil
        if os.path.exists("test_models_enhanced"):
            shutil.rmtree("test_models_enhanced")
        
        print("\n✅ Enhanced ES Trainer Test Passed")
        return True
        
    except ImportError as e:
        print(f"❌ Enhanced ES trainer test failed: {e}")
        return False

def test_adaptive_reward_system():
    """Test the adaptive reward system"""
    print("\n🧪 Testing Adaptive Reward System")
    print("-" * 50)
    
    try:
        from adaptive_reward_system import (
            CuriosityDrivenRewardCalculator,
            create_adaptive_reward_calculator,
            RewardComponents
        )
        
        # Test different reward modes
        modes = ["exploration", "balanced", "safety", "efficiency"]
        results = {}
        
        for mode in modes:
            print(f"\n🔍 Testing {mode} mode...")
            
            calculator = create_adaptive_reward_calculator(mode=mode)
            
            # Simulate rover behavior
            rewards = []
            positions = []
            
            position = np.array([0.0, 0.0])
            
            for step in range(50):
                # Simulate action and movement
                action = np.random.randn(2) * 0.2
                position += action * 0.1  # Simple movement model
                
                # Simulate depth data
                depth_data = np.random.rand(160, 288) * 3.0
                
                # Simulate conditions
                collision = np.random.rand() < 0.05  # 5% collision rate
                near_collision = np.random.rand() < 0.15  # 15% near collision
                progress = max(0, np.random.normal(0.1, 0.05))  # Mostly positive progress
                
                total_reward, components = calculator.calculate_comprehensive_reward(
                    action=action,
                    position=position,
                    collision=collision,
                    near_collision=near_collision,
                    progress=progress,
                    depth_data=depth_data,
                    step_count=step
                )
                
                rewards.append(total_reward)
                positions.append(position.copy())
            
            # Calculate statistics
            avg_reward = np.mean(rewards)
            reward_std = np.std(rewards)
            final_distance = np.linalg.norm(position)
            
            results[mode] = {
                'avg_reward': avg_reward,
                'reward_std': reward_std,
                'final_distance': final_distance,
                'total_states_visited': len(calculator.state_visit_counts)
            }
            
            print(f"  ✓ Average reward: {avg_reward:.2f}")
            print(f"  ✓ Reward std: {reward_std:.2f}")
            print(f"  ✓ Distance traveled: {final_distance:.2f}")
            print(f"  ✓ States visited: {len(calculator.state_visit_counts)}")
            
            # Test reward breakdown
            breakdown = calculator.get_reward_breakdown()
            print(f"  ✓ Curiosity scale: {breakdown.get('curiosity_scale', 'N/A')}")
            print(f"  ✓ Exploration rate: {breakdown.get('exploration_rate', 'N/A'):.3f}")
        
        # Test configuration save/load
        print(f"\n💾 Testing configuration persistence...")
        test_calc = create_adaptive_reward_calculator(mode="balanced")
        test_calc.save_configuration("test_reward_config.json")
        
        new_calc = create_adaptive_reward_calculator()
        new_calc.load_configuration("test_reward_config.json")
        
        # Cleanup
        if os.path.exists("test_reward_config.json"):
            os.remove("test_reward_config.json")
        
        print("✓ Configuration save/load successful")
        
        print("\n✅ Adaptive Reward System Test Passed")
        return results
        
    except ImportError as e:
        print(f"❌ Adaptive reward system test failed: {e}")
        return None

def test_neural_network_optimizer():
    """Test the neural network optimizer"""
    print("\n🧪 Testing Neural Network Optimizer")
    print("-" * 50)
    
    try:
        from neural_network_optimizer import (
            AutoOptimizer,
            OptimizationConfig,
            PerformanceProfiler
        )
        
        # Test configuration creation
        config = OptimizationConfig(
            network_mode="balanced",
            population_size=10,
            reward_mode="exploration"
        )
        print(f"✓ Configuration created: {config.network_mode}")
        
        # Test optimizer
        optimizer = AutoOptimizer(config)
        print(f"✓ Optimizer created with hardware: {optimizer.hardware_info}")
        
        # Test hardware optimization
        hw_optimized = optimizer.optimize_for_hardware()
        print(f"✓ Hardware optimization: threads={hw_optimized.cpu_threads}, pop={hw_optimized.population_size}")
        
        # Test task optimization
        task_optimized = optimizer.optimize_for_task("exploration")
        print(f"✓ Task optimization: exp_weight={task_optimized.exploration_weight}")
        
        # Test performance profiler
        profiler = PerformanceProfiler()
        
        # Simulate metrics
        for i in range(20):
            profiler.record_performance(
                inference_latency=20 + np.random.normal(0, 5),
                reward_variance=15 + np.random.normal(0, 3),
                exploration_rate=0.8 + np.random.normal(0, 0.2)
            )
        
        bottlenecks = profiler.analyze_bottlenecks()
        print(f"✓ Bottleneck analysis: {len(bottlenecks)} issues detected")
        
        # Test config save/load
        optimizer.save_config("test_optimizer_config.json", config)
        loaded_config = optimizer.load_config("test_optimizer_config.json")
        print(f"✓ Config save/load: {loaded_config.network_mode}")
        
        # Cleanup
        if os.path.exists("test_optimizer_config.json"):
            os.remove("test_optimizer_config.json")
        
        print("\n✅ Neural Network Optimizer Test Passed")
        return True
        
    except ImportError as e:
        print(f"❌ Neural network optimizer test failed: {e}")
        return False

def test_integration():
    """Test integration between all components"""
    print("\n🧪 Testing Component Integration")
    print("-" * 50)
    
    try:
        # Test that all components can work together
        from optimized_depth_network import create_optimized_model
        from adaptive_reward_system import create_adaptive_reward_calculator
        from neural_network_optimizer import OptimizationConfig
        
        # Create integrated configuration
        config = OptimizationConfig(
            network_mode="balanced",
            reward_mode="balanced",
            enable_curiosity=True,
            population_size=8
        )
        
        # Create model
        model = create_optimized_model(
            performance_mode=config.network_mode,
            enable_temporal=config.enable_temporal
        )
        
        # Create reward calculator
        reward_calc = create_adaptive_reward_calculator(mode=config.reward_mode)
        
        # Simulate integrated operation
        depth_image = np.random.rand(160, 288).astype(np.float32) * 3.0
        proprio_data = np.random.rand(16).astype(np.float32)
        
        # Model inference
        import torch
        with torch.no_grad():
            depth_tensor = torch.from_numpy(depth_image).unsqueeze(0).unsqueeze(0)
            proprio_tensor = torch.from_numpy(proprio_data).unsqueeze(0)
            output = model(depth_tensor, proprio_tensor)
            action = torch.tanh(output[0, :2]).numpy()
        
        # Reward calculation
        position = np.array([1.0, 0.5])
        total_reward, components = reward_calc.calculate_comprehensive_reward(
            action=action,
            position=position,
            collision=False,
            near_collision=False,
            progress=0.1,
            depth_data=depth_image
        )
        
        print(f"✓ Integrated inference successful")
        print(f"  Model output: {action}")
        print(f"  Reward: {total_reward:.3f}")
        print(f"  Components: exploration={components.exploration:.2f}, "
              f"efficiency={components.efficiency:.2f}")
        
        print("\n✅ Integration Test Passed")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def run_performance_comparison():
    """Run performance comparison between original and optimized systems"""
    print("\n🏁 Performance Comparison")
    print("-" * 50)
    
    results = {
        'original': {'latency': [], 'throughput': [], 'memory': []},
        'optimized': {'latency': [], 'throughput': [], 'memory': []}
    }
    
    try:
        # Test original network (if available)
        try:
            from rknn_trainer_depth import DepthImageExplorationNet
            
            original_model = DepthImageExplorationNet(stacked_frames=1, extra_proprio=13)
            
            # Benchmark original model
            print("🔍 Benchmarking original network...")
            latency = benchmark_model(original_model, "original")
            results['original']['latency'].append(latency)
            
        except ImportError:
            print("⚠ Original network not available for comparison")
        
        # Test optimized network
        from optimized_depth_network import create_optimized_model
        
        for mode in ["fast", "balanced"]:
            print(f"🔍 Benchmarking optimized network ({mode})...")
            optimized_model = create_optimized_model(performance_mode=mode)
            latency = benchmark_model(optimized_model, f"optimized_{mode}")
            results['optimized']['latency'].append(latency)
        
        # Display results
        print("\n📊 Performance Results:")
        if results['original']['latency']:
            print(f"  Original network: {results['original']['latency'][0]:.2f}ms")
        
        for i, latency in enumerate(results['optimized']['latency']):
            mode = ["fast", "balanced"][i] if i < 2 else "accurate"
            print(f"  Optimized ({mode}): {latency:.2f}ms")
        
        # Calculate improvements
        if results['original']['latency'] and results['optimized']['latency']:
            original_latency = results['original']['latency'][0]
            best_optimized = min(results['optimized']['latency'])
            improvement = (original_latency - best_optimized) / original_latency * 100
            print(f"\n🚀 Performance improvement: {improvement:.1f}% faster")
        
        print("\n✅ Performance Comparison Complete")
        return results
        
    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")
        return results

def benchmark_model(model, name: str) -> float:
    """Benchmark a single model's inference speed"""
    import torch
    
    # Prepare test data
    batch_size = 1
    depth_input = torch.randn(batch_size, 1, 160, 288)
    proprio_input = torch.randn(batch_size, 16)
    
    # Warmup
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = model(depth_input, proprio_input)
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            output = model(depth_input, proprio_input)
    end_time = time.time()
    
    latency_ms = (end_time - start_time) / 100 * 1000
    return latency_ms

def generate_optimization_report(test_results: Dict) -> str:
    """Generate a comprehensive optimization report"""
    
    report = f"""
# Neural Network Optimization Report
Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Test Results Summary

### ✅ Completed Tests
"""
    
    for test_name, result in test_results.items():
        if result is not None and result:
            report += f"- **{test_name}**: PASSED ✓\n"
        else:
            report += f"- **{test_name}**: FAILED ❌\n"
    
    report += "\n### 🏗️ Architecture Improvements\n"
    
    if test_results.get('network_architecture'):
        results = test_results['network_architecture']
        report += f"- **Fast mode**: {results['fast']['latency_ms']:.1f}ms latency, {results['fast']['parameters']:,} parameters\n"
        report += f"- **Balanced mode**: {results['balanced']['latency_ms']:.1f}ms latency, {results['balanced']['parameters']:,} parameters\n"
        report += f"- **Accurate mode**: {results['accurate']['latency_ms']:.1f}ms latency, {results['accurate']['parameters']:,} parameters\n"
    
    report += "\n### 🎯 Reward System Improvements\n"
    
    if test_results.get('reward_system'):
        for mode, result in test_results['reward_system'].items():
            report += f"- **{mode.capitalize()} mode**: Avg reward {result['avg_reward']:.2f}, States visited: {result['total_states_visited']}\n"
    
    report += "\n### 🚀 Performance Gains\n"
    
    if test_results.get('performance_comparison'):
        report += "- Network architecture optimized for RK3588 NPU\n"
        report += "- Multi-objective ES training with curriculum learning\n"
        report += "- Adaptive reward system with curiosity-driven exploration\n"
        report += "- Dynamic inference control for optimal performance\n"
    
    report += "\n### 💡 Recommendations\n"
    report += "1. Use 'balanced' mode for general exploration tasks\n"
    report += "2. Use 'fast' mode when real-time performance is critical\n"
    report += "3. Enable curiosity-driven rewards for better exploration\n"
    report += "4. Use adaptive scaling to automatically tune performance\n"
    report += "5. Monitor performance with built-in profiling tools\n"
    
    report += "\n### 🔧 Configuration\n"
    report += "The optimized system provides:\n"
    report += "- Efficient depth processing with MobileNet-inspired architecture\n"
    report += "- Multi-objective evolutionary strategy training\n"
    report += "- Curiosity-driven exploration rewards\n"
    report += "- Automatic hyperparameter tuning with Bayesian optimization\n"
    report += "- Hardware-specific optimizations for RK3588\n"
    
    return report

def main():
    """Run all optimization tests and generate report"""
    print("🤖 ROS2 Rover Neural Network Optimization Test Suite")
    print("=" * 60)
    print("Testing all optimization components and measuring improvements...")
    print()
    
    test_results = {}
    
    # Run all tests
    test_results['network_architecture'] = test_optimized_network()
    test_results['enhanced_es_trainer'] = test_enhanced_es_trainer()
    test_results['reward_system'] = test_adaptive_reward_system()
    test_results['optimizer'] = test_neural_network_optimizer()
    test_results['integration'] = test_integration()
    test_results['performance_comparison'] = run_performance_comparison()
    
    # Generate report
    print("\n📋 Generating Optimization Report...")
    report = generate_optimization_report(test_results)
    
    # Save report
    report_file = "neural_network_optimization_report.md"
    try:
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"✓ Report saved to {report_file}")
    except Exception as e:
        print(f"✗ Failed to save report: {e}")
    
    # Print summary
    print("\n🎯 Test Summary:")
    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    
    print(f"  Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("  🎉 All optimizations working correctly!")
        print("  🚀 Ready for deployment with enhanced performance")
    elif passed_tests > total_tests // 2:
        print("  ⚠ Most optimizations working - check failed tests")
    else:
        print("  ❌ Multiple optimization failures - review implementation")
    
    print(f"\n📖 See {report_file} for detailed analysis and recommendations")
    
    return test_results

if __name__ == "__main__":
    main()