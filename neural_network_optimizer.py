#!/usr/bin/env python3
"""
Neural Network Optimization Configuration for ROS2 Rover ES-Hybrid Mode
Automatically tunes ES parameters, network architecture, and reward systems for optimal learning
"""

import os
import json
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import argparse

@dataclass
class OptimizationConfig:
    """Configuration class for neural network optimization"""
    
    # Network Architecture
    network_mode: str = "balanced"  # "fast", "balanced", "accurate"
    width_multiplier: float = 1.0
    enable_temporal: bool = False
    stacked_frames: int = 1
    
    # ES Training Parameters
    population_size: int = 15
    sigma: float = 0.08
    learning_rate: float = 0.015
    evolution_frequency: int = 50
    elite_ratio: float = 0.25
    
    # Multi-Objective Weights
    exploration_weight: float = 0.4
    efficiency_weight: float = 0.3
    safety_weight: float = 0.2
    smoothness_weight: float = 0.1
    
    # Reward System
    reward_mode: str = "balanced"  # "exploration", "balanced", "safety", "efficiency"
    enable_curiosity: bool = True
    enable_adaptive_scaling: bool = True
    curriculum_learning: bool = True
    
    # Performance Optimization
    target_fps: float = 30.0
    max_inference_latency: float = 33.0  # ms
    enable_dynamic_inference: bool = True
    
    # Hardware Optimization (RK3588)
    cpu_threads: int = 6
    npu_optimization: bool = True
    quantization_enabled: bool = True

class PerformanceProfiler:
    """Profiles system performance and suggests optimizations"""
    
    def __init__(self):
        self.metrics = {
            'inference_latency': [],
            'training_throughput': [],
            'memory_usage': [],
            'cpu_usage': [],
            'reward_variance': [],
            'exploration_rate': [],
            'success_rate': []
        }
        self.optimization_history = []
    
    def record_performance(self, **metrics):
        """Record performance metrics"""
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def analyze_bottlenecks(self) -> Dict[str, str]:
        """Analyze performance bottlenecks and suggest optimizations"""
        bottlenecks = {}
        
        # Inference latency analysis
        if len(self.metrics['inference_latency']) > 10:
            avg_latency = np.mean(self.metrics['inference_latency'][-10:])
            if avg_latency > 35.0:
                bottlenecks['inference'] = "High inference latency detected - consider reducing network complexity"
            elif avg_latency < 15.0:
                bottlenecks['inference'] = "Low inference latency - can afford more complex network"
        
        # Training efficiency analysis
        if len(self.metrics['reward_variance']) > 20:
            reward_var = np.var(self.metrics['reward_variance'][-20:])
            if reward_var > 50.0:
                bottlenecks['training'] = "High reward variance - consider stabilizing reward system"
            elif reward_var < 2.0:
                bottlenecks['training'] = "Low reward variance - may need more exploration"
        
        # Exploration analysis
        if len(self.metrics['exploration_rate']) > 10:
            exp_rate = np.mean(self.metrics['exploration_rate'][-10:])
            if exp_rate < 0.3:
                bottlenecks['exploration'] = "Low exploration rate - increase curiosity rewards"
            elif exp_rate > 2.0:
                bottlenecks['exploration'] = "High exploration rate - focus more on efficiency"
        
        return bottlenecks
    
    def suggest_config_changes(self, current_config: OptimizationConfig) -> OptimizationConfig:
        """Suggest configuration changes based on performance analysis"""
        new_config = OptimizationConfig(**asdict(current_config))
        bottlenecks = self.analyze_bottlenecks()
        
        # Apply suggestions based on bottlenecks
        if 'inference' in bottlenecks and "High inference latency" in bottlenecks['inference']:
            new_config.width_multiplier *= 0.9
            new_config.network_mode = "fast"
            new_config.enable_temporal = False
        elif 'inference' in bottlenecks and "Low inference latency" in bottlenecks['inference']:
            new_config.width_multiplier *= 1.1
            if new_config.network_mode == "fast":
                new_config.network_mode = "balanced"
        
        if 'training' in bottlenecks and "High reward variance" in bottlenecks['training']:
            new_config.enable_adaptive_scaling = True
            new_config.sigma *= 0.9
            new_config.safety_weight *= 1.2
        elif 'training' in bottlenecks and "Low reward variance" in bottlenecks['training']:
            new_config.sigma *= 1.1
            new_config.exploration_weight *= 1.2
        
        if 'exploration' in bottlenecks:
            if "Low exploration rate" in bottlenecks['exploration']:
                new_config.exploration_weight *= 1.3
                new_config.enable_curiosity = True
                new_config.reward_mode = "exploration"
            elif "High exploration rate" in bottlenecks['exploration']:
                new_config.efficiency_weight *= 1.3
                new_config.exploration_weight *= 0.8
        
        return new_config

class AutoOptimizer:
    """Automatically optimizes neural network configuration for ROS2 rover"""
    
    def __init__(self, base_config: Optional[OptimizationConfig] = None):
        self.config = base_config or OptimizationConfig()
        self.profiler = PerformanceProfiler()
        self.optimization_iterations = 0
        self.best_config = None
        self.best_performance_score = -float('inf')
        
        # Hardware detection
        self.hardware_info = self._detect_hardware()
        
        # Optimization constraints
        self.min_performance_score = 5.0
        self.max_optimization_iterations = 10
        
    def _detect_hardware(self) -> Dict[str, any]:
        """Detect hardware capabilities"""
        hardware = {
            'cpu_cores': os.cpu_count() or 8,
            'has_npu': self._check_npu_availability(),
            'memory_gb': self._estimate_memory(),
            'platform': 'rk3588'  # Assume RK3588 for this rover
        }
        
        print(f"✓ Hardware detected: {hardware}")
        return hardware
    
    def _check_npu_availability(self) -> bool:
        """Check if NPU is available"""
        try:
            from rknn.api import RKNN
            return True
        except ImportError:
            return False
    
    def _estimate_memory(self) -> float:
        """Estimate available memory in GB"""
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        kb = int(line.split()[1])
                        return kb / (1024 * 1024)  # Convert to GB
        except:
            pass
        return 8.0  # Default assumption for RK3588
    
    def optimize_for_hardware(self) -> OptimizationConfig:
        """Optimize configuration for detected hardware"""
        config = OptimizationConfig(**asdict(self.config))
        
        # CPU optimization
        if self.hardware_info['cpu_cores'] >= 8:
            config.cpu_threads = 6  # Leave 2 cores for system
            config.population_size = min(20, config.population_size + 5)
        elif self.hardware_info['cpu_cores'] >= 4:
            config.cpu_threads = 3
            config.population_size = max(10, config.population_size - 3)
        
        # Memory optimization
        if self.hardware_info['memory_gb'] >= 16:
            config.population_size = min(25, config.population_size + 5)
        elif self.hardware_info['memory_gb'] < 8:
            config.population_size = max(8, config.population_size - 5)
            config.width_multiplier *= 0.8
        
        # NPU optimization
        if self.hardware_info['has_npu']:
            config.npu_optimization = True
            config.quantization_enabled = True
            config.enable_dynamic_inference = True
        else:
            config.npu_optimization = False
            config.quantization_enabled = False
            config.width_multiplier *= 0.9  # Smaller network for CPU-only
        
        return config
    
    def optimize_for_task(self, task_type: str = "exploration") -> OptimizationConfig:
        """Optimize configuration for specific task"""
        config = OptimizationConfig(**asdict(self.config))
        
        task_configs = {
            "exploration": {
                "exploration_weight": 0.45,
                "efficiency_weight": 0.25,
                "safety_weight": 0.2,
                "smoothness_weight": 0.1,
                "reward_mode": "exploration",
                "enable_curiosity": True,
                "sigma": 0.1,
                "population_size": 18
            },
            "navigation": {
                "exploration_weight": 0.25,
                "efficiency_weight": 0.4,
                "safety_weight": 0.25,
                "smoothness_weight": 0.1,
                "reward_mode": "balanced",
                "enable_curiosity": True,
                "sigma": 0.06,
                "population_size": 15
            },
            "mapping": {
                "exploration_weight": 0.35,
                "efficiency_weight": 0.3,
                "safety_weight": 0.25,
                "smoothness_weight": 0.1,
                "reward_mode": "exploration",
                "enable_curiosity": True,
                "curriculum_learning": True,
                "sigma": 0.08
            }
        }
        
        if task_type in task_configs:
            task_config = task_configs[task_type]
            for key, value in task_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        return config
    
    def run_performance_benchmark(self, config: OptimizationConfig, duration_minutes: float = 2.0) -> float:
        """Run performance benchmark with given configuration"""
        print(f"🔍 Benchmarking configuration (duration: {duration_minutes:.1f} min)...")
        
        # Simulate performance metrics (in real implementation, this would run actual training)
        metrics = self._simulate_benchmark(config, duration_minutes)
        
        # Record metrics
        for key, value in metrics.items():
            self.profiler.record_performance(**{key: value})
        
        # Calculate performance score
        score = self._calculate_performance_score(metrics)
        
        print(f"  Performance score: {score:.2f}")
        return score
    
    def _simulate_benchmark(self, config: OptimizationConfig, duration: float) -> Dict[str, float]:
        """Simulate benchmark metrics (replace with actual benchmark in real implementation)"""
        
        # Simulate based on configuration parameters
        base_latency = 25.0
        if config.network_mode == "fast":
            base_latency *= 0.7
        elif config.network_mode == "accurate":
            base_latency *= 1.4
        
        base_latency *= config.width_multiplier
        
        # Add some randomness to simulate real conditions
        inference_latency = base_latency + np.random.normal(0, 3)
        
        # Simulate other metrics
        metrics = {
            'inference_latency': max(5.0, inference_latency),
            'training_throughput': 100.0 / config.population_size + np.random.normal(0, 2),
            'memory_usage': config.population_size * config.width_multiplier * 50 + np.random.normal(0, 10),
            'reward_variance': abs(np.random.normal(20, 10)),
            'exploration_rate': config.exploration_weight * 2 + np.random.normal(0, 0.2),
            'success_rate': min(1.0, 0.3 + config.safety_weight + np.random.normal(0, 0.1))
        }
        
        return metrics
    
    def _calculate_performance_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall performance score from metrics"""
        
        # Normalize metrics and calculate weighted score
        score = 0.0
        
        # Inference latency (lower is better, target ~25ms)
        latency_score = max(0, 10 - abs(metrics['inference_latency'] - 25) / 5)
        score += latency_score * 0.3
        
        # Training throughput (higher is better)
        throughput_score = min(10, metrics['training_throughput'] / 10)
        score += throughput_score * 0.2
        
        # Exploration rate (target ~1.0)
        exploration_score = max(0, 10 - abs(metrics['exploration_rate'] - 1.0) * 5)
        score += exploration_score * 0.2
        
        # Success rate (higher is better)
        success_score = metrics['success_rate'] * 10
        score += success_score * 0.2
        
        # Reward stability (lower variance is better)
        stability_score = max(0, 10 - metrics['reward_variance'] / 5)
        score += stability_score * 0.1
        
        return score
    
    def auto_optimize(self, target_task: str = "exploration", 
                     benchmark_duration: float = 2.0) -> OptimizationConfig:
        """Automatically optimize configuration for best performance"""
        
        print("🚀 Starting automatic neural network optimization...")
        print(f"Target task: {target_task}")
        print(f"Hardware: {self.hardware_info}")
        
        # Start with hardware-optimized configuration
        current_config = self.optimize_for_hardware()
        current_config = self.optimize_for_task(target_task)
        
        print(f"\n📋 Initial configuration:")
        self._print_config(current_config)
        
        # Benchmark initial configuration
        current_score = self.run_performance_benchmark(current_config, benchmark_duration)
        self.best_config = current_config
        self.best_performance_score = current_score
        
        # Iterative optimization
        for iteration in range(self.max_optimization_iterations):
            print(f"\n🔄 Optimization iteration {iteration + 1}/{self.max_optimization_iterations}")
            
            # Get suggested improvements
            suggested_config = self.profiler.suggest_config_changes(current_config)
            
            # Benchmark suggested configuration
            suggested_score = self.run_performance_benchmark(suggested_config, benchmark_duration)
            
            # Update if improvement
            if suggested_score > current_score:
                current_config = suggested_config
                current_score = suggested_score
                print(f"✓ Improvement found! Score: {current_score:.2f}")
                
                # Update best if this is the best so far
                if current_score > self.best_performance_score:
                    self.best_config = current_config
                    self.best_performance_score = current_score
            else:
                print(f"  No improvement (score: {suggested_score:.2f})")
            
            # Early stopping if good enough
            if current_score > 8.0:
                print("✓ Excellent performance achieved, stopping optimization")
                break
        
        print(f"\n🎯 Optimization complete!")
        print(f"Best performance score: {self.best_performance_score:.2f}")
        print(f"\n📋 Optimized configuration:")
        self._print_config(self.best_config)
        
        return self.best_config
    
    def _print_config(self, config: OptimizationConfig):
        """Print configuration in readable format"""
        print(f"  Network: {config.network_mode} (width: {config.width_multiplier:.2f})")
        print(f"  ES: pop={config.population_size}, σ={config.sigma:.3f}, lr={config.learning_rate:.3f}")
        print(f"  Objectives: exp={config.exploration_weight:.2f}, eff={config.efficiency_weight:.2f}, "
              f"safe={config.safety_weight:.2f}")
        print(f"  Rewards: {config.reward_mode}, curiosity={config.enable_curiosity}")
        print(f"  Performance: {config.target_fps}fps, NPU={config.npu_optimization}")
    
    def save_config(self, filepath: str, config: Optional[OptimizationConfig] = None):
        """Save configuration to file"""
        config_to_save = config or self.best_config or self.config
        
        config_dict = asdict(config_to_save)
        config_dict['optimization_metadata'] = {
            'optimization_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'hardware_info': self.hardware_info,
            'performance_score': self.best_performance_score,
            'optimization_iterations': self.optimization_iterations
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
            print(f"✓ Configuration saved to {filepath}")
        except Exception as e:
            print(f"✗ Failed to save configuration: {e}")
    
    def load_config(self, filepath: str) -> OptimizationConfig:
        """Load configuration from file"""
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            
            # Remove metadata before creating config
            config_dict.pop('optimization_metadata', None)
            
            config = OptimizationConfig(**config_dict)
            print(f"✓ Configuration loaded from {filepath}")
            return config
        except Exception as e:
            print(f"✗ Failed to load configuration: {e}")
            return OptimizationConfig()

def create_launch_script(config: OptimizationConfig, output_path: str):
    """Create optimized launch script for the rover"""
    
    script_content = f"""#!/bin/bash
# Auto-generated optimized launch script for ROS2 Rover ES-Hybrid mode
# Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
# Performance score: {config.__dict__.get('performance_score', 'N/A')}

echo "🚀 Launching ROS2 Rover with optimized ES-Hybrid configuration"
echo "Network mode: {config.network_mode}"
echo "Population size: {config.population_size}"
echo "Reward mode: {config.reward_mode}"
echo ""

# Set optimized environment variables
export PYTORCH_NUM_THREADS={config.cpu_threads}
export OMP_NUM_THREADS={config.cpu_threads}
export RKNN_OPTIMIZATION={1 if config.npu_optimization else 0}

# Launch with optimized parameters
./start_npu_exploration_depth.sh es_hybrid \\
    --population_size {config.population_size} \\
    --sigma {config.sigma} \\
    --learning_rate {config.learning_rate} \\
    --network_mode {config.network_mode} \\
    --reward_mode {config.reward_mode} \\
    --enable_curiosity {1 if config.enable_curiosity else 0} \\
    --enable_adaptive_scaling {1 if config.enable_adaptive_scaling else 0} \\
    --target_fps {config.target_fps} \\
    --width_multiplier {config.width_multiplier}

echo "✓ Launch complete"
"""
    
    try:
        with open(output_path, 'w') as f:
            f.write(script_content)
        os.chmod(output_path, 0o755)  # Make executable
        print(f"✓ Optimized launch script created: {output_path}")
    except Exception as e:
        print(f"✗ Failed to create launch script: {e}")

def main():
    """Main optimization function"""
    parser = argparse.ArgumentParser(description="Neural Network Optimization for ROS2 Rover")
    parser.add_argument("--task", choices=["exploration", "navigation", "mapping"], 
                       default="exploration", help="Target task type")
    parser.add_argument("--duration", type=float, default=2.0, 
                       help="Benchmark duration in minutes")
    parser.add_argument("--config", type=str, help="Load configuration from file")
    parser.add_argument("--output", type=str, default="optimized_config.json", 
                       help="Output configuration file")
    parser.add_argument("--create_launcher", action="store_true", 
                       help="Create optimized launch script")
    
    args = parser.parse_args()
    
    print("🤖 ROS2 Rover Neural Network Optimizer")
    print("=" * 50)
    
    # Create optimizer
    optimizer = AutoOptimizer()
    
    # Load existing config if provided
    if args.config and os.path.exists(args.config):
        base_config = optimizer.load_config(args.config)
        optimizer.config = base_config
    
    # Run optimization
    optimized_config = optimizer.auto_optimize(
        target_task=args.task,
        benchmark_duration=args.duration
    )
    
    # Save results
    optimizer.save_config(args.output, optimized_config)
    
    # Create launch script if requested
    if args.create_launcher:
        script_path = "optimized_launch.sh"
        create_launch_script(optimized_config, script_path)
    
    # Print recommendations
    print("\n💡 Optimization Recommendations:")
    bottlenecks = optimizer.profiler.analyze_bottlenecks()
    if bottlenecks:
        for component, suggestion in bottlenecks.items():
            print(f"  • {component}: {suggestion}")
    else:
        print("  • No major bottlenecks detected - configuration is well-balanced")
    
    print(f"\n✅ Optimization complete! Use the configuration in {args.output}")
    
    if args.create_launcher:
        print(f"🚀 Run optimized rover with: ./optimized_launch.sh")

if __name__ == "__main__":
    main()