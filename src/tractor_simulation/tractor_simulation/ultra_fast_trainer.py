#!/usr/bin/env python3
"""
Ultra-Fast Training System
Combines Bayesian Optimization + Vectorized Simulation + Optimized Neural Networks
Expected 100-1000x speedup over original ES approach
"""

import torch
import numpy as np
import time
from typing import Dict, List, Tuple
import argparse
import json
from pathlib import Path

# Import our optimized components
from vectorized_simulation import VectorizedTractorSimulation, BatchSimulationTrainer
from optimized_model import OptimizedDepthModel, UltraFastTrainer, create_optimized_model
from bayesian_trainer import BayesianOptimizationTrainer

class UltraFastRobotTrainer:
    """
    Complete ultra-fast training system
    """
    
    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 64,
        model_params: int = 1000,  # Number of parameters to optimize
        use_bayesian: bool = True,
        save_dir: str = "models/ultra_fast"
    ):
        self.device = device
        self.batch_size = batch_size
        self.model_params = model_params
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Initializing Ultra-Fast Training System")
        print(f"   Device: {device}")
        print(f"   Batch size: {batch_size}")
        print(f"   Model parameters: {model_params}")
        
        # 1. Create optimized neural network
        self.model, self.nn_trainer = create_optimized_model(device=device)
        
        # 2. Create vectorized simulation  
        self.sim_trainer = BatchSimulationTrainer(
            model=self.model,
            batch_size=batch_size,
            device=device
        )
        
        # 3. Create optimization strategy
        if use_bayesian:
            self.optimizer = BayesianOptimizationTrainer(
                model_dim=model_params,
                bounds=torch.stack([
                    -2.0 * torch.ones(model_params),
                    2.0 * torch.ones(model_params)
                ]),
                batch_size=8,  # BO batch size  
                n_init=20
            )
            self.optimization_type = "Bayesian Optimization"
        else:
            # Fallback to improved evolutionary strategy
            self.optimizer = None
            self.optimization_type = "Vectorized ES"
            
        # Training state
        self.best_params = None
        self.best_fitness = -float('inf')
        self.training_history = []
        
        print(f"✓ System initialized with {self.optimization_type}")
        
    def objective_function(self, parameters: torch.Tensor) -> float:
        """
        Objective function that evaluates a set of model parameters
        This is where the magic happens - ultra-fast evaluation!
        """
        if parameters.dim() == 1:
            parameters = parameters.unsqueeze(0)  # Add batch dimension
            
        # Ensure parameters are on the correct device
        parameters = parameters.to(device=self.device, dtype=torch.float32)
            
        # Use vectorized simulation for evaluation
        fitness_scores = self.sim_trainer.evaluate_parameters_batch(parameters)
        
        # Return scalar for single parameter set, mean for batch
        if fitness_scores.shape[0] == 1:
            return fitness_scores.item()
        else:
            return fitness_scores.mean().item()
            
    def train_bayesian(self, n_iterations: int = 50) -> Dict:
        """Train using Bayesian Optimization"""
        print(f"\n🎯 Starting Bayesian Optimization Training")
        print(f"   Iterations: {n_iterations}")
        print(f"   Expected evaluation time per iteration: ~1-2 seconds")
        
        start_time = time.time()
        
        # Run Bayesian optimization
        best_params = self.optimizer.optimize(
            objective_fn=self.objective_function,
            n_iterations=n_iterations
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Get final results
        self.best_params = best_params
        self.best_fitness = self.optimizer.train_Y.max().item()
        
        # Save results
        results = {
            "optimization_type": "Bayesian Optimization",
            "total_time": total_time,
            "total_evaluations": len(self.optimizer.train_Y),
            "best_fitness": self.best_fitness,
            "time_per_evaluation": total_time / len(self.optimizer.train_Y),
            "evaluations_per_second": len(self.optimizer.train_Y) / total_time,
            "convergence_history": self.optimizer.train_Y.cpu().numpy().tolist()
        }
        
        print(f"\n🏆 Bayesian Optimization Complete!")
        print(f"   Total time: {total_time:.2f} seconds")
        print(f"   Total evaluations: {len(self.optimizer.train_Y)}")
        print(f"   Best fitness: {self.best_fitness:.4f}")
        print(f"   Avg time per evaluation: {results['time_per_evaluation']:.3f}s")
        print(f"   Evaluations per second: {results['evaluations_per_second']:.1f}")
        
        return results
        
    def train_vectorized_es(self, n_generations: int = 100, population_size: int = 32) -> Dict:
        """Train using vectorized evolutionary strategy"""
        print(f"\n🧬 Starting Vectorized ES Training")  
        print(f"   Generations: {n_generations}")
        print(f"   Population size: {population_size}")
        
        start_time = time.time()
        
        # Initialize population
        population = torch.randn(
            population_size, self.model_params, 
            device=self.device, dtype=torch.float32
        )
        
        best_fitness_history = []
        
        for generation in range(n_generations):
            # Evaluate entire population in parallel
            fitness_scores = self.sim_trainer.evaluate_parameters_batch(population)
            
            # Track best
            gen_best_idx = fitness_scores.argmax()
            gen_best_fitness = fitness_scores[gen_best_idx].item()
            
            if gen_best_fitness > self.best_fitness:
                self.best_fitness = gen_best_fitness
                self.best_params = population[gen_best_idx].clone()
                
            best_fitness_history.append(gen_best_fitness)
            
            # ES update (simplified)
            # Select top 25%
            top_k = population_size // 4
            top_indices = fitness_scores.topk(top_k).indices
            elite_params = population[top_indices]
            
            # Generate new population
            new_population = torch.zeros_like(population)
            for i in range(population_size):
                parent_idx = torch.randint(0, top_k, (1,)).item()
                parent = elite_params[parent_idx]
                
                # Add noise  
                noise = torch.randn_like(parent) * 0.1
                new_population[i] = parent + noise
                
            population = new_population
            
            # Print progress
            if (generation + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"   Generation {generation+1:3d}: Best = {gen_best_fitness:.4f}, "
                      f"Time = {elapsed:.1f}s, Speed = {(generation+1)*population_size/elapsed:.1f} eval/s")
                
        end_time = time.time()
        total_time = end_time - start_time
        total_evaluations = n_generations * population_size
        
        results = {
            "optimization_type": "Vectorized ES",
            "total_time": total_time,
            "total_evaluations": total_evaluations,
            "best_fitness": self.best_fitness,
            "time_per_evaluation": total_time / total_evaluations,
            "evaluations_per_second": total_evaluations / total_time,
            "convergence_history": best_fitness_history
        }
        
        print(f"\n🏆 Vectorized ES Complete!")
        print(f"   Total time: {total_time:.2f} seconds")
        print(f"   Total evaluations: {total_evaluations}")
        print(f"   Best fitness: {self.best_fitness:.4f}")
        print(f"   Avg time per evaluation: {results['time_per_evaluation']:.4f}s")
        print(f"   Evaluations per second: {results['evaluations_per_second']:.1f}")
        
        return results
        
    def save_results(self, results: Dict, filename: str = "training_results.json"):
        """Save training results"""
        save_path = self.save_dir / filename
        
        # Add best parameters to results
        if self.best_params is not None:
            results["best_parameters"] = self.best_params.cpu().numpy().tolist()
            
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"💾 Results saved to {save_path}")
        
    def benchmark_system(self):
        """Benchmark the entire system performance"""
        print(f"\n⏱️  Benchmarking System Performance")
        
        # Benchmark neural network inference
        print("   Benchmarking Neural Network:")
        for batch_size in [16, 32, 64, 128]:
            try:
                speed = self.nn_trainer.benchmark_inference(
                    batch_size=batch_size, 
                    num_iterations=20
                )
                print(f"     Batch {batch_size:3d}: {speed:7.0f} inferences/sec")
            except Exception as e:
                print(f"     Batch {batch_size:3d}: Failed ({str(e)[:30]})")
                break
                
        # Benchmark simulation
        print("\n   Benchmarking Vectorized Simulation:")
        test_params = torch.randn(64, self.model_params, device=self.device)
        
        start_time = time.time()
        for _ in range(10):
            _ = self.sim_trainer.evaluate_parameters_batch(test_params)
        end_time = time.time()
        
        sim_time = (end_time - start_time) / 10
        sim_speed = test_params.shape[0] / sim_time
        
        print(f"     Simulation: {sim_speed:.0f} episodes/sec ({sim_time:.3f}s per batch of {test_params.shape[0]})")
        
        return {
            "nn_inference_speed": speed,
            "simulation_speed": sim_speed
        }


def main():
    parser = argparse.ArgumentParser(description="Ultra-Fast Robot Training")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--batch-size", type=int, default=64, help="Simulation batch size")
    parser.add_argument("--model-params", type=int, default=1000, help="Number of model parameters")
    parser.add_argument("--method", choices=["bayesian", "es", "both"], default="bayesian", help="Optimization method")
    parser.add_argument("--iterations", type=int, default=50, help="Number of iterations/generations")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark only")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = UltraFastRobotTrainer(
        device=args.device,
        batch_size=args.batch_size,
        model_params=args.model_params,
        use_bayesian=(args.method in ["bayesian", "both"])
    )
    
    # Run benchmark
    if args.benchmark:
        trainer.benchmark_system()
        return
        
    # Run training
    if args.method == "bayesian":
        results = trainer.train_bayesian(n_iterations=args.iterations)
        trainer.save_results(results, "bayesian_results.json")
        
    elif args.method == "es":
        results = trainer.train_vectorized_es(n_generations=args.iterations)
        trainer.save_results(results, "vectorized_es_results.json")
        
    elif args.method == "both":
        # Compare both methods
        print("\n" + "="*60)
        print("COMPARISON: BAYESIAN VS VECTORIZED ES")
        print("="*60)
        
        # Bayesian first
        trainer_bo = UltraFastRobotTrainer(
            device=args.device, batch_size=args.batch_size, 
            model_params=args.model_params, use_bayesian=True
        )
        results_bo = trainer_bo.train_bayesian(n_iterations=args.iterations//2)
        
        # Then ES
        trainer_es = UltraFastRobotTrainer(
            device=args.device, batch_size=args.batch_size,
            model_params=args.model_params, use_bayesian=False
        )
        results_es = trainer_es.train_vectorized_es(n_generations=args.iterations//2)
        
        # Compare
        print(f"\n📊 COMPARISON RESULTS:")
        print(f"   Bayesian Optimization:")
        print(f"     Best fitness: {results_bo['best_fitness']:.4f}")
        print(f"     Time per eval: {results_bo['time_per_evaluation']:.4f}s")
        print(f"     Evals per sec: {results_bo['evaluations_per_second']:.1f}")
        print(f"\n   Vectorized ES:")
        print(f"     Best fitness: {results_es['best_fitness']:.4f}")
        print(f"     Time per eval: {results_es['time_per_evaluation']:.4f}s")
        print(f"     Evals per sec: {results_es['evaluations_per_second']:.1f}")
        
        # Save comparison
        comparison = {
            "bayesian": results_bo,
            "vectorized_es": results_es,
            "winner": "bayesian" if results_bo['best_fitness'] > results_es['best_fitness'] else "vectorized_es"
        }
        trainer.save_results(comparison, "method_comparison.json")


if __name__ == "__main__":
    main()