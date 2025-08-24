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
import logging
from datetime import datetime
import multiprocessing as mp

# Optimize CPU utilization for better performance
torch.set_num_threads(mp.cpu_count())  # Use all available CPU cores
torch.set_num_interop_threads(4)  # Optimize inter-operator parallelism

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

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
        # Use absolute path to save models in project root
        if not save_dir.startswith('/'):
            save_dir = f"/home/benson/Documents/ros2-rover/{save_dir}"
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🚀 Initializing Ultra-Fast Training System")
        print(f"   Device: {device}")
        print(f"   Batch size: {batch_size}")
        print(f"   Model parameters: {model_params}")
        
        # 1. Create rover-compatible neural network (exact architecture match)
        from rover_compatible_model import create_rover_compatible_model
        self.model, self.nn_trainer = create_rover_compatible_model(device=device, use_optimizations=True)
        
        # 2. Create rover-compatible vectorized simulation  
        from rover_vectorized_simulation import RoverBatchSimulationTrainer, RoverVectorizedSimulation
        self.sim_trainer = RoverBatchSimulationTrainer(
            model=self.model,
            batch_size=batch_size,
            device=device
        )
        
        # 2.5. Create parallel ES evaluator for much faster evaluation
        from parallel_es_evaluation import ParallelESEvaluator
        self.parallel_evaluator = ParallelESEvaluator(
            base_model=self.model,
            simulation=RoverVectorizedSimulation(batch_size=batch_size, device=device),
            device=device
        )
        
        # 3. Create optimization strategy (ES with optional Bayesian hyperparameter tuning)
        self.optimizer = None
        self.optimization_type = "Ultra-Fast ES"
        
        # ES parameters (matching rover's ES trainer)
        self.population_size = 64  # Increased to better utilize CPU cores
        self.sigma = 0.1  # Mutation strength
        self.learning_rate = 0.01
        self.generation = 0
        self.population = []
        self.fitness_history = []
        self.best_fitness = -float('inf')
        self.best_model_state = None
        
        # Elite preservation (matching rover)
        self.elite_ratio = 0.2
        self.n_elites = max(1, int(self.population_size * self.elite_ratio))
        self.elite_individuals = []
        self.elite_fitness_scores = []
        
        # Adaptive sigma parameters (matching rover)
        self.sigma_decay_rate = 0.99
        self.sigma_increase_rate = 1.05
        self.min_sigma = self.sigma * 0.1
        self.max_sigma = self.sigma * 3.0
        self.stagnation_counter = 0
        
        # Bayesian hyperparameter optimization (matching rover approach)
        self.bayesian_es_wrapper = None
        if use_bayesian:
            try:
                from ultra_fast_bayesian_es import UltraFastBayesianESWrapper, BOTORCH_AVAILABLE
                if BOTORCH_AVAILABLE:
                    self.bayesian_es_wrapper = UltraFastBayesianESWrapper(
                        es_trainer=self,
                        optimization_interval=5,  # Optimize every 5 generations
                        min_observations=3,
                        enable_debug=True
                    )
                    self.optimization_type = "Ultra-Fast ES + Bayesian Hyperparameters"
                    print("Using Bayesian optimization for ES hyperparameters (rover-compatible)")
                else:
                    print("BoTorch not available - using standard ES hyperparameter adaptation")
            except Exception as e:
                print(f"Failed to initialize Bayesian hyperparameter optimization: {e}")
                self.bayesian_es_wrapper = None
            
        # Training state
        self.best_params = None
        self.best_fitness = -float('inf')
        self.training_history = []
        
        # Set up logging
        log_dir = self.save_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        self.setup_logging(log_dir)
        
        print(f"✓ System initialized with {self.optimization_type}")
        
    def setup_logging(self, log_dir: Path):
        """Set up logging for training"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"training_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def apply_parameters_to_model(self, parameters: torch.Tensor):
        """Apply parameters to the neural network model"""
        try:
            # Get model parameters as a flat vector
            param_dict = dict(self.model.named_parameters())
            param_shapes = {name: param.shape for name, param in param_dict.items()}
            
            # Split parameters according to model architecture
            param_idx = 0
            with torch.no_grad():
                for name, param in param_dict.items():
                    param_size = param.numel()
                    if param_idx + param_size <= len(parameters):
                        # Reshape and assign parameters
                        new_param = parameters[param_idx:param_idx + param_size].reshape(param.shape)
                        param.copy_(new_param)
                        param_idx += param_size
                    else:
                        self.logger.warning(f"Not enough parameters for layer {name}")
                        break
                        
        except Exception as e:
            self.logger.error(f"Failed to apply parameters to model: {e}")
        
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
        
        # Store training history
        self.training_history = self.optimizer.train_Y.cpu().numpy().tolist()
        
        # Save results
        results = {
            "optimization_type": "Bayesian Optimization",
            "total_time": total_time,
            "total_evaluations": len(self.optimizer.train_Y),
            "best_fitness": self.best_fitness,
            "time_per_evaluation": total_time / len(self.optimizer.train_Y),
            "evaluations_per_second": len(self.optimizer.train_Y) / total_time,
            "convergence_history": self.training_history
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
        
        # Store training history
        self.training_history = best_fitness_history
        
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
        
    def train_ultra_fast_es(self, n_generations: int = 50) -> Dict:
        """Train using ultra-fast ES (matching rover's ES approach)"""
        print(f"\n🧬 Starting Ultra-Fast ES Training")  
        print(f"   Generations: {n_generations}")
        print(f"   Population size: {self.population_size}")
        print(f"   Using rover-compatible ES algorithm")
        
        start_time = time.time()
        
        # Initialize ES population
        self._initialize_es_population()
        
        best_fitness_history = []
        
        for generation in range(n_generations):
            gen_start_time = time.time()
            
            # Evaluate entire population in parallel using rover simulation
            fitness_scores = self._evaluate_es_population()
            
            # Track best
            gen_best_idx = fitness_scores.argmax()
            gen_best_fitness = fitness_scores[gen_best_idx].item()
            
            if gen_best_fitness > self.best_fitness:
                self.best_fitness = gen_best_fitness
                self.best_params = self.population[gen_best_idx].clone()
                
            best_fitness_history.append(gen_best_fitness)
            self.fitness_history.append(gen_best_fitness)
            
            # Apply rover-style ES evolution
            self._evolve_es_population(fitness_scores)
            
            # Apply Bayesian hyperparameter optimization (matching rover)
            if self.bayesian_es_wrapper is not None:
                self.bayesian_es_wrapper.apply_bayesian_optimization(generation, gen_best_fitness)
            
            # Update generation counter
            self.generation = generation + 1
            
            # Print progress
            gen_time = time.time() - gen_start_time
            print(f"   Generation {generation+1:3d}: Best = {gen_best_fitness:.4f}, "
                  f"Time = {gen_time:.1f}s, Sigma = {self.sigma:.6f}")
                
        end_time = time.time()
        total_time = end_time - start_time
        total_evaluations = n_generations * self.population_size
        
        results = {
            "optimization_type": self.optimization_type,
            "total_time": total_time,
            "total_evaluations": total_evaluations,
            "best_fitness": self.best_fitness,
            "time_per_evaluation": total_time / total_evaluations,
            "evaluations_per_second": total_evaluations / total_time,
            "convergence_history": best_fitness_history,
            "final_sigma": self.sigma,
            "stagnation_counter": self.stagnation_counter,
            "generations": n_generations
        }
        
        # Add Bayesian optimization statistics if available
        if self.bayesian_es_wrapper is not None:
            bayesian_stats = self.bayesian_es_wrapper.get_optimization_summary()
            results.update({f"bayesian_{k}": v for k, v in bayesian_stats.items()})
        
        print(f"\n🏆 Ultra-Fast ES Complete!")
        print(f"   Total time: {total_time:.2f} seconds")
        print(f"   Total evaluations: {total_evaluations}")
        print(f"   Best fitness: {self.best_fitness:.4f}")
        print(f"   Avg time per evaluation: {results['time_per_evaluation']:.4f}s")
        print(f"   Evaluations per second: {results['evaluations_per_second']:.1f}")
        
        # Save the best trained model
        self.save_model("ultra_fast_es_best_model.pth")
        
        return results
    
    def _initialize_es_population(self):
        """Initialize ES population (matching rover's approach)"""
        print(f"[ES] Extracting baseline model parameters...")
        # Get model parameters as baseline
        with torch.no_grad():
            baseline_params = []
            for param in self.model.parameters():
                baseline_params.append(param.data.flatten())
            baseline_params = torch.cat(baseline_params)
        
        print(f"[ES] Baseline has {baseline_params.numel():,} parameters")
        print(f"[ES] Creating population of {self.population_size} individuals...")
        
        # Create population as perturbations from baseline
        self.population = []
        for i in range(self.population_size):
            if i % 16 == 0:  # Progress every 16 individuals
                print(f"[ES] Creating individual {i+1}/{self.population_size}")
            # Random perturbation
            perturbation = torch.randn_like(baseline_params) * self.sigma
            individual = baseline_params + perturbation
            self.population.append(individual)
            
        print(f"[ES] ✓ Population initialized with {self.population_size} individuals")
    
    def _evaluate_es_population(self) -> torch.Tensor:
        """Evaluate ES population using parallel rover simulation"""
        print(f"[ES] Stacking population tensor...")
        population_tensor = torch.stack(self.population)  # [population_size, param_dim]
        print(f"[ES] Population tensor shape: {population_tensor.shape}")
        
        print(f"[ES] Starting PARALLEL evaluation of {self.population_size} individuals...")
        # Use parallel evaluation for much faster processing
        fitness_scores = self.parallel_evaluator.evaluate_population_parallel(population_tensor)
        print(f"[ES] ✓ Parallel evaluation complete - fitness range: {fitness_scores.min().item():.3f} to {fitness_scores.max().item():.3f}")
        
        return fitness_scores
    
    def _evolve_es_population(self, fitness_scores: torch.Tensor):
        """Evolve ES population (matching rover's evolution strategy)"""
        # Update elites
        self._update_es_elites(fitness_scores)
        
        # Adapt sigma based on progress
        self._adapt_es_sigma(fitness_scores.max().item())
        
        # Generate new population using rank-based selection
        fitness_ranks = torch.argsort(torch.argsort(-fitness_scores))  # Higher rank = better fitness
        weights = torch.maximum(torch.zeros_like(fitness_ranks.float()), 
                               torch.log(torch.tensor(self.population_size/2.0 + 1)) - torch.log(fitness_ranks.float() + 1))
        weights = weights / weights.sum()
        
        # Generate new population
        new_population = []
        
        # Keep elites
        n_elites_kept = 0
        if self.elite_individuals:
            for elite_params, _ in self.elite_individuals[:self.n_elites]:
                if n_elites_kept < self.n_elites:
                    new_population.append(elite_params.clone())
                    n_elites_kept += 1
        
        # Generate rest of population
        for i in range(n_elites_kept, self.population_size):
            # Weighted selection of parent
            parent_idx = torch.multinomial(weights, 1).item()
            parent = self.population[parent_idx].clone()
            
            # Add noise
            noise = torch.randn_like(parent) * self.sigma
            child = parent + noise
            new_population.append(child)
        
        self.population = new_population
    
    def _update_es_elites(self, fitness_scores: torch.Tensor):
        """Update elite individuals (matching rover's elite preservation)"""
        # Combine current population with fitness scores
        population_fitness_pairs = list(zip(self.population, fitness_scores))
        
        # Add previous elites
        if self.elite_individuals:
            population_fitness_pairs.extend(self.elite_individuals)
        
        # Sort by fitness (descending)
        population_fitness_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Select top performers as new elites
        self.elite_individuals = population_fitness_pairs[:self.n_elites]
        self.elite_fitness_scores = [fitness.item() for _, fitness in self.elite_individuals]
    
    def _adapt_es_sigma(self, current_best_fitness: float):
        """Adapt sigma based on fitness improvement (matching rover)"""
        if len(self.fitness_history) < 10:
            return
            
        # Check improvement over last 5 generations
        recent_avg = np.mean(self.fitness_history[-5:])
        earlier_avg = np.mean(self.fitness_history[-10:-5])
        
        improvement = recent_avg - earlier_avg
        
        if improvement >= 0.01:  # Good improvement
            self.sigma *= self.sigma_decay_rate
            self.stagnation_counter = 0
        else:  # Stagnation
            self.stagnation_counter += 1
            if self.stagnation_counter >= 3:
                self.sigma *= self.sigma_increase_rate
                self.stagnation_counter = 0
        
        # Clamp sigma to bounds
        self.sigma = np.clip(self.sigma, self.min_sigma, self.max_sigma)
        
    def save_results(self, results: Dict, filename: str = "training_results.json"):
        """Save training results"""
        save_path = self.save_dir / filename
        
        # Add best parameters to results
        if self.best_params is not None:
            results["best_parameters"] = self.best_params.cpu().numpy().tolist()
            
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"💾 Results saved to {save_path}")
    
    def apply_parameters_to_model(self, parameters: torch.Tensor):
        """Apply parameter tensor to model weights"""
        param_idx = 0
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param_size = param.numel()
                param_slice = parameters[param_idx:param_idx + param_size]
                param.data.copy_(param_slice.view(param.shape))
                param_idx += param_size
        
    def save_model(self, filename: str = "best_model.pth"):
        """Save the trained model with best parameters"""
        if self.best_params is None:
            self.logger.warning("No best parameters found - cannot save model")
            return
            
        try:
            # Apply best parameters to model
            self.apply_parameters_to_model(self.best_params)
            
            # Create model checkpoint
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'parameters': self.best_params.cpu(),
                'fitness': self.best_fitness,
                'architecture': {
                    'depth_dim': 64,
                    'proprio_dim': 8,
                    'hidden_dim': 128,
                    'action_dim': 2
                },
                'optimization_type': self.optimization_type,
                'training_history': self.training_history,
                'save_timestamp': datetime.now().isoformat()
            }
            
            save_path = self.save_dir / filename
            torch.save(checkpoint, save_path)
            
            print(f"💾 Model saved to {save_path}")
            self.logger.info(f"Model saved with fitness {self.best_fitness:.4f}")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            
    def save_convergence_plot(self, results: Dict, filename: str = "convergence_plot.png"):
        """Save convergence plot"""
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib not available - skipping convergence plot")
            return
            
        try:
            plt.figure(figsize=(10, 6))
            
            if 'convergence_history' in results:
                history = results['convergence_history']
                # Flatten nested lists if needed
                if isinstance(history[0], list):
                    history = [item[0] if isinstance(item, list) else item for item in history]
                
                plt.plot(history, 'b-', linewidth=2, label='Fitness')
                plt.xlabel('Evaluation')
                plt.ylabel('Fitness Score')
                plt.title('Training Convergence - {}'.format(results.get("optimization_type", "Unknown")))
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # Add best fitness line
                best_fitness = max(history)
                plt.axhline(y=best_fitness, color='r', linestyle='--', alpha=0.7, 
                           label='Best: {:.4f}'.format(best_fitness))
                plt.legend()
                
            save_path = self.save_dir / filename
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Convergence plot saved to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save convergence plot: {e}")
        
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
        # Use ES with Bayesian hyperparameter optimization (rover-compatible)
        results = trainer.train_ultra_fast_es(n_generations=args.iterations)
        trainer.save_results(results, "rover_es_bayesian_results.json")
        trainer.save_model("rover_es_bayesian_best_model.pth")
        trainer.save_convergence_plot(results, "rover_es_bayesian_convergence.png")
        
    elif args.method == "es":
        # Use pure ES (rover-compatible)
        results = trainer.train_ultra_fast_es(n_generations=args.iterations)
        trainer.save_results(results, "rover_es_results.json")
        trainer.save_model("rover_es_best_model.pth")
        trainer.save_convergence_plot(results, "rover_es_convergence.png")
        
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
        
        # Save the better model
        if results_bo['best_fitness'] > results_es['best_fitness']:
            trainer_bo.save_model("comparison_best_model.pth")
        else:
            trainer_es.save_model("comparison_best_model.pth")


if __name__ == "__main__":
    main()