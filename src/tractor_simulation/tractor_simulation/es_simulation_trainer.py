#!/usr/bin/env python3
"""
ES Training Integration with PyBullet Simulation
Runs Evolutionary Strategy training using realistic simulation
"""

import sys
import os
import numpy as np
import time
import json
import argparse
from typing import Dict, List, Tuple
import torch
import multiprocessing as mp
import concurrent.futures
from functools import partial
import threading
from collections import deque

# Add the workspace to Python path
sys.path.insert(0, '/workspace/src')
sys.path.insert(0, '/workspace/install/tractor_bringup/lib/python3.12/site-packages')
sys.path.insert(0, '/workspace/install/tractor_simulation/lib/python3.12/site-packages')

try:
    from tractor_bringup.es_trainer_depth import EvolutionaryStrategyTrainer
    from tractor_bringup.improved_reward_system import ImprovedRewardCalculator
    from tractor_simulation.tractor_simulation.bullet_simulation import TractorSimulation
    print("✓ Successfully imported ES trainer components")
    print("✓ Successfully imported PyBullet simulation")
except ImportError as e:
    print(f"✗ Failed to import components: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

class ESSimulationTrainer:
    def __init__(self, 
                 population_size: int = 10,
                 sigma: float = 0.1,
                 learning_rate: float = 0.01,
                 max_generations: int = 1000,
                 use_gui: bool = True,
                 environment_type: str = "indoor"):
        """
        Initialize ES training with PyBullet simulation
        
        Args:
            population_size: Number of individuals in ES population
            sigma: Standard deviation for parameter perturbations
            learning_rate: Learning rate for parameter updates
            max_generations: Maximum number of generations to train
            use_gui: Whether to show PyBullet GUI
            environment_type: Type of environment ("indoor", "outdoor", "mixed")
        """
        self.max_generations = max_generations
        self.environment_type = environment_type
        self.use_gui = use_gui
        self.generation = 0
        self.best_fitness = -float('inf')
        self.best_model_params = None
        
        # Initialize ES trainer
        self.es_trainer = EvolutionaryStrategyTrainer(
            model_dir="models/simulation",
            population_size=population_size,
            sigma=sigma,
            learning_rate=learning_rate,
            enable_debug=True
        )
        
        # Initialize simulation
        self.simulation = TractorSimulation(use_gui=use_gui, enable_visualization=True)
        
        # Initialize CPU thread pool for parallel processing
        self.cpu_thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(8, mp.cpu_count()),
            thread_name_prefix="cpu_worker"
        )
        self._thread_local = threading.local()
        
        # Initialize reward calculator
        try:
            self.reward_calculator = ImprovedRewardCalculator()
            print("✓ Using improved reward system")
        except Exception as e:
            self.reward_calculator = None
            print(f"⚠ Using basic reward system: {e}")
        
        # Setup environment
        self._setup_environment()
        
        # Training statistics
        self.generation_stats = []
        self.fitness_history = []
        
        print(f"ES Simulation Trainer initialized:")
        print(f"  Population size: {population_size}")
        print(f"  Sigma: {sigma}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Max generations: {max_generations}")
        print(f"  Environment: {environment_type}")
        print(f"  GUI enabled: {use_gui}")
        
    def _setup_environment(self):
        """Setup simulation environment based on type"""
        if self.environment_type == "indoor":
            self.simulation.add_indoor_environment()
            print("✓ Indoor environment loaded")
        elif self.environment_type == "outdoor":
            self.simulation.add_outdoor_environment()
            print("✓ Outdoor environment loaded")
        elif self.environment_type == "mixed":
            self.simulation.add_indoor_environment()
            self.simulation.add_outdoor_environment()
            print("✓ Mixed indoor/outdoor environment loaded")
        else:
            # Default to indoor
            self.simulation.add_indoor_environment()
            print("✓ Default indoor environment loaded")
            
    def _compute_collision_analysis(self, depth_data):
        """Analyze collision data in parallel"""
        if depth_data is None:
            return False, 1.0
        
        try:
            valid_depths = depth_data[(depth_data > 0.1) & (depth_data < 5.0)]
            if len(valid_depths) > 0:
                min_distance = np.min(valid_depths)
                near_collision = 0.3 < min_distance < 0.5
                return near_collision, min_distance
        except:
            pass
        return False, 1.0
    
    def _calculate_reward(self, 
                         action: np.ndarray,
                         collision: bool,
                         progress: float,
                         exploration_bonus: float,
                         position: np.ndarray = None,
                         depth_data: np.ndarray = None,
                         wheel_velocities: Tuple[float, float] = None) -> float:
        """
        Calculate reward for current action and state
        """
        if self.reward_calculator is not None and position is not None:
            # Use improved reward system with parallel collision analysis
            collision_future = self.cpu_thread_pool.submit(self._compute_collision_analysis, depth_data)
            near_collision, min_distance = collision_future.result()
            
            total_reward, reward_breakdown = self.reward_calculator.calculate_comprehensive_reward(
                action=action,
                position=position,
                collision=collision,
                near_collision=near_collision,
                progress=progress,
                depth_data=depth_data,
                wheel_velocities=wheel_velocities
            )
            
            return total_reward
        else:
            # Fallback to basic reward system
            reward = 0.0
            
            # Progress reward
            reward += progress * 10.0
            
            # Collision penalty
            if collision:
                reward -= 50.0
                
            # Exploration bonus
            reward += exploration_bonus * 5.0
            
            # Smooth control reward
            if len(self.es_trainer.action_history) > 0:
                last_action = self.es_trainer.action_history[-1]
                action_smoothness = -np.linalg.norm(action - last_action) * 2.0
                reward += action_smoothness
                
            # Speed efficiency
            speed = abs(action[0])
            if 0.05 < speed < 0.3:
                reward += 2.0
            elif speed < 0.02:
                reward -= 5.0
                
            if speed > 0.03:
                speed_factor = min(speed, 0.25)
                speed_bonus = 15.0 * (speed_factor ** 1.5)
                reward += speed_bonus
                
            self.es_trainer.action_history.append(action)
            return reward
            
    def _evaluate_individual(self, perturbation: np.ndarray) -> float:
        """
        Evaluate fitness of an individual in the population
        """
        # Save original parameters
        original_params = self.es_trainer._get_flat_params()
        
        # Apply perturbation
        perturbed_params = original_params + perturbation
        self.es_trainer._set_flat_params(perturbed_params)
        
        # Run simulation episode
        fitness = self._run_simulation_episode()
        
        # Restore original parameters
        self.es_trainer._set_flat_params(original_params)
        
        return fitness
        
    def _run_simulation_episode(self, max_steps: int = 200) -> float:
        """
        Run a single episode in simulation and return fitness
        
        Args:
            max_steps: Maximum number of simulation steps
            
        Returns:
            Fitness score for the episode
        """
        # Reset simulation with error handling
        try:
            self.simulation.reset()
        except Exception as e:
            print(f"    Simulation reset failed: {e}")
            return -50.0  # Penalty for failed simulation
        
        total_reward = 0.0
        prev_position = np.array([0.0, 0.0])
        collision_count = 0
        exploration_bonus = 0.0
        visited_positions = set()
        
        # Run episode
        for step in range(max_steps):
            # Get current state
            sim_state = self.simulation.step()
            robot_state = sim_state["robot_state"]
            depth_image = sim_state["depth_image"]
            
            # Check for collisions (simplified)
            collision = self._check_collision(depth_image)
            if collision:
                collision_count += 1
                
            # Calculate progress
            # Safe position extraction
            try:
                pos_data = robot_state.get("position", [0.0, 0.0])
                if isinstance(pos_data, dict):
                    current_position = np.array([pos_data.get("x", 0.0), pos_data.get("y", 0.0)])
                else:
                    current_position = np.array(pos_data[:2]) if len(pos_data) >= 2 else np.array([0.0, 0.0])
            except:
                current_position = np.array([0.0, 0.0])
            progress = np.linalg.norm(current_position - prev_position)
            prev_position = current_position.copy()
            
            # Exploration bonus (visit new areas)
            pos_key = (int(current_position[0] * 10), int(current_position[1] * 10))
            if pos_key not in visited_positions:
                exploration_bonus += 0.1
                visited_positions.add(pos_key)
                
            # Get proprioceptive data
            proprioceptive = self._get_proprioceptive_data(robot_state, depth_image)
            
            # Get action from ES trainer
            try:
                action, confidence = self.es_trainer.inference(depth_image, proprioceptive)
            except Exception as e:
                print(f"⚠ Inference failed: {e}")
                action = np.array([0.0, 0.0])  # Stop action
                confidence = 0.0
                
            # Apply action to simulation
            # Use fixed values for max_speed and angular_scale since they're not in the ES trainer
            max_speed = 0.25  # m/s - reasonable max speed for indoor navigation
            angular_scale = 1.0  # rad/s scaling factor
            self.simulation.set_velocity(action[0] * max_speed, 
                                       action[1] * angular_scale)
            
            # Calculate reward
            reward = self._calculate_reward(
                action=action,
                collision=collision,
                progress=progress,
                exploration_bonus=exploration_bonus,
                position=current_position,
                depth_data=depth_image,
                wheel_velocities=(
                    robot_state.get("linear_velocity", [0.0, 0.0])[0] if len(robot_state.get("linear_velocity", [])) > 0 else 0.0,
                    robot_state.get("linear_velocity", [0.0, 0.0])[1] if len(robot_state.get("linear_velocity", [])) > 1 else 0.0
                )
            )
            
            total_reward += reward
            
            # Add experience to buffer (for training the actual model)
            if step > 10:  # Skip initial steps
                self.es_trainer.add_experience(
                    depth_image=depth_image,
                    proprioceptive=proprioceptive,
                    action=action,
                    reward=reward,
                    done=False,
                    collision=collision
                )
                
            # Small delay for visualization (removed for speed)
            if self.use_gui:
                pass  # No delay for maximum speed
        # Normalize fitness by episode length
        avg_reward = total_reward / max_steps
        
        # Penalty for too many collisions
        collision_penalty = collision_count * 2.0
        
        # Bonus for exploration
        exploration_score = len(visited_positions) * 0.01
        
        final_fitness = avg_reward - collision_penalty + exploration_score
        
        print(f"  Episode completed - Avg Reward: {avg_reward:.2f}, "
              f"Collisions: {collision_count}, "
              f"Exploration: {exploration_score:.2f}, "
              f"Final Fitness: {final_fitness:.2f}")
        
        return final_fitness
        
    def _check_collision(self, depth_image: np.ndarray) -> bool:
        """
        Check if robot is in collision based on depth image
        
        Args:
            depth_image: Depth image from camera
            
        Returns:
            True if collision detected
        """
        if depth_image is None or depth_image.size == 0:
            return False
            
        # Check front center region for close obstacles
        h, w = depth_image.shape
        center_region = depth_image[int(h*0.3):int(h*0.7), int(w*0.4):int(w*0.6)]
        valid_depths = center_region[(center_region > 0.05) & (center_region < 4.0)]
        
        if len(valid_depths) > 0:
            min_distance = np.min(valid_depths)
            return min_distance < 0.2  # Collision if closer than 20cm
            
        return False
        
    def _compute_depth_stats(self, valid_depths):
        """Compute depth statistics in parallel"""
        if len(valid_depths) == 0:
            return 4.0, 2.0, 0.0
        
        min_depth = np.min(valid_depths)
        mean_depth = np.mean(valid_depths)
        near_collision = 1.0 if np.percentile(valid_depths, 5) < 0.25 else 0.0
        return min_depth, mean_depth, near_collision

    def _compute_velocity_stats(self, robot_state):
        """Compute velocity statistics in parallel"""
        def safe_mean(v):
            v_array = np.array(v)
            return float(np.mean(v_array)) if v_array.size > 0 else 0.0
        
        linear_vel = robot_state.get("linear_velocity", [0.0, 0.0, 0.0])
        angular_vel = robot_state.get("angular_velocity", [0.0, 0.0, 0.0])
        
        # Convert to numpy arrays for safe operations
        linear_vel_array = np.array(linear_vel)
        angular_vel_array = np.array(angular_vel)
        
        return (
            safe_mean(linear_vel),
            safe_mean(angular_vel),
            abs(linear_vel_array[0]) if linear_vel_array.size > 0 else 0.0,
            abs(angular_vel_array[2]) if angular_vel_array.size > 2 else 0.0
        )

    def _get_proprioceptive_data(self, robot_state: Dict, depth_image: np.ndarray) -> np.ndarray:
        """
        Extract proprioceptive data from robot state
        
        Args:
            robot_state: Dictionary with robot state information
            depth_image: Depth image from camera
            
        Returns:
            Proprioceptive data array
        """
        # Extract relevant data (with safe indexing)
        try:
            linear_vel_data = robot_state.get("linear_velocity", [0.0, 0.0, 0.0])
            linear_vel = float(linear_vel_data[0]) if isinstance(linear_vel_data, (list, tuple, np.ndarray)) and len(linear_vel_data) > 0 else 0.0
        except:
            linear_vel = 0.0
            
        try:
            angular_vel_data = robot_state.get("angular_velocity", [0.0, 0.0, 0.0])
            angular_vel = float(angular_vel_data[2]) if isinstance(angular_vel_data, (list, tuple, np.ndarray)) and len(angular_vel_data) > 2 else 0.0
        except:
            angular_vel = 0.0
            
        yaw = robot_state.get("yaw", 0.0)
        
        # Calculate depth statistics and velocity stats in parallel (with fallback)
        valid_depths = depth_image[(depth_image > 0.05) & (depth_image < 4.0)]
        
        try:
            # Submit parallel computations
            depth_future = self.cpu_thread_pool.submit(self._compute_depth_stats, valid_depths)
            velocity_future = self.cpu_thread_pool.submit(self._compute_velocity_stats, robot_state)
            
            # Process side bands while waiting
            h, w = depth_image.shape
            left_band = depth_image[int(h*0.3):int(h*0.7), int(w*0.15):int(w*0.35)]
            right_band = depth_image[int(h*0.3):int(h*0.7), int(w*0.65):int(w*0.85)]
            
            # Get results from parallel computations
            min_depth, mean_depth, near_collision = depth_future.result()
            avg_linear_vel, avg_angular_vel, abs_linear_vel, abs_angular_vel = velocity_future.result()
        except Exception as e:
            # Fallback to sequential processing
            h, w = depth_image.shape
            left_band = depth_image[int(h*0.3):int(h*0.7), int(w*0.15):int(w*0.35)]
            right_band = depth_image[int(h*0.3):int(h*0.7), int(w*0.65):int(w*0.85)]
            
            min_depth, mean_depth, near_collision = self._compute_depth_stats(valid_depths)
            avg_linear_vel, avg_angular_vel, abs_linear_vel, abs_angular_vel = self._compute_velocity_stats(robot_state)
        
        # Last action (if available)
        last_action = self.es_trainer.last_action if hasattr(self.es_trainer, 'last_action') else np.array([0.0, 0.0])
        
        # Wheel velocity difference (simplified)
        wheel_diff = 0.0  # Would come from actual wheel encoders
        
        # Emergency stop flag
        emergency_stop = 1.0 if min_depth < 0.18 else 0.0
        center_band = depth_image[int(h*0.3):int(h*0.7), int(w*0.45):int(w*0.55)]
        
        def free_metric(band):
            v = band[(band > 0.05) & (band < 4.0)]
            return float(np.mean(v)) if v.size else 0.0
            
        left_free = free_metric(left_band)
        right_free = free_metric(right_band)
        center_free = free_metric(center_band)
        
        # Create proprioceptive vector (16 elements to match ES trainer)
        proprioceptive = np.array([
            linear_vel,
            angular_vel,
            float(self.generation % 100) / 100.0,  # Generation progress
            last_action[0],
            last_action[1],
            wheel_diff,
            min_depth,
            mean_depth,
            near_collision,
            emergency_stop,
            left_free,
            right_free,
            center_free,
            0.0,  # Placeholder for gradient info
            0.0,  # Placeholder
            0.0   # Placeholder
        ], dtype=np.float32)
        
        return proprioceptive

    def _evaluate_individual_threaded(self, perturbation):
        """
        Evaluate fitness of an individual using threading (shares GPU context)
        """
        # Create a new simulation instance for this thread
        sim = TractorSimulation(use_gui=False, enable_visualization=False)
        
        # Add the same environment
        if self.environment_type == "indoor":
            sim.add_indoor_environment()
        elif self.environment_type == "outdoor":
            sim.add_outdoor_environment()
        elif self.environment_type == "mixed":
            sim.add_indoor_environment()
            sim.add_outdoor_environment()
        else:
            sim.add_indoor_environment()
        
        # Apply perturbation to current model (shared GPU context)
        original_params = self.es_trainer._get_flat_params()
        perturbed_params = original_params + perturbation
        self.es_trainer._set_flat_params(perturbed_params)
        
        # Run simulation episode with the main ES trainer (GPU accelerated)
        fitness = self._run_simulation_episode_threaded(sim)
        
        # Restore original parameters
        self.es_trainer._set_flat_params(original_params)
        
        # Clean up
        sim.close()
        
        return fitness
    
    def _run_simulation_episode_threaded(self, simulation, max_steps: int = 200) -> float:
        """
        Run a single episode in simulation using shared GPU context
        """
        # Reset simulation
        simulation.reset()
        
        total_reward = 0.0
        prev_position = np.array([0.0, 0.0])
        collision_count = 0
        exploration_bonus = 0.0
        visited_positions = set()
        
        # Run episode with timeout protection
        start_time = time.time()
        action = np.array([0.0, 0.0])  # Initialize action
        
        for step in range(max_steps):
            # Check for timeout (prevent infinite loops)
            if time.time() - start_time > 60:  # 1 minute max per episode
                print(f"    Threaded episode timeout at step {step}")
                break
                
            # Get current state
            try:
                sim_state = simulation.step()
                robot_state = sim_state["robot_state"]
                depth_image = sim_state["depth_image"]
            except Exception as e:
                print(f"    Threaded simulation step error: {e}")
                break
            
            # Check for collisions (simplified)
            if robot_state.get("collision", False):
                collision_count += 1
                if collision_count > 3:  # Reduced collision threshold
                    total_reward -= 10.0  # Collision penalty
                    break
            
            # Get action every few steps for efficiency (reduce neural network calls)
            if step % 5 == 0:  # Update action every 5 steps
                try:
                    # Get proprioceptive data
                    proprioceptive = self._get_proprioceptive_data(robot_state, depth_image)
                    # Get action from ES trainer (GPU accelerated!)
                    action, confidence = self.es_trainer.inference(depth_image, proprioceptive)
                except Exception as e:
                    action = np.array([0.0, 0.0])  # Stop action
                    confidence = 0.0
                
            # Apply action to simulation
            max_speed = 0.25  # m/s
            angular_scale = 2.0  # rad/s
            linear_velocity = float(action[0] * max_speed)
            angular_velocity = float(action[1] * angular_scale)
            
            simulation.set_velocity(linear_velocity, angular_velocity)
            
            # Calculate reward
            # Safe position extraction for threaded simulation
            try:
                pos_data = robot_state.get("position", {"x": 0.0, "y": 0.0})
                if isinstance(pos_data, dict):
                    current_position = np.array([pos_data.get("x", 0.0), pos_data.get("y", 0.0)])
                else:
                    current_position = np.array(pos_data[:2]) if len(pos_data) >= 2 else np.array([0.0, 0.0])
            except:
                current_position = np.array([0.0, 0.0])
            distance_moved = np.linalg.norm(current_position - prev_position)
            
            # Distance-based reward
            total_reward += distance_moved * 10.0
            
            # Exploration bonus
            pos_key = (int(current_position[0] * 2), int(current_position[1] * 2))
            if pos_key not in visited_positions:
                visited_positions.add(pos_key)
                exploration_bonus += 1.0
                total_reward += 2.0
            
            prev_position = current_position
            
            # No sleep for maximum speed
            # time.sleep(0.005)  # Removed sleep for speed
            
            # Less frequent progress indicator  
            if step % 200 == 0 and step > 0:
                print(f"      Step {step}/{max_steps}")
        
        return total_reward
        
    def _evaluate_individual_parallel(self, perturbation_data):
        """
        Evaluate fitness of an individual in parallel (used with multiprocessing)
        """
        perturbation, model_state_dict, environment_type = perturbation_data
        
        # Create a new simulation instance for this process
        sim = TractorSimulation(use_gui=False, enable_visualization=False)
        
        # Add the same environment
        if environment_type == "indoor":
            sim.add_indoor_environment()
        elif environment_type == "outdoor":
            sim.add_outdoor_environment()
        elif environment_type == "mixed":
            sim.add_indoor_environment()
            sim.add_outdoor_environment()
        else:
            sim.add_indoor_environment()
        
        # Create a temporary ES trainer for this process
        temp_trainer = EvolutionaryStrategyTrainer(
            model_dir="models/simulation",
            population_size=1,  # We only need one individual
            sigma=self.es_trainer.sigma,
            learning_rate=self.es_trainer.learning_rate,
            enable_debug=False
        )
        
        # Load the model state (convert numpy back to tensors on CPU)
        cpu_model_state_dict = {}
        for k, v in model_state_dict.items():
            if isinstance(v, np.ndarray):
                cpu_model_state_dict[k] = torch.from_numpy(v).cpu()
            else:
                cpu_model_state_dict[k] = v.cpu() if hasattr(v, 'cpu') else v
        
        temp_trainer.model.load_state_dict(cpu_model_state_dict)
        # Force CPU-only processing for multiprocessing workers
        temp_trainer.model.to('cpu')
        temp_trainer.device = torch.device('cpu')
        
        # Apply perturbation
        original_params = temp_trainer._get_flat_params()
        perturbed_params = original_params + perturbation
        temp_trainer._set_flat_params(perturbed_params)
        
        # Run simulation episode
        fitness = self._run_simulation_episode_parallel(sim, temp_trainer)
        
        # Clean up
        sim.close()
        
        return fitness

    def _run_simulation_episode_parallel(self, simulation, es_trainer, max_steps: int = 200) -> float:
        """
        Run a single episode in simulation and return fitness (parallel version)
        
        Args:
            simulation: TractorSimulation instance
            es_trainer: EvolutionaryStrategyTrainer instance
            max_steps: Maximum number of simulation steps
            
        Returns:
            Fitness score for the episode
        """
        # Reset simulation
        simulation.reset()
        
        total_reward = 0.0
        prev_position = np.array([0.0, 0.0])
        collision_count = 0
        exploration_bonus = 0.0
        visited_positions = set()
        
        # Run episode
        for step in range(max_steps):
            # Get current state
            sim_state = simulation.step()
            robot_state = sim_state["robot_state"]
            depth_image = sim_state["depth_image"]
            
            # Check for collisions (simplified)
            collision = self._check_collision(depth_image)
            if collision:
                collision_count += 1
                
            # Calculate progress
            # Safe position extraction
            try:
                pos_data = robot_state.get("position", [0.0, 0.0])
                if isinstance(pos_data, dict):
                    current_position = np.array([pos_data.get("x", 0.0), pos_data.get("y", 0.0)])
                else:
                    current_position = np.array(pos_data[:2]) if len(pos_data) >= 2 else np.array([0.0, 0.0])
            except:
                current_position = np.array([0.0, 0.0])
            progress = np.linalg.norm(current_position - prev_position)
            prev_position = current_position.copy()
            
            # Exploration bonus (visit new areas)
            pos_key = (int(current_position[0] * 10), int(current_position[1] * 10))
            if pos_key not in visited_positions:
                exploration_bonus += 0.1
                visited_positions.add(pos_key)
                
            # Get proprioceptive data
            proprioceptive = self._get_proprioceptive_data(robot_state, depth_image)
            
            # Get action from ES trainer
            try:
                # Force CPU-only inference to avoid GPU context issues in multiprocessing
                with torch.no_grad():
                    es_trainer.model.eval()
                    processed = es_trainer.preprocess_depth_for_model(depth_image)
                    depth_tensor = torch.from_numpy(processed).unsqueeze(0).float().to('cpu')
                    sensor_tensor = torch.from_numpy(proprioceptive).float().unsqueeze(0).to('cpu')
                    output = es_trainer.model(depth_tensor, sensor_tensor)
                    raw_output = output.cpu().numpy()[0]
                    action = np.tanh(raw_output[:2])
                    confidence = 1.0 / (1.0 + np.exp(-raw_output[2]))  # sigmoid
            except Exception as e:
                action = np.array([0.0, 0.0])  # Stop action
                confidence = 0.0
                
            # Apply action to simulation
            # Use fixed values for max_speed and angular_scale since they're not in the ES trainer
            max_speed = 0.25  # m/s - reasonable max speed for indoor navigation
            angular_scale = 1.0  # rad/s scaling factor
            simulation.set_velocity(action[0] * max_speed, 
                                   action[1] * angular_scale)
            
            # Calculate reward
            reward = self._calculate_reward(
                action=action,
                collision=collision,
                progress=progress,
                exploration_bonus=exploration_bonus,
                position=current_position,
                depth_data=depth_image,
                wheel_velocities=(
                    robot_state.get("linear_velocity", [0.0, 0.0])[0] if len(robot_state.get("linear_velocity", [])) > 0 else 0.0,
                    robot_state.get("linear_velocity", [0.0, 0.0])[1] if len(robot_state.get("linear_velocity", [])) > 1 else 0.0
                )
            )
            
            total_reward += reward
            
            # Small delay for visualization (skip in parallel processing)
                
        # Normalize fitness by episode length
        avg_reward = total_reward / max_steps
        
        # Penalty for too many collisions
        collision_penalty = collision_count * 2.0
        
        # Bonus for exploration
        exploration_score = len(visited_positions) * 0.01
        
        final_fitness = avg_reward - collision_penalty + exploration_score
        
        return final_fitness
    
    def _process_simulation_batch(self, batch_data):
        """Process a batch of simulation steps in parallel"""
        sim_states, robot_states, depth_images = batch_data
        
        # Process proprioceptive data for all states in parallel
        proprioceptive_futures = []
        for robot_state, depth_image in zip(robot_states, depth_images):
            future = self.cpu_thread_pool.submit(self._get_proprioceptive_data, robot_state, depth_image)
            proprioceptive_futures.append(future)
        
        # Collect results
        proprioceptive_data = [future.result() for future in proprioceptive_futures]
        
        return proprioceptive_data
    
    def __del__(self):
        """Cleanup method to ensure thread pools are properly closed"""
        try:
            if hasattr(self, 'cpu_thread_pool'):
                self.cpu_thread_pool.shutdown(wait=False)
        except:
            pass

    def train(self):
        """
        Main training loop
        """
        print("Starting ES training with PyBullet simulation...")
        print("=" * 60)
        
        try:
            for generation in range(self.max_generations):
                self.generation = generation
                print(f"\nGeneration {generation + 1}/{self.max_generations}")
                print("-" * 40)
                
                # Evaluate population fitness
                fitness_scores = []
                population_size = len(self.es_trainer.population)
                
                # Use parallel processing for better performance
                if population_size > 1:
                    print(f"  Evaluating population of size {population_size} in parallel...")
                    
                    # Try CPU multiprocessing first (more stable with PyBullet)
                    try:
                        print("  Using CPU multiprocessing (PyBullet-safe)...")
                        
                        # Get current model state for sharing with processes (force CPU and numpy conversion)
                        model_state_dict = {}
                        for k, v in self.es_trainer.model.state_dict().items():
                            # Convert to CPU, detach, and then to numpy to avoid HIP sharing issues
                            cpu_tensor = v.cpu().detach()
                            model_state_dict[k] = cpu_tensor.numpy().copy()  # Convert to numpy for safe sharing
                        
                        # Prepare data for parallel processing
                        perturbation_data = [(pert, model_state_dict, self.environment_type) for pert in self.es_trainer.population]
                        
                        # Use multiprocessing to evaluate individuals in parallel
                        with mp.Pool(processes=min(population_size, 2)) as pool:  # Limited to 2 processes
                            fitness_scores = pool.map(self._evaluate_individual_parallel, perturbation_data)
                        
                        # Convert to numpy arrays
                        fitness_scores = np.array(fitness_scores)
                        print("  ✓ CPU multiprocessing evaluation successful!")
                        
                    except Exception as e:
                        print(f"  ⚠ CPU multiprocessing failed: {e}")
                        print("  Falling back to sequential evaluation...")
                        # Fallback to sequential evaluation
                        fitness_scores = []
                        for i, perturbation in enumerate(self.es_trainer.population):
                            print(f"  Evaluating individual {i+1}/{population_size}...")
                            try:
                                fitness = self._evaluate_individual(perturbation)
                                fitness_scores.append(fitness)
                            except Exception as eval_error:
                                print(f"    Individual {i+1} evaluation failed: {eval_error}")
                                fitness_scores.append(-100.0)  # Penalty for failed evaluation
                        
                        # Convert to numpy arrays
                        fitness_scores = np.array(fitness_scores)
                else:
                    # Fallback to sequential evaluation for single individual
                    for i, perturbation in enumerate(self.es_trainer.population):
                        print(f"  Evaluating individual {i+1}/{population_size}...")
                        fitness = self._evaluate_individual(perturbation)
                        fitness_scores.append(fitness)
                    
                    # Convert to numpy arrays
                    fitness_scores = np.array(fitness_scores)
                
                # Update best model
                best_idx = np.argmax(fitness_scores)
                current_best_fitness = fitness_scores[best_idx]
                
                if current_best_fitness > self.best_fitness:
                    self.best_fitness = current_best_fitness
                    self.best_model_params = self.es_trainer._get_flat_params() + self.es_trainer.population[best_idx]
                    print(f"  🏆 New best fitness: {self.best_fitness:.4f}")
                
                # Update ES trainer with fitness scores
                self.es_trainer.fitness_history.extend(fitness_scores)
                
                # Perform evolution
                print(f"  Evolving population...")
                evolution_stats = self.es_trainer.evolve_population()
                
                # Record statistics
                avg_fitness = float(np.mean(fitness_scores))
                fitness_std = float(np.std(fitness_scores))
                
                generation_stats = {
                    "generation": generation + 1,
                    "avg_fitness": avg_fitness,
                    "best_fitness": float(self.best_fitness),
                    "fitness_std": fitness_std,
                    "samples": self.es_trainer.buffer_size,
                    "sigma": float(self.es_trainer.sigma)
                }
                
                self.generation_stats.append(generation_stats)
                self.fitness_history.append(avg_fitness)
                
                # Print generation summary
                print(f"  Generation Summary:")
                print(f"    Average Fitness: {avg_fitness:.4f}")
                print(f"    Best Fitness: {self.best_fitness:.4f}")
                print(f"    Fitness Std: {fitness_std:.4f}")
                print(f"    Sigma: {self.es_trainer.sigma:.6f}")
                print(f"    Buffer Size: {self.es_trainer.buffer_size}")
                
                # Save model periodically
                if (generation + 1) % 10 == 0:
                    self.es_trainer.save_model()
                    self._save_training_stats()
                    print(f"  💾 Model saved at generation {generation + 1}")
                
                # Check for convergence
                if self._check_convergence():
                    print(f"  🎯 Convergence detected at generation {generation + 1}")
                    break
                    
                # Small delay between generations
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
        except Exception as e:
            print(f"\n\nTraining error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Save final model and cleanup
            self._save_final_model()
            try:
                self.simulation.close()
            except:
                pass  # Ignore close errors
            # Shutdown thread pools
            print("Shutting down CPU thread pool...")
            self.cpu_thread_pool.shutdown(wait=True)
            print("\nTraining completed!")
            
    def _check_convergence(self, window_size: int = 20, threshold: float = 0.01) -> bool:
        """
        Check if training has converged
        
        Args:
            window_size: Number of generations to check for improvement
            threshold: Minimum improvement threshold
            
        Returns:
            True if converged
        """
        if len(self.fitness_history) < window_size:
            return False
            
        recent_fitness = self.fitness_history[-window_size:]
        avg_recent = np.mean(recent_fitness)
        
        if len(self.fitness_history) >= 2 * window_size:
            earlier_fitness = self.fitness_history[-2*window_size:-window_size]
            avg_earlier = np.mean(earlier_fitness)
            
            improvement = avg_recent - avg_earlier
            return improvement < threshold
            
        return False
        
    def _save_training_stats(self):
        """Save training statistics to file"""
        stats_file = os.path.join(self.es_trainer.model_dir, "training_stats.json")
        try:
            with open(stats_file, 'w') as f:
                json.dump({
                    "generation_stats": self.generation_stats,
                    "fitness_history": self.fitness_history,
                    "best_fitness": float(self.best_fitness),
                    "final_generation": self.generation
                }, f, indent=2)
            print(f"  📊 Training stats saved to {stats_file}")
        except Exception as e:
            print(f"  ⚠ Failed to save training stats: {e}")
            
    def _save_final_model(self):
        """Save the final trained model"""
        try:
            # Apply best parameters if available
            if self.best_model_params is not None:
                self.es_trainer._set_flat_params(self.best_model_params)
                print("  ✓ Applied best model parameters")
                
            # Save model
            self.es_trainer.save_model()
            
            # Export to ONNX for RKNN conversion
            self._export_onnx_model()
            
            # Save training stats
            self._save_training_stats()
            
            print("  💾 Final model saved successfully")
        except Exception as e:
            print(f"  ⚠ Failed to save final model: {e}")
            
    def _export_onnx_model(self):
        """Export model to ONNX format for RKNN conversion"""
        try:
            self.es_trainer.convert_to_rknn()
            print("  🔄 Model exported to ONNX/RKNN format")
        except Exception as e:
            print(f"  ⚠ Failed to export model: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="ES Training with PyBullet Simulation")
    parser.add_argument("--population-size", type=int, default=10, help="ES population size")
    parser.add_argument("--sigma", type=float, default=0.1, help="ES sigma parameter")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--max-generations", type=int, default=100, help="Maximum generations")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI")
    parser.add_argument("--environment", choices=["indoor", "outdoor", "mixed"], 
                       default="indoor", help="Environment type")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = ESSimulationTrainer(
        population_size=args.population_size,
        sigma=args.sigma,
        learning_rate=args.learning_rate,
        max_generations=args.max_generations,
        use_gui=not args.no_gui,
        environment_type=args.environment
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
