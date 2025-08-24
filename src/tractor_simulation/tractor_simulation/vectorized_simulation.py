#!/usr/bin/env python3
"""
Vectorized Simulation for Ultra-Fast Batch Evaluation
Replaces PyBullet with vectorized physics for 100x+ speedup
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List
import time
import logging

class VectorizedTractorSimulation:
    def __init__(self, batch_size=32, device="cuda", dtype=torch.float32):
        """
        Vectorized simulation that can run many episodes in parallel
        
        Args:
            batch_size: Number of parallel simulations
            device: "cuda" or "cpu"  
            dtype: Precision (float32 recommended for speed)
        """
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        
        # Environment parameters
        self.dt = 0.02  # 50 Hz simulation
        self.max_steps = 100  # Further reduced for speed
        
        # Robot parameters
        self.max_linear_vel = 0.25  # m/s
        self.max_angular_vel = 2.0  # rad/s
        self.robot_radius = 0.15  # m
        
        # Initialize batch state
        self.reset_batch()
        self.setup_environment()
        
    def reset_batch(self):
        """Reset all simulations in the batch"""
        # Positions: [batch_size, 2] (x, y)
        self.positions = torch.zeros(self.batch_size, 2, device=self.device, dtype=self.dtype)
        
        # Orientations: [batch_size] (yaw angle)
        self.orientations = torch.zeros(self.batch_size, device=self.device, dtype=self.dtype)
        
        # Velocities: [batch_size, 2] (linear, angular)
        self.velocities = torch.zeros(self.batch_size, 2, device=self.device, dtype=self.dtype)
        
        # Episode info
        self.step_count = torch.zeros(self.batch_size, device=self.device, dtype=torch.long)
        self.episode_rewards = torch.zeros(self.batch_size, device=self.device, dtype=self.dtype)
        self.collision_flags = torch.zeros(self.batch_size, device=self.device, dtype=torch.bool)
        self.done_flags = torch.zeros(self.batch_size, device=self.device, dtype=torch.bool)
        
        # History for reward calculation
        self.prev_positions = self.positions.clone()
        self.visited_cells = torch.zeros(self.batch_size, 20, 20, device=self.device, dtype=torch.bool)  # 20x20 grid
        
    def setup_environment(self):
        """Create vectorized environment (obstacles, walls, etc.)"""
        # Room bounds
        self.room_size = 4.0  # 4x4 meter room
        self.wall_bounds = torch.tensor([
            [-self.room_size/2, -self.room_size/2],  # min x, min y
            [self.room_size/2, self.room_size/2]     # max x, max y
        ], device=self.device, dtype=self.dtype)
        
        # Simple obstacle setup - few rectangles
        self.obstacles = torch.tensor([
            # [x_min, y_min, x_max, y_max] for each obstacle
            [-0.5, -0.5, 0.5, -0.3],   # Horizontal bar
            [0.8, -1.0, 1.0, 1.0],     # Vertical bar
            [-1.2, 0.5, -0.8, 1.2],    # Small box
        ], device=self.device, dtype=self.dtype)
        
    def check_collisions(self) -> torch.Tensor:
        """Check collisions for all robots in batch"""
        batch_collisions = torch.zeros(self.batch_size, device=self.device, dtype=torch.bool)
        
        # Wall collisions
        wall_collision = (
            (self.positions[:, 0] - self.robot_radius < self.wall_bounds[0, 0]) |
            (self.positions[:, 0] + self.robot_radius > self.wall_bounds[1, 0]) |
            (self.positions[:, 1] - self.robot_radius < self.wall_bounds[0, 1]) |
            (self.positions[:, 1] + self.robot_radius > self.wall_bounds[1, 1])
        )
        batch_collisions |= wall_collision
        
        # Obstacle collisions (vectorized)
        for obs in self.obstacles:
            x_min, y_min, x_max, y_max = obs
            
            # Check if robot center is close enough to obstacle
            close_x = (self.positions[:, 0] + self.robot_radius > x_min) & (self.positions[:, 0] - self.robot_radius < x_max)
            close_y = (self.positions[:, 1] + self.robot_radius > y_min) & (self.positions[:, 1] - self.robot_radius < y_max)
            
            obstacle_collision = close_x & close_y
            batch_collisions |= obstacle_collision
            
        return batch_collisions
        
    def generate_depth_images(self) -> torch.Tensor:
        """Generate simplified depth images for all robots"""
        # Simplified: just return distance to nearest obstacle in 64 directions
        n_rays = 64
        angles = torch.linspace(0, 2*torch.pi, n_rays, device=self.device, dtype=self.dtype)
        
        # Create rays for all robots: [batch_size, n_rays, 2]
        cos_angles = torch.cos(angles).unsqueeze(0).expand(self.batch_size, -1)  # [batch_size, n_rays]
        sin_angles = torch.sin(angles).unsqueeze(0).expand(self.batch_size, -1)
        
        # Adjust for robot orientation
        robot_cos = torch.cos(self.orientations).unsqueeze(1)  # [batch_size, 1]
        robot_sin = torch.sin(self.orientations).unsqueeze(1)
        
        # Rotate ray directions by robot orientation
        ray_dirs_x = cos_angles * robot_cos - sin_angles * robot_sin
        ray_dirs_y = cos_angles * robot_sin + sin_angles * robot_cos
        
        # Calculate distances (simplified - just check walls)
        max_range = 2.0
        
        # Distance to walls in each direction
        dist_to_walls = torch.full((self.batch_size, n_rays), max_range, device=self.device, dtype=self.dtype)
        
        # X walls
        pos_x_dist = torch.where(
            ray_dirs_x > 0,
            (self.wall_bounds[1, 0] - self.positions[:, 0:1]) / (ray_dirs_x + 1e-6),
            (self.wall_bounds[0, 0] - self.positions[:, 0:1]) / (ray_dirs_x - 1e-6)
        )
        
        # Y walls  
        pos_y_dist = torch.where(
            ray_dirs_y > 0,
            (self.wall_bounds[1, 1] - self.positions[:, 1:2]) / (ray_dirs_y + 1e-6),
            (self.wall_bounds[0, 1] - self.positions[:, 1:2]) / (ray_dirs_y - 1e-6)
        )
        
        # Take minimum valid distance
        valid_x = pos_x_dist > 0
        valid_y = pos_y_dist > 0
        
        dist_to_walls = torch.where(valid_x, torch.minimum(dist_to_walls, pos_x_dist), dist_to_walls)
        dist_to_walls = torch.where(valid_y, torch.minimum(dist_to_walls, pos_y_dist), dist_to_walls)
        
        # Clamp to max range
        depth_images = torch.clamp(dist_to_walls, 0.05, max_range)
        
        return depth_images  # [batch_size, n_rays]
        
    def calculate_rewards(self) -> torch.Tensor:
        """Calculate vectorized rewards for all robots"""
        rewards = torch.zeros(self.batch_size, device=self.device, dtype=self.dtype)
        
        # Movement reward - encourage forward progress
        displacement = self.positions - self.prev_positions
        distance_moved = torch.norm(displacement, dim=1)
        rewards += distance_moved * 20.0  # Increased movement reward
        
        # Exploration reward - visiting new cells
        cell_x = ((self.positions[:, 0] + self.room_size/2) / self.room_size * 20).long().clamp(0, 19)
        cell_y = ((self.positions[:, 1] + self.room_size/2) / self.room_size * 20).long().clamp(0, 19)
        
        # Check if cell is new
        batch_indices = torch.arange(self.batch_size, device=self.device)
        is_new_cell = ~self.visited_cells[batch_indices, cell_x, cell_y]
        
        # Mark cell as visited
        self.visited_cells[batch_indices, cell_x, cell_y] = True
        
        # Reward for new cells
        rewards += is_new_cell.float() * 10.0  # Increased exploration reward
        
        # Survival bonus - reward for staying alive
        rewards += (~self.collision_flags).float() * 2.0
        
        # Collision penalty - only when actually colliding
        collision_penalty = self.collision_flags.float() * -50.0  # Strong penalty
        rewards += collision_penalty
        
        # Small time penalty to encourage efficiency
        rewards -= 0.05  # Reduced time penalty
        
        # Debug: Log rewards occasionally for first robot
        if torch.rand(1).item() < 0.01:  # 1% chance
            logger = logging.getLogger(__name__)
            logger.info(f"Sample rewards - Move: {distance_moved[0]*20:.2f}, "
                       f"Explore: {is_new_cell[0]*10:.2f}, "
                       f"Collision: {collision_penalty[0]:.2f}, "
                       f"Total: {rewards[0]:.2f}")
        
        return rewards
        
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Step all simulations forward
        
        Args:
            actions: [batch_size, 2] - (linear_vel, angular_vel) commands
            
        Returns:
            depth_images: [batch_size, n_rays]
            rewards: [batch_size]
            done_flags: [batch_size]
        """
        # Store previous positions for reward calculation
        self.prev_positions = self.positions.clone()
        
        # Apply actions (with limits)
        linear_vel = torch.clamp(actions[:, 0], -self.max_linear_vel, self.max_linear_vel)
        angular_vel = torch.clamp(actions[:, 1], -self.max_angular_vel, self.max_angular_vel)
        
        # Update orientations
        self.orientations += angular_vel * self.dt
        
        # Update positions (simple integration)
        cos_orient = torch.cos(self.orientations)
        sin_orient = torch.sin(self.orientations)
        
        self.positions[:, 0] += linear_vel * cos_orient * self.dt
        self.positions[:, 1] += linear_vel * sin_orient * self.dt
        
        # Check collisions
        self.collision_flags = self.check_collisions()
        
        # Calculate rewards
        step_rewards = self.calculate_rewards()
        self.episode_rewards += step_rewards
        
        # Update step count
        self.step_count += 1
        
        # Check done conditions
        self.done_flags = (
            (self.step_count >= self.max_steps) |
            self.collision_flags
        )
        
        # Generate depth images
        depth_images = self.generate_depth_images()
        
        return depth_images, step_rewards, self.done_flags
        
    def get_proprioceptive_data(self) -> torch.Tensor:
        """Get proprioceptive sensor data for all robots"""
        # [batch_size, feature_dim]
        proprioceptive = torch.stack([
            self.velocities[:, 0],  # linear velocity
            self.velocities[:, 1],  # angular velocity
            torch.cos(self.orientations),  # orientation cos
            torch.sin(self.orientations),  # orientation sin
            self.positions[:, 0],  # x position (normalized)
            self.positions[:, 1],  # y position (normalized)
            self.step_count.float() / self.max_steps,  # time progress
            self.collision_flags.float(),  # collision flag
        ], dim=1)
        
        return proprioceptive


class BatchSimulationTrainer:
    """Trainer that uses vectorized simulation for ultra-fast evaluation"""
    
    def __init__(self, model, batch_size=32, device="cuda"):
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.sim = VectorizedTractorSimulation(batch_size=batch_size, device=device)
        
        # Pre-compute parameter mapping for efficiency
        self.param_mapping = self._compute_parameter_mapping()
        self.logger = logging.getLogger(__name__)
        
    def _compute_parameter_mapping(self):
        """Pre-compute how to map flat parameters to model layers"""
        param_info = []
        total_params = 0
        
        for name, param in self.model.named_parameters():
            param_info.append({
                'name': name,
                'shape': param.shape,
                'size': param.numel(),
                'start_idx': total_params,
                'end_idx': total_params + param.numel()
            })
            total_params += param.numel()
            
        return param_info, total_params
        
    def evaluate_parameters_batch(self, parameter_batch: torch.Tensor) -> torch.Tensor:
        """
        Evaluate multiple parameter sets in parallel
        
        Args:
            parameter_batch: [num_params, param_dim] - batch of parameters to evaluate
            
        Returns:
            fitness_scores: [num_params] - fitness for each parameter set
        """
        num_params = parameter_batch.shape[0]
        all_fitness = []
        
        # Process in smaller batches for better memory management
        eval_batch_size = min(self.batch_size, 16)  # Reduced for parameter injection overhead
        
        for i in range(0, num_params, eval_batch_size):
            end_idx = min(i + eval_batch_size, num_params)
            current_batch_size = end_idx - i
            
            # Set up simulation for this batch size
            if current_batch_size != self.sim.batch_size:
                self.sim = VectorizedTractorSimulation(batch_size=current_batch_size, device=self.device)
                
            # Reset simulation
            self.sim.reset_batch()
            
            batch_fitness = torch.zeros(current_batch_size, device=self.device)
            
            # Run episodes with early stopping
            consecutive_low_rewards = torch.zeros(current_batch_size, device=self.device)
            
            for step in range(self.sim.max_steps):
                # Get observations
                depth_images = self.sim.generate_depth_images()  # [batch_size, 64]
                proprioceptive = self.sim.get_proprioceptive_data()  # [batch_size, 8]
                
                # Get actions from models (this is where parameters matter)
                with torch.no_grad():
                    actions = self.model_forward_batch(
                        parameter_batch[i:end_idx], 
                        depth_images, 
                        proprioceptive
                    )
                
                # Step simulation
                _, rewards, done_flags = self.sim.step(actions)
                
                # Accumulate fitness
                batch_fitness += rewards
                
                # Early stopping for poor performers
                consecutive_low_rewards = torch.where(
                    rewards < -5.0,  # Poor reward threshold
                    consecutive_low_rewards + 1,
                    torch.zeros_like(consecutive_low_rewards)
                )
                
                early_stop = (consecutive_low_rewards > 10) | done_flags
                
                # Early termination if all episodes are done or performing poorly
                if early_stop.all():
                    break
                    
            all_fitness.append(batch_fitness)
            
        return torch.cat(all_fitness)
        
    def apply_parameters_to_model(self, parameters: torch.Tensor):
        """Apply flat parameter vector to model weights"""
        param_info, total_expected = self.param_mapping
        
        if len(parameters) < total_expected:
            # Pad parameters if too short
            padded = torch.zeros(total_expected, device=parameters.device, dtype=parameters.dtype)
            padded[:len(parameters)] = parameters
            parameters = padded
        elif len(parameters) > total_expected:
            # Truncate if too long
            parameters = parameters[:total_expected]
            
        # Apply parameters to model
        with torch.no_grad():
            for info in param_info:
                start_idx, end_idx = info['start_idx'], info['end_idx']
                param_slice = parameters[start_idx:end_idx].reshape(info['shape'])
                
                # Find the corresponding model parameter
                for name, param in self.model.named_parameters():
                    if name == info['name']:
                        param.copy_(param_slice)
                        break
                        
    def model_forward_batch(self, parameters: torch.Tensor, depth: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with different parameters for each batch item
        
        For efficiency, we'll evaluate one parameter set at a time
        and use the model directly rather than parameter injection
        """
        batch_size = depth.shape[0]
        actions = torch.zeros(batch_size, 2, device=self.device)
        
        try:
            # Store original model state
            original_state = {name: param.clone() for name, param in self.model.named_parameters()}
            
            for i in range(batch_size):
                # Apply parameters for this batch item
                self.apply_parameters_to_model(parameters[i])
                
                # Forward pass for single item
                with torch.no_grad():
                    single_depth = depth[i:i+1]  # Keep batch dimension
                    single_proprio = proprio[i:i+1]
                    action, _ = self.model(single_depth, single_proprio)
                    actions[i] = action.squeeze(0)
                    
            # Restore original model state
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    param.copy_(original_state[name])
                    
        except Exception as e:
            self.logger.error(f"Model forward pass failed: {e}")
            # Fallback to simple parameter-based actions
            for i in range(batch_size):
                # Use first few parameters as coefficients
                depth_coeff = parameters[i][:64] if len(parameters[i]) >= 64 else torch.zeros(64, device=self.device)
                proprio_coeff = parameters[i][64:72] if len(parameters[i]) >= 72 else torch.zeros(8, device=self.device)
                
                # Simple linear combination
                depth_response = torch.sum(depth[i] * depth_coeff[:depth.shape[1]])
                proprio_response = torch.sum(proprio[i] * proprio_coeff[:proprio.shape[1]])
                
                actions[i, 0] = torch.tanh(depth_response * 0.1)  # linear velocity
                actions[i, 1] = torch.tanh(proprio_response * 0.1)  # angular velocity
                
        return actions