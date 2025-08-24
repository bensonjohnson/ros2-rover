#!/usr/bin/env python3
"""
Rover-Compatible Vectorized Simulation
Matches your rover's training environment and reward system
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List
import time
import logging
import cv2


class RoverVectorizedSimulation:
    """
    Vectorized simulation matching your rover's training environment
    - Depth images: 160x288 resolution
    - Proprioceptive: 16 features (3 base + 13 extras)
    - Reward system: Matches your ES training
    """
    
    def __init__(self, batch_size=16, device="cuda", dtype=torch.float32):
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        
        # Rover-specific parameters
        self.depth_h = 160
        self.depth_w = 288
        self.clip_max_distance = 4.0  # Matches your rover
        self.dt = 0.02  # 50 Hz simulation
        self.max_steps = 150  # Reduced for faster evolution
        
        # Robot parameters (matching your rover)
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
        self.visited_cells = torch.zeros(self.batch_size, 30, 30, device=self.device, dtype=torch.bool)  # 30x30 grid
        
        # Rover-specific state
        self.wheel_velocities = torch.zeros(self.batch_size, 2, device=self.device, dtype=self.dtype)  # left, right
        
    def setup_environment(self):
        """Create vectorized environment matching rover conditions"""
        # Room bounds (larger for rover exploration)
        self.room_size = 6.0  # 6x6 meter room
        self.wall_bounds = torch.tensor([
            [-self.room_size/2, -self.room_size/2],  # min x, min y
            [self.room_size/2, self.room_size/2]     # max x, max y
        ], device=self.device, dtype=self.dtype)
        
        # Obstacles matching rover environment
        self.obstacles = torch.tensor([
            # [x_min, y_min, x_max, y_max] for each obstacle
            [-1.0, -1.0, 1.0, -0.7],   # Horizontal barrier
            [1.5, -2.0, 1.8, 2.0],     # Vertical wall
            [-2.0, 1.0, -1.5, 1.5],    # Small obstacle
            [0.5, 0.5, 1.0, 1.0],      # Box obstacle
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
        
    def generate_rover_depth_images(self) -> torch.Tensor:
        """
        Generate depth images matching rover format
        Returns: [batch_size, 160, 288] depth images
        """
        # Simplified depth generation - raycast from robot position
        depth_images = torch.full((self.batch_size, self.depth_h, self.depth_w), 
                                 self.clip_max_distance, device=self.device, dtype=self.dtype)
        
        # Generate rays for each pixel in the depth image
        # Assume 60-degree horizontal FOV, 45-degree vertical FOV
        h_fov = np.pi / 3  # 60 degrees
        v_fov = np.pi / 4  # 45 degrees
        
        # Create angle arrays for each pixel
        h_angles = torch.linspace(-h_fov/2, h_fov/2, self.depth_w, device=self.device)
        v_angles = torch.linspace(-v_fov/2, v_fov/2, self.depth_h, device=self.device)
        
        # For each robot, calculate distances to walls/obstacles
        for batch_idx in range(self.batch_size):
            robot_pos = self.positions[batch_idx]
            robot_orient = self.orientations[batch_idx]
            
            # Generate depth for center horizontal line (simplified)
            for w_idx, h_angle in enumerate(h_angles):
                # Ray direction in world coordinates
                world_angle = robot_orient + h_angle
                ray_dir = torch.tensor([torch.cos(world_angle), torch.sin(world_angle)], 
                                     device=self.device, dtype=self.dtype)
                
                # Find intersection with walls
                distances = []
                
                # Check wall intersections
                if ray_dir[0] != 0:
                    # X walls
                    for wall_x in [self.wall_bounds[0, 0], self.wall_bounds[1, 0]]:
                        t = (wall_x - robot_pos[0]) / ray_dir[0]
                        if t > 0:
                            y_intersect = robot_pos[1] + t * ray_dir[1]
                            if self.wall_bounds[0, 1] <= y_intersect <= self.wall_bounds[1, 1]:
                                distances.append(t)
                
                if ray_dir[1] != 0:
                    # Y walls  
                    for wall_y in [self.wall_bounds[0, 1], self.wall_bounds[1, 1]]:
                        t = (wall_y - robot_pos[1]) / ray_dir[1]
                        if t > 0:
                            x_intersect = robot_pos[0] + t * ray_dir[0]
                            if self.wall_bounds[0, 0] <= x_intersect <= self.wall_bounds[1, 0]:
                                distances.append(t)
                
                # Find minimum valid distance
                if distances:
                    min_dist = min(distances)
                    min_dist = min(min_dist, self.clip_max_distance)
                else:
                    min_dist = self.clip_max_distance
                
                # Fill vertical column with this distance (simplified)
                center_h = self.depth_h // 2
                depth_images[batch_idx, center_h-10:center_h+10, w_idx] = min_dist
        
        return depth_images
        
    def get_rover_proprioceptive_data(self) -> torch.Tensor:
        """
        Get 16-feature proprioceptive data matching your rover
        [batch_size, 16] - same as your rover's sensor setup
        """
        # Base 3 features
        base_features = torch.stack([
            self.velocities[:, 0],  # linear velocity
            self.velocities[:, 1],  # angular velocity  
            torch.cos(self.orientations),  # orientation cos
        ], dim=1)
        
        # 13 extra features (matching your rover's extra_proprio=13)
        extra_features = torch.stack([
            torch.sin(self.orientations),  # orientation sin
            self.positions[:, 0] / self.room_size,  # normalized x position
            self.positions[:, 1] / self.room_size,  # normalized y position
            self.step_count.float() / self.max_steps,  # time progress
            self.collision_flags.float(),  # collision flag
            self.wheel_velocities[:, 0],  # left wheel velocity
            self.wheel_velocities[:, 1],  # right wheel velocity
            torch.norm(self.positions - self.prev_positions, dim=1),  # movement magnitude
            (self.visited_cells.sum(dim=(1,2)).float() / (30*30)),  # exploration progress
            torch.sin(self.step_count.float() * 0.1),  # time-based signal
            torch.cos(self.step_count.float() * 0.05), # slower time signal
            torch.randn(self.batch_size, device=self.device) * 0.1,  # noise/uncertainty
            torch.zeros(self.batch_size, device=self.device),  # reserved feature
        ], dim=1)
        
        # Combine to 16 features total
        return torch.cat([base_features, extra_features], dim=1)
        
    def calculate_rover_rewards(self) -> torch.Tensor:
        """Calculate rewards matching your rover's ES reward system"""
        rewards = torch.zeros(self.batch_size, device=self.device, dtype=self.dtype)
        
        # Movement reward - encourage forward progress (matching ES trainer)
        displacement = self.positions - self.prev_positions
        distance_moved = torch.norm(displacement, dim=1)
        progress_reward = distance_moved * 10.0  # Same as ES trainer
        rewards += progress_reward
        
        # Exploration reward - visiting new cells
        cell_x = ((self.positions[:, 0] + self.room_size/2) / self.room_size * 30).long().clamp(0, 29)
        cell_y = ((self.positions[:, 1] + self.room_size/2) / self.room_size * 30).long().clamp(0, 29)
        
        # Check if cell is new
        batch_indices = torch.arange(self.batch_size, device=self.device)
        is_new_cell = ~self.visited_cells[batch_indices, cell_x, cell_y]
        
        # Mark cell as visited
        self.visited_cells[batch_indices, cell_x, cell_y] = True
        
        # Exploration bonus (matching ES trainer)
        exploration_reward = is_new_cell.float() * 5.0
        rewards += exploration_reward
        
        # Speed efficiency reward (matching ES trainer logic)
        speed = torch.abs(self.velocities[:, 0])
        speed_mask = (speed > 0.05) & (speed < 0.3)
        speed_reward = speed_mask.float() * 2.0
        rewards += speed_reward
        
        # Penalize being too slow
        too_slow_mask = speed < 0.02
        slow_penalty = too_slow_mask.float() * -5.0
        rewards += slow_penalty
        
        # Speed magnitude bonus (encourage meaningful movement)
        meaningful_movement = speed > 0.03
        speed_factor = torch.clamp(speed, 0, 0.25)
        speed_bonus = meaningful_movement.float() * 15.0 * (speed_factor ** 1.5)
        rewards += speed_bonus
        
        # Smooth control reward (avoid jerky movements)
        if hasattr(self, 'prev_actions'):
            action_diff = torch.norm(self.velocities - self.prev_actions, dim=1)
            smoothness_reward = -action_diff * 2.0
            rewards += smoothness_reward
            
        # Store actions for next smoothness calculation
        self.prev_actions = self.velocities.clone()
        
        # Collision penalty (strong penalty like ES trainer)
        collision_penalty = self.collision_flags.float() * -50.0
        rewards += collision_penalty
        
        # Survival bonus
        survival_bonus = (~self.collision_flags).float() * 1.0
        rewards += survival_bonus
        
        # Small time penalty to encourage efficiency
        rewards -= 0.1
        
        return rewards
        
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Step all simulations forward
        
        Args:
            actions: [batch_size, 2] - (linear_vel, angular_vel) commands
            
        Returns:
            depth_images: [batch_size, 160, 288]  
            proprioceptive: [batch_size, 16]
            rewards: [batch_size]
            done_flags: [batch_size]
        """
        # Store previous positions for reward calculation
        self.prev_positions = self.positions.clone()
        
        # Apply actions (with limits matching rover)
        linear_vel = torch.clamp(actions[:, 0], -self.max_linear_vel, self.max_linear_vel)
        angular_vel = torch.clamp(actions[:, 1], -self.max_angular_vel, self.max_angular_vel)
        
        # Store velocities
        self.velocities = torch.stack([linear_vel, angular_vel], dim=1)
        
        # Simulate differential drive (approximate wheel velocities)
        wheel_base = 0.3  # meters
        self.wheel_velocities[:, 0] = linear_vel - angular_vel * wheel_base / 2  # left wheel
        self.wheel_velocities[:, 1] = linear_vel + angular_vel * wheel_base / 2  # right wheel
        
        # Update orientations
        self.orientations += angular_vel * self.dt
        
        # Update positions (simple integration)
        cos_orient = torch.cos(self.orientations)
        sin_orient = torch.sin(self.orientations)
        
        self.positions[:, 0] += linear_vel * cos_orient * self.dt
        self.positions[:, 1] += linear_vel * sin_orient * self.dt
        
        # Check collisions
        self.collision_flags = self.check_collisions()
        
        # Calculate rewards (matching rover ES system)
        step_rewards = self.calculate_rover_rewards()
        self.episode_rewards += step_rewards
        
        # Update step count
        self.step_count += 1
        
        # Check done conditions
        self.done_flags = (
            (self.step_count >= self.max_steps) |
            self.collision_flags
        )
        
        # Generate rover-format depth images and proprioceptive data
        depth_images = self.generate_rover_depth_images()
        proprioceptive = self.get_rover_proprioceptive_data()
        
        return depth_images, proprioceptive, step_rewards, self.done_flags


class RoverBatchSimulationTrainer:
    """Trainer that uses rover-compatible vectorized simulation"""
    
    def __init__(self, model, batch_size=16, device="cuda"):
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.sim = RoverVectorizedSimulation(batch_size=batch_size, device=device)
        
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
                        
    def evaluate_parameters_batch(self, parameter_batch: torch.Tensor) -> torch.Tensor:
        """
        Evaluate multiple parameter sets in parallel using rover simulation
        
        Args:
            parameter_batch: [num_params, param_dim] - batch of parameters to evaluate
            
        Returns:
            fitness_scores: [num_params] - fitness for each parameter set
        """
        num_params = parameter_batch.shape[0]
        print(f"[SIM] Evaluating {num_params} individuals...")
        all_fitness = []
        
        # Process in smaller batches for memory management
        eval_batch_size = min(self.batch_size, 8)  # Smaller for rover-sized images
        print(f"[SIM] Using evaluation batch size: {eval_batch_size}")
        
        for i in range(0, num_params, eval_batch_size):
            end_idx = min(i + eval_batch_size, num_params)
            current_batch_size = end_idx - i
            print(f"[SIM] Processing batch {i//eval_batch_size + 1}/{(num_params-1)//eval_batch_size + 1} ({current_batch_size} individuals)")
            
            # Set up simulation for this batch size
            if current_batch_size != self.sim.batch_size:
                print(f"[SIM] Recreating simulation with batch size {current_batch_size}")
                self.sim = RoverVectorizedSimulation(batch_size=current_batch_size, device=self.device)
                
            # Reset simulation
            print(f"[SIM] Resetting simulation...")
            self.sim.reset_batch()
            
            batch_fitness = torch.zeros(current_batch_size, device=self.device)
            
            # Store original model state
            original_state = {name: param.clone() for name, param in self.model.named_parameters()}
            
            # Run episodes with early stopping
            print(f"[SIM] Running {self.sim.max_steps} simulation steps...")
            for step in range(self.sim.max_steps):
                if step % 50 == 0:  # Progress every 50 steps
                    print(f"[SIM]   Step {step}/{self.sim.max_steps}")
                # Apply parameters for each individual in batch (simplified approach)
                for j in range(current_batch_size):
                    self.apply_parameters_to_model(parameter_batch[i + j])
                    
                    # Get single robot's observations
                    depth_img = self.sim.generate_rover_depth_images()[j:j+1]  # [1, 160, 288]
                    proprio = self.sim.get_rover_proprioceptive_data()[j:j+1]   # [1, 16]
                    
                    # Add channel dimension for depth
                    depth_tensor = depth_img.unsqueeze(1)  # [1, 1, 160, 288]
                    
                    # Get action from model
                    with torch.no_grad():
                        output = self.model(depth_tensor, proprio)
                        action = torch.tanh(output[0, :2])  # [2] 
                        
                    # Store action for this robot
                    if j == 0:
                        actions = action.unsqueeze(0)
                    else:
                        actions = torch.cat([actions, action.unsqueeze(0)], dim=0)
                
                # Step simulation with all actions
                depth_images, proprioceptive, rewards, done_flags = self.sim.step(actions)
                
                # Accumulate fitness
                batch_fitness += rewards
                
                # Early termination if all episodes are done
                if done_flags.all():
                    break
                    
            # Restore original model state
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    param.copy_(original_state[name])
                    
            all_fitness.append(batch_fitness)
            
        return torch.cat(all_fitness)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    if args.benchmark:
        # Test the rover simulation
        sim = RoverVectorizedSimulation(batch_size=args.batch_size, device=args.device)
        
        # Run a few simulation steps
        for step in range(10):
            # Random actions
            actions = torch.randn(args.batch_size, 2, device=args.device) * 0.1
            
            # Step simulation
            start_time = time.time()
            depth_imgs, proprio, rewards, done = sim.step(actions)
            step_time = time.time() - start_time
            
            logger.info(f"Step {step}: {step_time*1000:.2f}ms, "
                       f"depth shape: {depth_imgs.shape}, "
                       f"proprio shape: {proprio.shape}, "
                       f"avg reward: {rewards.mean():.3f}")
                       
        logger.info("Rover simulation benchmark complete!")
    else:
        # Just create and show info
        sim = RoverVectorizedSimulation(batch_size=4)
        logger.info("Rover-compatible vectorized simulation created")
        logger.info(f"Depth images: [batch, 160, 288]")
        logger.info(f"Proprioceptive: [batch, 16] features")
        logger.info(f"Matches your rover's training environment")