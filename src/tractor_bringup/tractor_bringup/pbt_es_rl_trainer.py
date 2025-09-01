"""
Population-Based Training (PBT) with ES and RL Hybrid Trainer
"""

import numpy as np
import copy
import random
from collections import deque

# Assuming RKNNTrainerBEV is in a file that can be imported
from .rknn_trainer_bev import RKNNTrainerBEV

class PBT_ES_RL_Trainer:
    def __init__(self, population_size=4, bev_channels=4, **kwargs):
        self.population_size = population_size
        self.bev_channels = bev_channels
        self.kwargs = kwargs
        
        self.population = [self._create_agent() for _ in range(self.population_size)]
        self.active_agent_idx = 0
        
        # PBT parameters
        self.pbt_interval = 1000  # steps before considering a PBT update (increased for stability)
        self.perturb_prob = 0.8
        self.resample_prob = 0.2
        
        # Hyperparameter perturbation factors
        self.hyperparam_perturbations = {
            'learning_rate': [0.8, 1.2],
            'batch_size': [0.75, 1.5],
            'discount_factor': [0.98, 1.0]
        }
        
        print(f"PBT ES-RL Hybrid Trainer initialized with population size: {self.population_size}")

    def _create_agent(self):
        """Creates a new agent (trainer instance)"""
        agent = RKNNTrainerBEV(bev_channels=self.bev_channels, **self.kwargs)
        # You can randomize initial hyperparameters here if you want
        return {
            'trainer': agent,
            'fitness': 0.0,
            'steps': 0,
            'history': deque(maxlen=100) # Store recent rewards
        }

    def get_active_agent(self):
        """Returns the currently active agent's trainer"""
        return self.population[self.active_agent_idx]['trainer']

    @property
    def buffer_size(self):
        """Returns the buffer size of the active agent."""
        return self.get_active_agent().buffer_size

    @property
    def batch_size(self):
        """Returns the batch size of the active agent."""
        return self.get_active_agent().batch_size

    def select_active_agent(self):
        """Selects an agent to be active for the next episode/steps"""
        # Simple round-robin for now
        self.active_agent_idx = (self.active_agent_idx + 1) % self.population_size
        return self.get_active_agent()

    def add_experience(self, **kwargs):
        """Add experience to the active agent's replay buffer"""
        agent_data = self.population[self.active_agent_idx]
        agent_data['trainer'].add_experience(**kwargs)
        
        # Update agent's step count and fitness
        agent_data['steps'] += 1
        reward = kwargs.get('reward', 0)
        agent_data['history'].append(reward)
        
        # Simple fitness: average reward over last 100 steps
        if len(agent_data['history']) > 0:
            agent_data['fitness'] = np.mean(agent_data['history'])

    def train_step(self):
        """Perform a training step for the active agent"""
        agent_data = self.population[self.active_agent_idx]
        
        # Standard RL training step
        stats = agent_data['trainer'].train_step()
        
        # Check if it's time for a PBT update
        if agent_data['steps'] > 0 and agent_data['steps'] % self.pbt_interval == 0:
            self.pbt_update(self.active_agent_idx)
            
        return stats

    def pbt_update(self, agent_idx):
        """Perform a PBT exploit-and-explore step"""
        agent_data = self.population[agent_idx]
        
        # Find a better performing agent
        contenders = [p for i, p in enumerate(self.population) if i != agent_idx]
        if not contenders:
            return
            
        best_contender = max(contenders, key=lambda x: x['fitness'])
        
        if agent_data['fitness'] < best_contender['fitness']:
            print(f"PBT Update for Agent {agent_idx}: Fitness {agent_data['fitness']:.2f} < Best Contender {best_contender['fitness']:.2f}")
            # Exploit: copy weights from the better agent
            agent_data['trainer'].copy_weights_from(best_contender['trainer'])
            
            # Explore: perturb hyperparameters and/or weights
            self._perturb_hyperparameters(agent_data['trainer'])
            self._perturb_weights(agent_data['trainer'])

    def _perturb_hyperparameters(self, trainer):
        """Perturb the hyperparameters of a trainer instance"""
        for param, factors in self.hyperparam_perturbations.items():
            if hasattr(trainer, param):
                current_value = getattr(trainer, param)
                factor = random.choice(factors)
                new_value = current_value * factor
                
                # Handle integer params like batch_size
                if isinstance(current_value, int):
                    new_value = max(1, int(new_value))
                    
                setattr(trainer, param, new_value)
                print(f"  - Perturbed {param}: {current_value} -> {getattr(trainer, param)}")

    def _perturb_weights(self, trainer, noise_std=0.01):
        """ES-style weight perturbation"""
        model = trainer.get_model()
        with trainer.torch.no_grad():
            for param in model.parameters():
                noise = trainer.torch.randn_like(param) * noise_std
                param.add_(noise)
        print(f"  - Perturbed model weights with noise_std={noise_std}")

    def inference(self, bev_image, proprioceptive):
        """Perform inference using the active agent"""
        return self.get_active_agent().inference(bev_image, proprioceptive)

    def calculate_reward(self, **kwargs):
        """Calculate reward using the active agent's reward function"""
        return self.get_active_agent().calculate_reward(**kwargs)

    def safe_save(self):
        """Save all models in the population"""
        for i, agent_data in enumerate(self.population):
            agent_data['trainer'].model_path = f"models/pbt_agent_{i}_model.pth"
            agent_data['trainer'].safe_save()
        print("Saved all PBT population models.")

    def get_training_stats(self):
        """Get aggregated training stats for the population"""
        active_agent_stats = self.get_active_agent().get_training_stats()
        active_agent_stats['active_agent'] = self.active_agent_idx
        active_agent_stats['population_size'] = self.population_size
        
        fitness_scores = [p['fitness'] for p in self.population]
        active_agent_stats['avg_population_fitness'] = np.mean(fitness_scores)
        active_agent_stats['max_population_fitness'] = np.max(fitness_scores)
        
        return active_agent_stats