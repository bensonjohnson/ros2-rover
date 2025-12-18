#!/usr/bin/env python3
"""V700 ES-SAC Hybrid Trainer.

Combines Evolutionary Strategies (ES) with Soft Actor-Critic (SAC).
- Population of Actors evolves via Mutation/Crossover.
- One "Meta-Learner" (SAC Agent) trains continuously on the Replay Buffer.
- SAC Agent "injects" its learned policy into the population to guide evolution.
- SAC Critic "evaluates" population members on previous data to estimate fitness.
"""

print("🔧 Loading imports...", flush=True)
import os
import sys
import time
import json
import copy
import random
import threading
import asyncio
import argparse
import warnings
import gc
from typing import Dict, List, Tuple, Optional
from collections import deque
from pathlib import Path

print("  → Importing numpy...", flush=True)
import numpy as np
print("  → Importing torch (may take a moment on ROCm)...", flush=True)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
print("  → Importing tqdm...", flush=True)
from tqdm import tqdm
print("  → Importing nats...", flush=True)
import nats
from nats.js.api import StreamConfig

print("  → Importing model architectures...", flush=True)
# Import architectures
from model_architectures import DualEncoderPolicyNetwork, DualEncoderQNetwork

print("  → Importing serialization utils...", flush=True)
# Import serialization
from serialization_utils import (
    deserialize_batch, serialize_model_update,
    serialize_metadata, deserialize_metadata
)
print("✅ All imports loaded", flush=True)

# -----------------------------------------------------------------------------
# Population Manager
# -----------------------------------------------------------------------------

class PopulationManager:
    """Manages the population of policies for Evolution."""
    
    def __init__(self, population_size: int, device: torch.device):
        self.pop_size = population_size
        self.device = device
        self.population = [] # List of {'model': state_dict, 'fitness': float, 'id': int}
        self.next_id = 0
        self.generation = 0
        
        # Hyperparams
        self.mutation_rate = 0.02
        self.mutation_noise = 0.05
        
    def initialize_population(self, template_model: nn.Module):
        """Create initial random population."""
        print(f"🧬 Initializing population of {self.pop_size}...")
        self.population = []
        for _ in range(self.pop_size):
            # Create random weights
            individual = copy.deepcopy(template_model)
            # Randomize slightly different initializations? 
            # PyTorch init is random, so just creating new instances is enough if we extracted state_dict
            # But better to take one template and perturb it to ensure valid range
            
            # Actually, standard ES starts with one mean and perturbs.
            # Here we want diversity. Let's just use standard PyTorch Kaiming init for all.
            wrapper = {
                'model': copy.deepcopy(individual.state_dict()),
                'fitness': -float('inf'), # Unknown
                'id': self.next_id,
                'source': 'init'
            }
            self.population.append(wrapper)
            self.next_id += 1
            
    def inject_agent(self, agent_state_dict: dict):
        """Inject a trained agent into the population (Replacing the worst)."""
        # Sort current population by fitness (descending)
        # Any with -inf fitness are effectively at the bottom
        sorted_pop = sorted(self.population, key=lambda x: x['fitness'], reverse=True)
        
        # Replace the worst one
        worst = sorted_pop[-1]
        print(f"💉 Injecting SAC Agent (replacing ID {worst['id']}, fit={worst['fitness']:.2f})")
        
        worst['model'] = copy.deepcopy(agent_state_dict)
        worst['fitness'] = -float('inf') # Needs re-evaluation
        worst['source'] = 'sac_injection'
        worst['id'] = self.next_id
        self.next_id += 1
        
    def get_candidate(self) -> dict:
        """Select a candidate for evaluation on the rover.
        
        Strategy: Round-robin through unevaluated, then tournament for re-evaluation?
        Simple: Return a random member that has fitness == -inf (unevaluated).
        If all evaluated, return top member to confirm? Or trigger evolution?
        """
        unevaluated = [p for p in self.population if p['fitness'] == -float('inf')]
        if unevaluated:
            return random.choice(unevaluated)
        else:
            # All evaluated. Return best for demo/showing off until evolution triggers
            return max(self.population, key=lambda x: x['fitness'])

    def update_fitness(self, individual_id: int, fitness: float):
        """Update fitness for an individual."""
        for p in self.population:
            if p['id'] == individual_id:
                p['fitness'] = fitness
                break # Found

    def evolve(self, elite_fraction=0.2):
        """Perform one generation of evolution."""
        self.generation += 1
        
        # 1. Sort by fitness
        sorted_pop = sorted(self.population, key=lambda x: x['fitness'], reverse=True)
        
        # 2. Elitism
        num_elites = max(1, int(self.pop_size * elite_fraction))
        elites = sorted_pop[:num_elites]
        
        print(f"🧬 Generation {self.generation}: Best Fitness = {elites[0]['fitness']:.2f}")
        
        next_pop = []
        
        # Keep elites
        for elite in elites:
            wrapper = copy.deepcopy(elite)
            wrapper['source'] = 'elite'
            # Retain fitness? Or re-evaluate to handle noise?
            # Let's reset fitness to force re-evaluation if environment is noisy
            wrapper['fitness'] = -float('inf') 
            next_pop.append(wrapper)
            
        # Fill rest via mutation/crossover of elites
        while len(next_pop) < self.pop_size:
            parent = random.choice(elites)
            
            # Create child
            child_state = copy.deepcopy(parent['model'])
            self._mutate(child_state)
            
            wrapper = {
                'model': child_state,
                'fitness': -float('inf'),
                'id': self.next_id,
                'source': f"mut_{parent['id']}"
            }
            self.next_id += 1
            next_pop.append(wrapper)
            
        self.population = next_pop
        
    def _mutate(self, state_dict):
        """Apply Gaussian noise to weights."""
        for name, param in state_dict.items():
            if 'weight' in name: # Only mutate weights, maybe bias too?
                noise = torch.randn_like(param) * self.mutation_noise
                param.add_(noise)
                
# -----------------------------------------------------------------------------
# SAC Replay Buffer (Standard)
# -----------------------------------------------------------------------------

class ReplayBuffer:
    """Experience Replay Buffer for SAC."""
    def __init__(self, capacity: int, proprio_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Storage (CPU RAM to save VRAM)
        self.laser = torch.zeros((capacity, 1, 128, 128), dtype=torch.uint8)
        self.depth = torch.zeros((capacity, 1, 100, 848), dtype=torch.uint8)
        self.proprio = torch.zeros((capacity, proprio_dim), dtype=torch.float32)
        self.actions = torch.zeros((capacity, 2), dtype=torch.float32)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32)
        
    def add(self, batch_data: dict):
        """Add batch of transitions."""
        # Assume batch_data has keys corresponding to storage
        # Handles single step or batch
        N = len(batch_data['rewards'])
        if N == 0: return

        # Indices
        indices = np.arange(self.ptr, self.ptr + N) % self.capacity
        
        # Copy data
        # Note: Inputs expected to be tensors or numpy arrays
        self.laser[indices] = torch.as_tensor(batch_data['laser'], dtype=torch.uint8)
        self.depth[indices] = torch.as_tensor(batch_data['depth'], dtype=torch.uint8)
        self.proprio[indices] = torch.as_tensor(batch_data['proprio'], dtype=torch.float32)
        self.actions[indices] = torch.as_tensor(batch_data['actions'], dtype=torch.float32)
        self.rewards[indices] = torch.as_tensor(batch_data['rewards'], dtype=torch.float32).reshape(-1, 1)
        self.dones[indices] = torch.as_tensor(batch_data['dones'], dtype=torch.float32).reshape(-1, 1)
        
        self.ptr = (self.ptr + N) % self.capacity
        self.size = min(self.size + N, self.capacity)
        
    def sample(self, batch_size):
        """Sample batch for SAC training."""
        idx = np.random.randint(0, self.size-1, size=batch_size)
        
        # Next state is idx + 1 (unless done, handled by mask)
        next_idx = (idx + 1) % self.capacity
        
        batch = {
            'laser': self.laser[idx].float().to(self.device), # 0/1 -> float
            'depth': self.depth[idx].float().div(255.0).to(self.device),
            'proprio': self.proprio[idx].to(self.device),
            'action': self.actions[idx].to(self.device),
            'reward': self.rewards[idx].to(self.device),
            'done': self.dones[idx].to(self.device),
            'next_laser': self.laser[next_idx].float().to(self.device),
            'next_depth': self.depth[next_idx].float().div(255.0).to(self.device),
            'next_proprio': self.proprio[next_idx].to(self.device),
        }
        return batch

# -----------------------------------------------------------------------------
# Main Trainer
# -----------------------------------------------------------------------------

class ESSACTrainer:
    def __init__(self, args):
        self.args = args
        print("  → Setting up device...", flush=True)
        self.setup_device()

        # Components
        self.proprio_dim = 10
        self.action_dim = 2

        # 1. SAC Agent (The "Teacher")
        print("  → Creating actor network...", flush=True)
        self.actor = DualEncoderPolicyNetwork(self.action_dim, self.proprio_dim).to(self.device)
        print("  → Creating critic networks...", flush=True)
        self.critic1 = DualEncoderQNetwork(self.action_dim, self.proprio_dim).to(self.device)
        self.critic2 = DualEncoderQNetwork(self.action_dim, self.proprio_dim).to(self.device)
        print("  → Creating target critics...", flush=True)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        
        print("  → Creating optimizers...", flush=True)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=3e-4)

        # Alpha (Entropy)
        self.log_alpha = torch.tensor(np.log(0.1), requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -float(self.action_dim)

        # 2. Population (The "Students")
        print("  → Initializing population...", flush=True)
        self.pop_manager = PopulationManager(population_size=10, device=self.device)  # Full population for diversity
        self.pop_manager.initialize_population(self.actor) # Init with random policies

        # 3. Buffer
        print("  → Creating replay buffer...", flush=True)
        self.buffer = ReplayBuffer(100000, self.proprio_dim, torch.device('cpu'))

        # 4. State
        self.total_steps = 0
        self.active_requests = {} # request_id -> model_id

        # GPU Serialization Lock - prevents concurrent GPU operations
        self.gpu_lock = asyncio.Lock()

        # NATS
        self.nc = None
        self.js = None

        # Logs
        print("  → Setting up TensorBoard...", flush=True)
        self.writer = SummaryWriter(args.log_dir)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        print("✅ Trainer initialized successfully", flush=True)

    def setup_device(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("⚠ Using CPU")

    async def connect_nats(self):
        self.nc = await nats.connect(self.args.nats_server, max_reconnect_attempts=-1)
        self.js = self.nc.jetstream()
        print("✓ Connected to NATS")
        
        # Create Streams
        try:
            await self.js.add_stream(name="es_training", subjects=["es.*"])
        except Exception:
            pass # Stream might exist

    async def run(self):
        await self.connect_nats()
        
        # Subscribers
        await self.nc.subscribe("es.request_model", cb=self.handle_model_request) # Keep for legacy/hybrid
        await self.nc.subscribe("es.episode_result", cb=self.handle_episode_result)
        await self.nc.subscribe("es.step_inference", cb=self.handle_step_inference)
        
        print("🚀 ES-SAC Trainer Running (Remote Inference Mode supported)...")
        
        # Background Training Loop for SAC
        train_task = asyncio.create_task(self.sac_training_loop())
        
        # Keep alive
        while True:
            await asyncio.sleep(1)

    async def handle_model_request(self, msg):
        """Rover asks: 'Give me a model to test'."""
        request = json.loads(msg.data.decode())
        rover_id = request.get('rover_id', 'unknown')
        
        # Select candidate
        candidate = self.pop_manager.get_candidate()
        model_id = candidate['id']
        fitness_src = candidate['source']
        
        print(f"Selecting Model {model_id} ({fitness_src}) for Rover {rover_id}")
        
        # Serialize Weights
        # We assume the rover side controls 'reset' logic, so we just send weights.
        # To make it fast, we might compress.
        # For now, standard pickle/torch serialization. 
        # WARNING: sending full weights over NATS might be heavy (20MB+).
        # We should use Object Store or compressed binary.
        # Let's use a specialized binary serialization.
        
        import io
        buffer = io.BytesIO()
        torch.save(candidate['model'], buffer)
        weights_bytes = buffer.getvalue()
        
        # Reply with weights
        # Current NATS max payload defaults are often 1MB. We likely adjusted this?
        # If not, this will fail. Assuming User configured NATS for large payloads.
        response = {
            'model_id': model_id,
            'source': fitness_src
        }
        
        # Send header + binary body? Or just raw binary with ID in header?
        # Simple NATS Request-Reply pattern
        # Be careful with size. 
        # Standard SAC actor is ~5MB? DualEncoder might be larger (20MB?).
        # We need ObjectStore if > 1MB usually.
        # Or chunking.
        
        # Start simple: Assume server configured for 64MB messages (as per previous convos)
        
        # Reply structure: JSON Header len (4 bytes) + JSON Header + Binary
        header = json.dumps(response).encode()
        header_len = len(header).to_bytes(4, 'big')
        payload = header_len + header + weights_bytes
        
        await self.nc.publish(msg.reply, payload)

    async def handle_episode_result(self, msg):
        """Rover reports: 'Finished episode with Model X, here is data'."""
        # This is likely a large message with compressed batch data
        data = deserialize_batch(msg.data) # Reusing existing util which handles zstd
        
        model_id = data['metadata'].get('model_id')
        total_reward = float(np.sum(data['rewards']))
        
        if self.buffer.size > 1000:
            # "Base the fitness on some sort of reinforcement logic"
            # We calculate the Mean Q-Value of this model on a batch of replay data.
            # This estimates "future success" based on the collective knowledge of the SAC agent.
            q_score = self.evaluate_model_critic(model_id)
            print(f"    Simple Reward: {total_reward:.2f}, Q-Score: {q_score:.2f}")
            
            # Hybrid Fitness: Real Reward + Critic Estimate
            # Scaling: Rewards are ~10-50? Q-values are ~10-50?
            # Let's simple sum for now, effectively doubling the signal if they agree.
            final_fitness = total_reward + q_score
        else:
            final_fitness = total_reward
            
        print(f"📝 Result for Model {model_id}: Fitness = {final_fitness:.2f} (R:{total_reward:.1f} + Q:{final_fitness-total_reward:.1f})")
        
        # 1. Update Population Fitness
        self.pop_manager.update_fitness(model_id, final_fitness)
        self.writer.add_scalar('ES/EpisodeReward', total_reward, self.total_steps)
        self.writer.add_scalar('ES/Fitness', final_fitness, self.total_steps)
        
        # 2. Add to Replay Buffer (for SAC training)
        with torch.no_grad():
             # Convert data to format ReplayBuffer expects
            self.buffer.add({
                'laser': data['laser'], # (N, 1, 128, 128)
                'depth': data['depth'], # (N, 1, 100, 848)
                'proprio': data['proprio'],
                'actions': data['actions'],
                'rewards': data['rewards'],
                'dones': data['dones']
            })
            
        # 3. Check for Evolution Trigger
        # If all population evaluated (fitness != -inf), trigger evolution
        unevaluated = [p for p in self.pop_manager.population if p['fitness'] == -float('inf')]
        if len(unevaluated) == 0:
            print("🔄 Population fully evaluated. Evolving...")
            
            # Inject SAC Agent before evolving
            sac_state = self.actor.state_dict()
            self.pop_manager.inject_agent(sac_state)
            
            # Evolve
            self.pop_manager.evolve()

            # Log max fitness
            best_fit = max(p['fitness'] for p in self.pop_manager.population)
            self.writer.add_scalar('ES/GenMaxFitness', best_fit, self.pop_manager.generation)

    async def handle_step_inference(self, msg):
        """Rover sends Obs, Server runs Inference -> Returns Action."""
        t0 = time.perf_counter()

        # Always send exactly 8 bytes (2 float32) no matter what
        zero_action = np.zeros(2, dtype=np.float32)

        try:
            # 1. Deserialize (outside lock - no GPU needed)
            data = msg.data
            header_len = int.from_bytes(data[:2], 'big')
            header = json.loads(data[2:2+header_len])
            rover_id = header.get('rover_id', 'rover_1')

            # 2. Get Assigned Model
            if rover_id not in self.active_requests:
                candidate = self.pop_manager.get_candidate()
                self.active_requests[rover_id] = candidate
                print(f"Assigning Model {candidate['id']} to {rover_id}", flush=True)

            candidate = self.active_requests[rover_id]

            # 3. Parse Tensors (outside lock - no GPU needed)
            raw_bytes = data[2+header_len:]
            offset = 0

            # Laser (128x128 uint8)
            laser_size = 128 * 128
            laser_np = np.frombuffer(raw_bytes, dtype=np.uint8, count=laser_size, offset=offset).reshape(1, 1, 128, 128)
            offset += laser_size

            # Depth (100x848 uint8)
            depth_size = 100 * 848
            depth_np = np.frombuffer(raw_bytes, dtype=np.uint8, count=depth_size, offset=offset).reshape(1, 1, 100, 848)
            offset += depth_size

            # Proprio (10 float32)
            proprio_size = 10
            proprio_np = np.frombuffer(raw_bytes, dtype=np.float32, count=proprio_size, offset=offset).reshape(1, 10)

            # 4. Inference - ACQUIRE GPU LOCK
            async with self.gpu_lock:
                with torch.no_grad():
                    # Convert to tensor on GPU
                    l_t = torch.from_numpy(laser_np.copy()).to(self.device, dtype=torch.float32)
                    d_t = torch.from_numpy(depth_np.copy()).to(self.device, dtype=torch.float32).div_(255.0)
                    p_t = torch.from_numpy(proprio_np.copy()).to(self.device, dtype=torch.float32)

                    # Maintain inference actor (only reload weights when model ID changes)
                    if not hasattr(self, 'inference_actor'):
                        self.inference_actor = copy.deepcopy(self.actor)
                        self.inference_actor_id = -1

                    if self.inference_actor_id != candidate['id']:
                        self.inference_actor.load_state_dict(candidate['model'])
                        self.inference_actor_id = candidate['id']
                        self.inference_actor.eval()

                    # Run inference
                    action_mean, _ = self.inference_actor(l_t, d_t, p_t)
                    action = torch.tanh(action_mean).cpu().numpy()[0]  # (2,)

                    # Explicitly delete GPU tensors
                    del l_t, d_t, p_t, action_mean

            # 5. Reply - MUST be exactly 8 bytes
            reply_bytes = action.astype(np.float32).tobytes()
            assert len(reply_bytes) == 8, f"Reply must be 8 bytes, got {len(reply_bytes)}"
            await self.nc.publish(msg.reply, reply_bytes)

            # Latency check
            dt = (time.perf_counter() - t0) * 1000
            if dt > 100:
                print(f"⚠ Inference took {dt:.1f}ms (lock wait + GPU)", flush=True)

        except Exception as e:
            # Log error with more detail
            import traceback
            print(f"✗ Inference error: {type(e).__name__}: {e}", flush=True)
            print(f"   Traceback: {traceback.format_exc()}", flush=True)

            # ALWAYS send exactly 8 bytes (zero action)
            reply_bytes = zero_action.tobytes()
            assert len(reply_bytes) == 8, f"Error reply must be 8 bytes, got {len(reply_bytes)}"
            await self.nc.publish(msg.reply, reply_bytes)

    async def sac_training_loop(self):
        """Continuous SAC training on Replay Buffer."""
        print("🧠 SAC Training Loop started...")
        while True:
            if self.buffer.size < 1000:
                await asyncio.sleep(1)
                continue

            # ACQUIRE GPU LOCK for training
            async with self.gpu_lock:
                # Train step
                try:
                    metrics = self.train_step()
                    self.total_steps += 1

                    if self.total_steps % 100 == 0:
                        self.writer.add_scalar('SAC/ActorLoss', metrics['actor_loss'], self.total_steps)
                        self.writer.add_scalar('SAC/CriticLoss', metrics['critic_loss'], self.total_steps)

                    # Periodic garbage collection
                    if self.total_steps % 500 == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"⚠ Training OOM, clearing cache and skipping step", flush=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        await asyncio.sleep(0.5)
                        continue
                    else:
                        raise

            # Yield to event loop after releasing lock - allows inference to get priority
            await asyncio.sleep(0.01)  # Short sleep to let inference requests in

    @torch.no_grad()
    def evaluate_model_critic(self, model_id: int) -> float:
        """Evaluate a population member using the SAC Critic."""
        # Find model
        entry = next((m for m in self.pop_manager.population if m['id'] == model_id), None)
        if not entry: return 0.0

        # Load weights into a temp actor
        temp_actor = copy.deepcopy(self.actor)
        temp_actor.load_state_dict(entry['model'])
        temp_actor.eval()

        # Sample batch
        batch = self.buffer.sample(batch_size=256)

        # Compute Action
        action, _ = temp_actor(batch['laser'], batch['depth'], batch['proprio'])

        # Compute Q
        q1 = self.critic1(batch['laser'], batch['depth'], batch['proprio'], action)
        q2 = self.critic2(batch['laser'], batch['depth'], batch['proprio'], action)
        min_q = torch.min(q1, q2)

        result = min_q.mean().item()

        # Clean up to prevent memory leak
        del temp_actor, batch, action, q1, q2, min_q

        return result

    def train_step(self):
        """Single SAC Gradient Step."""
        batch = self.buffer.sample(self.args.batch_size)
        
        # 1. Update Critics
        with torch.no_grad():
            next_action, next_log_prob = self.actor(batch['next_laser'], batch['next_depth'], batch['next_proprio'])
            target_q1 = self.target_critic1(batch['next_laser'], batch['next_depth'], batch['next_proprio'], next_action)
            target_q2 = self.target_critic2(batch['next_laser'], batch['next_depth'], batch['next_proprio'], next_action)
            target_q = torch.min(target_q1, target_q2) - torch.exp(self.log_alpha) * next_log_prob
            target_q = batch['reward'] + (1 - batch['done']) * 0.99 * target_q
            
        current_q1 = self.critic1(batch['laser'], batch['depth'], batch['proprio'], batch['action'])
        current_q2 = self.critic2(batch['laser'], batch['depth'], batch['proprio'], batch['action'])
        
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        # 2. Update Actor
        pred_action, log_prob = self.actor(batch['laser'], batch['depth'], batch['proprio'])
        q1 = self.critic1(batch['laser'], batch['depth'], batch['proprio'], pred_action)
        q2 = self.critic2(batch['laser'], batch['depth'], batch['proprio'], pred_action)
        min_q = torch.min(q1, q2)
        
        actor_loss = (torch.exp(self.log_alpha) * log_prob - min_q).mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
        # 3. Update Alpha
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        
        # 4. Soft Update Targets
        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)
            
        return {'actor_loss': actor_loss.item(), 'critic_loss': critic_loss.item()}

if __name__ == "__main__":
    print("🔧 Parsing arguments...", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--nats_server', type=str, default="nats://nats.gokickrocks.org:4222")
    parser.add_argument('--checkpoint_dir', type=str, default="./checkpoints_es")
    parser.add_argument('--log_dir', type=str, default="./logs_es")
    parser.add_argument('--batch_size', type=int, default=256)  # Large batch for better gradients
    args = parser.parse_args()

    print("🔧 Initializing trainer...", flush=True)
    trainer = ESSACTrainer(args)
    print("🔧 Starting async event loop...", flush=True)
    try:
        asyncio.run(trainer.run())
    except KeyboardInterrupt:
        print("🛑 Stopping...")
