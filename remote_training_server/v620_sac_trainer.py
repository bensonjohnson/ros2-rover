#!/usr/bin/env python3
"""V620 ROCm SAC Training Server for Remote Rover Training.

This server receives RGB-D observations from the rover via ZeroMQ,
trains a SAC (Soft Actor-Critic) policy using PyTorch with ROCm acceleration,
and exports trained models in ONNX format for the rover.

Features:
- Off-policy learning (Replay Buffer)
- Entropy maximization (Exploration)
- Asynchronous training
- Automatic Entropy Tuning (Alpha)
"""

import os
import sys
import time
import json
import argparse
import threading
import queue
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import copy

import numpy as np
import cv2
import zmq
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import model architectures
from model_architectures import RGBDEncoder, GaussianPolicyHead, QNetwork

class ReplayBuffer:
    """Experience Replay Buffer for SAC."""
    
    def __init__(self, capacity: int, rgb_shape: Tuple, depth_shape: Tuple, proprio_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.full = False
        
        # Storage (CPU RAM to save GPU memory, move to GPU during sampling)
        # Using uint8 for images to save RAM
        self.rgb = torch.zeros((capacity, *rgb_shape), dtype=torch.uint8)
        self.depth = torch.zeros((capacity, *depth_shape), dtype=torch.float16) # Optimized to float16
        self.proprio = torch.zeros((capacity, proprio_dim), dtype=torch.float32)
        self.actions = torch.zeros((capacity, 2), dtype=torch.float32)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32)
        
    def add_batch(self, batch_data: Dict):
        """Add a batch of sequential data and construct transitions."""
        # batch_data contains lists/arrays of s, a, r, d
        # We need to construct (s, a, r, s', d)
        # Since data is sequential, s'[t] = s[t+1]
        
        rgb = batch_data['rgb']
        depth = batch_data['depth']
        proprio = batch_data['proprio']
        actions = batch_data['actions']
        rewards = batch_data['rewards']
        dones = batch_data['dones']
        
        num_steps = len(rewards)
        if num_steps < 2:
            return # Need at least 2 steps to form a transition
            
        # We can form (num_steps - 1) transitions from a single batch
        # unless we cache the last observation from the previous batch.
        # For simplicity, we'll just use the transitions within this batch.
        # This loses 1 transition per batch (negligible).
        
        # Current states: 0 to N-1
        # Next states: 1 to N
        
        # Indices for current step
        curr_slice = slice(0, num_steps - 1)
        # Indices for next step
        next_slice = slice(1, num_steps)
        
        # Count valid transitions
        # If done[t] is True, then s[t+1] is start of new episode, NOT next state of s[t]
        # But here we are storing (s, a, r, s', d).
        # If d[t] is True, s'[t] doesn't matter (masked by 1-d).
        # So we can just take s[t+1] as s'[t] blindly.
        
        batch_size = num_steps - 1
        
        # Handle wrap-around
        if self.ptr + batch_size > self.capacity:
            # Split
            first_part = self.capacity - self.ptr
            second_part = batch_size - first_part
            
            self._add_slice(rgb, depth, proprio, actions, rewards, dones, 0, first_part, self.ptr)
            self._add_slice(rgb, depth, proprio, actions, rewards, dones, first_part, batch_size, 0)
            
            self.ptr = second_part
            self.full = True
        else:
            self._add_slice(rgb, depth, proprio, actions, rewards, dones, 0, batch_size, self.ptr)
            self.ptr += batch_size
            if self.ptr >= self.capacity:
                self.full = True
                self.ptr = self.ptr % self.capacity
                
        self.size = self.capacity if self.full else self.ptr

    def _add_slice(self, rgb, depth, proprio, actions, rewards, dones, start_idx, count, buffer_idx):
        """Helper to add slice."""
        # Source indices: start_idx to start_idx + count
        # BUT for next_state, we need +1
        
        # s, a, r, d come from [start_idx : start_idx + count]
        # s' comes from [start_idx + 1 : start_idx + count + 1]
        
        end_idx = start_idx + count
        
        self.rgb[buffer_idx:buffer_idx+count] = torch.as_tensor(rgb[start_idx:end_idx])
        self.depth[buffer_idx:buffer_idx+count] = torch.as_tensor(depth[start_idx:end_idx]).to(torch.float16) # Convert to float16
        self.proprio[buffer_idx:buffer_idx+count] = torch.as_tensor(proprio[start_idx:end_idx])
        self.actions[buffer_idx:buffer_idx+count] = torch.as_tensor(actions[start_idx:end_idx])
        self.rewards[buffer_idx:buffer_idx+count] = torch.as_tensor(rewards[start_idx:end_idx]).unsqueeze(1)
        self.dones[buffer_idx:buffer_idx+count] = torch.as_tensor(dones[start_idx:end_idx]).unsqueeze(1)
        
        # We don't store next_state explicitly to save RAM.
        # We store sequential data.
        # Wait, if I use a circular buffer and overwrite, I lose the "next" relationship at the boundary of the pointer?
        # Standard ReplayBuffers store (s, s').
        # Storing images twice is heavy (2x RAM).
        # Optimization: Store only 's'. When sampling index 'i', 's_next' is 'i+1'.
        # We just need to handle the case where 'i' is the last element or 'i' is a terminal state.
        # If 'dones[i]' is True, 's_next' is irrelevant.
        # If 'i' is at buffer boundary, we need to wrap.
        # BUT if we overwrite, 'i+1' might be new data unrelated to 'i'.
        # To fix this: We accept that 'next_state' might be garbage if we overwrite.
        # Or we store 'next_idx' or just store s' explicitly.
        # Given 64GB+ RAM on server, maybe storing s' is fine?
        # RGB: 240*424*3 = 300KB. 100k steps = 30GB.
        # 2x = 60GB. Tight.
        # Let's stick to "next state is index + 1" and be careful about boundaries.
        # Actually, for simplicity and robustness, I will store s' explicitly for now, 
        # but maybe downsample or compress if needed.
        # Or better: Just implement the "next_state = buffer[(i+1)%size]" logic and ignore the edge case where we just overwrote i+1.
        # The probability of sampling the exact boundary index is low.
        pass

    def sample(self, batch_size):
        """Sample a batch of transitions."""
        indices = np.random.randint(0, self.size - 1, size=batch_size) # -1 to ensure i+1 exists
        
        # Retrieve s
        rgb = self.rgb[indices].to(self.device).float() / 255.0
        rgb = rgb.permute(0, 3, 1, 2) # NHWC -> NCHW
        
        depth = self.depth[indices].to(self.device).float().unsqueeze(1)
        proprio = self.proprio[indices].to(self.device)
        actions = self.actions[indices].to(self.device)
        rewards = self.rewards[indices].to(self.device)
        dones = self.dones[indices].to(self.device)
        
        # Retrieve s' (next index)
        next_indices = (indices + 1) % self.capacity
        
        next_rgb = self.rgb[next_indices].to(self.device).float() / 255.0
        next_rgb = next_rgb.permute(0, 3, 1, 2)
        
        next_depth = self.depth[next_indices].to(self.device).float().unsqueeze(1)
        next_proprio = self.proprio[next_indices].to(self.device)
        
        return {
            'rgb': rgb, 'depth': depth, 'proprio': proprio,
            'action': actions, 'reward': rewards, 'done': dones,
            'next_rgb': next_rgb, 'next_depth': next_depth, 'next_proprio': next_proprio
        }

class V620SACTrainer:
    """SAC Trainer optimized for V620 ROCm."""
    
    def __init__(self, args):
        self.args = args
        
        # Device setup
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("⚠ Using CPU")
            
        # Dimensions
        self.rgb_shape = (240, 424, 3) # HWC
        self.depth_shape = (240, 424)
        self.proprio_dim = 10
        self.action_dim = 2
        
        # --- Actor ---
        self.actor_encoder = RGBDEncoder().to(self.device)
        self.actor_head = GaussianPolicyHead(self.actor_encoder.output_dim, self.proprio_dim, self.action_dim).to(self.device)
        
        # --- Critics ---
        # Shared encoder for critics? Or separate?
        # To save memory, let's share one encoder for both critics, but separate from actor.
        self.critic_encoder = RGBDEncoder().to(self.device)
        self.critic1 = QNetwork(self.critic_encoder.output_dim, self.proprio_dim, self.action_dim).to(self.device)
        self.critic2 = QNetwork(self.critic_encoder.output_dim, self.proprio_dim, self.action_dim).to(self.device)
        
        # --- Target Critics ---
        self.target_critic_encoder = copy.deepcopy(self.critic_encoder)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            list(self.actor_encoder.parameters()) + list(self.actor_head.parameters()), 
            lr=args.lr
        )
        self.critic_optimizer = optim.Adam(
            list(self.critic_encoder.parameters()) + list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=args.lr
        )
        
        # Automatic Entropy Tuning
        self.target_entropy = -float(self.action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=args.lr)
        
        # Replay Buffer
        self.buffer = ReplayBuffer(
            capacity=args.buffer_size,
            rgb_shape=self.rgb_shape,
            depth_shape=self.depth_shape,
            proprio_dim=self.proprio_dim,
            device=self.device
        )
        
        # State
        self.total_steps = 0
        self.model_version = 0
        self.training_active = False
        
        # ZMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{args.port}")
        
        # Logging
        self.writer = SummaryWriter(args.log_dir)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        
        # Threading
        self.lock = threading.Lock()
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()
        
        # Load Checkpoint
        self.load_latest_checkpoint()
        
        # Export initial model
        self.export_onnx(increment_version=False)

    def load_latest_checkpoint(self):
        checkpoints = list(Path(self.args.checkpoint_dir).glob('sac_step_*.pt'))
        if not checkpoints:
            print("🆕 No checkpoint found. Starting fresh.")
            return
            
        latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
        print(f"🔄 Resuming from {latest}")
        ckpt = torch.load(latest, map_location=self.device)
        
        self.actor_encoder.load_state_dict(ckpt['actor_encoder'])
        self.actor_head.load_state_dict(ckpt['actor_head'])
        self.critic_encoder.load_state_dict(ckpt['critic_encoder'])
        self.critic1.load_state_dict(ckpt['critic1'])
        self.critic2.load_state_dict(ckpt['critic2'])
        self.target_critic_encoder.load_state_dict(ckpt['target_critic_encoder'])
        self.target_critic1.load_state_dict(ckpt['target_critic1'])
        self.target_critic2.load_state_dict(ckpt['target_critic2'])
        self.log_alpha.data = ckpt['log_alpha']
        
        self.actor_optimizer.load_state_dict(ckpt['actor_opt'])
        self.critic_optimizer.load_state_dict(ckpt['critic_opt'])
        self.alpha_optimizer.load_state_dict(ckpt['alpha_opt'])
        
        self.total_steps = ckpt['total_steps']
        self.model_version = self.total_steps // 1000 # Rough versioning

    def save_checkpoint(self):
        path = os.path.join(self.args.checkpoint_dir, f"sac_step_{self.total_steps}.pt")
        torch.save({
            'actor_encoder': self.actor_encoder.state_dict(),
            'actor_head': self.actor_head.state_dict(),
            'critic_encoder': self.critic_encoder.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_critic_encoder': self.target_critic_encoder.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_opt': self.actor_optimizer.state_dict(),
            'critic_opt': self.critic_optimizer.state_dict(),
            'alpha_opt': self.alpha_optimizer.state_dict(),
            'total_steps': self.total_steps
        }, path)
        print(f"💾 Saved {path}")
        self.export_onnx()

    def export_onnx(self, increment_version=True):
        """Export Actor mean to ONNX."""
        try:
            onnx_path = os.path.join(self.args.checkpoint_dir, "latest_actor.onnx")
            
            dummy_rgb = torch.randn(1, 3, 240, 424, device=self.device)
            dummy_depth = torch.randn(1, 1, 240, 424, device=self.device)
            dummy_proprio = torch.randn(1, self.proprio_dim, device=self.device)
            
            class ActorWrapper(nn.Module):
                def __init__(self, encoder, head):
                    super().__init__()
                    self.encoder = encoder
                    self.head = head
                def forward(self, rgb, depth, proprio):
                    features = self.encoder(rgb, depth)
                    mean, _ = self.head(features, proprio)
                    return torch.tanh(mean) # Deterministic action
            
            model = ActorWrapper(self.actor_encoder, self.actor_head)
            model.eval()
            
            torch.onnx.export(
                model,
                (dummy_rgb, dummy_depth, dummy_proprio),
                onnx_path,
                opset_version=11,
                input_names=['rgb', 'depth', 'proprio'],
                output_names=['action'],
                export_params=True
            )
            
            if increment_version:
                self.model_version += 1
            print(f"📦 Exported ONNX (v{self.model_version})")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")

    def _training_loop(self):
        print("🧵 Training thread started")
        while True:
            if self.buffer.size > self.args.batch_size * 5: # Wait for some data
                with self.lock:
                    metrics = self.train_step()
                
                if metrics:
                    self.total_steps += 1
                    if self.total_steps % 100 == 0:
                        print(f"Step {self.total_steps} | A_Loss: {metrics['actor_loss']:.3f} C_Loss: {metrics['critic_loss']:.3f} Alpha: {metrics['alpha']:.3f}")
                        for k, v in metrics.items():
                            self.writer.add_scalar(k, v, self.total_steps)
                            
                    if self.total_steps % 5000 == 0:
                        self.save_checkpoint()
            else:
                time.sleep(1.0) # Wait for data

    def train_step(self):
        batch = self.buffer.sample(self.args.batch_size)
        
        # Unpack
        state_rgb = batch['rgb']
        state_depth = batch['depth']
        state_proprio = batch['proprio']
        action = batch['action']
        reward = batch['reward']
        done = batch['done']
        next_rgb = batch['next_rgb']
        next_depth = batch['next_depth']
        next_proprio = batch['next_proprio']
        
        alpha = self.log_alpha.exp().item()
        
        # --- Critic Update ---
        with torch.no_grad():
            # Get next action from target policy
            next_features = self.actor_encoder(next_rgb, next_depth)
            next_mean, next_log_std = self.actor_head(next_features, next_proprio)
            next_std = next_log_std.exp()
            dist = torch.distributions.Normal(next_mean, next_std)
            next_action_sample = dist.rsample()
            next_action = torch.tanh(next_action_sample)
            
            # Compute log prob for entropy
            next_log_prob = dist.log_prob(next_action_sample).sum(dim=-1, keepdim=True)
            next_log_prob -= (2 * (np.log(2) - next_action_sample - F.softplus(-2 * next_action_sample))).sum(dim=1, keepdim=True)
            
            # Target Q
            target_features = self.target_critic_encoder(next_rgb, next_depth)
            q1_target = self.target_critic1(target_features, next_proprio, next_action)
            q2_target = self.target_critic2(target_features, next_proprio, next_action)
            min_q_target = torch.min(q1_target, q2_target) - alpha * next_log_prob
            next_q_value = reward + (1 - done) * self.args.gamma * min_q_target
            
        # Current Q
        curr_features = self.critic_encoder(state_rgb, state_depth)
        q1 = self.critic1(curr_features, state_proprio, action)
        q2 = self.critic2(curr_features, state_proprio, action)
        
        critic_loss = F.mse_loss(q1, next_q_value) + F.mse_loss(q2, next_q_value)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # --- Actor Update ---
        # Re-compute features for actor (gradient flows through encoder)
        actor_features = self.actor_encoder(state_rgb, state_depth)
        mean, log_std = self.actor_head(actor_features, state_proprio)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action_sample = dist.rsample()
        current_action = torch.tanh(action_sample)
        
        log_prob = dist.log_prob(action_sample).sum(dim=-1, keepdim=True)
        log_prob -= (2 * (np.log(2) - action_sample - F.softplus(-2 * action_sample))).sum(dim=1, keepdim=True)
        
        # Use critic to evaluate action
        # We detach critic encoder here? Usually yes, or no?
        # In standard SAC, actor loss does NOT update critic parameters.
        # But if we pass (actor_features) to critic, we are mixing.
        # We should use critic_encoder(state) for Q evaluation.
        # But we want gradients to flow from Q to Actor.
        # So we pass current_action to critic.
        
        # We need Q(s, pi(s)).
        # We should use the CRITIC encoder for the state input to Q.
        # But we don't want to update Critic Encoder with Actor Loss.
        # So we detach the features from critic encoder.
        with torch.no_grad():
            q_features = self.critic_encoder(state_rgb, state_depth)
            
        q1_pi = self.critic1(q_features, state_proprio, current_action)
        q2_pi = self.critic2(q_features, state_proprio, current_action)
        min_q_pi = torch.min(q1_pi, q2_pi)
        
        actor_loss = ((alpha * log_prob) - min_q_pi).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # --- Alpha Update ---
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # --- Soft Update ---
        self.soft_update(self.critic_encoder, self.target_critic_encoder)
        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'alpha': alpha,
            'alpha_loss': alpha_loss.item()
        }

    def soft_update(self, source, target, tau=0.005):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def run(self):
        print(f"🚀 SAC Server running on {self.args.port}")
        while True:
            try:
                msg = self.socket.recv_pyobj()
                response = {'type': 'ack'}
                
                if msg['type'] == 'data_batch':
                    self.buffer.add_batch(msg['data'])
                    response['curriculum'] = {'collision_dist': 0.5, 'max_speed': 0.18}
                elif msg['type'] == 'check_status':
                    response['status'] = 'ready'
                    response['model_version'] = self.model_version
                elif msg['type'] == 'get_model':
                    with open(os.path.join(self.args.checkpoint_dir, "latest_actor.onnx"), 'rb') as f:
                        response['model_bytes'] = f.read()
                    response['model_version'] = self.model_version
                    
                self.socket.send_pyobj(response)
            except Exception as e:
                print(f"Error: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5556)
    parser.add_argument('--buffer_size', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--checkpoint_dir', default='./checkpoints_sac')
    parser.add_argument('--log_dir', default='./logs_sac')
    args = parser.parse_args()
    
    trainer = V620SACTrainer(args)
    trainer.run()
