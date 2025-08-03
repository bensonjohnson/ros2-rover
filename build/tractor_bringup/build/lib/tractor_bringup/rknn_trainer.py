#!/usr/bin/env python3
"""
RKNN Training System for Autonomous Rover Exploration
Implements real-time learning with multi-modal sensor fusion
"""

import numpy as np
import time
import os
import pickle
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import cv2

try:
    from rknn.api import RKNN
    RKNN_AVAILABLE = True
except ImportError:
    RKNN_AVAILABLE = False

class MultiModalExplorationNet(nn.Module):
    """
    Neural network for multi-modal rover exploration (NO RGB)
    Inputs: point cloud, IMU data, proprioceptive data
    Outputs: Linear velocity, angular velocity, exploration confidence
    """
    
    def __init__(self):
        super().__init__()
        
        # Point cloud branch (PointNet-like) - Enhanced since no RGB
        self.pc_conv = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 512, 1),  # More capacity without RGB
            nn.ReLU()
        )
        self.pc_fc = nn.Linear(512, 512)
        
        # IMU + Proprioceptive branch - Enhanced
        self.sensor_fc = nn.Sequential(
            nn.Linear(10, 128),  # 6 IMU + 4 proprioceptive
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Fusion and output - Point cloud + sensors only
        self.fusion = nn.Sequential(
            nn.Linear(512 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # [linear_vel, angular_vel, confidence]
        )
        
    def forward(self, pointcloud, sensor_data):
        # Point cloud processing (global max pooling)
        pc_features = self.pc_conv(pointcloud)
        pc_features = torch.max(pc_features, dim=2)[0]
        pc_out = self.pc_fc(pc_features)
        
        # Sensor processing
        sensor_out = self.sensor_fc(sensor_data)
        
        # Fusion
        fused = torch.cat([pc_out, sensor_out], dim=1)
        output = self.fusion(fused)
        
        return output

class RKNNTrainer:
    """
    Handles model training, data collection, and RKNN conversion
    """
    
    def __init__(self, model_dir="/home/ubuntu/ros2-rover/models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Neural network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MultiModalExplorationNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # Experience replay buffer
        self.buffer_size = 10000
        self.experience_buffer = deque(maxlen=self.buffer_size)
        self.batch_size = 32
        
        # Training state
        self.training_step = 0
        self.last_save_time = time.time()
        self.save_interval = 300  # Save every 5 minutes
        self.last_rknn_conversion = 0
        self.rknn_conversion_interval = 1800  # Convert to RKNN every 30 minutes
        
        # Data collection
        self.reward_history = deque(maxlen=1000)
        self.action_history = deque(maxlen=100)
        
        # Load existing model if available
        self.load_latest_model()
        
    def add_experience(self, 
                      pointcloud: np.ndarray, 
                      imu_data: np.ndarray,
                      proprioceptive: np.ndarray,
                      action: np.ndarray,
                      reward: float,
                      next_pointcloud: np.ndarray = None,
                      done: bool = False):
        """Add experience to replay buffer"""
        
        experience = {
            'pointcloud': pointcloud.copy(),
            'imu': imu_data.copy(),
            'proprioceptive': proprioceptive.copy(),
            'action': action.copy(),
            'reward': reward,
            'next_pointcloud': next_pointcloud.copy() if next_pointcloud is not None else None,
            'done': done
        }
        
        self.experience_buffer.append(experience)
        self.reward_history.append(reward)
        
    def calculate_reward(self, 
                        action: np.ndarray,
                        collision: bool,
                        progress: float,
                        exploration_bonus: float) -> float:
        """Calculate reward for reinforcement learning"""
        
        reward = 0.0
        
        # Progress reward (forward movement is good)
        reward += progress * 10.0
        
        # Collision penalty
        if collision:
            reward -= 50.0
            
        # Exploration bonus (new areas are good)
        reward += exploration_bonus * 5.0
        
        # Smooth control reward (avoid jerky movements)
        if len(self.action_history) > 0:
            last_action = self.action_history[-1]
            action_smoothness = -np.linalg.norm(action - last_action) * 2.0
            reward += action_smoothness
            
        # Speed efficiency (not too slow, not too fast)
        speed = abs(action[0])
        if 0.05 < speed < 0.3:
            reward += 2.0
        elif speed < 0.02:
            reward -= 5.0  # Penalize being too slow
            
        self.action_history.append(action)
        return reward
        
    def train_step(self) -> Dict[str, float]:
        """Perform one training step if enough data available"""
        
        if len(self.experience_buffer) < self.batch_size:
            return {"loss": 0.0, "samples": len(self.experience_buffer)}
            
        # Sample random batch
        indices = np.random.choice(len(self.experience_buffer), self.batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in indices]
        
        # Prepare batch data
        pc_batch = torch.FloatTensor([exp['pointcloud'] for exp in batch]).to(self.device)
        sensor_batch = torch.FloatTensor([
            np.concatenate([exp['imu'], exp['proprioceptive']]) 
            for exp in batch
        ]).to(self.device)
        action_batch = torch.FloatTensor([exp['action'] for exp in batch]).to(self.device)
        reward_batch = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)
        
        # Forward pass
        predicted_actions = self.model(pc_batch, sensor_batch)
        
        # Calculate target actions with reward weighting
        target_actions = action_batch.clone()
        confidence_weight = torch.sigmoid(reward_batch).unsqueeze(1)
        target_actions = target_actions * confidence_weight
        
        # Loss calculation
        action_loss = self.criterion(predicted_actions[:, :2], target_actions)
        confidence_loss = self.criterion(predicted_actions[:, 2], torch.sigmoid(reward_batch))
        total_loss = action_loss + 0.1 * confidence_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        self.training_step += 1
        
        # Periodic saving
        current_time = time.time()
        if current_time - self.last_save_time > self.save_interval:
            self.save_model()
            self.last_save_time = current_time
            
        # Periodic RKNN conversion
        if current_time - self.last_rknn_conversion > self.rknn_conversion_interval:
            self.convert_to_rknn()
            self.last_rknn_conversion = current_time
            
        return {
            "loss": total_loss.item(),
            "action_loss": action_loss.item(),
            "confidence_loss": confidence_loss.item(),
            "samples": len(self.experience_buffer),
            "training_steps": self.training_step,
            "avg_reward": np.mean(self.reward_history) if self.reward_history else 0.0
        }
        
    def inference(self, 
                  pointcloud: np.ndarray,
                  imu_data: np.ndarray,
                  proprioceptive: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run inference to get action and confidence"""
        
        self.model.eval()
        with torch.no_grad():
            # Prepare inputs
            pc_tensor = torch.FloatTensor(pointcloud).unsqueeze(0).to(self.device)
            sensor_tensor = torch.FloatTensor(
                np.concatenate([imu_data, proprioceptive])
            ).unsqueeze(0).to(self.device)
            
            # Forward pass
            output = self.model(pc_tensor, sensor_tensor)
            
            # Extract action and confidence
            action = output[0, :2].cpu().numpy()
            confidence = torch.sigmoid(output[0, 2]).cpu().numpy()
            
        self.model.train()
        return action, float(confidence)
        
    def save_model(self):
        """Save PyTorch model and training state"""
        
        timestamp = int(time.time())
        
        # Save PyTorch model
        model_path = os.path.join(self.model_dir, f"exploration_model_{timestamp}.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'buffer_size': len(self.experience_buffer)
        }, model_path)
        
        # Save latest symlink
        latest_path = os.path.join(self.model_dir, "exploration_model_latest.pth")
        if os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(model_path, latest_path)
        
        print(f"Model saved: {model_path}")
        
    def load_latest_model(self):
        """Load the latest saved model"""
        
        latest_path = os.path.join(self.model_dir, "exploration_model_latest.pth")
        
        if os.path.exists(latest_path):
            try:
                checkpoint = torch.load(latest_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.training_step = checkpoint.get('training_step', 0)
                print(f"Loaded model from {latest_path}")
                print(f"Training steps: {self.training_step}")
            except Exception as e:
                print(f"Failed to load model: {e}")
                
    def convert_to_rknn(self):
        """Convert PyTorch model to RKNN format for NPU inference"""
        
        if not RKNN_AVAILABLE:
            print("RKNN not available - skipping conversion")
            return
            
        try:
            # Export to ONNX first
            dummy_pc = torch.randn(1, 3, 512).to(self.device)
            dummy_sensor = torch.randn(1, 10).to(self.device)
            
            onnx_path = os.path.join(self.model_dir, "exploration_model.onnx")
            
            self.model.eval()
            torch.onnx.export(
                self.model,
                (dummy_pc, dummy_sensor),
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['pointcloud', 'sensor'],
                output_names=['action_confidence']
            )
            
            # Convert ONNX to RKNN
            rknn = RKNN(verbose=False)
            rknn.config(
                mean_values=[[123.675, 116.28, 103.53], [0], [0]],
                std_values=[[58.395, 57.12, 57.375], [1], [1]],
                target_platform='rk3588'
            )
            
            rknn.load_onnx(model=onnx_path)
            rknn.build(do_quantization=True, dataset='./dataset.txt')
            
            rknn_path = os.path.join(self.model_dir, "exploration_model.rknn")
            rknn.export_rknn(rknn_path)
            rknn.release()
            
            print(f"RKNN model saved: {rknn_path}")
            
        except Exception as e:
            print(f"RKNN conversion failed: {e}")
            
    def get_training_stats(self) -> Dict[str, float]:
        """Get current training statistics"""
        
        return {
            "training_steps": self.training_step,
            "buffer_size": len(self.experience_buffer),
            "avg_reward": np.mean(self.reward_history) if self.reward_history else 0.0,
            "buffer_full": len(self.experience_buffer) / self.buffer_size
        }