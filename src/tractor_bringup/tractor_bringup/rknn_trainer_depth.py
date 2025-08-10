#!/usr/bin/env python3
"""
RKNN Training System for Autonomous Rover Exploration (Depth Image Version)
Implements real-time learning with multi-modal sensor fusion using depth images
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

try:
    from .improved_reward_system import ImprovedRewardCalculator
    IMPROVED_REWARDS = True
except ImportError:
    IMPROVED_REWARDS = False
    print("Improved reward system not available - using basic rewards")

class DepthImageExplorationNet(nn.Module):
    """
    Neural network for depth image-based rover exploration
    Inputs: depth image, IMU data, proprioceptive data
    Outputs: Linear velocity, angular velocity, exploration confidence
    """
    
    def __init__(self, stacked_frames: int = 1, extra_proprio: int = 0):
        super().__init__()
        self.stacked_frames = stacked_frames
        in_channels = stacked_frames  # depth frames stacked along channel dim
        # Depth image branch (CNN)
        self.depth_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(3, 3), stride=(3, 3)),
            nn.Flatten()
        )
        # After conv stack with input 160x288 -> sizes: /2=80x144 /2=40x72 /2=20x36 /2=10x18 then pool /3 -> 3x6
        self.depth_fc = nn.Linear(256 * 3 * 6, 512)
        proprio_inputs = 3 + extra_proprio  # base + added features
        self.sensor_fc = nn.Sequential(
            nn.Linear(proprio_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.fusion = nn.Sequential(
            nn.Linear(512 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )
        
    def forward(self, depth_image, sensor_data):
        # Depth image processing
        depth_features = self.depth_conv(depth_image)
        depth_out = self.depth_fc(depth_features)
        
        # Sensor processing
        sensor_out = self.sensor_fc(sensor_data)
        
        # Fusion
        fused = torch.cat([depth_out, sensor_out], dim=1)
        output = self.fusion(fused)
        
        return output

class RKNNTrainerDepth:
    """
    Handles model training, data collection, and RKNN conversion for depth images
    """
    
    def __init__(self, model_dir="models", stacked_frames: int = 1, enable_debug: bool = False):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.enable_debug = enable_debug
        self.target_h = 160
        self.target_w = 288
        self.clip_max_distance = 4.0
        self.stacked_frames = stacked_frames
        self.frame_stack: deque = deque(maxlen=stacked_frames)
        # Extended proprio feature count: last_action(2) + wheel_diff(1) + min_d(1) + mean_d(1) + near_collision(1) = 6
        self.extra_proprio = 6
        # Neural network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DepthImageExplorationNet(stacked_frames=stacked_frames, extra_proprio=self.extra_proprio).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # Experience replay buffer
        self.buffer_size = 10000
        self.experience_buffer = deque(maxlen=self.buffer_size)
        self.batch_size = 32
        
        # Training state
        self.training_step = 0
        self.last_save_time = time.time()
        self.save_interval = 300  # Save every 5 minutes (time-based)
        self.step_save_interval = 250  # NEW: fallback checkpoint every N steps
        self.last_rknn_conversion = 0
        self.rknn_conversion_interval = 300  # hard max interval (seconds) (was 1800)
        # Metric-gated export parameters
        self.rknn_min_interval = 600  # minimum seconds between exports if improvement seen
        self.rknn_loss_improve_ratio = 0.90  # export if median loss improved by >=10%
        self.rknn_reward_improve_delta = 0.5  # or avg reward improved by this amount
        self.loss_history = deque(maxlen=200)
        self.last_export_loss = None
        self.last_export_avg_reward = None
        
        # Data collection
        self.reward_history = deque(maxlen=1000)
        self.action_history = deque(maxlen=100)
        
        # Initialize improved reward calculator if available
        if IMPROVED_REWARDS:
            self.reward_calculator = ImprovedRewardCalculator()
            print("Using improved reward system")
        else:
            self.reward_calculator = None
            print("Using basic reward system")
        
        # Load existing model if available
        self.load_latest_model()
        
    def add_experience(self,
                      depth_image: np.ndarray,
                      proprioceptive: np.ndarray,
                      action: np.ndarray,
                      reward: float,
                      next_depth_image: np.ndarray = None,
                      done: bool = False):
        """Add experience to replay buffer"""

        # Preprocess depth images into (C,H,W)
        processed = depth_image if depth_image.ndim == 3 else self.preprocess_depth_for_storage(depth_image)
        processed_next = None
        if next_depth_image is not None:
            processed_next = next_depth_image if next_depth_image.ndim == 3 else self.preprocess_depth_for_storage(next_depth_image)
        experience = {
            'depth_image': processed.copy(),
            'proprioceptive': proprioceptive.copy(),
            'action': action.copy(),
            'reward': reward,
            'next_depth_image': processed_next.copy() if processed_next is not None else None,
            'done': done
        }
        
        self.experience_buffer.append(experience)
        self.reward_history.append(reward)
        
    def calculate_reward(self, 
                        action: np.ndarray,
                        collision: bool,
                        progress: float,
                        exploration_bonus: float,
                        position: Optional[np.ndarray] = None,
                        depth_data: Optional[np.ndarray] = None,
                        wheel_velocities: Optional[Tuple[float, float]] = None) -> float:
        """Calculate reward for reinforcement learning"""
        
        # Use improved reward system if available
        if self.reward_calculator is not None and position is not None:
            # Use comprehensive reward calculation
            near_collision = False  # You can enhance this based on depth data
            if depth_data is not None:
                try:
                    valid_depths = depth_data[(depth_data > 0.1) & (depth_data < 5.0)]
                    if len(valid_depths) > 0:
                        min_distance = np.min(valid_depths)
                        near_collision = 0.3 < min_distance < 0.5  # Close but not colliding
                except:
                    pass
            
            total_reward, reward_breakdown = self.reward_calculator.calculate_comprehensive_reward(
                action=action,
                position=position,
                collision=collision,
                near_collision=near_collision,
                progress=progress,
                depth_data=depth_data,
                wheel_velocities=wheel_velocities
            )
            
            # Log reward breakdown occasionally for debugging
            if self.training_step % 100 == 0:
                print(f"Reward breakdown: {reward_breakdown}")
                
            return total_reward
        
        # Fallback to basic reward system
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
        
        # Detect legacy shape and purge if necessary
        if any(exp['depth_image'].ndim != 3 for exp in batch):
            self.experience_buffer.clear()
            return {"loss": 0.0, "samples": 0}
        
        # Prepare batch data
        depth_batch = torch.FloatTensor(np.array([exp['depth_image'] for exp in batch]))  # (B,C,H,W)
        sensor_batch = torch.FloatTensor(np.array([exp['proprioceptive'] for exp in batch]))
        action_batch = torch.FloatTensor(np.array([exp['action'] for exp in batch]))
        reward_batch = torch.FloatTensor(np.array([exp['reward'] for exp in batch]))
        depth_batch = depth_batch.to(self.device)
        sensor_batch = sensor_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        
        # Forward pass
        predicted_actions = self.model(depth_batch, sensor_batch)
        
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
        # Track loss history
        self.loss_history.append(float(total_loss.item()))
        
        self.training_step += 1
        
        # NEW: Step-based periodic saving (fallback for early training before metrics improve)
        if self.step_save_interval > 0 and (self.training_step % self.step_save_interval == 0):
            self.save_model()
            self.last_save_time = time.time()  # Sync last_save_time to avoid immediate time-based save
        
        # Periodic saving (time-based)
        current_time = time.time()
        if current_time - self.last_save_time > self.save_interval:
            self.save_model()
            self.last_save_time = current_time
        # Metric-gated RKNN conversion
        force_due_interval = (current_time - self.last_rknn_conversion) > self.rknn_conversion_interval
        can_check_improvement = (current_time - self.last_rknn_conversion) > self.rknn_min_interval
        median_loss = float(np.median(self.loss_history)) if self.loss_history else None
        avg_reward = float(np.mean(self.reward_history)) if self.reward_history else 0.0
        improved = False
        if can_check_improvement and median_loss is not None:
            loss_improved = (self.last_export_loss is None) or (median_loss < self.last_export_loss * self.rknn_loss_improve_ratio)
            reward_improved = (self.last_export_avg_reward is None) or (avg_reward > self.last_export_avg_reward + self.rknn_reward_improve_delta)
            if loss_improved or reward_improved:
                improved = True
        if (improved or force_due_interval) and median_loss is not None:
            try:
                self.convert_to_rknn()
                self.last_rknn_conversion = current_time
                self.last_export_loss = median_loss
                self.last_export_avg_reward = avg_reward
                if self.enable_debug:
                    print(f"RKNN export triggered (improved={improved}, force={force_due_interval}) median_loss={median_loss:.4f} avg_reward={avg_reward:.2f}")
            except Exception as e:
                print(f"RKNN export attempt failed: {e}")
        return {
            "loss": total_loss.item(),
            "action_loss": action_loss.item(),
            "confidence_loss": confidence_loss.item(),
            "samples": len(self.experience_buffer),
            "training_steps": self.training_step,
            "avg_reward": avg_reward,
            "median_loss": median_loss if median_loss is not None else 0.0
        }
        
    def preprocess_depth_for_model(self, depth_image: np.ndarray) -> np.ndarray:
        """Normalize, resize, clip and optionally stack frames.
        Input depth_image: (H,W) meters.
        Returns (stacked_frames, target_h, target_w) float32 in [0,1]."""
        try:
            # Clip far ranges to reduce dynamic range then normalize
            depth = np.nan_to_num(depth_image, nan=0.0, posinf=self.clip_max_distance, neginf=0.0)
            depth = np.clip(depth, 0.0, self.clip_max_distance)
            # Resize to target
            depth_resized = cv2.resize(depth, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)
            # Scale to [0,1]
            depth_norm = depth_resized / self.clip_max_distance
            # Append to stack
            self.frame_stack.append(depth_norm.astype(np.float32))
            # If stack not full yet, pad with copies of first
            while len(self.frame_stack) < self.stacked_frames:
                self.frame_stack.append(self.frame_stack[0])
            stacked = np.stack(list(self.frame_stack), axis=0)  # (C,H,W)
            return stacked
        except Exception:
            # Fallback zero tensor
            return np.zeros((self.stacked_frames, self.target_h, self.target_w), dtype=np.float32)
    
    def preprocess_depth_for_storage(self, depth_image: np.ndarray) -> np.ndarray:
        """Preprocess without mutating frame stack (single frame replicated)."""
        try:
            depth = np.nan_to_num(depth_image, nan=0.0, posinf=self.clip_max_distance, neginf=0.0)
            depth = np.clip(depth, 0.0, self.clip_max_distance)
            depth_resized = cv2.resize(depth, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)
            depth_norm = depth_resized / self.clip_max_distance
            if self.stacked_frames == 1:
                return depth_norm.astype(np.float32)[None, ...]
            else:
                return np.repeat(depth_norm[None, ...], self.stacked_frames, axis=0)
        except Exception:
            return np.zeros((self.stacked_frames, self.target_h, self.target_w), dtype=np.float32)
        
    def inference(self, depth_image: np.ndarray, proprioceptive: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run inference to get action and confidence (Phase1: still returns 3 outputs)."""
        self.model.eval()
        with torch.no_grad():
            processed = self.preprocess_depth_for_model(depth_image)
            depth_tensor = torch.from_numpy(processed).unsqueeze(0).to(self.device)
            sensor_tensor = torch.FloatTensor(proprioceptive).unsqueeze(0).to(self.device)
            output = self.model(depth_tensor, sensor_tensor)
            # Apply tanh squashing to first two outputs for bounded actions
            action = torch.tanh(output[0, :2]).cpu().numpy()
            confidence = torch.sigmoid(output[0, 2]).item()
        self.model.train()
        return action, float(confidence)
        
    def save_model(self):
        """Save PyTorch model and training state"""
        try:
            # Ensure model directory exists
            os.makedirs(self.model_dir, exist_ok=True)
            
            timestamp = int(time.time())
            
            # Save PyTorch model
            model_path = os.path.join(self.model_dir, f"exploration_model_depth_{timestamp}.pth")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_step': self.training_step,
                'buffer_size': len(self.experience_buffer)
            }, model_path)
            
            # Save latest symlink
            latest_path = os.path.join(self.model_dir, "exploration_model_depth_latest.pth")
            if os.path.exists(latest_path):
                os.remove(latest_path)
            os.symlink(os.path.basename(model_path), latest_path)
            
            print(f"Model saved: {model_path}")
        except Exception as e:
            print(f"Failed to save model: {e}")
        
    def load_latest_model(self):
        """Load the latest saved model"""
        try:
            latest_path = os.path.join(self.model_dir, "exploration_model_depth_latest.pth")
            
            if os.path.exists(latest_path) and os.path.islink(latest_path):
                # Resolve symlink to actual file
                actual_model_path = os.path.join(self.model_dir, os.readlink(latest_path))
                if os.path.exists(actual_model_path):
                    checkpoint = torch.load(actual_model_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.training_step = checkpoint.get('training_step', 0)
                    print(f"Loaded model from {actual_model_path}")
                    print(f"Training steps: {self.training_step}")
                else:
                    print(f"Model file {actual_model_path} not found")
            elif os.path.exists(latest_path):
                # Direct file (not symlink)
                checkpoint = torch.load(latest_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.training_step = checkpoint.get('training_step', 0)
                print(f"Loaded model from {latest_path}")
                print(f"Training steps: {self.training_step}")
            else:
                print("No saved model found, starting with fresh model")
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Starting with fresh model")
            
    def _proprio_feature_size(self):
        """Return proprioceptive feature size consistent with network construction.
        Matches 'proprio_inputs = 3 + extra_proprio' used in __init__."""
        return 3 + self.extra_proprio

    def convert_to_rknn(self):
        """Convert PyTorch model to RKNN format for NPU inference (fixed input shapes) with detailed debug."""
        if not RKNN_AVAILABLE:
            print("RKNN not available - skipping conversion")
            # Still export ONNX for inspection
            try:
                onnx_path = os.path.join(self.model_dir, "exploration_model_depth.onnx")
                dummy_depth = torch.randn(1, self.stacked_frames, self.target_h, self.target_w).to(self.device)
                dummy_sensor = torch.randn(1, self._proprio_feature_size()).to(self.device)
                torch.onnx.export(
                    self.model,
                    (dummy_depth, dummy_sensor),
                    onnx_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['depth_image', 'sensor'],
                    output_names=['action_confidence'],
                    dynamic_axes=None
                )
                print(f"ONNX exported (without RKNN toolkit): {onnx_path}")
            except Exception as e:
                print(f"ONNX export failed (no RKNN toolkit): {e}")
            return
        try:
            if self.enable_debug:
                print("[RKNN] Starting conversion pipeline")
            os.makedirs(self.model_dir, exist_ok=True)
            dummy_depth = torch.randn(1, self.stacked_frames, self.target_h, self.target_w).to(self.device)
            proprio_size = self._proprio_feature_size()
            dummy_sensor = torch.randn(1, proprio_size).to(self.device)
            if self.enable_debug:
                print(f"[RKNN] Dummy tensors created depth_shape={tuple(dummy_depth.shape)} sensor_shape={tuple(dummy_sensor.shape)}")
            # Sanity check flatten size
            with torch.no_grad():
                self.model.eval()
                depth_features = self.model.depth_conv(dummy_depth)
                # depth_conv ends with Flatten(), so depth_features is (B, F)
                flat_dim = depth_features.shape[1]
                expected = self.model.depth_fc.in_features
                if self.enable_debug:
                    print(f"[RKNN] Depth feature flatten size={flat_dim} expected={expected}")
                if flat_dim != expected:
                    print(f"[RKNN Export] WARNING: depth flatten size {flat_dim} != expected {expected}. Aborting export.")
                    self.model.train()
                    return
            onnx_path = os.path.join(self.model_dir, "exploration_model_depth.onnx")
            if self.enable_debug:
                print(f"[RKNN] Exporting ONNX to {onnx_path}")
            torch.onnx.export(
                self.model,
                (dummy_depth, dummy_sensor),
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['depth_image', 'sensor'],
                output_names=['action_confidence'],
                dynamic_axes=None
            )
            if self.enable_debug:
                print("[RKNN] ONNX export complete")
            dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'dataset.txt')
            dataset_path = os.path.abspath(dataset_path)
            if self.enable_debug:
                print(f"[RKNN] Dataset path: {dataset_path}")
            rknn = RKNN(verbose=self.enable_debug)
            # First try simplest config (some toolkit versions choke on multi-input mean/std lists)
            try:
                if self.enable_debug:
                    print("[RKNN] Calling config (simple)")
                rknn.config(target_platform='rk3588')
            except Exception as e:
                print(f"[RKNN] Simple config failed: {e}. Trying depth-only mean/std.")
                try:
                    mean_values = [[0.0] * self.stacked_frames]
                    std_values = [[1.0] * self.stacked_frames]
                    rknn.config(mean_values=mean_values, std_values=std_values, target_platform='rk3588')
                except Exception as e2:
                    print(f"[RKNN] Depth-only config failed: {e2}. Aborting.")
                    rknn.release()
                    self.model.train()
                    return
            if self.enable_debug:
                print("[RKNN] Loading ONNX model")
            ret = rknn.load_onnx(model=onnx_path)
            if ret != 0:
                print("[RKNN] Failed to load ONNX model (ret != 0)")
                rknn.release()
                self.model.train()
                return
            if self.enable_debug:
                print("[RKNN] ONNX loaded successfully")
            do_quantization = False
            if os.path.exists(dataset_path):
                try:
                    with open(dataset_path, 'r') as f:
                        if f.read().strip():
                            do_quantization = True
                except Exception as e:
                    print(f"[RKNN] Error reading dataset file: {e}")
            if self.enable_debug:
                print(f"[RKNN] Building (quantization={do_quantization})")
            if do_quantization:
                ret = rknn.build(do_quantization=True, dataset=dataset_path)
            else:
                ret = rknn.build(do_quantization=False)
            if ret != 0:
                print("[RKNN] Failed to build RKNN model (ret != 0)")
                rknn.release()
                self.model.train()
                return
            if self.enable_debug:
                print("[RKNN] Build successful, exporting RKNN")
            rknn_path = os.path.join(self.model_dir, "exploration_model_depth.rknn")
            ret = rknn.export_rknn(rknn_path)
            if ret != 0:
                print("[RKNN] Failed to export RKNN model")
            else:
                print(f"RKNN model saved: {rknn_path}")
            rknn.release()
            if self.enable_debug:
                print("[RKNN] Pipeline complete")
            self.model.train()
        except Exception as e:
            print(f"RKNN conversion failed (exception): {e}")
            import traceback
            traceback.print_exc()
            self.model.train()
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get current training statistics"""
        
        return {
            "training_steps": self.training_step,
            "buffer_size": len(self.experience_buffer),
            "avg_reward": np.mean(self.reward_history) if self.reward_history else 0.0,
            "buffer_full": len(self.experience_buffer) / self.buffer_size
        }
    
    def safe_save(self):
        try:
            self.save_model()
        except Exception as e:
            print(f"Model save failed during shutdown: {e}")
