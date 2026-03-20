#!/usr/bin/env python3
"""PPO BEV Inference for NPU Deployment.

This script loads an RKNN model and performs inference for autonomous
rover navigation using the Unified BEV architecture.

Usage:
    python bev_inference.py --rknn checkpoints_ppo/latest_actor.rknn
"""

import os
import sys
import argparse
import time
import numpy as np
from typing import Tuple, Optional

# Try to import RKNN runtime
try:
    from rknn.runtime import RKNN, rtConfig
    HAS_RKNN = True
except ImportError:
    HAS_RKNN = False
    print("⚠ RKNN runtime not available - will use PyTorch fallback")

import torch
import torch.nn.functional as F

# Import model architecture
from model_architectures import UnifiedBEVPPOPolicy


class BEVInferenceEngine:
    """Inference engine for PPO BEV policy."""
    
    def __init__(self, model_path: str, use_rknn: bool = True, device: str = 'cpu'):
        """Initialize inference engine.
        
        Args:
            model_path: Path to .rknn or .onnx or .pt model
            use_rknn: Whether to use RKNN NPU (if available)
            device: Device for PyTorch fallback
        """
        self.model_path = model_path
        self.use_rknn = use_rknn and HAS_RKNN
        self.device = device
        self.rknn_model = None
        
        # Check model type
        if model_path.endswith('.rknn') and self.use_rkNN:
            self._load_rknn()
        elif model_path.endswith('.onnx'):
            self._load_onnx()
        else:
            self._load_pytorch()
    
    def _load_rknn(self):
        """Load RKNN model for NPU inference."""
        print(f"🔄 Loading RKNN model: {self.model_path}")
        
        self.rknn_model = RKNN()
        self.rknn_model.load_rknn(self.model_path)
        self.rknn_model.init_runtime()
        
        print("✅ RKNN model loaded successfully")
    
    def _load_onnx(self):
        """Load ONNX model."""
        try:
            import onnxruntime as ort
            self.ort_session = ort.InferenceSession(self.model_path)
            print(f"✅ ONNX model loaded: {self.model_path}")
        except ImportError:
            print("❌ onnxruntime not available!")
            print("   Install with: pip install onnxruntime")
            sys.exit(1)
    
    def _load_pytorch(self):
        """Load PyTorch model."""
        print(f"🔄 Loading PyTorch model: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        policy_state_dict = checkpoint.get('policy_state_dict', checkpoint)
        
        # Create policy
        self.policy = UnifiedBEVPPOPolicy(action_dim=2, proprio_dim=6)
        self.policy.load_state_dict(policy_state_dict)
        self.policy.to(self.device)
        self.policy.eval()
        
        print(f"✅ PyTorch model loaded")
    
    def infer(self, bev: np.ndarray, proprio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference to get action.
        
        Args:
            bev: (2, 128, 128) BEV grid (float 0-1)
            proprio: (6,) proprioception values
            
        Returns:
            action: (2,) [linear_vel, angular_vel]
            log_std: (2,) action log std
        """
        if self.use_rknn and self.rknn_model:
            return self._infer_rknn(bev, proprio)
        elif hasattr(self, 'ort_session'):
            return self._infer_onnx(bev, proprio)
        else:
            return self._infer_pytorch(bev, proprio)
    
    def _infer_rknn(self, bev: np.ndarray, proprio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference on RKNN NPU."""
        # Prepare inputs
        inputs = [
            {'name': 'bev', 'data': bev.astype(np.float32)},
            {'name': 'proprio', 'data': proprio.astype(np.float32)}
        ]
        
        # Run inference
        outputs = self.rknn_model.infer(inputs=inputs)
        
        # Extract outputs
        action_mean = outputs['action_mean'][0]
        log_std = outputs['log_std'][0]
        
        # Apply tanh to get action in [-1, 1]
        action = np.tanh(action_mean)
        
        return action, log_std
    
    def _infer_onnx(self, bev: np.ndarray, proprio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference on ONNX Runtime."""
        # Prepare inputs
        inputs = {
            'bev': bev.astype(np.float32).reshape(1, 2, 128, 128),
            'proprio': proprio.astype(np.float32).reshape(1, 6)
        }
        
        # Run inference
        outputs = self.ort_session.run(None, inputs)
        
        action_mean = outputs[0][0]
        log_std = outputs[1][0]
        
        # Apply tanh
        action = np.tanh(action_mean)
        
        return action, log_std
    
    def _infer_pytorch(self, bev: np.ndarray, proprio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference on PyTorch."""
        with torch.no_grad():
            # Convert to tensors
            bev_tensor = torch.from_numpy(bev).float().unsqueeze(0).to(self.device)
            proprio_tensor = torch.from_numpy(proprio).float().unsqueeze(0).to(self.device)
            
            # Forward pass
            action_mean, log_std, _ = self.policy(bev_tensor, proprio_tensor)
            
            # Apply tanh
            action = torch.tanh(action_mean).cpu().numpy()[0]
            log_std = log_std.cpu().numpy()[0]
        
        return action, log_std
    
    def release(self):
        """Release resources."""
        if self.rknn_model:
            self.rknn_model.release()


class RoverController:
    """Simple rover controller using BEV inference."""
    
    def __init__(self, model_path: str, max_speed: float = 0.18):
        """Initialize controller.
        
        Args:
            model_path: Path to model file
            max_speed: Maximum linear speed (m/s)
        """
        self.engine = BEVInferenceEngine(model_path)
        self.max_speed = max_speed
        
        # State
        self.prev_linear = 0.0
        self.prev_angular = 0.0
        
    def get_action(self, bev: np.ndarray, proprio: np.ndarray) -> Tuple[float, float]:
        """Get control action from policy.
        
        Args:
            bev: (2, 128, 128) BEV grid
            proprio: (6,) proprioception
            
        Returns:
            linear_vel: Linear velocity (m/s)
            angular_vel: Angular velocity (rad/s)
        """
        action, _ = self.engine.infer(bev, proprio)
        
        # Scale actions to physical units
        # Action space: [-1, 1] for both linear and angular
        linear_vel = action[0] * self.max_speed
        angular_vel = action[1] * 2.0  # Max angular: ±2.0 rad/s
        
        # Smooth transitions (optional)
        linear_vel = 0.9 * self.prev_linear + 0.1 * linear_vel
        angular_vel = 0.9 * self.prev_angular + 0.1 * angular_vel
        
        self.prev_linear = linear_vel
        self.prev_angular = angular_vel
        
        return linear_vel, angular_vel
    
    def release(self):
        """Release resources."""
        self.engine.release()


def demo_inference():
    """Demo inference with random data."""
    print("🧪 PPO BEV Inference Demo")
    print("="*50)
    
    # Load model
    model_path = "checkpoints_ppo/latest_actor.rknn"
    if not os.path.exists(model_path):
        # Try PyTorch fallback
        model_path = "checkpoints_ppo/latest_actor.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("   Please train a model first or provide --rknn path")
        return
    
    controller = RoverController(model_path)
    
    # Simulate inference loop
    print("\n📊 Running inference demo...")
    print("   Generating random BEV and proprioception...")
    
    bev = np.random.rand(2, 128, 128).astype(np.float32)
    proprio = np.random.rand(6).astype(np.float32)
    
    # Warmup
    for _ in range(5):
        controller.get_action(bev, proprio)
    
    # Timing
    print("\n⏱️  Inference timing (100 iterations):")
    times = []
    for i in range(100):
        start = time.time()
        linear, angular = controller.get_action(bev, proprio)
        elapsed = time.time() - start
        times.append(elapsed)
    
    avg_time = np.mean(times) * 1000  # ms
    fps = 1.0 / np.mean(times)
    
    print(f"   Average inference time: {avg_time:.2f} ms")
    print(f"   Inference FPS: {fps:.1f}")
    print(f"   Sample action: linear={linear:.3f}, angular={angular:.3f}")
    
    controller.release()
    print("\n✅ Demo complete!")


def main():
    parser = argparse.ArgumentParser(description='PPO BEV Inference')
    parser.add_argument('--rknn', type=str, help='Path to RKNN model')
    parser.add_argument('--onnx', type=str, help='Path to ONNX model')
    parser.add_argument('--pt', type=str, help='Path to PyTorch checkpoint')
    parser.add_argument('--max_speed', type=float, default=0.18, help='Max linear speed (m/s)')
    parser.add_argument('--demo', action='store_true', help='Run demo inference')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_inference()
        return
    
    # Determine model path
    model_path = None
    if args.rknn:
        model_path = args.rknn
    elif args.onnx:
        model_path = args.onnx
    elif args.pt:
        model_path = args.pt
    else:
        # Default path
        model_path = "checkpoints_ppo/latest_actor.rknn"
        if not os.path.exists(model_path):
            model_path = "checkpoints_ppo/latest_actor.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)
    
    print(f"🚀 PPO BEV Inference Engine")
    print(f"   Model: {model_path}")
    print(f"   Max speed: {args.max_speed} m/s")
    print("="*50)
    
    # Create controller
    controller = RoverController(model_path, max_speed=args.max_speed)
    
    print("\n✅ Inference engine ready!")
    print("   Use controller.get_action(bev, proprio) to get actions")
    print("   Press Ctrl+C to exit")
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        controller.release()
        print("✅ Done")


if __name__ == '__main__':
    main()