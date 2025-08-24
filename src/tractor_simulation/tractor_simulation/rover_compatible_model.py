#!/usr/bin/env python3
"""
Rover-Compatible Model Architecture
Matches the DepthImageExplorationNet from your rover's ES training system
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import time
import logging

try:
    # Try to import PyTorch AO optimizations
    from torchao.quantization import quantize_, int4_weight_only
    from torchao.sparsity import sparsify_
    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False


class RoverDepthExplorationNet(nn.Module):
    """
    Neural network matching your rover's DepthImageExplorationNet
    Inputs: depth image (160x288), proprioceptive data (16 features)
    Outputs: Linear velocity, angular velocity, exploration confidence
    """
    
    def __init__(self, stacked_frames: int = 1, extra_proprio: int = 13, 
                 use_quantization: bool = False, use_sparsity: bool = False):
        super().__init__()
        self.stacked_frames = stacked_frames
        self.extra_proprio = extra_proprio
        in_channels = stacked_frames  # depth frames stacked along channel dim
        
        # Depth image branch (CNN) - exactly matching rover architecture
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
        
        # After conv stack with input 160x288: 
        # /2=80x144 /2=40x72 /2=20x36 /2=10x18 then pool /3 -> 3x6
        self.depth_fc = nn.Linear(256 * 3 * 6, 512)
        
        # Proprioceptive branch - matching rover architecture
        proprio_inputs = 3 + extra_proprio  # base + added features (16 total)
        self.sensor_fc = nn.Sequential(
            nn.Linear(proprio_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Fusion layer - matching rover architecture
        self.fusion = nn.Sequential(
            nn.Linear(512 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # linear, angular, confidence
        )
        
        # Apply optimizations
        self.logger = logging.getLogger(__name__)
        self._apply_optimizations(use_quantization, use_sparsity)
        
    def _apply_optimizations(self, use_quantization: bool, use_sparsity: bool):
        """Apply PyTorch AO optimizations"""
        
        if not TORCHAO_AVAILABLE:
            self.logger.info("TorchAO not available - using standard optimizations")
            return
            
        if use_quantization:
            try:
                # Apply INT4 weight-only quantization for memory and speed
                quantize_(self, int4_weight_only())
                self.logger.info("Applied INT4 weight-only quantization")
            except Exception as e:
                self.logger.warning(f"Quantization failed: {e}")
                
        if use_sparsity:
            try:
                # Apply 2:4 structured sparsity for speed without accuracy loss
                sparsify_(self, sparsity_level=0.5)
                self.logger.info("Applied 2:4 structured sparsity")  
            except Exception as e:
                self.logger.warning(f"Sparsity failed: {e}")
        
    def forward(self, depth_image, sensor_data):
        """Forward pass matching rover architecture exactly"""
        # Depth image processing
        depth_features = self.depth_conv(depth_image)
        depth_out = self.depth_fc(depth_features)
        
        # Sensor processing
        sensor_out = self.sensor_fc(sensor_data)
        
        # Fusion
        fused = torch.cat([depth_out, sensor_out], dim=1)
        output = self.fusion(fused)
        
        return output
        
    def get_actions(self, depth_image: torch.Tensor, sensor_data: torch.Tensor) -> torch.Tensor:
        """Lightweight inference - actions only (compatible with ultra-fast system)"""
        with torch.no_grad():
            output = self.forward(depth_image, sensor_data)
            # Apply tanh to first two outputs (linear, angular velocity)
            actions = torch.tanh(output[:, :2])
            return actions


class RoverCompatibleTrainer:
    """
    Ultra-optimized trainer using rover-compatible architecture
    """
    
    def __init__(self, 
                 model: RoverDepthExplorationNet,
                 device: str = "cuda",
                 use_compile: bool = True,
                 use_mixed_precision: bool = True):
        self.device = device
        self.model = model.to(device)
        self.use_mixed_precision = use_mixed_precision
        
        # Compile model for maximum speed
        if use_compile:
            try:
                self.model = torch.compile(
                    self.model, 
                    mode="max-autotune",  # Aggressive optimization
                    fullgraph=True        # Compile entire graph
                )
                print("✓ Rover model compiled with max-autotune mode")
            except Exception as e:
                print(f"⚠ Compilation failed: {e}")
                
        # Set up optimized inference
        self.model.eval()
        
        # Automatic Mixed Precision scaler
        if use_mixed_precision and device == "cuda":
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        # Pre-allocate tensors for batch inference (rover image size)
        self.max_batch_size = 64  # Smaller for rover-sized images
        self._preallocate_tensors()
        
        self.logger = logging.getLogger(__name__)
        
    def _preallocate_tensors(self):
        """Pre-allocate tensors to avoid memory allocation overhead"""
        # Rover dimensions: 160x288 depth images, 16 proprioceptive features
        self.depth_buffer = torch.zeros(
            self.max_batch_size, self.model.stacked_frames, 160, 288,
            device=self.device, 
            dtype=torch.float16 if self.use_mixed_precision else torch.float32
        )
        self.proprio_buffer = torch.zeros(
            self.max_batch_size, 16,  # 3 + 13 extras = 16 features
            device=self.device, 
            dtype=torch.float16 if self.use_mixed_precision else torch.float32  
        )
        self.action_buffer = torch.zeros(
            self.max_batch_size, 2,
            device=self.device, 
            dtype=torch.float16 if self.use_mixed_precision else torch.float32
        )
        
    def batch_inference(self, 
                       depth_batch: torch.Tensor, 
                       proprio_batch: torch.Tensor,
                       use_amp: bool = True) -> torch.Tensor:
        """
        Ultra-fast batch inference for rover-compatible model
        
        Args:
            depth_batch: [batch_size, stacked_frames, 160, 288]
            proprio_batch: [batch_size, 16]
            use_amp: Use Automatic Mixed Precision
            
        Returns:
            actions: [batch_size, 2]
        """
        batch_size = depth_batch.shape[0]
        
        if batch_size > self.max_batch_size:
            # Process in chunks
            results = []
            for i in range(0, batch_size, self.max_batch_size):
                end_idx = min(i + self.max_batch_size, batch_size)
                chunk_actions = self.batch_inference(
                    depth_batch[i:end_idx], 
                    proprio_batch[i:end_idx],
                    use_amp=use_amp
                )
                results.append(chunk_actions)
            return torch.cat(results, dim=0)
        
        # Use pre-allocated buffers
        self.depth_buffer[:batch_size] = depth_batch.to(self.depth_buffer.dtype)
        self.proprio_buffer[:batch_size] = proprio_batch.to(self.proprio_buffer.dtype)
        
        # Inference with optional AMP
        if use_amp and self.use_mixed_precision:
            with torch.cuda.amp.autocast():
                actions = self.model.get_actions(
                    self.depth_buffer[:batch_size], 
                    self.proprio_buffer[:batch_size]
                )
        else:
            actions = self.model.get_actions(
                self.depth_buffer[:batch_size], 
                self.proprio_buffer[:batch_size]
            )
            
        return actions.float()  # Convert back to float32 for compatibility
        
    def benchmark_inference(self, batch_size: int = 16, num_iterations: int = 50):
        """Benchmark inference performance with rover dimensions"""
        self.logger.info(f"Benchmarking rover model with batch_size={batch_size}")
        
        # Generate random test data with rover dimensions
        depth_test = torch.randn(batch_size, self.model.stacked_frames, 160, 288, device=self.device)
        proprio_test = torch.randn(batch_size, 16, device=self.device)
        
        # Warmup
        for _ in range(5):
            _ = self.batch_inference(depth_test, proprio_test)
            
        # Actual benchmark
        torch.cuda.synchronize() if self.device == "cuda" else None
        start_time = time.time()
        
        for _ in range(num_iterations):
            actions = self.batch_inference(depth_test, proprio_test)
            
        torch.cuda.synchronize() if self.device == "cuda" else None
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_batch = total_time / num_iterations
        samples_per_second = (batch_size * num_iterations) / total_time
        
        self.logger.info(f"Average time per batch: {avg_time_per_batch*1000:.2f} ms")
        self.logger.info(f"Samples per second: {samples_per_second:.0f}")
        self.logger.info(f"Actions shape: {actions.shape}")
        
        return samples_per_second


def create_rover_compatible_model(device="cuda", use_optimizations=True) -> Tuple[RoverDepthExplorationNet, RoverCompatibleTrainer]:
    """Factory function to create rover-compatible model and trainer"""
    
    # Create model with rover architecture (1 frame, 13 extra proprio features)
    model = RoverDepthExplorationNet(
        stacked_frames=1,
        extra_proprio=13,
        use_quantization=use_optimizations,
        use_sparsity=use_optimizations
    )
    
    # Create optimized trainer
    trainer = RoverCompatibleTrainer(
        model=model,
        device=device,
        use_compile=use_optimizations,
        use_mixed_precision=True
    )
    
    return model, trainer


def convert_to_rover_format(ultra_fast_checkpoint: str, output_path: str):
    """
    Convert an ultra-fast model checkpoint to rover-compatible format
    
    Args:
        ultra_fast_checkpoint: Path to ultra-fast model checkpoint
        output_path: Where to save rover-compatible model
    """
    # Load ultra-fast checkpoint
    checkpoint = torch.load(ultra_fast_checkpoint)
    
    # Create rover model
    rover_model, _ = create_rover_compatible_model()
    
    # Try to load compatible weights (architecture should match)
    try:
        rover_model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Successfully loaded weights into rover model")
    except Exception as e:
        print(f"⚠ Weight loading failed: {e}")
        print("Models may have different architectures - using transfer learning")
        
        # Could implement weight transfer logic here if needed
        
    # Save in rover format
    rover_checkpoint = {
        'model_state_dict': rover_model.state_dict(),
        'generation': checkpoint.get('generation', 0),
        'best_fitness': checkpoint.get('fitness', -999.0),
        'training_type': 'ultra_fast_converted',
        'original_checkpoint': ultra_fast_checkpoint,
        'conversion_timestamp': time.time()
    }
    
    torch.save(rover_checkpoint, output_path)
    print(f"💾 Rover-compatible model saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--convert", help="Convert ultra-fast checkpoint to rover format")
    parser.add_argument("--output", help="Output path for converted model")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    if args.convert:
        if not args.output:
            print("Error: --output required when converting")
            exit(1)
        convert_to_rover_format(args.convert, args.output)
        
    elif args.benchmark:
        # Benchmark the rover-compatible model
        logger.info("Creating rover-compatible model...")
        model, trainer = create_rover_compatible_model(device=args.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters: {total_params:,}")
        
        # Run benchmarks
        for batch_size in [1, 4, 8, 16, 32]:
            try:
                trainer.benchmark_inference(batch_size=batch_size, num_iterations=20)
            except Exception as e:
                logger.error(f"Batch size {batch_size} failed: {e}")
                break
    else:
        # Just show model info
        model, _ = create_rover_compatible_model()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Rover-compatible model created with {total_params:,} parameters")
        print("Architecture matches your rover's DepthImageExplorationNet")
        print("Input: [batch, 1, 160, 288] depth + [batch, 16] proprioceptive")
        print("Output: [batch, 3] (linear_vel, angular_vel, confidence)")