#!/usr/bin/env python3
"""
Ultra-Optimized Neural Network using PyTorch AO techniques
Includes quantization, sparsity, and compilation optimizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import time

try:
    # Try to import PyTorch AO optimizations
    from torchao.quantization import quantize_, int4_weight_only
    from torchao.sparsity import sparsify_
    TORCHAO_AVAILABLE = True
except ImportError:
    print("TorchAO not available - using standard PyTorch optimizations")
    TORCHAO_AVAILABLE = False

class OptimizedDepthModel(nn.Module):
    """
    Highly optimized depth-based navigation model
    Uses techniques from PyTorch AO for maximum performance
    """
    
    def __init__(
        self, 
        depth_dim: int = 64,
        proprio_dim: int = 8, 
        hidden_dim: int = 128,
        action_dim: int = 2,
        use_quantization: bool = True,
        use_sparsity: bool = True
    ):
        super().__init__()
        
        self.depth_dim = depth_dim
        self.proprio_dim = proprio_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        # Depth processing branch - CNN-style but flattened for speed
        self.depth_encoder = nn.Sequential(
            nn.Linear(depth_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        
        # Proprioceptive branch
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
            nn.ReLU(inplace=True),
        )
        
        # Fusion and action output
        fusion_dim = hidden_dim // 2 + hidden_dim // 4
        self.action_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()  # Actions in [-1, 1]
        )
        
        # Value head for advantage estimation (helps with training)
        self.value_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Apply optimizations
        self._apply_optimizations(use_quantization, use_sparsity)
        
    def _apply_optimizations(self, use_quantization: bool, use_sparsity: bool):
        """Apply PyTorch AO optimizations"""
        
        if not TORCHAO_AVAILABLE:
            print("Skipping TorchAO optimizations - not available")
            return
            
        if use_quantization:
            try:
                # Apply INT4 weight-only quantization for memory and speed
                quantize_(self, int4_weight_only())
                print("✓ Applied INT4 weight-only quantization")
            except Exception as e:
                print(f"⚠ Quantization failed: {e}")
                
        if use_sparsity:
            try:
                # Apply 2:4 structured sparsity for speed without accuracy loss
                sparsify_(self, sparsity_level=0.5)
                print("✓ Applied 2:4 structured sparsity")  
            except Exception as e:
                print(f"⚠ Sparsity failed: {e}")
        
    def forward(self, depth: torch.Tensor, proprio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            depth: [batch_size, depth_dim] - depth sensor readings
            proprio: [batch_size, proprio_dim] - proprioceptive data
            
        Returns:
            actions: [batch_size, action_dim] - predicted actions
            values: [batch_size, 1] - state values (for training)
        """
        # Encode depth
        depth_features = self.depth_encoder(depth)
        
        # Encode proprioceptive
        proprio_features = self.proprio_encoder(proprio)
        
        # Fuse features
        fused = torch.cat([depth_features, proprio_features], dim=-1)
        
        # Generate actions and values
        actions = self.action_head(fused)
        values = self.value_head(fused)
        
        return actions, values
        
    def get_actions(self, depth: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """Lightweight inference - actions only"""
        with torch.no_grad():
            actions, _ = self.forward(depth, proprio)
            return actions


class UltraFastTrainer:
    """
    Ultra-optimized trainer using all PyTorch performance tricks
    """
    
    def __init__(
        self, 
        model: OptimizedDepthModel,
        device: str = "cuda",
        use_compile: bool = True,
        use_mixed_precision: bool = True
    ):
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
                print("✓ Model compiled with max-autotune mode")
            except Exception as e:
                print(f"⚠ Compilation failed: {e}")
                
        # Set up optimized inference
        self.model.eval()
        
        # Automatic Mixed Precision scaler
        if use_mixed_precision and device == "cuda":
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        # Pre-allocate tensors for batch inference
        self.max_batch_size = 256
        self._preallocate_tensors()
        
    def _preallocate_tensors(self):
        """Pre-allocate tensors to avoid memory allocation overhead"""
        self.depth_buffer = torch.zeros(
            self.max_batch_size, 64, 
            device=self.device, dtype=torch.float16 if self.use_mixed_precision else torch.float32
        )
        self.proprio_buffer = torch.zeros(
            self.max_batch_size, 8,
            device=self.device, dtype=torch.float16 if self.use_mixed_precision else torch.float32  
        )
        self.action_buffer = torch.zeros(
            self.max_batch_size, 2,
            device=self.device, dtype=torch.float16 if self.use_mixed_precision else torch.float32
        )
        
    def batch_inference(
        self, 
        depth_batch: torch.Tensor, 
        proprio_batch: torch.Tensor,
        use_amp: bool = True
    ) -> torch.Tensor:
        """
        Ultra-fast batch inference
        
        Args:
            depth_batch: [batch_size, 64]
            proprio_batch: [batch_size, 8]
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
        
    def benchmark_inference(self, batch_size: int = 32, num_iterations: int = 100):
        """Benchmark inference performance"""
        print(f"Benchmarking inference with batch_size={batch_size}")
        
        # Generate random test data
        depth_test = torch.randn(batch_size, 64, device=self.device)
        proprio_test = torch.randn(batch_size, 8, device=self.device)
        
        # Warmup
        for _ in range(10):
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
        
        print(f"Average time per batch: {avg_time_per_batch*1000:.2f} ms")
        print(f"Samples per second: {samples_per_second:.0f}")
        print(f"Actions shape: {actions.shape}")
        
        return samples_per_second


def create_optimized_model(device="cuda", smaller_model=False) -> Tuple[OptimizedDepthModel, UltraFastTrainer]:
    """Factory function to create optimized model and trainer"""
    
    # Adjust model size based on requirements
    if smaller_model:
        hidden_dim = 64  # Smaller for faster parameter optimization
        use_quantization = False  # Skip for smaller models
        use_sparsity = False
    else:
        hidden_dim = 128
        use_quantization = True
        use_sparsity = True
    
    # Create model with optimizations
    model = OptimizedDepthModel(
        depth_dim=64,
        proprio_dim=8,
        hidden_dim=hidden_dim,
        action_dim=2,
        use_quantization=use_quantization,
        use_sparsity=use_sparsity
    )
    
    # Create optimized trainer
    trainer = UltraFastTrainer(
        model=model,
        device=device,
        use_compile=not smaller_model,  # Skip compilation for smaller models
        use_mixed_precision=True
    )
    
    return model, trainer


if __name__ == "__main__":
    # Benchmark the optimized model
    print("Creating optimized model...")
    model, trainer = create_optimized_model()
    
    # Run benchmarks
    for batch_size in [1, 8, 16, 32, 64, 128]:
        try:
            trainer.benchmark_inference(batch_size=batch_size, num_iterations=50)
            print()
        except Exception as e:
            print(f"Batch size {batch_size} failed: {e}")
            break