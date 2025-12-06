#!/usr/bin/env python3
"""
Benchmark script for SAC Actor inference on RK3588 NPU.
Tests if the optimized model can achieve 30Hz inference.

Usage:
    python3 benchmark_sac_npu.py

Requirements:
    - RKNNLite installed (for NPU benchmark)
    - ONNX Runtime installed (optional, for CPU benchmark)
    - PyTorch (optional, for creating ONNX model)
    - ONNX model file (sac_actor_benchmark.onnx)
"""

import os
import time
import numpy as np
import subprocess

# Optional PyTorch imports
try:
    import torch
    import torch.nn as nn
    from model_architectures import OccupancyGridEncoder, GaussianPolicyHead
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠ PyTorch not available - will skip model creation")

# RKNN support
try:
    from rknnlite.api import RKNNLite
    HAS_RKNN = True
except ImportError:
    HAS_RKNN = False
    print("⚠ RKNNLite not available - cannot run NPU benchmark")

def source_rknn_env():
    """Source RKNN environment if available."""
    env_files = [
        "/opt/rknn-toolkit2/envsetup.sh",
        "/usr/bin/envsetup.sh",
        "/opt/rknn/envsetup.sh"
    ]
    
    for env_file in env_files:
        if os.path.exists(env_file):
            print(f"🔧 Sourcing RKNN environment: {env_file}")
            try:
                result = subprocess.run(f"source {env_file} && env", shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    # Parse and set environment variables
                    for line in result.stdout.split('\n'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key] = value
                    print("✓ RKNN environment sourced successfully")
                    return True
                else:
                    print(f"⚠ Failed to source {env_file}")
            except Exception as e:
                print(f"⚠ Error sourcing {env_file}: {e}")
    
    print("ℹ No RKNN environment file found, proceeding anyway...")
    return False

# Define model class only if PyTorch is available
if HAS_TORCH:
    class SACActor(nn.Module):
        """SAC Actor model for benchmarking."""
        def __init__(self):
            super().__init__()
            self.encoder = OccupancyGridEncoder(input_channels=4)
            self.policy_head = GaussianPolicyHead(
                feature_dim=self.encoder.output_dim,
                proprio_dim=10,
                action_dim=2
            )

        def forward(self, grid, proprio):
            features = self.encoder(grid)
            mean, log_std = self.policy_head(features, proprio)
            return torch.tanh(mean)  # Deterministic action
else:
    SACActor = None

def create_onnx_model():
    """Create and export ONNX model."""
    if not HAS_TORCH:
        print("⚠ Cannot create ONNX model - PyTorch not available")
        print("   Please create the model on a machine with PyTorch and transfer sac_actor_benchmark.onnx")
        return None

    print("Creating ONNX model...")

    model = SACActor()
    model.eval()

    # Dummy inputs
    dummy_grid = torch.randn(1, 4, 64, 64)
    dummy_proprio = torch.randn(1, 10)

    onnx_path = "sac_actor_benchmark.onnx"
    torch.onnx.export(
        model,
        (dummy_grid, dummy_proprio),
        onnx_path,
        opset_version=11,
        input_names=['grid', 'proprio'],
        output_names=['action'],
        export_params=True,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        verbose=False,
        dynamo=False
    )

    print(f"✅ ONNX model exported to {onnx_path}")
    return onnx_path

def benchmark_cpu(model_path, num_runs=1000):
    """Benchmark on CPU using ONNX Runtime."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("⚠ ONNX Runtime not available, skipping CPU benchmark")
        return

    print("Benchmarking on CPU...")

    # Create session
    session = ort.InferenceSession(model_path)

    # Dummy inputs
    grid = np.random.randn(1, 4, 64, 64).astype(np.float32)
    proprio = np.random.randn(1, 10).astype(np.float32)

    # Warmup
    for _ in range(10):
        session.run(None, {'grid': grid, 'proprio': proprio})

    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        session.run(None, {'grid': grid, 'proprio': proprio})
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs
    hz = 1.0 / avg_time

    print(".4f")
    print(".1f")
    return hz

def benchmark_rknn(onnx_path, num_runs=1000):
    """Benchmark on RK3588 NPU using RKNN with FP16."""
    if not HAS_RKNN:
        print("⚠ RKNN not available")
        return

    print("Benchmarking on RK3588 NPU (FP16)...")

    # Convert to RKNN
    rknn_path = "sac_actor_benchmark_fp16.rknn"

    # RKNN conversion
    rknn = RKNNLite()

    print("Loading ONNX model...")
    # Try different RKNN API methods for different versions
    try:
        # RKNN Toolkit Lite2 2.3.2 API
        ret = rknn.load_onnx_model(onnx_path)
    except AttributeError:
        try:
            # Older API
            ret = rknn.load_onnx(onnx_path)
        except AttributeError:
            print("❌ RKNN API not supported. This version may require different loading method.")
            return
    if ret != 0:
        print(f"❌ Load ONNX failed: {ret}")
        return

    print("Building RKNN model (FP16, no quantization)...")
    # Use FP16 for faster inference without calibration data
    ret = rknn.build(
        do_quantization=False,  # FP16 mode
        rknn_platform='rk3588',
        optimization_level=3  # Maximum optimization
    )
    if ret != 0:
        print(f"❌ Build RKNN failed: {ret}")
        return

    print("Exporting RKNN model...")
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print(f"❌ Export RKNN failed: {ret}")
        return

    print("Initializing RKNN runtime...")
    # Use all 3 NPU cores (core_mask=7 = binary 111 for cores 0,1,2)
    # This maximizes performance for the RK3588
    ret = rknn.init_runtime(core_mask=7)
    if ret != 0:
        print(f"❌ Init runtime failed: {ret}")
        return

    # Dummy inputs for inference
    grid = np.random.randn(1, 4, 64, 64).astype(np.float32)
    proprio = np.random.randn(1, 10).astype(np.float32)

    # Warmup
    print("Warming up...")
    for _ in range(10):
        outputs = rknn.inference(inputs=[grid, proprio])

    # Benchmark
    print(f"Running {num_runs} inferences...")
    start_time = time.time()
    for _ in range(num_runs):
        outputs = rknn.inference(inputs=[grid, proprio])
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs
    hz = 1.0 / avg_time

    print(".4f")
    print(".1f")

    if hz >= 30.0:
        print("✅ Target 30Hz ACHIEVED!")
    else:
        print(".1f")

    return hz

def main():
    print("==================================================")
    print("   SAC Actor NPU Benchmark (RK3588)              ")
    print("==================================================")

    # Source RKNN environment if available
    source_rknn_env()

    # Create model if needed
    onnx_path = "sac_actor_benchmark.onnx"
    if not os.path.exists(onnx_path):
        onnx_path = create_onnx_model()
        if onnx_path is None:
            print("❌ Cannot proceed without ONNX model")
            return
    else:
        print(f"Using existing ONNX model: {onnx_path}")

    # CPU benchmark
    cpu_hz = benchmark_cpu(onnx_path)

    # NPU benchmark (FP16, no calibration needed)
    npu_hz = benchmark_rknn(onnx_path)

    print("\n==================================================")
    print("Benchmark Results:")
    if cpu_hz:
        print(".1f")
    if npu_hz:
        print(".1f")
        if npu_hz >= 30:
            print("🎉 SUCCESS: Model meets 30Hz requirement!")
        else:
            print(".1f")
    print("==================================================")

if __name__ == '__main__':
    main()