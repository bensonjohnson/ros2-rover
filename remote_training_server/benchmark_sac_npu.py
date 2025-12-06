#!/usr/bin/env python3
"""
Benchmark script for SAC Actor inference on RK3588 NPU.
Tests if the optimized model can achieve 30Hz inference.

Usage:
    python3 benchmark_sac_npu.py

Requirements:
    - RKNNLite installed
    - ONNX model (will be created if not exists)
    - Calibration data for quantization
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Import model architectures
from model_architectures import OccupancyGridEncoder, GaussianPolicyHead

# RKNN support
try:
    from rknnlite.api import RKNNLite
    HAS_RKNN = True
except ImportError:
    HAS_RKNN = False
    print("⚠ RKNNLite not available - cannot run NPU benchmark")

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

def create_onnx_model():
    """Create and export ONNX model."""
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
    ret = rknn.load_onnx(onnx_path)
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

    # Create model if needed
    onnx_path = "sac_actor_benchmark.onnx"
    if not os.path.exists(onnx_path):
        create_onnx_model()
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