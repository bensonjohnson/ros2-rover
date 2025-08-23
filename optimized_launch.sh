#!/bin/bash
# Auto-generated optimized launch script for ROS2 Rover ES-Hybrid mode
# Generated on: 2025-08-23 05:05:20
# Performance score: N/A

echo "🚀 Launching ROS2 Rover with optimized ES-Hybrid configuration"
echo "Network mode: balanced"
echo "Population size: 18"
echo "Reward mode: exploration"
echo ""

# Set optimized environment variables
export PYTORCH_NUM_THREADS=6
export OMP_NUM_THREADS=6
export RKNN_OPTIMIZATION=1

# Launch with optimized parameters
./start_npu_exploration_depth.sh es_hybrid \
    --population_size 18 \
    --sigma 0.1 \
    --learning_rate 0.015 \
    --network_mode balanced \
    --reward_mode exploration \
    --enable_curiosity 1 \
    --enable_adaptive_scaling 1 \
    --target_fps 30.0 \
    --width_multiplier 1.0

echo "✓ Launch complete"
