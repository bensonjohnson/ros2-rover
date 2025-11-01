#!/bin/bash

# Model Deployment Script (Run on V620 Server)
# Converts trained model to RKNN and deploys to rover

set -e

echo "=================================================="
echo "V620 Model Deployment to Rover"
echo "=================================================="

# Configuration
CHECKPOINT=${1:-"./checkpoints/ppo_v620_update_10.pt"}
ROVER_IP=${2:-"192.168.1.50"}
ROVER_USER=${3:-"benson"}

if [ ! -f "$CHECKPOINT" ]; then
  echo "❌ Checkpoint not found: ${CHECKPOINT}"
  echo "Available checkpoints:"
  ls -lh checkpoints/*.pt 2>/dev/null || echo "  (no checkpoints found)"
  exit 1
fi

echo "Checkpoint: ${CHECKPOINT}"
echo "Rover: ${ROVER_USER}@${ROVER_IP}"
echo ""

# Extract checkpoint name for output files
CHECKPOINT_NAME=$(basename ${CHECKPOINT} .pt)
ONNX_PATH="./export/${CHECKPOINT_NAME}.onnx"
RKNN_PATH="./export/${CHECKPOINT_NAME}.rknn"

mkdir -p export

# Step 1: Export to ONNX (already done by training server)
if [ ! -f "${CHECKPOINT%.pt}.onnx" ]; then
  echo "Step 1: Exporting PyTorch → ONNX..."
  python3 -c "
import torch
from v620_ppo_trainer import V620PPOTrainer

trainer = V620PPOTrainer()
trainer.load_checkpoint('${CHECKPOINT}')
trainer.export_onnx('${ONNX_PATH}')
"
  echo "✓ ONNX exported: ${ONNX_PATH}"
else
  ONNX_PATH="${CHECKPOINT%.pt}.onnx"
  echo "✓ Using existing ONNX: ${ONNX_PATH}"
fi

# Step 2: Convert ONNX → RKNN
echo ""
echo "Step 2: Converting ONNX → RKNN (with quantization)..."
echo "This step requires RKNN-Toolkit2 (x86_64 Linux only)"

if ! python3 -c "from rknn.api import RKNN" 2>/dev/null; then
  echo "❌ RKNN-Toolkit2 not installed!"
  echo ""
  echo "Install from: https://github.com/rockchip-linux/rknn-toolkit2"
  echo ""
  echo "Alternative: Convert on a machine with RKNN-Toolkit2 installed,"
  echo "then copy the .rknn file to the rover manually."
  exit 1
fi

python3 export_to_rknn.py \
  "${ONNX_PATH}" \
  --output "${RKNN_PATH}" \
  --target rk3588 \
  --optimization-level 3

if [ ! -f "${RKNN_PATH}" ]; then
  echo "❌ RKNN conversion failed"
  exit 1
fi

echo "✓ RKNN model created: ${RKNN_PATH}"
echo "  Size: $(du -h ${RKNN_PATH} | cut -f1)"

# Step 3: Deploy to rover
echo ""
echo "Step 3: Deploying to rover..."
echo "Testing SSH connection to rover..."

if ! ssh -o ConnectTimeout=5 ${ROVER_USER}@${ROVER_IP} "echo '✓ Connected to rover'" 2>/dev/null; then
  echo "❌ Cannot connect to rover at ${ROVER_USER}@${ROVER_IP}"
  echo ""
  echo "Manual deployment:"
  echo "  scp ${RKNN_PATH} ${ROVER_USER}@${ROVER_IP}:~/Documents/ros2-rover/models/"
  exit 1
fi

# Create models directory on rover
ssh ${ROVER_USER}@${ROVER_IP} "mkdir -p ~/Documents/ros2-rover/models"

# Copy RKNN model
echo "Copying model to rover..."
scp ${RKNN_PATH} ${ROVER_USER}@${ROVER_IP}:~/Documents/ros2-rover/models/

# Create symlink to latest model
ssh ${ROVER_USER}@${ROVER_IP} "cd ~/Documents/ros2-rover/models && ln -sf $(basename ${RKNN_PATH}) remote_trained.rknn"

echo "✓ Model deployed to rover!"
echo ""
echo "Deployed files:"
ssh ${ROVER_USER}@${ROVER_IP} "ls -lh ~/Documents/ros2-rover/models/"

echo ""
echo "=================================================="
echo "Deployment Complete!"
echo "=================================================="
echo ""
echo "Run on rover:"
echo "  cd ~/Documents/ros2-rover"
echo "  ./start_remote_trained_inference.sh"
echo ""
echo "Or reload model in running system:"
echo "  ros2 service call /reload_remote_model std_srvs/srv/Trigger"
