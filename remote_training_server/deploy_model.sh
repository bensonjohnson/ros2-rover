#!/bin/bash

# Model Deployment Script (Run on V620 Server)
# Exports ONNX and deploys to rover for conversion

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

mkdir -p export

# Step 1: Export to ONNX
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

echo "  Size: $(du -h ${ONNX_PATH} | cut -f1)"

# Step 2: Deploy ONNX to rover
echo ""
echo "Step 2: Deploying ONNX to rover..."
echo "Testing SSH connection to rover..."

if ! ssh -o ConnectTimeout=5 ${ROVER_USER}@${ROVER_IP} "echo '✓ Connected to rover'" 2>/dev/null; then
  echo "❌ Cannot connect to rover at ${ROVER_USER}@${ROVER_IP}"
  echo ""
  echo "Manual deployment:"
  echo "  scp ${ONNX_PATH} ${ROVER_USER}@${ROVER_IP}:~/Documents/ros2-rover/models/"
  echo "  ssh ${ROVER_USER}@${ROVER_IP}"
  echo "  cd ~/Documents/ros2-rover"
  echo "  ./convert_onnx_to_rknn.sh models/$(basename ${ONNX_PATH})"
  exit 1
fi

# Create models directory on rover
ssh ${ROVER_USER}@${ROVER_IP} "mkdir -p ~/Documents/ros2-rover/models"

# Copy ONNX model
echo "Copying ONNX to rover..."
scp ${ONNX_PATH} ${ROVER_USER}@${ROVER_IP}:~/Documents/ros2-rover/models/

echo "✓ ONNX deployed to rover"

# Step 3: Convert ONNX → RKNN on rover
echo ""
echo "Step 3: Converting ONNX → RKNN on rover..."
echo "(RKNN conversion must happen on the RK3588)"

ssh ${ROVER_USER}@${ROVER_IP} "cd ~/Documents/ros2-rover && ./convert_onnx_to_rknn.sh models/$(basename ${ONNX_PATH})"

if [ $? -ne 0 ]; then
  echo "❌ RKNN conversion failed on rover"
  echo ""
  echo "Manual conversion on rover:"
  echo "  ssh ${ROVER_USER}@${ROVER_IP}"
  echo "  cd ~/Documents/ros2-rover"
  echo "  ./convert_onnx_to_rknn.sh models/$(basename ${ONNX_PATH})"
  exit 1
fi

echo "✓ Model converted to RKNN on rover"

# Check deployed files
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
