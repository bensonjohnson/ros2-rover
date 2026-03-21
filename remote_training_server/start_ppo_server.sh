#!/bin/bash

# PPO Training Server Startup Script
# Auto-detects hardware (ROCm V620 / CUDA) and configures AMP FP16 training

set -e

# Detect hardware backend
if command -v nvidia-smi &> /dev/null; then
  CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  echo "=================================================="
  echo "PPO Training Server (CUDA)"
  echo "=================================================="
  echo ""
  echo "CUDA GPU Detected:"
  echo "  Driver: ${CUDA_VERSION}"
  echo "  GPU: ${GPU_NAME}"
elif command -v rocm-smi &> /dev/null; then
  ROCM_VERSION=$(rocm-smi --version 2>/dev/null | head -1)
  GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | grep -A1 "GPU" | tail -1 | tr -d ' ' || echo "Unknown")
  echo "=================================================="
  echo "PPO Training Server (ROCm) - V620 Unified BEV"
  echo "=================================================="
  echo ""
  echo "ROCm GPU Detected:"
  echo "  Version: ${ROCM_VERSION}"
  echo "  GPU: ${GPU_NAME}"
else
  echo "=================================================="
  echo "PPO Training Server (No GPU) - CPU Fallback"
  echo "=================================================="
fi

# Configuration
NATS_SERVER="nats://nats.gokickrocks.org:4222"
CHECKPOINT_DIR=${1:-./checkpoints_ppo}
LOG_DIR=${2:-./logs_ppo}

echo ""
echo "Configuration:"
echo "  NATS Server: ${NATS_SERVER}"
echo "  Checkpoint Dir: ${CHECKPOINT_DIR}"
echo "  Log Dir: ${LOG_DIR}"
echo "  Architecture: Unified BEV (LiDAR + Depth fusion)"
echo "  Algorithm: PPO (on-policy, rollout-based)"
echo ""

# Check we're in the right directory
if [ ! -f "v620_ppo_trainer.py" ]; then
  echo "Error: Please run this script from remote_training_server directory"
  echo "Current directory: $(pwd)"
  exit 1
fi

# Create directories
mkdir -p ${CHECKPOINT_DIR} ${LOG_DIR}

# Checkpoint cleanup menu
echo "=================================================="
echo "Clean up checkpoint directory?"
echo "=================================================="
echo ""
echo "  1) Keep everything (default)"
echo "  2) Delete PPO checkpoints only"
echo "  3) Delete ALL (checkpoints + ONNX)"
echo ""
read -rp "Select [1-3] (default: 1): " PRUNE_CHOICE
PRUNE_CHOICE=${PRUNE_CHOICE:-1}

if [[ "${PRUNE_CHOICE}" =~ ^[2-3]$ ]]; then
  PPO_COUNT=$(find "${CHECKPOINT_DIR}" -maxdepth 1 -name "ppo_update_*.pt" 2>/dev/null | wc -l | tr -d ' ')
  ONNX_COUNT=$(find "${CHECKPOINT_DIR}" -maxdepth 1 -name "*.onnx" 2>/dev/null | wc -l | tr -d ' ')

  echo ""
  echo "The following will be deleted:"
  case "${PRUNE_CHOICE}" in
    2)
      echo "  - PPO checkpoints (${PPO_COUNT} files)"
      ;;
    3)
      echo "  - PPO checkpoints (${PPO_COUNT} files)"
      echo "  - ONNX exports (${ONNX_COUNT} files)"
      ;;
  esac

  read -rp "Confirm deletion? [y/N]: " CONFIRM
  if [[ "${CONFIRM}" =~ ^[Yy]$ ]]; then
    case "${PRUNE_CHOICE}" in
      2) rm -f "${CHECKPOINT_DIR}"/ppo_update_*.pt ;;
      3) rm -f "${CHECKPOINT_DIR}"/ppo_update_*.pt "${CHECKPOINT_DIR}"/*.onnx ;;
    esac
    echo "Cleanup complete"
  else
    echo "  -> Skipped cleanup"
  fi
fi

# Check Python and packages
echo ""
echo "Checking Python version..."
python3 --version

echo ""
echo "Checking required packages..."
MISSING_PACKAGES=()
python3 -c "import torch" 2>/dev/null || MISSING_PACKAGES+=("torch")
python3 -c "import nats" 2>/dev/null || MISSING_PACKAGES+=("nats-py")
python3 -c "import numpy" 2>/dev/null || MISSING_PACKAGES+=("numpy")
python3 -c "import tensorboard" 2>/dev/null || MISSING_PACKAGES+=("tensorboard")
python3 -c "import msgpack" 2>/dev/null || MISSING_PACKAGES+=("msgpack")
python3 -c "import zstandard" 2>/dev/null || MISSING_PACKAGES+=("zstandard")

if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
  echo "Missing packages: ${MISSING_PACKAGES[@]}"
  echo "Install with: pip3 install ${MISSING_PACKAGES[@]}"
  exit 1
fi
echo "All required packages installed"

# Verify PyTorch backend
echo ""
echo "Verifying PyTorch acceleration..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'AMP FP16: Enabled')
else:
    print('No GPU - CPU training (slower)')
"

# Check for existing processes
echo ""
if pgrep -f v620_ppo_trainer > /dev/null; then
  echo "Found old v620_ppo_trainer process(es), killing..."
  pkill -f v620_ppo_trainer
  sleep 2
fi

# Hardware-specific optimizations
OS_TYPE=$(uname -s)
if [ "$OS_TYPE" = "Linux" ]; then
  if command -v nvidia-smi &> /dev/null; then
    echo "Applying CUDA optimizations..."
    export CUDA_MODULE_LOADING=LAZY
    export CUDA_LAUNCH_BLOCKING=0
    export CUDA_VISIBLE_DEVICES=0
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  elif command -v rocm-smi &> /dev/null; then
    echo "Applying ROCm optimizations..."
    export HSA_FORCE_FINE_GRAIN_PCIE=1
    export MIOPEN_FIND_ENFORCE=NONE
    export MIOPEN_DISABLE_CACHE=0
    export PYTORCH_ROCM_ARCH="gfx1030"
    export MIOPEN_DEBUG_DISABLE_FIND_DB=0
    export HSA_ENABLE_SDMA=0
    export MIOPEN_FIND_MODE=NORMAL
    export HSA_OVERRIDE_GFX_VERSION=10.3.0
    export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
  fi
fi

# Raise file descriptor limit
if ulimit -n 65535 2>/dev/null; then
  echo "Raised file descriptor limit to 65535"
fi

# Start TensorBoard
echo ""
echo "Starting TensorBoard..."
tensorboard --logdir "$(pwd)/${LOG_DIR}" --port 6007 --bind_all > /dev/null 2>&1 &
TB_PID=$!
echo "TensorBoard running on http://localhost:6007 (PID: ${TB_PID})"

LOG_FILE="${LOG_DIR}/ppo_server_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "Starting V620 PPO Trainer..."

python3 -u v620_ppo_trainer.py \
  --nats_server "${NATS_SERVER}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --log_dir "${LOG_DIR}" \
  "$@" > >(tee "${LOG_FILE}") 2>&1 &

TRAINER_PID=$!

echo "PPO server PID: ${TRAINER_PID}"
echo "Log file: ${LOG_FILE}"
echo ""

cleanup() {
  echo ""
  echo "Shutting down PPO server..."
  kill -TERM $TRAINER_PID 2>/dev/null || true
  wait $TRAINER_PID 2>/dev/null
  kill $TB_PID 2>/dev/null || true
  echo "Shutdown complete"
  exit 0
}

trap cleanup SIGINT SIGTERM

wait $TRAINER_PID
