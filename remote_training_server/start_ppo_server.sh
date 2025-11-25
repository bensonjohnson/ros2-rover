#!/bin/bash

# PPO Training Server Startup Script for V620
# Trains a continuous PPO policy using ROCm acceleration

set -e

echo "=================================================="
echo "PPO Training Server (V620)"
echo "=================================================="

# Configuration
PORT=${1:-5556}
CHECKPOINT_DIR=${2:-./checkpoints_ppo}
LOG_DIR=${3:-./logs_ppo}

echo "Configuration:"
echo "  ZeroMQ Port: ${PORT}"
echo "  Checkpoint Dir: ${CHECKPOINT_DIR}"
echo "  Log Dir: ${LOG_DIR}"
echo ""

# Check we're in the right directory
if [ ! -f "v620_ppo_trainer.py" ]; then
  echo "❌ Error: Please run this script from remote_training_server directory"
  echo "Current directory: $(pwd)"
  exit 1
fi

# Create directories
echo "Creating directories..."
mkdir -p ${CHECKPOINT_DIR} ${LOG_DIR}

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python version: ${PYTHON_VERSION}"

# Check Hardware Acceleration Backend
echo ""
echo "Checking hardware acceleration backend..."

# Detect OS
OS_TYPE=$(uname -s)
echo "  OS: ${OS_TYPE}"

# Check ROCm GPU on Linux
if [ "$OS_TYPE" = "Linux" ]; then
  if command -v rocm-smi &> /dev/null; then
    echo ""
    echo "ROCm GPU detected:"
    rocm-smi --showproductname 2>/dev/null | grep -A1 "GPU" || echo "  (rocm-smi output unavailable)"
  fi
fi

# Verify PyTorch backend
echo ""
echo "Verifying PyTorch acceleration backend..."
python3 -c "
import torch
import sys

# Check CUDA (includes ROCm)
if torch.cuda.is_available():
    print(f'✓ CUDA/ROCm GPU detected: {torch.cuda.get_device_name(0)}')
    print(f'  Device count: {torch.cuda.device_count()}')
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  Backend: CUDA')
    sys.exit(0)

# Fallback to CPU
print('⚠ No GPU acceleration detected! Will use CPU (slower)')
print('  Backend: CPU')
" 2>&1

# Check required packages
echo ""
echo "Checking required packages..."

MISSING_PACKAGES=()

python3 -c "import torch" 2>/dev/null || MISSING_PACKAGES+=("torch")
python3 -c "import zmq" 2>/dev/null || MISSING_PACKAGES+=("pyzmq")
python3 -c "import numpy" 2>/dev/null || MISSING_PACKAGES+=("numpy")
python3 -c "import tensorboard" 2>/dev/null || MISSING_PACKAGES+=("tensorboard")

if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
  echo "❌ Missing packages: ${MISSING_PACKAGES[@]}"
  echo ""
  echo "Install with:"
  echo "  pip3 install ${MISSING_PACKAGES[@]}"
  exit 1
fi

echo "✓ All required packages installed"

# Check network port availability
echo ""
echo "Checking port availability..."

if pgrep -f v620_ppo_trainer > /dev/null; then
  echo "⚠ Found old v620_ppo_trainer process(es), killing..."
  pkill -f v620_ppo_trainer
  sleep 2
fi

if netstat -tuln 2>/dev/null | grep -q ":${PORT} "; then
  echo "⚠ Warning: Port ${PORT} still in use after cleanup"
  if command -v lsof &> /dev/null; then
    lsof -ti:${PORT} | xargs kill -9 2>/dev/null || true
    sleep 1
  fi
fi

echo "✓ Port ${PORT} available"

# ROCm optimizations (environment variables)
if [ "$OS_TYPE" = "Linux" ] && command -v rocm-smi &> /dev/null; then
  echo ""
  echo "Applying ROCm optimizations..."
  export HSA_FORCE_FINE_GRAIN_PCIE=1
  export MIOPEN_FIND_ENFORCE=NONE
  export MIOPEN_DISABLE_CACHE=1
  echo "✓ ROCm environment variables set"
fi

# Set up signal handling
cleanup() {
  echo ""
  echo "🛑 Shutting down PPO server..."
  kill $TRAINER_PID 2>/dev/null || true
  echo "✅ Shutdown complete"
  exit 0
}

trap cleanup SIGINT SIGTERM

# Start PPO server
echo ""
echo "=================================================="
echo "Starting PPO Trainer"
echo "=================================================="
echo ""
echo "Listening for experience batches on port ${PORT}"
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo "Logs: ${LOG_DIR}"
echo ""
echo "Monitor training with:"
echo "  tensorboard --logdir ${LOG_DIR}"
echo ""
echo "Press Ctrl+C to stop"
echo ""

LOG_FILE="${LOG_DIR}/ppo_server_$(date +%Y%m%d_%H%M%S).log"

python3 v620_ppo_trainer.py \
  --port ${PORT} \
  --checkpoint_dir ${CHECKPOINT_DIR} \
  --log_dir ${LOG_DIR} \
  --calibration_dir ./calibration_data \
  2>&1 | tee "${LOG_FILE}" &

TRAINER_PID=$!

echo "PPO server PID: ${TRAINER_PID}"
echo "Log file: ${LOG_FILE}"
echo ""

# Wait for trainer
wait $TRAINER_PID
