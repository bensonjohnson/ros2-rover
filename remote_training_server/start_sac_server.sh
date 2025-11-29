#!/bin/bash

# SAC Training Server Startup Script for V620
# Trains a continuous SAC policy using ROCm acceleration

set -e

echo "=================================================="
echo "SAC Training Server (V620)"
echo "=================================================="

# Configuration
PORT=${1:-5556}
CHECKPOINT_DIR=${2:-./checkpoints_sac}
LOG_DIR=${3:-./logs_sac}

echo "Configuration:"
echo "  ZeroMQ Port: ${PORT}"
echo "  Checkpoint Dir: ${CHECKPOINT_DIR}"
echo "  Log Dir: ${LOG_DIR}"
echo ""

# Check we're in the right directory
if [ ! -f "v620_sac_trainer.py" ]; then
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

if pgrep -f v620_sac_trainer > /dev/null; then
  echo "⚠ Found old v620_sac_trainer process(es), killing..."
  pkill -f v620_sac_trainer
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
  export MIOPEN_DISABLE_CACHE=0 # Enable cache to speed up startup after first run
  echo "✓ ROCm environment variables set"
fi

# Check and warn about file descriptor limits
CURRENT_ULIMIT=$(ulimit -n)
echo ""
echo "Checking file descriptor limits..."
echo "  Current ulimit: ${CURRENT_ULIMIT}"
if [ ${CURRENT_ULIMIT} -lt 65535 ]; then
  echo "  ⚠ Warning: ulimit is below recommended 65535 for MIOpen benchmark"
  echo "  Attempting to raise limit for training process..."
fi

# Start TensorBoard
echo "Starting TensorBoard..."
tensorboard --logdir "$(pwd)/${LOG_DIR}" --port 6006 --bind_all > /dev/null 2>&1 &
TB_PID=$!
echo "TensorBoard running on http://localhost:6006 (PID: ${TB_PID})"

LOG_FILE="${LOG_DIR}/sac_server_$(date +%Y%m%d_%H%M%S).log"

# Launch Python with proper ulimit applied to the process
bash -c "ulimit -n 65535 && exec python3 -u v620_sac_trainer.py \
  --port ${PORT} \
  --checkpoint_dir ${CHECKPOINT_DIR} \
  --log_dir ${LOG_DIR} \
  $* \
  2>&1" | tee "${LOG_FILE}" &

TRAINER_PID=$!

echo "SAC server PID: ${TRAINER_PID}"
echo "Log file: ${LOG_FILE}"
echo ""

# Set up signal handling
cleanup() {
  echo ""
  echo "🛑 Shutting down SAC server..."
  kill $TRAINER_PID 2>/dev/null || true
  kill $TB_PID 2>/dev/null || true
  echo "✅ Shutdown complete"
  exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for trainer
wait $TRAINER_PID
