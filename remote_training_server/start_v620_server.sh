#!/bin/bash

# V620 Training Server Startup Script
# This script starts the PPO training server on the V620 GPU machine

set -e

echo "=================================================="
echo "V620 Remote Training Server Startup"
echo "=================================================="

# Configuration
PORT=${1:-5555}
UPDATE_INTERVAL=${2:-8192}
CHECKPOINT_INTERVAL=${3:-10}
TENSORBOARD_PORT=${4:-6006}

echo "Configuration:"
echo "  ZeroMQ Port: ${PORT}"
echo "  Update Interval: ${UPDATE_INTERVAL} samples"
echo "  Checkpoint Interval: ${CHECKPOINT_INTERVAL} updates"
echo "  TensorBoard Port: ${TENSORBOARD_PORT}"
echo ""

# Check we're in the right directory
if [ ! -f "v620_ppo_trainer.py" ]; then
  echo "❌ Error: Please run this script from remote_training_server directory"
  echo "Current directory: $(pwd)"
  echo "Expected files: v620_ppo_trainer.py, export_to_rknn.py"
  exit 1
fi

# Create directories
echo "Creating directories..."
mkdir -p checkpoints export runs logs

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python version: ${PYTHON_VERSION}"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
  echo "⚠ Warning: Python 3.10+ recommended, you have ${PYTHON_VERSION}"
fi

# Check ROCm GPU
echo ""
echo "Checking ROCm GPU..."
if command -v rocm-smi &> /dev/null; then
  rocm-smi --showproductname 2>/dev/null | grep -A1 "GPU" || echo "  (rocm-smi output unavailable)"
else
  echo "⚠ rocm-smi not found - is ROCm installed?"
fi

# Verify PyTorch can see GPU
echo ""
echo "Verifying PyTorch + ROCm..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✓ GPU detected: {torch.cuda.get_device_name(0)}')
    print(f'  Device count: {torch.cuda.device_count()}')
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  PyTorch version: {torch.__version__}')
else:
    print('❌ No GPU detected! Training will be VERY slow on CPU.')
    print('Install PyTorch with ROCm:')
    print('  pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0')
" 2>&1

# Check required packages
echo ""
echo "Checking required packages..."

MISSING_PACKAGES=()

# Check PyTorch
python3 -c "import torch" 2>/dev/null || MISSING_PACKAGES+=("torch")

# Check ZeroMQ
python3 -c "import zmq" 2>/dev/null || MISSING_PACKAGES+=("pyzmq")

# Check NumPy
python3 -c "import numpy" 2>/dev/null || MISSING_PACKAGES+=("numpy")

# Check OpenCV
python3 -c "import cv2" 2>/dev/null || MISSING_PACKAGES+=("opencv-python")

# Check TensorBoard
python3 -c "import tensorboard" 2>/dev/null || MISSING_PACKAGES+=("tensorboard")

if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
  echo "❌ Missing packages: ${MISSING_PACKAGES[@]}"
  echo ""
  echo "Install with:"
  echo "  pip3 install ${MISSING_PACKAGES[@]}"
  echo ""
  echo "Or use requirements file:"
  echo "  pip3 install -r requirements.txt"
  echo ""
  echo "For PyTorch with ROCm:"
  echo "  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0"
  exit 1
fi

echo "✓ All required packages installed"

# Check network port availability
echo ""
echo "Checking port availability..."

# Kill any old v620_ppo_trainer processes
if pgrep -f v620_ppo_trainer > /dev/null; then
  echo "⚠ Found old v620_ppo_trainer process(es), killing..."
  pkill -f v620_ppo_trainer
  sleep 2
fi

if netstat -tuln 2>/dev/null | grep -q ":${PORT} "; then
  echo "⚠ Warning: Port ${PORT} still in use after cleanup"
  echo "Running processes:"
  lsof -i :${PORT} 2>/dev/null || netstat -tuln | grep ":${PORT}"
  echo ""
  echo "Attempting to free port..."
  if command -v lsof &> /dev/null; then
    lsof -ti:${PORT} | xargs kill -9 2>/dev/null || true
    sleep 1
  fi

  if netstat -tuln 2>/dev/null | grep -q ":${PORT} "; then
    echo "❌ Failed to free port ${PORT}"
    exit 1
  fi
fi

echo "✓ Port ${PORT} available"

# Start TensorBoard in background
echo ""
echo "Starting TensorBoard on port ${TENSORBOARD_PORT}..."
pkill -f "tensorboard.*${TENSORBOARD_PORT}" 2>/dev/null || true
tensorboard --logdir ./runs --port ${TENSORBOARD_PORT} --bind_all > logs/tensorboard.log 2>&1 &
TENSORBOARD_PID=$!
sleep 2

if ps -p $TENSORBOARD_PID > /dev/null; then
  echo "✓ TensorBoard started (PID: ${TENSORBOARD_PID})"

  # Get all network interfaces
  echo ""
  echo "TensorBoard accessible at:"
  hostname -I | tr ' ' '\n' | grep -E '^[0-9]' | while read ip; do
    echo "  http://${ip}:${TENSORBOARD_PORT}"
  done
else
  echo "⚠ TensorBoard failed to start (check logs/tensorboard.log)"
fi

# Set up signal handling
cleanup() {
  echo ""
  echo "🛑 Shutting down training server..."
  kill $TRAINING_PID 2>/dev/null || true
  kill $TENSORBOARD_PID 2>/dev/null || true
  echo "✅ Shutdown complete"
  exit 0
}

trap cleanup SIGINT SIGTERM

# Start training server
echo ""
echo "=================================================="
echo "Starting PPO Training Server"
echo "=================================================="
echo ""
echo "Listening for rover data on port ${PORT}"
echo "Updates every ${UPDATE_INTERVAL} samples"
echo "Checkpoints saved every ${CHECKPOINT_INTERVAL} updates"
echo ""
echo "Press Ctrl+C to stop"
echo ""

LOG_FILE="logs/training_$(date +%Y%m%d_%H%M%S).log"

python3 v620_ppo_trainer.py \
  --port ${PORT} \
  --update-interval ${UPDATE_INTERVAL} \
  --checkpoint-interval ${CHECKPOINT_INTERVAL} \
  --checkpoint-dir ./checkpoints \
  --tensorboard-dir ./runs \
  2>&1 | tee "${LOG_FILE}" &

TRAINING_PID=$!

echo "Training server PID: ${TRAINING_PID}"
echo "Log file: ${LOG_FILE}"
echo ""
echo "Useful commands:"
echo "  - Monitor GPU: watch -n 1 rocm-smi"
echo "  - View logs: tail -f ${LOG_FILE}"
echo "  - Deploy model: ./deploy_model.sh checkpoints/ppo_v620_update_XX.pt ROVER_IP USER"
echo ""

# Wait for training server
wait $TRAINING_PID
