#!/bin/bash

# SAC Training Server Startup Script for V620
# Trains a continuous SAC policy using ROCm acceleration with Unified BEV architecture

set -e

echo "=================================================="
echo "SAC Training Server (V620) - Unified BEV Architecture"
echo "=================================================="

# Configuration
NATS_SERVER=${1:-"nats://nats.gokickrocks.org:4222"}
CHECKPOINT_DIR=${2:-./checkpoints_sac}
LOG_DIR=${3:-./logs_sac}
BATCH_SIZE=${4:-1536}  # Reduced from 1024 for faster gradient steps and lower memory usage
BUFFER_SIZE=${5:-500000}  # 500k samples (~16.5GB VRAM on 32GB GPU)
GPU_BUFFER=true  # Store buffer on GPU for faster sampling

echo "Configuration:"
echo "  NATS Server: ${NATS_SERVER}"
echo "  Checkpoint Dir: ${CHECKPOINT_DIR}"
echo "  Log Dir: ${LOG_DIR}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Buffer Size: ${BUFFER_SIZE}"
echo "  GPU Buffer: ${GPU_BUFFER}"
echo "  Architecture: Unified BEV (LiDAR + Depth fusion)"
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

# Check MPS (Apple Silicon)
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('✓ MPS (Apple Silicon) GPU detected')
    print(f'  MPS built: {torch.backends.mps.is_built()}')
    print('  Backend: MPS')
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
python3 -c "import flask" 2>/dev/null || MISSING_PACKAGES+=("flask")

if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
  echo "❌ Missing packages: ${MISSING_PACKAGES[@]}"
  echo ""
  echo "Install with:"
  echo "  pip3 install ${MISSING_PACKAGES[@]}"
  exit 1
fi

echo "✓ All required packages installed"

# Check for existing processes
echo ""
echo "Checking for existing processes..."

if pgrep -f v620_sac_trainer > /dev/null; then
  echo "⚠ Found old v620_sac_trainer process(es), killing..."
  pkill -f v620_sac_trainer
  sleep 2
fi

echo "✓ Ready to start"

# ROCm optimizations (environment variables)
if [ "$OS_TYPE" = "Linux" ] && command -v rocm-smi &> /dev/null; then
  echo ""
  echo "Applying ROCm optimizations..."
  export HSA_FORCE_FINE_GRAIN_PCIE=1
  export MIOPEN_FIND_ENFORCE=NONE
  export MIOPEN_DISABLE_CACHE=0 # Enable cache to speed up startup after first run

  # V620-specific optimizations
  export PYTORCH_ROCM_ARCH="gfx1030"  # V620 architecture (Navi 21)
  export MIOPEN_DEBUG_DISABLE_FIND_DB=0  # Enable find-db for faster ops
  export HSA_ENABLE_SDMA=0  # Disable SDMA for better performance
  export MIOPEN_FIND_MODE=NORMAL  # Use normal find mode
  export HSA_OVERRIDE_GFX_VERSION=10.3.0  # Ensure correct GFX version

  # Memory allocator tuning for torch.compile + training workloads
  export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"

  # torch.compile backend for ROCm
  export TORCHINDUCTOR_FORCE_DISABLE_CACHES=0  # Enable inductor caching

  echo "✓ V620-optimized ROCm environment variables set"

  # Check available VRAM
  echo ""
  echo "Checking GPU memory..."
  if rocm-smi --showmeminfo vram 2>/dev/null | grep -q "Total"; then
    TOTAL_VRAM=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "VRAM Total Memory" | awk '{print $5}')
    echo "  Total VRAM: ${TOTAL_VRAM} MB"
    # Note: Batch size of 256 requires ~8GB
  fi
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

# Raise file descriptor limit if possible
if ! ulimit -n 65535 2>/dev/null; then
    echo "⚠ Warning: Failed to raise ulimit to 65535. Current limit: $(ulimit -n)"
    echo "  Proceeding anyway..."
else
    echo "✓ Raised file descriptor limit to 65535"
fi

echo "Starting V620 SAC Trainer..."

# Build GPU buffer flag
GPU_BUFFER_FLAG=""
if [ "${GPU_BUFFER}" = "true" ]; then
  GPU_BUFFER_FLAG="--gpu-buffer"
fi

# Launch Python with process substitution so $! captures Python's PID (not tee's).
# This ensures kill $TRAINER_PID sends SIGTERM to Python, which runs the buffer save handler.
python3 -u v620_sac_trainer.py \
  --nats_server "${NATS_SERVER}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --log_dir "${LOG_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --buffer_size "${BUFFER_SIZE}" \
  ${GPU_BUFFER_FLAG} \
  "$@" > >(tee "${LOG_FILE}") 2>&1 &

TRAINER_PID=$!

echo "SAC server PID: ${TRAINER_PID}"
echo "Log file: ${LOG_FILE}"
echo ""

# Set up signal handling — send SIGTERM to Python directly
cleanup() {
  echo ""
  echo "🛑 Shutting down SAC server..."
  kill -TERM $TRAINER_PID 2>/dev/null || true
  # Wait for Python to finish saving replay buffer
  wait $TRAINER_PID 2>/dev/null
  kill $TB_PID 2>/dev/null || true
  echo "✅ Shutdown complete"
  exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for trainer
wait $TRAINER_PID
