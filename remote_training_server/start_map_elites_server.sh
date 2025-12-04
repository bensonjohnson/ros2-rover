#!/bin/bash

# Single-Population Evolution Server Startup Script for V620
# Trains a single adaptive rover driving policy using evolutionary algorithms

set -e

echo "=================================================="
echo "Evolution Training Server (V620)"
echo "=================================================="

# Configuration
PORT=${1:-5556}
NUM_EVALUATIONS=${2:-1000}
CHECKPOINT_DIR=${3:-./checkpoints}
INITIAL_POP=${4:-10}
MAX_POP=${5:-25}

echo "Configuration:"
echo "  ZeroMQ Port: ${PORT}"
echo "  Target Evaluations: ${NUM_EVALUATIONS}"
echo "  Checkpoint Dir: ${CHECKPOINT_DIR}"
echo "  Population Size: ${INITIAL_POP} → ${MAX_POP} (adaptive)"
echo ""

# Check we're in the right directory
if [ ! -f "v620_map_elites_trainer.py" ]; then
  echo "❌ Error: Please run this script from remote_training_server directory"
  echo "Current directory: $(pwd)"
  exit 1
fi

# Create directories
echo "Creating directories..."
mkdir -p ${CHECKPOINT_DIR} export logs

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
    print('✓ Apple MPS (Metal) backend detected')
    print(f'  Device: {torch.device(\"mps\")}')
    print('  Backend: MPS')
    sys.exit(0)

# Fallback to CPU
print('⚠ No GPU acceleration detected! Will use CPU (slower)')
print('  Supported backends: CUDA/ROCm (Linux/Windows), MPS (macOS)')
print('  Backend: CPU')
" 2>&1

# Check required packages
echo ""
echo "Checking required packages..."

MISSING_PACKAGES=()

python3 -c "import torch" 2>/dev/null || MISSING_PACKAGES+=("torch")
python3 -c "import zmq" 2>/dev/null || MISSING_PACKAGES+=("pyzmq")
python3 -c "import numpy" 2>/dev/null || MISSING_PACKAGES+=("numpy")

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

if pgrep -f v620_map_elites_trainer > /dev/null; then
  echo "⚠ Found old v620_map_elites_trainer process(es), killing..."
  pkill -f v620_map_elites_trainer
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

  # Disable MIOpen auto-tuning completely (use default kernels)
  export MIOPEN_FIND_ENFORCE=NONE
  export MIOPEN_DISABLE_CACHE=1

  # Additional memory management for new PyTorch/ROCm versions
  export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
  export HIP_VISIBLE_DEVICES=0  # Use only first GPU
  export MIOPEN_LOG_LEVEL=3     # Reduce MIOpen verbosity

  echo "✓ ROCm environment variables set (MIOpen auto-tuning disabled)"
  echo "✓ Memory management: max_split_size_mb=512"
fi

# Set up signal handling
cleanup() {
  echo ""
  echo "🛑 Shutting down Evolution server..."
  kill $TRAINER_PID 2>/dev/null || true
  echo "✅ Shutdown complete"
  exit 0
}

trap cleanup SIGINT SIGTERM

# Start Evolution server
echo ""
echo "=================================================="
echo "Starting Evolution Trainer"
echo "=================================================="
echo ""
echo "Listening for episode results on port ${PORT}"
echo "Target: ${NUM_EVALUATIONS} episode evaluations"
echo "Population size: ${POPULATION_SIZE} models"
echo ""
echo "Press Ctrl+C to stop"
echo ""

LOG_FILE="logs/evolution_$(date +%Y%m%d_%H%M%S).log"

python3 v620_map_elites_trainer.py \
  --port ${PORT} \
  --num-evaluations ${NUM_EVALUATIONS} \
  --checkpoint-dir ${CHECKPOINT_DIR} \
  --initial-population ${INITIAL_POP} \
  --max-population ${MAX_POP} \
  2>&1 | tee "${LOG_FILE}" &

TRAINER_PID=$!

echo "Evolution server PID: ${TRAINER_PID}"
echo "Log file: ${LOG_FILE}"
echo ""
echo "Advanced features enabled:"
echo "  ✓ Adaptive population (10→25 models)"
echo "  ✓ Adaptive tournament sizes (75-500 candidates)"
echo "  ✓ Multi-tournament for champions (3x parallel)"
echo "  ✓ Diversity bonus (explores novel behaviors)"
echo "  ✓ Warmup acceleration (heuristic seeding)"
echo "  ✓ Variable episode duration (30-75s)"
echo ""
echo "The model will learn to:"
echo "  - Navigate collision-free through obstacles"
echo "  - Adapt speed based on clearance (fast when safe, slow when risky)"
echo "  - Use smooth, efficient motion patterns"
echo "  - Prefer forward movement over spinning"
echo ""
echo "Best model will be exported to:"
echo "  checkpoints/best_models/best_final.pt"
echo ""

# Wait for trainer
wait $TRAINER_PID
