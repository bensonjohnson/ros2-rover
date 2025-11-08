#!/bin/bash

# PGPE Evolution Server Startup Script for V620
# Trains a single adaptive rover driving policy using PGPE (EvoTorch)

set -e

echo "=================================================="
echo "PGPE Evolution Training Server (V620 + EvoTorch)"
echo "=================================================="

# Configuration
PORT=${1:-5556}
NUM_EVALUATIONS=${2:-1000}
CHECKPOINT_DIR=${3:-./checkpoints}
POPULATION_SIZE=${4:-20}
CENTER_LR=${5:-0.01}
STDEV_LR=${6:-0.001}

echo "Configuration:"
echo "  ZeroMQ Port: ${PORT}"
echo "  Target Evaluations: ${NUM_EVALUATIONS}"
echo "  Checkpoint Dir: ${CHECKPOINT_DIR}"
echo "  Population Size: ${POPULATION_SIZE}"
echo "  Center LR: ${CENTER_LR}"
echo "  Stdev LR: ${STDEV_LR}"
echo ""

# Check we're in the right directory
if [ ! -f "v620_pgpe_trainer.py" ]; then
  echo "❌ Error: Please run this script from remote_training_server directory"
  echo "Current directory: $(pwd)"
  exit 1
fi

# Create directories
echo "Creating directories..."
mkdir -p ${CHECKPOINT_DIR} export logs

# Activate virtual environment
echo "Activating virtual environment..."
if [ -d "/root/rocm-train/bin" ]; then
  source /root/rocm-train/bin/activate
  echo "✓ Virtual environment activated"
else
  echo "❌ Error: /root/rocm-train virtual environment not found"
  exit 1
fi

# Check Python version
echo ""
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
python3 -c "import evotorch" 2>/dev/null || MISSING_PACKAGES+=("evotorch")

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

if pgrep -f v620_pgpe_trainer > /dev/null; then
  echo "⚠ Found old v620_pgpe_trainer process(es), killing..."
  pkill -f v620_pgpe_trainer
  sleep 2
fi

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

# Set up signal handling
cleanup() {
  echo ""
  echo "🛑 Shutting down PGPE server..."
  kill $TRAINER_PID 2>/dev/null || true
  echo "✅ Shutdown complete"
  exit 0
}

trap cleanup SIGINT SIGTERM

# Start PGPE server
echo ""
echo "=================================================="
echo "Starting PGPE Evolution Trainer"
echo "=================================================="
echo ""
echo "Listening for episode results on port ${PORT}"
echo "Target: ${NUM_EVALUATIONS} episode evaluations"
echo "Population size: ${POPULATION_SIZE} models"
echo ""
echo "Press Ctrl+C to stop"
echo ""

LOG_FILE="logs/pgpe_$(date +%Y%m%d_%H%M%S).log"

python3 v620_pgpe_trainer.py \
  --port ${PORT} \
  --num-evaluations ${NUM_EVALUATIONS} \
  --checkpoint-dir ${CHECKPOINT_DIR} \
  --population-size ${POPULATION_SIZE} \
  --center-lr ${CENTER_LR} \
  --stdev-lr ${STDEV_LR} \
  2>&1 | tee "${LOG_FILE}" &

TRAINER_PID=$!

echo "PGPE server PID: ${TRAINER_PID}"
echo "Log file: ${LOG_FILE}"
echo ""
echo "Algorithm features:"
echo "  ✓ PGPE (Policy Gradients with Parameter Exploration)"
echo "  ✓ Adam optimizer for distribution updates"
echo "  ✓ Centered ranking for gradient estimates"
echo "  ✓ Adaptive learning rates"
echo "  ✓ Population size: ${POPULATION_SIZE}"
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
echo "Advantages over MAP-Elites:"
echo "  ✓ Gradient-based distribution updates (faster convergence)"
echo "  ✓ Sample efficient (~20 pop vs 25)"
echo "  ✓ Production-ready infrastructure (EvoTorch)"
echo "  ✓ Less custom code to maintain"
echo ""

# Wait for trainer
wait $TRAINER_PID
