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
else:
    print('⚠ No GPU detected! Will use CPU (slower)')
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
