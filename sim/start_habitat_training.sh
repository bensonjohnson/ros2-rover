#!/bin/bash

# Habitat Simulation Training Startup Script
# Runs MAP-Elites training using Habitat 3.0 simulator

set -e

echo "=================================================="
echo "Habitat MAP-Elites Training"
echo "=================================================="

# Activate conda environment
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate habitat
    echo "✓ Activated conda environment: habitat"
else
    echo "⚠ Warning: Conda not found at $HOME/miniconda3"
    echo "  Make sure habitat-lab and dependencies are installed for system python3"
fi

# Force OpenGL rendering for AMD GPU (disable EGL/CUDA)
export MAGNUM_DEVICE="GLX"
export HABITAT_SIM_LOG="quiet"
export MAGNUM_LOG="quiet"

# Configuration
SERVER_ADDR=${1:-tcp://localhost:5556}
NUM_EPISODES=${2:-1000}
EPISODE_DURATION=${3:-60.0}
DEVICE=${4:-cuda}

echo "Configuration:"
echo "  Server Address: ${SERVER_ADDR}"
echo "  Episodes: ${NUM_EPISODES}"
echo "  Episode Duration: ${EPISODE_DURATION}s"
echo "  Device: ${DEVICE}"
echo ""

# Check we're in the right directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "${SCRIPT_DIR}/habitat_episode_runner.py" ]; then
  echo "❌ Error: Please run this script from the sim directory"
  echo "Current directory: $(pwd)"
  exit 1
fi

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python version: ${PYTHON_VERSION}"

# Check for Habitat installation
echo ""
echo "Checking Habitat installation..."
python3 -c "import habitat; import habitat_sim" 2>/dev/null && echo "  ✓ Habitat installed" || {
  echo "  ❌ Habitat not installed!"
  echo ""
  echo "Install with:"
  echo "  conda install habitat-sim -c conda-forge -c aihabitat"
  echo "  pip install habitat-lab"
  echo ""
  echo "Or see: https://github.com/facebookresearch/habitat-sim"
  exit 1
}

# Check for PyTorch
echo ""
echo "Checking PyTorch..."
python3 -c "import torch; print(f'  ✓ PyTorch {torch.__version__}')" 2>/dev/null || {
  echo "  ❌ PyTorch not installed!"
  echo ""
  echo "Install with:"
  echo "  pip install torch torchvision"
  exit 1
}

# Check for ZeroMQ
echo ""
echo "Checking ZeroMQ..."
python3 -c "import zmq; print(f'  ✓ PyZMQ {zmq.zmq_version()}')" 2>/dev/null || {
  echo "  ❌ PyZMQ not installed!"
  echo ""
  echo "Install with:"
  echo "  pip install pyzmq"
  exit 1
}

# Check for Zstandard (compression)
echo ""
echo "Checking Zstandard..."
python3 -c "import zstandard; print('  ✓ Zstandard installed')" 2>/dev/null || {
  echo "  ❌ Zstandard not installed!"
  echo ""
  echo "Install with:"
  echo "  pip install zstandard"
  exit 1
}

# Check GPU availability (if using CUDA)
if [ "$DEVICE" = "cuda" ]; then
  echo ""
  echo "Checking GPU availability..."
  python3 -c "
import torch
if torch.cuda.is_available():
    print(f'  ✓ GPU: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA version: {torch.version.cuda}')
else:
    print('  ⚠ No GPU detected! Will use CPU (slower)')
    print('  Consider setting DEVICE=cpu explicitly')
" 2>&1
fi

# Check if server is reachable
echo ""
echo "Checking server connectivity..."
if [[ $SERVER_ADDR == tcp://* ]]; then
  # Extract host and port
  HOST_PORT="${SERVER_ADDR#tcp://}"
  HOST="${HOST_PORT%:*}"
  PORT="${HOST_PORT#*:}"

  # Try to connect (timeout 2s)
  timeout 2 bash -c "cat < /dev/null > /dev/tcp/${HOST}/${PORT}" 2>/dev/null && \
    echo "  ✓ Server reachable at ${SERVER_ADDR}" || \
    echo "  ⚠ Cannot reach server at ${SERVER_ADDR} (may not be started yet)"
else
  echo "  ⚠ Non-TCP address, skipping connectivity check"
fi

# Set up signal handling
cleanup() {
  echo ""
  echo "🛑 Shutting down Habitat training..."
  kill $RUNNER_PID 2>/dev/null || true
  echo "✅ Shutdown complete"
  exit 0
}

trap cleanup SIGINT SIGTERM

# Create log directory
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/habitat_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "=================================================="
echo "Starting Habitat Episode Runner"
echo "=================================================="
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run habitat episode runner
python3 "${SCRIPT_DIR}/habitat_episode_runner.py" \
  --server "${SERVER_ADDR}" \
  --episodes "${NUM_EPISODES}" \
  --duration "${EPISODE_DURATION}" \
  --device "${DEVICE}" \
  2>&1 | tee "${LOG_FILE}" &

RUNNER_PID=$!

echo "Runner PID: ${RUNNER_PID}"
echo "Log file: ${LOG_FILE}"
echo ""

# Wait for runner
wait $RUNNER_PID

echo ""
echo "✅ Training complete!"
