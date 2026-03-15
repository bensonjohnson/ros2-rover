#!/bin/bash

# SAC Training Server Startup Script
# Auto-detects hardware (ROCm V620 / CUDA Grace Blackwell) and configures FP16 training

set -e

# Detect hardware backend
CUDA_AVAILABLE=0
ROCM_AVAILABLE=0
CUDA_VERSION=""
ROCm_VERSION=""

if command -v nvidia-smi &> /dev/null; then
  CUDA_AVAILABLE=1
  CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  echo "=================================================="
  echo "SAC Training Server (CUDA) - Grace Blackwell / H100"
  echo "=================================================="
  echo ""
  echo "CUDA GPU Detected:"
  echo "  Driver: ${CUDA_VERSION}"
  echo "  GPU: ${GPU_NAME}"
elif command -v rocm-smi &> /dev/null; then
  ROCM_AVAILABLE=1
  ROCM_VERSION=$(rocm-smi --version 2>/dev/null | head -1)
  GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | grep -A1 "GPU" | tail -1 | tr -d ' ' || echo "Unknown")
  echo "=================================================="
  echo "SAC Training Server (ROCm) - V620 Unified BEV Architecture"
  echo "=================================================="
  echo ""
  echo "ROCm GPU Detected:"
  echo "  Version: ${ROCM_VERSION}"
  echo "  GPU: ${GPU_NAME}"
else
  echo "=================================================="
  echo "SAC Training Server (No GPU) - CPU Fallback"
  echo "=================================================="
fi

# Configuration
NATS_SERVER=${1:-"nats://nats.gokickrocks.org:4222"}
CHECKPOINT_DIR=${2:-./checkpoints_sac}
LOG_DIR=${3:-./logs_sac}
BATCH_SIZE=${4:-256}  # Standard SAC batch size; more gradient steps/sec > larger batches for off-policy RL
BUFFER_SIZE=${5:-750000}  # 750k samples (~24GB VRAM) - Optimized for 32GB GPU with model overhead
GPU_BUFFER=true  # Store buffer on GPU to maximize VRAM utilization (32GB GPU can handle ~950k samples)

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

# --- Checkpoint Pruning Menu ---
echo ""
echo "=================================================="
echo "Clean up checkpoint directory?"
echo "=================================================="
echo ""
echo "  1) Keep everything (default)"
echo "  2) Delete SAC checkpoints only"
echo "  3) Delete replay buffer only"
echo "  4) Delete SAC checkpoints + replay buffer"
echo "  5) Delete ALL (checkpoints + buffer + ONNX + VAE)"
echo ""
read -rp "Select [1-5] (default: 1): " PRUNE_CHOICE
PRUNE_CHOICE=${PRUNE_CHOICE:-1}

if [[ "${PRUNE_CHOICE}" =~ ^[2-5]$ ]]; then
  # Count files that would be deleted
  SAC_COUNT=$(find "${CHECKPOINT_DIR}" -maxdepth 1 -name "sac_step_*.pt" 2>/dev/null | wc -l | tr -d ' ')
  ONNX_COUNT=$(find "${CHECKPOINT_DIR}" -maxdepth 1 -name "*.onnx" 2>/dev/null | wc -l | tr -d ' ')
  BUFFER_EXISTS="no"
  [ -f "${CHECKPOINT_DIR}/replay_buffer.pt" ] && BUFFER_EXISTS="yes"
  VAE_EXISTS="no"
  [ -d "${CHECKPOINT_DIR}/vae" ] && VAE_EXISTS="yes"

  echo ""
  echo "The following will be deleted:"
  case "${PRUNE_CHOICE}" in
    2)
      echo "  - SAC checkpoints (${SAC_COUNT} files)"
      echo "  - ONNX exports (${ONNX_COUNT} files)"
      ;;
    3)
      echo "  - Replay buffer (exists: ${BUFFER_EXISTS})"
      ;;
    4)
      echo "  - SAC checkpoints (${SAC_COUNT} files)"
      echo "  - ONNX exports (${ONNX_COUNT} files)"
      echo "  - Replay buffer (exists: ${BUFFER_EXISTS})"
      ;;
    5)
      echo "  - SAC checkpoints (${SAC_COUNT} files)"
      echo "  - ONNX exports (${ONNX_COUNT} files)"
      echo "  - Replay buffer (exists: ${BUFFER_EXISTS})"
      echo "  - VAE directory (exists: ${VAE_EXISTS})"
      ;;
  esac

  read -rp "Confirm deletion? [y/N]: " CONFIRM
  if [[ "${CONFIRM}" =~ ^[Yy]$ ]]; then
    case "${PRUNE_CHOICE}" in
      2)
        rm -f "${CHECKPOINT_DIR}"/sac_step_*.pt
        rm -f "${CHECKPOINT_DIR}"/*.onnx
        ;;
      3)
        rm -f "${CHECKPOINT_DIR}"/replay_buffer.pt
        ;;
      4)
        rm -f "${CHECKPOINT_DIR}"/sac_step_*.pt
        rm -f "${CHECKPOINT_DIR}"/*.onnx
        rm -f "${CHECKPOINT_DIR}"/replay_buffer.pt
        ;;
      5)
        rm -f "${CHECKPOINT_DIR}"/sac_step_*.pt
        rm -f "${CHECKPOINT_DIR}"/*.onnx
        rm -f "${CHECKPOINT_DIR}"/replay_buffer.pt
        rm -rf "${CHECKPOINT_DIR}"/vae
        ;;
    esac
    echo "✓ Cleanup complete"
  else
    echo "  -> Skipped cleanup"
  fi
fi

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
    device_name = torch.cuda.get_device_name(0)
    device_count = torch.cuda.device_count()
    cuda_version = torch.version.cuda
    
    print(f'✓ CUDA/ROCm GPU detected: {device_name}')
    print(f'  Device count: {device_count}')
    print(f'  CUDA version: {cuda_version}')
    print(f'  Backend: CUDA')
    
    # Check for Grace Blackwell (H100/B100 series)
    if 'H100' in device_name or 'B100' in device_name or 'Grace' in device_name:
        print(f'  GPU Type: Grace Blackwell / H100 Series')
        print(f'  FP16 Support: Hardware tensor cores (2x speedup)')
    elif 'V620' in device_name or 'Navi' in device_name or 'RX' in device_name:
        print(f'  GPU Type: AMD ROCm (V620)')
        print(f'  FP16 Support: HIP FP16 (2x speedup)')
    else:
        print(f'  GPU Type: Standard CUDA GPU')
    
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

# --- Pre-trained BEV Encoder Selection ---
ENCODER_PATH="${CHECKPOINT_DIR}/vae/best_bev_encoder.pt"
PRETRAINED_FLAGS=""

if [ -f "${ENCODER_PATH}" ]; then
  echo ""
  echo "=================================================="
  echo "Pre-trained BEV encoder found:"
  echo "  ${ENCODER_PATH}"
  echo "=================================================="
  echo ""
  echo "  1) Use pre-trained encoder (recommended)"
  echo "  2) Use pre-trained encoder + freeze weights"
  echo "  3) Skip (train from scratch)"
  echo ""
  read -rp "Select [1/2/3] (default: 1): " ENCODER_CHOICE
  ENCODER_CHOICE=${ENCODER_CHOICE:-1}

  case "${ENCODER_CHOICE}" in
    1)
      PRETRAINED_FLAGS="--pretrained_encoder ${ENCODER_PATH}"
      echo "  -> Will load pre-trained encoder (fine-tunable)"
      ;;
    2)
      PRETRAINED_FLAGS="--pretrained_encoder ${ENCODER_PATH} --freeze_encoder"
      echo "  -> Will load pre-trained encoder (frozen)"
      ;;
    3)
      echo "  -> Skipping pre-trained encoder, training from scratch"
      ;;
    *)
      echo "  -> Invalid choice, defaulting to option 1"
      PRETRAINED_FLAGS="--pretrained_encoder ${ENCODER_PATH}"
      ;;
  esac
else
  echo ""
  echo "ℹ️  No pre-trained encoder found at ${ENCODER_PATH}"
  echo "   Run start_vae_training.sh first to pre-train, or training from scratch."
fi

# Hardware-specific optimizations (FP16 enabled for both CUDA/ROCm)
if [ "$OS_TYPE" = "Linux" ]; then
  if command -v nvidia-smi &> /dev/null; then
    # CUDA (Grace Blackwell / H100) optimizations
    echo ""
    echo "Applying CUDA optimizations (FP16 enabled)..."
    export CUDA_MODULE_LOADING=LAZY
    export CUDA_LAUNCH_BLOCKING=0
    export NCCL_NVLS_ENABLE=1
    export NCCL_NVLS_SELF=0
    export CUDA_VISIBLE_DEVICES=0
    
    # FP16/TF32 tuning for Blackwell/H100 tensor cores
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    
    # CuDNN frontend API for faster kernel selection
    export CUDNN_FRONTEND_API=1
    export CUDNN_FRONTEND_LOG_LEVEL=0
    
    # Memory optimization
    export CUDA_MAX_WORKER_RANK=0
    
    echo "✓ Grace Blackwell/H100-optimized CUDA environment variables set"
    echo "  FP16 mode: Enabled via AMP in Python trainer"
    
  elif command -v rocm-smi &> /dev/null; then
    # ROCm (V620) optimizations
    echo ""
    echo "Applying ROCm optimizations (FP16 enabled)..."
    export HSA_FORCE_FINE_GRAIN_PCIE=1
    export MIOPEN_FIND_ENFORCE=NONE
    export MIOPEN_DISABLE_CACHE=0
    
    # V620-specific optimizations
    export PYTORCH_ROCM_ARCH="gfx1030"  # V620 architecture (Navi 21)
    export MIOPEN_DEBUG_DISABLE_FIND_DB=0
    export HSA_ENABLE_SDMA=0
    export MIOPEN_FIND_MODE=NORMAL
    export HSA_OVERRIDE_GFX_VERSION=10.3.0
    
    # Memory allocator tuning for ROCm - prevent OOM from fragmentation
    # expandable_segments:True - allows memory reuse across different tensor sizes
    export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
    
    # torch.compile backend for ROCm - reduce memory pressure
    # Disable caching allocator to prevent fragmentation
    export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
    # Limit inductor memory usage
    export TORCHINDUCTOR_CACHE_DIR="/tmp/torch_inductor_cache"
    
    # Prevent memory fragmentation during hipblas operations
    export HSA_SIGNAL_POOL_SIZE=1048576
    
    echo "✓ V620-optimized ROCm environment variables set"
    echo "  FP16 mode: Enabled via AMP in Python trainer"
    echo "  Memory allocator: expandable segments + round-robin"
    echo "  Inductor cache: disabled (prevent fragmentation)"
  fi
fi

# Check available VRAM
echo ""
echo "Checking GPU memory..."
if command -v nvidia-smi &> /dev/null; then
  TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | grep -oE '[0-9]+' | head -1)
  echo "  Total VRAM: ${TOTAL_VRAM} MB"
elif command -v rocm-smi &> /dev/null; then
  if rocm-smi --showmeminfo vram 2>/dev/null | grep -q "Total"; then
    TOTAL_VRAM=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "VRAM Total Memory" | awk '{print $5}')
    echo "  Total VRAM: ${TOTAL_VRAM} MB"
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
  ${PRETRAINED_FLAGS} \
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
