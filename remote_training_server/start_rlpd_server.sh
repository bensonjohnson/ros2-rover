#!/bin/bash

# RLPD + HIL-SERL Training Server Startup Script
# Auto-detects hardware (ROCm V620 / CUDA / Blackwell) and configures AMP
# training. Mirrors start_dreamer_server.sh but targets v620_rlpd_trainer.py
# with separate checkpoint/log directories and bumped dashboard/TensorBoard
# ports so Dreamer and RLPD can coexist on the same host.

set -e

# Detect hardware backend
if command -v nvidia-smi &> /dev/null; then
  CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  echo "=================================================="
  echo "RLPD + HIL-SERL Training Server (CUDA)"
  echo "=================================================="
  echo ""
  echo "CUDA GPU Detected:"
  echo "  Driver: ${CUDA_VERSION}"
  echo "  GPU: ${GPU_NAME}"
elif command -v rocm-smi &> /dev/null; then
  ROCM_VERSION=$(rocm-smi --version 2>/dev/null | head -1)
  GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | grep -A1 "GPU" | tail -1 | tr -d ' ' || echo "Unknown")
  echo "=================================================="
  echo "RLPD + HIL-SERL Training Server (ROCm) - V620"
  echo "=================================================="
  echo ""
  echo "ROCm GPU Detected:"
  echo "  Version: ${ROCM_VERSION}"
  echo "  GPU: ${GPU_NAME}"
else
  echo "=================================================="
  echo "RLPD + HIL-SERL Training Server (No GPU) - CPU Fallback"
  echo "=================================================="
fi

# Configuration
CHECKPOINT_DIR=${1:-./checkpoints_rlpd}
LOG_DIR=${2:-./logs_rlpd}

echo ""
echo "Configuration:"
echo "  ZMQ PULL port: 5555 (chunks from rover)"
echo "  ZMQ PUB port:  5556 (models to rover)"
echo "  Dashboard:     http://localhost:8081"
echo "  TensorBoard:   http://localhost:6008"
echo "  Checkpoint Dir: ${CHECKPOINT_DIR}"
echo "  Log Dir: ${LOG_DIR}"
echo "  Architecture: RLPD v3 (SAC + critic ensemble; Depth 96×72 + 1D LiDAR 360 + proprio)"
echo "  Algorithm: HIL-SERL (50/50 demo replay + intervention reward)"
echo ""

if [ ! -f "v620_rlpd_trainer.py" ]; then
  echo "Error: Please run this script from remote_training_server directory"
  echo "Current directory: $(pwd)"
  exit 1
fi

mkdir -p ${CHECKPOINT_DIR} ${LOG_DIR}

# Checkpoint cleanup menu
echo "=================================================="
echo "Clean up RLPD checkpoint directory?"
echo "=================================================="
echo ""
echo "  1) Keep everything (default)"
echo "  2) Delete RLPD checkpoints only"
echo "  3) Delete checkpoints + ONNX (keeps demos.npz)"
echo "  4) Delete ALL including demos.npz [DESTROYS HUMAN DATA]"
echo ""
read -rp "Select [1-4] (default: 1): " PRUNE_CHOICE
PRUNE_CHOICE=${PRUNE_CHOICE:-1}

if [[ "${PRUNE_CHOICE}" =~ ^[2-4]$ ]]; then
  CKPT_COUNT=$(find "${CHECKPOINT_DIR}" -maxdepth 1 -name "rlpd_update_*.pt" 2>/dev/null | wc -l | tr -d ' ')
  ONNX_COUNT=$(find "${CHECKPOINT_DIR}" -maxdepth 1 -name "*.onnx" 2>/dev/null | wc -l | tr -d ' ')
  HAS_SIMHASH=$([ -f "${CHECKPOINT_DIR}/lifetime_simhash.pkl" ] && echo yes || echo no)
  HAS_DEMOS=$([ -f "${CHECKPOINT_DIR}/demos.npz" ] && echo yes || echo no)
  DEMO_SIZE=$([ -f "${CHECKPOINT_DIR}/demos.npz" ] && du -h "${CHECKPOINT_DIR}/demos.npz" | cut -f1 || echo "0")

  echo ""
  echo "The following will be deleted:"
  case "${PRUNE_CHOICE}" in
    2)
      echo "  - RLPD checkpoints (${CKPT_COUNT} files)"
      ;;
    3)
      echo "  - RLPD checkpoints (${CKPT_COUNT} files)"
      echo "  - ONNX/RKNN exports (${ONNX_COUNT} files)"
      echo "  - Lifetime SimHash novelty memory (${HAS_SIMHASH})"
      ;;
    4)
      echo "  - RLPD checkpoints (${CKPT_COUNT} files)"
      echo "  - ONNX/RKNN exports (${ONNX_COUNT} files)"
      echo "  - Lifetime SimHash novelty memory (${HAS_SIMHASH})"
      echo "  - DEMO REPLAY BUFFER (${HAS_DEMOS}, ${DEMO_SIZE}) [EXPENSIVE TO REPLACE]"
      ;;
  esac

  if [ "${PRUNE_CHOICE}" = "4" ]; then
    echo ""
    echo "*** WARNING ***  demos.npz contains hand-collected teleop data."
    echo "Deleting it means re-driving the rover by hand to rebuild the demo buffer."
    read -rp "Type 'DELETE DEMOS' (literal) to confirm: " DEMO_CONFIRM
    if [ "${DEMO_CONFIRM}" != "DELETE DEMOS" ]; then
      echo "  -> Demo deletion aborted; falling back to option 3"
      PRUNE_CHOICE=3
    fi
  fi

  read -rp "Confirm deletion? [y/N]: " CONFIRM
  if [[ "${CONFIRM}" =~ ^[Yy]$ ]]; then
    case "${PRUNE_CHOICE}" in
      2) rm -f "${CHECKPOINT_DIR}"/rlpd_update_*.pt "${CHECKPOINT_DIR}"/latest_rlpd.pt ;;
      3)
        rm -f "${CHECKPOINT_DIR}"/rlpd_update_*.pt "${CHECKPOINT_DIR}"/*.onnx \
              "${CHECKPOINT_DIR}"/latest_rlpd.pt "${CHECKPOINT_DIR}"/latest_actor.onnx \
              "${CHECKPOINT_DIR}"/latest_actor.rknn "${CHECKPOINT_DIR}"/lifetime_simhash.pkl
        ;;
      4)
        rm -f "${CHECKPOINT_DIR}"/rlpd_update_*.pt "${CHECKPOINT_DIR}"/*.onnx \
              "${CHECKPOINT_DIR}"/latest_rlpd.pt "${CHECKPOINT_DIR}"/latest_actor.onnx \
              "${CHECKPOINT_DIR}"/latest_actor.rknn "${CHECKPOINT_DIR}"/lifetime_simhash.pkl \
              "${CHECKPOINT_DIR}"/demos.npz
        ;;
    esac
    echo "Cleanup complete"
  else
    echo "  -> Skipped cleanup"
  fi
fi

echo ""
echo "Checking Python version..."
python3 --version

echo ""
echo "Checking required packages..."
MISSING_PACKAGES=()
python3 -c "import torch" 2>/dev/null || MISSING_PACKAGES+=("torch")
python3 -c "import torchvision" 2>/dev/null || MISSING_PACKAGES+=("torchvision")
python3 -c "import zmq" 2>/dev/null || MISSING_PACKAGES+=("pyzmq")
python3 -c "import numpy" 2>/dev/null || MISSING_PACKAGES+=("numpy")
python3 -c "import tensorboard" 2>/dev/null || MISSING_PACKAGES+=("tensorboard")
python3 -c "import msgpack" 2>/dev/null || MISSING_PACKAGES+=("msgpack")
python3 -c "import zstandard" 2>/dev/null || MISSING_PACKAGES+=("zstandard")
python3 -c "import onnx" 2>/dev/null || MISSING_PACKAGES+=("onnx")

if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
  echo "Missing packages: ${MISSING_PACKAGES[@]}"
  echo "Install with: pip3 install ${MISSING_PACKAGES[@]}"
  exit 1
fi
echo "All required packages installed"

echo ""
echo "Verifying PyTorch acceleration..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'AMP: Enabled')
else:
    print('No GPU - CPU training (slower)')
"

echo ""
if pgrep -f v620_rlpd_trainer > /dev/null; then
  echo "Found old v620_rlpd_trainer process(es), killing..."
  pkill -f v620_rlpd_trainer
  sleep 2
fi

# Hardware-specific optimizations (identical to start_dreamer_server.sh)
OS_TYPE=$(uname -s)
if [ "$OS_TYPE" = "Linux" ]; then
  if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    echo "Applying CUDA optimizations for ${GPU_NAME}..."
    export CUDA_MODULE_LOADING=LAZY
    export CUDA_LAUNCH_BLOCKING=0
    export CUDA_VISIBLE_DEVICES=0
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

    if echo "${GPU_NAME}" | grep -qiE "B200|B100|GB10|GB200|Blackwell"; then
      echo "  Blackwell detected - BF16 + TF32 (no GradScaler)"
      export CUDNN_FRONTEND_API=1
      export CUDNN_FRONTEND_LOG_LEVEL=0
      export CUDNN_MATMUL_USE_THREE_RING=1
      export CUDA_DEVICE_MAX_CONNECTIONS=32
      export TORCH_CUDNN_V8_API_ENABLED=1
      export PYTORCH_JIT=1
      export CUDNN_BENCHMARK=1
      export CUDNN_DETERMINISTIC=0
      export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,pinned_use_cuda_host_register:False,garbage_collection_threshold:0.8"
      echo "  Blackwell multi-chip interconnect: ENABLED"
      echo "  cuDNN v8 API: ENABLED"
      echo "  TorchScript JIT: ENABLED"
    fi
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

if ulimit -n 65535 2>/dev/null; then
  echo "Raised file descriptor limit to 65535"
fi

echo ""
echo "Starting TensorBoard..."
tensorboard --logdir "$(pwd)/${LOG_DIR}" --port 6008 --bind_all > /dev/null 2>&1 &
TB_PID=$!
echo "TensorBoard running on http://localhost:6008 (PID: ${TB_PID})"

LOG_FILE="${LOG_DIR}/rlpd_server_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "Starting V620 RLPD Trainer..."

python3 -u v620_rlpd_trainer.py \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --log_dir "${LOG_DIR}" \
  "${@:3}" > >(tee "${LOG_FILE}") 2>&1 &

TRAINER_PID=$!

echo "RLPD server PID: ${TRAINER_PID}"
echo "Log file: ${LOG_FILE}"
echo ""
echo "=================================================="
echo "Monitoring:"
echo "  Dashboard: http://$(hostname -I | awk '{print $1}'):8081"
echo "  TensorBoard: http://localhost:6008"
echo "=================================================="
echo ""

cleanup() {
  echo ""
  echo "Shutting down RLPD server..."
  kill -TERM $TRAINER_PID 2>/dev/null || true
  wait $TRAINER_PID 2>/dev/null
  kill $TB_PID 2>/dev/null || true
  echo "Shutdown complete"
  exit 0
}

trap cleanup SIGINT SIGTERM

wait $TRAINER_PID
