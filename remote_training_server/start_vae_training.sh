#!/bin/bash
# Script to easily launch the VAE/AE pre-training

BUFFER_PATH=${1:-"checkpoints_sac/replay_buffer.pt"}
SAVE_DIR=${2:-"checkpoints_sac/vae"}
EPOCHS=${3:-50}

echo "=========================================="
echo "Starting BEV Autoencoder Pre-training"
echo "=========================================="
echo "Buffer Path: ${BUFFER_PATH}"
echo "Save Dir: ${SAVE_DIR}"
echo "Epochs: ${EPOCHS}"

# Raise file descriptor limit for MIOpen SQLite cache
ulimit -n 65535 2>/dev/null && echo "✓ Raised ulimit to 65535" || echo "⚠ Could not raise ulimit"

# ROCm environment (match SAC server settings)
if command -v rocm-smi &> /dev/null; then
  export MIOPEN_DISABLE_CACHE=0
  export MIOPEN_FIND_MODE=NORMAL
  export MIOPEN_DEBUG_DISABLE_FIND_DB=0
  export HSA_ENABLE_SDMA=0
  export HSA_OVERRIDE_GFX_VERSION=10.3.0
  export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
  echo "✓ ROCm environment set"
fi

# Run the training script
python3 train_bev_vae.py \
    --buffer_path "${BUFFER_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --epochs ${EPOCHS} \
    --batch_size 256 \
    --lr 0.001 \
    --noise_factor 0.1 \
    --gpu_buffer \
    --forward_prediction \
    --fwd_weight 1.0
