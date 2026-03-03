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

# Run the training script
python3 train_bev_vae.py \
    --buffer_path "${BUFFER_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --epochs ${EPOCHS} \
    --batch_size 256 \
    --lr 0.001 \
    --noise_factor 0.1 \
    --gpu_buffer
