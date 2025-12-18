#!/bin/bash
# Startup script for V700 ES-SAC Server

NATS_SERVER=${1:-"nats://nats.gokickrocks.org:4222"}

echo "=============================================="
echo "Starting ES-SAC Hybrid Trainer (V700)"
echo "=============================================="

# Ensure output directory exists
mkdir -p logs_es checkpoints_es

# Enable ROCm Optimizations
export HSA_FORCE_FINE_GRAIN_PCIE=1
export MIOPEN_FIND_ENFORCE=NONE

# Run
python3 v700_es_sac_trainer.py \
    --nats_server "${NATS_SERVER}" \
    --checkpoint_dir "./checkpoints_es" \
    --log_dir "./logs_es" \
    2>&1 | tee "logs_es/server_$(date +%Y%m%d_%H%M%S).log"
