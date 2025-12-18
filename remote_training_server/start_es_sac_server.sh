#!/bin/bash
# Startup script for V700 ES-SAC Server

set -e  # Exit on error

NATS_SERVER=${1:-"nats://nats.gokickrocks.org:4222"}

echo "=============================================="
echo "Starting ES-SAC Hybrid Trainer (V700)"
echo "=============================================="
echo "NATS Server: ${NATS_SERVER}"
echo ""

# Get script directory (works even if called from elsewhere)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

# Verify trainer script exists
if [ ! -f "v700_es_sac_trainer.py" ]; then
    echo "Error: v700_es_sac_trainer.py not found in ${SCRIPT_DIR}"
    exit 1
fi

# Check Python dependencies
echo "Verifying dependencies..."
python3 -c "import torch, nats, numpy" 2>/dev/null || {
    echo "Error: Missing required Python packages (torch, nats, numpy)"
    echo "Please install dependencies first."
    exit 1
}

# Ensure output directories exist
mkdir -p logs_es checkpoints_es
echo "Output directories ready: ./logs_es, ./checkpoints_es"

# Enable ROCm Optimizations (if ROCm is available)
if command -v rocm-smi &> /dev/null; then
    echo "ROCm detected - configuring for inference workload"
    export HSA_FORCE_FINE_GRAIN_PCIE=1

    # Disable MIOpen kernel tuning/finding (uses too much workspace memory)
    export MIOPEN_FIND_MODE=1              # Use immediate mode (no tuning)
    export MIOPEN_DEBUG_DISABLE_FIND_DB=1  # Disable find database
    export MIOPEN_FIND_ENFORCE=NONE        # Don't enforce find
    export MIOPEN_USER_DB_PATH=/tmp/miopen # Temp DB path

    # Disable convolution workspace to reduce memory
    export MIOPEN_CONV_PRECISE_ROCBLAS_TIMING=0

    # PyTorch Memory Management for ROCm
    export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    echo "✓ MIOpen kernel tuning disabled"
    echo "✓ GPU memory management configured"
fi

# Run trainer
echo ""
echo "Starting trainer..."
python3 -u v700_es_sac_trainer.py \
    --nats_server "${NATS_SERVER}" \
    --checkpoint_dir "./checkpoints_es" \
    --log_dir "./logs_es" \
    2>&1 | tee "logs_es/server_$(date +%Y%m%d_%H%M%S).log"
