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
    echo "Applying ROCm optimizations for V620..."

    # HSA optimizations
    export HSA_FORCE_FINE_GRAIN_PCIE=1
    export HSA_ENABLE_SDMA=0  # Disable SDMA for better performance
    export HSA_OVERRIDE_GFX_VERSION=10.3.0  # Ensure correct GFX version

    # V620-specific architecture (Navi 21)
    export PYTORCH_ROCM_ARCH="gfx1030"

    # MIOpen optimizations
    export MIOPEN_FIND_ENFORCE=NONE  # Let MIOpen choose best kernels
    export MIOPEN_DISABLE_CACHE=0  # Enable cache to speed up startup after first run
    export MIOPEN_DEBUG_DISABLE_FIND_DB=0  # Enable find-db for faster ops
    export MIOPEN_FIND_MODE=NORMAL  # Use normal find mode
    export MIOPEN_USER_DB_PATH=/tmp/miopen  # Cache tuned kernels

    # PyTorch Memory Management for ROCm
    export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512

    echo "✓ V620 (gfx1030) optimizations applied"
    echo "✓ MIOpen caching enabled"
    echo "✓ GPU memory management configured"

    # Check available VRAM
    echo ""
    echo "Checking GPU memory..."
    if rocm-smi --showmeminfo vram 2>/dev/null | grep -q "Total"; then
        TOTAL_VRAM=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "VRAM Total Memory" | awk '{print $5}')
        echo "  Total VRAM: ${TOTAL_VRAM} MB"
    fi
fi

# Check and raise file descriptor limits for MIOpen
CURRENT_ULIMIT=$(ulimit -n)
echo ""
echo "Checking file descriptor limits..."
echo "  Current ulimit: ${CURRENT_ULIMIT}"
if [ ${CURRENT_ULIMIT} -lt 65535 ]; then
    echo "  ⚠ Warning: ulimit is below recommended 65535 for MIOpen benchmark"
    echo "  Attempting to raise limit for training process..."
fi

# Run trainer with increased file descriptor limit
echo ""
echo "Starting trainer..."

{
    if ! ulimit -n 65535 2>/dev/null; then
        echo "⚠ Warning: Failed to raise ulimit to 65535. Current limit: $(ulimit -n)"
        echo "  Proceeding anyway..."
    else
        echo "✓ Raised file descriptor limit to 65535"
    fi

    exec python3 -u v700_es_sac_trainer.py \
        --nats_server "${NATS_SERVER}" \
        --checkpoint_dir "./checkpoints_es" \
        --log_dir "./logs_es" \
        --cpu_inference
} 2>&1 | tee "logs_es/server_$(date +%Y%m%d_%H%M%S).log"
