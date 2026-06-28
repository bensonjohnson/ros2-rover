#!/bin/bash
# ============================================================================
# Deep Explorer Network — Remote Training Server Startup
# ============================================================================
# Starts the GPU training server that:
#   1. Receives experience chunks from the rover (ZMQ PULL on port 5557)
#   2. Trains the Deep Explorer Network with TD-MPC2
#   3. Exports ONNX -> RKNN every 5 min (auto)
#   4. Broadcasts new RKNN model to rover (ZMQ PUB on port 5558)
#   5. Serves a training dashboard
#
# The rover receives models automatically — no scp needed.
#
# Usage:
#   ./start_explorer_server.sh [options]
#   ./start_explorer_server.sh --data-dir ~/.ros/explorer_chunks/  # local data
#
# Options:
#   --port <n>             ZMQ PULL port for rover data (default 5557)
#   --pub-port <n>         ZMQ PUB port for model broadcast (default 5558)
#   --data-dir <path>      Load pre-collected chunks from directory
#   --checkpoint-dir <path> Checkpoint directory (default ./checkpoints)
#   --no-dashboard         Disable web dashboard
# ============================================================================

echo "=================================================="
echo "Deep Explorer Network — Remote Training Server"
echo "=================================================="

# Detect GPU
if command -v nvidia-smi &> /dev/null; then
  echo "NVIDIA GPU detected:"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1
elif command -v rocm-smi &> /dev/null; then
  echo "AMD ROCm GPU detected:"
  rocm-smi --showproductname 2>/dev/null | head -2
else
  echo "No GPU detected — training will run on CPU (slower)"
fi

echo ""
echo "Python: $(python3 --version)"

# Defaults
PORT=5557
PUB_PORT=5558
DATA_DIR=""
CHECKPOINT_DIR="./checkpoints_explorer"
LOG_DIR="./logs_explorer"
DASHBOARD="true"

while [ $# -gt 0 ]; do
  case "$1" in
    --port)       PORT="$2"; shift 2 ;;
    --pub-port)   PUB_PORT="$2"; shift 2 ;;
    --data-dir)   DATA_DIR="$2"; shift 2 ;;
    --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --log-dir)    LOG_DIR="$2"; shift 2 ;;
    --no-dashboard) DASHBOARD="false"; shift ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"

ARGS=""
[ -n "$PORT" ] && [ "$PORT" != "0" ] && ARGS="$ARGS --port $PORT --pub-port $PUB_PORT"
[ -n "$DATA_DIR" ] && ARGS="$ARGS --data-dir $DATA_DIR"
[ "$DASHBOARD" = "true" ] && ARGS="$ARGS --dashboard-port 8085"

echo ""
echo "Configuration:"
echo "  ZMQ PULL (rover->server):  ${PORT}"
echo "  ZMQ PUB  (server->rover):  ${PUB_PORT}"
echo "  Data Dir:                  ${DATA_DIR:-<live mode>}"
echo "  Checkpoint Dir:            ${CHECKPOINT_DIR}"
echo "  Log Dir:                   ${LOG_DIR}"
echo "  Auto-export:               every 300s (ONNX + RKNN -> rover)"
echo "  Dashboard:                 ${DASHBOARD}"
echo ""
echo "Starting training server..."
echo ""

python3 "$(dirname "$0")/explorer_trainer.py" \
  $ARGS \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --log-dir "$LOG_DIR"
