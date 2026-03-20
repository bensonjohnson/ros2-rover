#!/bin/bash

# PPO BEV Rover Startup Script
# Dual-mode: TRAIN or INFERENCE

echo "=================================================="
echo "ROS2 Rover - PPO BEV Training/Inference"
echo "=================================================="
echo ""
echo "Select mode:"
echo "  1) TRAIN - Train PPO policy locally with BEV architecture"
echo "  2) INFERENCE - Load RKNN model and drive autonomously"
echo ""
read -p "Enter choice [1/2]: " MODE

if [ -z "$MODE" ]; then
  echo "❌ Invalid choice. Please enter 1 or 2."
  exit 1
fi

# Configuration
NATS_SERVER=${NATS_SERVER:-"nats://nats.gokickrocks.org:4222"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"./checkpoints_ppo"}
LOG_DIR=${LOG_DIR:-"./logs_ppo"}
MAX_SPEED=${MAX_SPEED:-"0.18"}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-200}

case $MODE in
  1)
    echo ""
    echo "=================================================="
    echo "🎯 PPO BEV TRAINING MODE"
    echo "=================================================="
    echo ""
    echo "Configuration:"
    echo "  NATS Server: ${NATS_SERVER}"
    echo "  Checkpoint Dir: ${CHECKPOINT_DIR}"
    echo "  Log Dir: ${LOG_DIR}"
    echo "  Checkpoint Interval: ${CHECKPOINT_INTERVAL} steps"
    echo "  Architecture: Unified BEV (LiDAR + Depth fusion)"
    echo ""
    echo "⚠ WARNING: This will train PPO on the rover!"
    echo "  - Requires GPU/ROCm for fast training"
    echo "  - Checkpoints saved every ${CHECKPOINT_INTERVAL} steps"
    echo "  - Automatically exports ONNX and RKNN models"
    echo ""
    read -p "Press Enter to start training or Ctrl+C to abort..."

    # Build workspace if needed
    if [ -d "install" ] && [ -f "install/setup.bash" ]; then
      echo "Sourcing ROS2 environment..."
      source /opt/ros/jazzy/setup.bash
      source install/setup.bash
    fi

    # Create directories
    mkdir -p ${CHECKPOINT_DIR} ${LOG_DIR}

    # Start TensorBoard (optional)
    echo "Starting TensorBoard..."
    tensorboard --logdir "${LOG_DIR}" --port 6006 --bind_all > /dev/null 2>&1 &
    TB_PID=$!
    echo "TensorBoard running on http://localhost:6006 (PID: ${TB_PID})"

    # Launch PPO trainer (local version - no NATS required)
    LOG_FILE="${LOG_DIR}/ppo_bev_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo "🚀 Starting PPO BEV Local Trainer (PyTorch only)..."
    echo "Log file: ${LOG_FILE}"
    echo ""

    python3 -u ppo_bev_trainer_local.py \
      --checkpoint_dir "${CHECKPOINT_DIR}" \
      --log_dir "${LOG_DIR}" \
      --checkpoint_interval ${CHECKPOINT_INTERVAL} \
      2>&1 | tee "${LOG_FILE}" &

    TRAINER_PID=$!

    # Signal handling
    trap 'echo; echo "🛑 Stopping PPO trainer..."; kill $TRAINER_PID 2>/dev/null; kill $TB_PID 2>/dev/null; sleep 2; echo "✅ Stopped"; exit 0' SIGINT SIGTERM

    echo "PPO trainer running (PID: $TRAINER_PID)"
    echo ""
    echo "Monitoring:"
    echo "  - Checkpoints: ${CHECKPOINT_DIR}/"
    echo "  - TensorBoard: http://localhost:6006"
    echo "  - Logs: ${LOG_FILE}"
    echo ""
    echo "Press Ctrl+C to stop training"

    wait $TRAINER_PID
    ;;

  2)
    echo ""
    echo "=================================================="
    echo "🤖 PPO BEV INFERENCE MODE"
    echo "=================================================="
    echo ""
    echo "Configuration:"
    echo "  Max Speed: ${MAX_SPEED} m/s"
    echo "  Model: ${CHECKPOINT_DIR}/latest_actor.rknn"
    echo ""
    echo "⚠ WARNING: Rover will drive AUTONOMOUSLY!"
    echo "  - Uses RKNN model for NPU inference"
    echo "  - Falls back to PyTorch if RKNN unavailable"
    echo "  - Keep emergency stop ready"
    echo ""
    read -p "Press Enter to start inference or Ctrl+C to abort..."

    # Build workspace if needed
    if [ -d "install" ] && [ -f "install/setup.bash" ]; then
      echo "Sourcing ROS2 environment..."
      source /opt/ros/jazzy/setup.bash
      source install/setup.bash
    fi

    # Check for model
    RKNN_MODEL="${CHECKPOINT_DIR}/latest_actor.rknn"
    ONNX_MODEL="${CHECKPOINT_DIR}/latest_actor.onnx"
    PT_MODEL="${CHECKPOINT_DIR}/latest_actor.pt"

    if [ -f "$RKNN_MODEL" ]; then
      MODEL_PATH="$RKNN_MODEL"
      echo "✅ Found RKNN model: ${RKNN_MODEL}"
    elif [ -f "$ONNX_MODEL" ]; then
      MODEL_PATH="$ONNX_MODEL"
      echo "⚠ Using ONNX model: ${ONNX_MODEL}"
    elif [ -f "$PT_MODEL" ]; then
      MODEL_PATH="$PT_MODEL"
      echo "⚠ Using PyTorch model: ${PT_MODEL}"
    else
      echo "❌ No model found! Please train first or provide a model."
      echo "   Run with mode 1 (TRAIN) to create a model."
      exit 1
    fi

    echo ""
    echo "🚀 Starting PPO BEV Inference..."
    echo ""

    python3 -u bev_inference.py \
      --rknn "$RKNN_MODEL" \
      --onnx "$ONNX_MODEL" \
      --pt "$PT_MODEL" \
      --max_speed "${MAX_SPEED}"

    ;;

  *)
    echo "❌ Invalid choice. Please enter 1 or 2."
    exit 1
    ;;
esac