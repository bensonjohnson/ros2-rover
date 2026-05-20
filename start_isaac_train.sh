#!/bin/bash

# Isaac Lab Training Startup Script (tractor rover, RGB Arducam)
# Trains an RL navigation policy in Isaac Sim using rsl_rl PPO.
# Runs on the DGX Spark (Blackwell). Outputs ONNX checkpoint that
# gets converted to RKNN and deployed to the rover.

echo "=================================================="
echo "Isaac Lab - Tractor Navigation Training"
echo "=================================================="

# Paths — adjust ISAACLAB_PATH if Isaac Lab lives elsewhere
ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/IsaacLab}"
TASK="${1:-Isaac-Tractor-Nav-v0}"
NUM_ENVS="${2:-4096}"
MAX_ITERATIONS="${3:-3000}"
SEED="${4:-42}"

if [ ! -f "$ISAACLAB_PATH/isaaclab.sh" ]; then
  echo "Error: Isaac Lab not found at $ISAACLAB_PATH"
  echo "Set ISAACLAB_PATH or clone to \$HOME/IsaacLab"
  exit 1
fi

# libgomp preload — Isaac Sim on aarch64 trips OpenMP symbol resolution
# without this. Harmless on x86_64 if the path exists.
if [ -f "/lib/aarch64-linux-gnu/libgomp.so.1" ]; then
  export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"
fi

# Make our custom task importable. The tractor env package lives in this repo.
export PYTHONPATH="$(pwd)/isaac_lab_tasks:$PYTHONPATH"

echo "Configuration:"
echo "  Isaac Lab path: $ISAACLAB_PATH"
echo "  Task:           $TASK"
echo "  Parallel envs:  $NUM_ENVS"
echo "  Max iterations: $MAX_ITERATIONS"
echo "  Seed:           $SEED"
echo "  Camera:         Arducam RGB (matched to /dev/video0 on rover)"
echo ""

mkdir -p log
LOG_FILE="log/isaac_train_$(date +%Y%m%d_%H%M%S).log"

echo "Launching training..."
echo "Log: $LOG_FILE"
echo ""
echo "Monitor:"
echo "  - tensorboard --logdir $ISAACLAB_PATH/logs/rsl_rl"
echo "  - tail -f $LOG_FILE"
echo ""

cd "$ISAACLAB_PATH"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task="$TASK" \
  --num_envs="$NUM_ENVS" \
  --max_iterations="$MAX_ITERATIONS" \
  --seed="$SEED" \
  --headless \
  2>&1 | tee "$OLDPWD/$LOG_FILE"
