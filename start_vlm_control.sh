#!/usr/bin/env bash
set -e

# Vision Language Model Control Startup Script
# This script launches the rover with VLM-based control using rkllama server

# Ensure ROS 2 + workspace environment is sourced
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
  source /opt/ros/jazzy/setup.bash
fi

if [ -f "install/setup.bash" ]; then
  source install/setup.bash
else
  echo "Workspace not built yet; building tractor_bringup (fast)..."
  colcon build --packages-select tractor_bringup --cmake-args -DCMAKE_BUILD_TYPE=Release
  source install/setup.bash
fi

# VLM control mode configuration
WITH_TELEOP=false
WITH_MOTOR=true
WITH_SAFETY=true
WITH_VLM=true
RKLLAMA_URL="https://ollama.gokickrocks.org"
MODEL_NAME="qwen2.5vl:7b"

# Parse command line arguments
for arg in "$@"; do
  case "$arg" in
    --teleop)
      WITH_TELEOP=true ;;
    --no-motor)
      WITH_MOTOR=false ;;
    --no-safety)
      WITH_SAFETY=false ;;
    --no-vlm)
      WITH_VLM=false ;;
    --rkllama-url=*)
      RKLLAMA_URL="${arg#*=}" ;;
    --model-name=*)
      MODEL_NAME="${arg#*=}" ;;
  esac
done

echo "Launching VLM Control mode..."
echo "  Motor driver: ${WITH_MOTOR}"
echo "  Safety monitor: ${WITH_SAFETY}"
echo "  VLM control: ${WITH_VLM}"
echo "  Teleop backup: ${WITH_TELEOP}"
echo "  Ollama URL: ${RKLLAMA_URL}"
echo "  Model name: ${MODEL_NAME}"
echo ""

# Launch the VLM control configuration
ros2 launch tractor_bringup vlm_control.launch.py \
  with_teleop:=${WITH_TELEOP} \
  with_motor:=${WITH_MOTOR} \
  with_safety:=${WITH_SAFETY} \
  with_vlm:=${WITH_VLM} \
  rkllama_url:=${RKLLAMA_URL} \
  model_name:=${MODEL_NAME}