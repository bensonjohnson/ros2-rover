#!/bin/bash

# Remote Trained Model Inference Script
# Runs rover with model trained on V620 server, deployed to RK3588 NPU

echo "=================================================="
echo "ROS2 Rover - Remote Trained Model Inference"
echo "=================================================="

if [ ! -f "install/setup.bash" ]; then
  echo "Error: Please run this script from the ros2-rover directory"
  exit 1
fi

# Configuration
MODEL_PATH=${1:-"./models/remote_trained.rknn"}
MAX_SPEED=${2:-0.18}
SAFETY_DISTANCE=${3:-0.25}
INFERENCE_RATE=${4:-10.0}
USE_NPU=${5:-true}

echo "Configuration:"
echo "  Model Path: ${MODEL_PATH}"
echo "  Max Speed: ${MAX_SPEED} m/s"
echo "  Safety Distance: ${SAFETY_DISTANCE} m"
echo "  Inference Rate: ${INFERENCE_RATE} Hz"
echo "  Use NPU: ${USE_NPU}"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
  echo "❌ Model not found: ${MODEL_PATH}"
  echo ""
  echo "Available models:"
  ls -lh models/*.rknn 2>/dev/null || echo "  (no models found)"
  echo ""
  echo "Deploy a model first:"
  echo "  1. Train on V620 server"
  echo "  2. Convert ONNX → RKNN"
  echo "  3. Copy .rknn file to rover's models/ directory"
  exit 1
fi

echo "✓ Model found: ${MODEL_PATH}"
echo "  Size: $(du -h ${MODEL_PATH} | cut -f1)"

# Build workspace
echo "Building workspace..."
colcon build --packages-select tractor_bringup tractor_control tractor_sensors --cmake-args -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
if [ $? -ne 0 ]; then
  echo "❌ Build failed!"
  exit 1
fi
echo "✓ Build complete"

# Source environment
echo "Sourcing ROS2 environment..."
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# Configure RealSense USB power
echo "Configuring USB power management for RealSense..."
USB_DEVICE_PATH=""
for device in /sys/bus/usb/devices/*/idProduct; do
  if [ -f "$device" ] && [ "$(cat $device 2>/dev/null)" = "0b3a" ]; then
    USB_DEVICE_PATH=$(dirname $device)
    echo "✓ Found D435i at USB path: $USB_DEVICE_PATH"
    break
  fi
done

if [ -n "$USB_DEVICE_PATH" ]; then
  echo "on" | sudo tee $USB_DEVICE_PATH/power/control > /dev/null 2>&1
  echo "-1" | sudo tee $USB_DEVICE_PATH/power/autosuspend > /dev/null 2>&1
  echo "✓ USB power management configured"
fi

# Check RKNN runtime
if [ "$USE_NPU" = "true" ]; then
  echo "Checking RKNN runtime..."
  python3 -c "from rknnlite.api import RKNNLite; print('✓ RKNNLite available')" 2>/dev/null
  if [ $? -ne 0 ]; then
    echo "⚠ RKNNLite not installed - will use CPU fallback (very slow!)"
    echo "Install RKNN runtime for RK3588:"
    echo "  https://github.com/rockchip-linux/rknn-toolkit2/tree/master/rknpu2"
  fi
fi

# Launch
echo "Launching remote trained inference..."
mkdir -p log
LOG_FILE="log/remote_inference_$(date +%Y%m%d_%H%M%S).log"

ros2 launch tractor_bringup remote_trained_inference.launch.py \
  model_path:=${MODEL_PATH} \
  max_speed:=${MAX_SPEED} \
  safety_distance:=${SAFETY_DISTANCE} \
  inference_rate_hz:=${INFERENCE_RATE} \
  use_npu:=${USE_NPU} \
  2>&1 | tee "$LOG_FILE" &

LAUNCH_PID=$!

trap 'echo; echo "🛑 Stopping inference..."; kill $LAUNCH_PID 2>/dev/null; sleep 2; echo "✅ Stopped"; exit 0' SIGINT SIGTERM

echo "Remote trained inference running (PID: $LAUNCH_PID)"
echo "Log file: $LOG_FILE"
echo ""
echo "Monitor:"
echo "  - ros2 topic echo /cmd_vel_ai"
echo "  - ros2 topic hz /cmd_vel_ai"
echo "  - ros2 service call /reload_remote_model std_srvs/srv/Trigger"
echo ""
echo "Press Ctrl+C to stop"

wait $LAUNCH_PID
