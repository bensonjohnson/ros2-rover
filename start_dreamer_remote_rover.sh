#!/bin/bash

# DreamerV3 Remote Autonomous Rover Startup Script
# Rover collects short chunks via RKNN NPU inference (with persistent RSSM
# state), ships them to the GPU server via ZMQ for off-policy world-model
# training. Updated ONNX/RKNN models are pushed back.

echo "=================================================="
echo "ROS2 Rover - DreamerV3 Remote Training"
echo "=================================================="

if [ ! -f "install/setup.bash" ]; then
  echo "Error: Please run this script from the ros2-rover directory"
  exit 1
fi

# Configuration
SERVER_ADDR=${1:-"192.168.1.100"}
MAX_SPEED=${2:-"0.18"}
CHUNK_LEN=${3:-"64"}

echo "Configuration:"
echo "  Server Address: ${SERVER_ADDR}"
echo "  Max Speed: ${MAX_SPEED} m/s"
echo "  Chunk Length: ${CHUNK_LEN} steps"
echo "  Architecture: DreamerV3 (BEV + Depth + RGB + RSSM)"
echo "  Training: Remote world-model (GPU server via ZMQ, off-policy)"
echo ""

echo "WARNING: Rover will drive AUTONOMOUSLY!"
echo "  - Chunks of ${CHUNK_LEN} steps shipped to ${SERVER_ADDR} via ZMQ"
echo "  - Updated ONNX/RKNN models downloaded automatically"
echo "  - Random exploration until first model arrives"
echo "  - RSSM state is maintained between ticks; resets on episode boundary"
echo "  - Keep emergency stop ready"
echo ""
read -p "Press Enter to continue or Ctrl+C to abort..."

echo "Building workspace..."
colcon build --packages-select tractor_bringup tractor_control tractor_sensors --cmake-args -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
if [ $? -ne 0 ]; then
  echo "Build failed!"
  exit 1
fi
echo "Build complete"

echo "Sourcing ROS2 environment..."
source /opt/ros/jazzy/setup.bash
source install/setup.bash

echo "Configuring USB power management for RealSense..."
USB_DEVICE_PATH=""
for device in /sys/bus/usb/devices/*/idProduct; do
  if [ -f "$device" ] && [ "$(cat $device 2>/dev/null)" = "0b3a" ]; then
    USB_DEVICE_PATH=$(dirname $device)
    echo "Found D435i at USB path: $USB_DEVICE_PATH"
    break
  fi
done

if [ -n "$USB_DEVICE_PATH" ]; then
  echo "on" | sudo tee $USB_DEVICE_PATH/power/control > /dev/null 2>&1
  echo "-1" | sudo tee $USB_DEVICE_PATH/power/autosuspend > /dev/null 2>&1
  echo "USB power management configured"
fi

echo "Launching DreamerV3 remote autonomous training..."
mkdir -p log
LOG_FILE="log/dreamer_remote_rover_$(date +%Y%m%d_%H%M%S).log"

ros2 launch tractor_bringup dreamer_remote_autonomous.launch.py \
  server_addr:=${SERVER_ADDR} \
  max_speed:=${MAX_SPEED} \
  chunk_len:=${CHUNK_LEN} \
  2>&1 | tee "$LOG_FILE" &

LAUNCH_PID=$!

trap 'echo; echo "Stopping Dreamer remote training..."; kill $LAUNCH_PID 2>/dev/null; sleep 2; echo "Stopped"; exit 0' SIGINT SIGTERM

echo ""
echo "Dreamer remote training running (PID: $LAUNCH_PID)"
echo "Log file: $LOG_FILE"
echo ""
echo "What's happening:"
echo "  1. Rover runs Dreamer actor + encoder + RSSM step on NPU (30Hz)"
echo "  2. Collects ${CHUNK_LEN}-step chunks (BEV + Proprio + RGB + is_first)"
echo "  3. Ships chunks continuously to GPU server via ZMQ (off-policy)"
echo "  4. Server trains world-model + actor-critic in imagination"
echo "  5. Rover receives updated ONNX/RKNN and swaps in-place"
echo ""
echo "Monitor:"
echo "  - Dashboard: http://$(hostname -I | awk '{print $1}'):8080"
echo "  - ros2 topic echo /track_cmd_ai"
echo "  - ros2 topic echo /emergency_stop"
echo ""
echo "Press Ctrl+C to stop"

wait $LAUNCH_PID
