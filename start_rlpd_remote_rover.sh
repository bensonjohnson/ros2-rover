#!/bin/bash

# RLPD + HIL-SERL Remote Autonomous Rover Startup Script
# Rover runs model-free SAC inference on the RKNN NPU and ships chunks to
# the GPU server. The Xbox controller remains live so the operator can hold
# RB to intervene; those interventions are logged with is_intervention=True
# and a -1 reward in the intervention channel — the policy learns to avoid
# needing correction (HIL-SERL pattern).

echo "=================================================="
echo "ROS2 Rover - RLPD + HIL-SERL Remote Training"
echo "=================================================="

if [ ! -f "install/setup.bash" ]; then
  echo "Error: Please run this script from the ros2-rover directory"
  exit 1
fi

# Configuration
SERVER_ADDR=${1:-"192.168.1.100"}
MAX_SPEED=${2:-"0.25"}
CHUNK_LEN=${3:-"64"}
FRAME_STACK=${4:-"4"}

echo "Configuration:"
echo "  Server Address: ${SERVER_ADDR}"
echo "  Max Speed: ${MAX_SPEED} m/s"
echo "  Chunk Length: ${CHUNK_LEN} steps"
echo "  Frame Stack: ${FRAME_STACK} frames"
echo "  Architecture: RLPD v3 (Depth 96×72 + 1D LiDAR 360 + proprio)"
echo "  Training: Remote model-free SAC (GPU server via ZMQ, off-policy)"
echo "  HIL: Xbox controller live; hold RB to intervene"
echo ""

echo "WARNING: Rover will drive AUTONOMOUSLY!"
echo "  - Chunks of ${CHUNK_LEN} steps shipped to ${SERVER_ADDR} via ZMQ"
echo "  - Updated ONNX/RKNN models downloaded automatically"
echo "  - Random exploration until first model arrives"
echo "  - Hold RB on Xbox controller to override policy (HIL-SERL intervention)"
echo "  - Frame stacks reset on episode boundary"
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

echo "Launching RLPD remote autonomous training..."
mkdir -p log
LOG_FILE="log/rlpd_remote_rover_$(date +%Y%m%d_%H%M%S).log"

ros2 launch tractor_bringup rlpd_remote_autonomous.launch.py \
  server_addr:=${SERVER_ADDR} \
  max_speed:=${MAX_SPEED} \
  chunk_len:=${CHUNK_LEN} \
  frame_stack:=${FRAME_STACK} \
  2>&1 | tee "$LOG_FILE" &

LAUNCH_PID=$!

trap 'echo; echo "Stopping RLPD remote training..."; kill $LAUNCH_PID 2>/dev/null; sleep 2; echo "Stopped"; exit 0' SIGINT SIGTERM

echo ""
echo "RLPD remote training running (PID: $LAUNCH_PID)"
echo "Log file: $LOG_FILE"
echo ""
echo "What's happening:"
echo "  1. Rover runs RLPD v3 actor (depth CNN + 1D lidar CNN + proprio MLP) on NPU (30Hz)"
echo "  2. Collects ${CHUNK_LEN}-step chunks (Depth + LiDAR + Proprio + is_first + is_intervention)"
echo "  3. Ships chunks continuously to GPU server via ZMQ (off-policy)"
echo "  4. Server trains SAC critic ensemble (N=10, LayerNorm) + actor with 50/50 demo replay"
echo "  5. Rover receives updated ONNX/RKNN and swaps in-place"
echo "  6. Hold RB to override the policy — those steps get a -1 reward and are stored"
echo "     in demos.npz so the policy learns to avoid needing intervention"
echo ""
echo "Monitor:"
echo "  - Dashboard: http://$(hostname -I | awk '{print $1}'):8081"
echo "  - ros2 topic echo /track_cmd_ai"
echo "  - ros2 topic echo /emergency_stop"
echo "  - ros2 topic echo /joy"
echo ""
echo "Press Ctrl+C to stop"

wait $LAUNCH_PID
