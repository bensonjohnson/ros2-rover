#!/bin/bash

# PPO Remote Autonomous Rover Startup Script
# Rover collects rollouts via RKNN NPU inference, ships to V620 server via NATS

echo "=================================================="
echo "ROS2 Rover - PPO Remote Training (Unified BEV)"
echo "=================================================="

if [ ! -f "install/setup.bash" ]; then
  echo "Error: Please run this script from the ros2-rover directory"
  exit 1
fi

# Configuration
NATS_SERVER="nats://nats.gokickrocks.org:4222"
MAX_SPEED=${1:-"0.18"}
ROLLOUT_STEPS=${2:-"2048"}

echo "Configuration:"
echo "  NATS Server: ${NATS_SERVER}"
echo "  Max Speed: ${MAX_SPEED} m/s"
echo "  Rollout Steps: ${ROLLOUT_STEPS}"
echo "  Architecture: Unified BEV (LiDAR + Depth fusion)"
echo "  Training: Remote PPO (V620 GPU server)"
echo ""

echo "WARNING: Rover will drive AUTONOMOUSLY!"
echo "  - Rollouts collected on-device, trained on V620 server"
echo "  - PPO ships ${ROLLOUT_STEPS}-step rollouts via NATS"
echo "  - Updated ONNX models downloaded automatically"
echo "  - Early episodes use random exploration"
echo "  - Keep emergency stop ready"
echo ""
read -p "Press Enter to continue or Ctrl+C to abort..."

# Build workspace
echo "Building workspace..."
colcon build --packages-select tractor_bringup tractor_control tractor_sensors --cmake-args -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
if [ $? -ne 0 ]; then
  echo "Build failed!"
  exit 1
fi
echo "Build complete"

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
    echo "Found D435i at USB path: $USB_DEVICE_PATH"
    break
  fi
done

if [ -n "$USB_DEVICE_PATH" ]; then
  echo "on" | sudo tee $USB_DEVICE_PATH/power/control > /dev/null 2>&1
  echo "-1" | sudo tee $USB_DEVICE_PATH/power/autosuspend > /dev/null 2>&1
  echo "USB power management configured"
fi

# Launch
echo "Launching PPO remote autonomous training..."
mkdir -p log
LOG_FILE="log/ppo_remote_rover_$(date +%Y%m%d_%H%M%S).log"

ros2 launch tractor_bringup ppo_remote_autonomous.launch.py \
  nats_server:=${NATS_SERVER} \
  max_speed:=${MAX_SPEED} \
  rollout_steps:=${ROLLOUT_STEPS} \
  2>&1 | tee "$LOG_FILE" &

LAUNCH_PID=$!

trap 'echo; echo "Stopping PPO remote training..."; kill $LAUNCH_PID 2>/dev/null; sleep 2; echo "Stopped"; exit 0' SIGINT SIGTERM

echo ""
echo "PPO remote training running (PID: $LAUNCH_PID)"
echo "Log file: $LOG_FILE"
echo ""
echo "What's happening:"
echo "  1. Rover runs PPO policy on NPU (30Hz)"
echo "  2. Collects ${ROLLOUT_STEPS}-step rollouts (BEV + Proprioception)"
echo "  3. Ships complete rollouts to V620 server via NATS"
echo "  4. Server trains PPO on GPU, sends updated ONNX model back"
echo "  5. Rover converts ONNX to RKNN and resumes with new policy"
echo ""
echo "Monitor:"
echo "  - Dashboard: http://$(hostname -I | awk '{print $1}'):8080"
echo "  - ros2 topic echo /track_cmd_ai"
echo "  - ros2 topic echo /emergency_stop"
echo ""
echo "Press Ctrl+C to stop"

wait $LAUNCH_PID
