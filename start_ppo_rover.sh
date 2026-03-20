#!/bin/bash

# PPO Local Autonomous Rover Startup Script
# Fully on-device PPO training + inference using Unified BEV architecture
# No external server required (no NATS, no ZMQ)

echo "=================================================="
echo "ROS2 Rover - PPO Local Training (Unified BEV)"
echo "=================================================="

if [ ! -f "install/setup.bash" ]; then
  echo "Error: Please run this script from the ros2-rover directory"
  exit 1
fi

# Configuration
MAX_SPEED=${1:-"0.18"}
CHECKPOINT_DIR=${2:-"./checkpoints_ppo"}
LOG_DIR=${3:-"./logs_ppo"}
ROLLOUT_STEPS=${4:-"2048"}

echo "Configuration:"
echo "  Max Speed: ${MAX_SPEED} m/s"
echo "  Checkpoint Dir: ${CHECKPOINT_DIR}"
echo "  Log Dir: ${LOG_DIR}"
echo "  Rollout Steps: ${ROLLOUT_STEPS}"
echo "  Architecture: Unified BEV (LiDAR + Depth fusion)"
echo "  Training: Local PPO (CPU, on-device)"
echo ""

echo "WARNING: Rover will drive AUTONOMOUSLY!"
echo "  - No external server required"
echo "  - PPO trains locally on CPU after every ${ROLLOUT_STEPS} steps"
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

# Create directories
mkdir -p ${CHECKPOINT_DIR} ${LOG_DIR} log

# Launch
echo "Launching PPO local autonomous training..."
LOG_FILE="log/ppo_rover_$(date +%Y%m%d_%H%M%S).log"

ros2 launch tractor_bringup ppo_local_autonomous.launch.py \
  max_speed:=${MAX_SPEED} \
  checkpoint_dir:=${CHECKPOINT_DIR} \
  log_dir:=${LOG_DIR} \
  rollout_steps:=${ROLLOUT_STEPS} \
  2>&1 | tee "$LOG_FILE" &

LAUNCH_PID=$!

trap 'echo; echo "Stopping PPO training..."; kill $LAUNCH_PID 2>/dev/null; sleep 2; echo "Stopped"; exit 0' SIGINT SIGTERM

echo ""
echo "PPO local training running (PID: $LAUNCH_PID)"
echo "Log file: $LOG_FILE"
echo ""
echo "What's happening:"
echo "  1. Rover collects experience using BEV observations (30Hz)"
echo "  2. After ${ROLLOUT_STEPS} steps, PPO trains locally on CPU"
echo "  3. Policy updates in-place, rover resumes driving"
echo "  4. Checkpoints saved to ${CHECKPOINT_DIR}/"
echo ""
echo "Monitor:"
echo "  - ros2 topic echo /track_cmd_ai"
echo "  - ros2 topic echo /emergency_stop"
echo "  - tensorboard --logdir ${LOG_DIR}"
echo ""
echo "Press Ctrl+C to stop"

wait $LAUNCH_PID
