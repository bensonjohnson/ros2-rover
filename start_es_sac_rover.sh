#!/bin/bash
# ES-SAC Autonomous Rover Startup Script
# Runs continuous ES-SAC remote inference and data collection

set -e # Exit on error

echo "=================================================="
echo "ROS2 Rover - ES-SAC Autonomous Training"
echo "=================================================="

if [ ! -f "install/setup.bash" ]; then
  echo "Error: Please run this script from the ros2-rover directory"
  exit 1
fi

# Configuration
NATS_SERVER=${1:-"nats://nats.gokickrocks.org:4222"}
MAX_EPISODE_STEPS=${2:-"1000"}
export ROS_DOMAIN_ID=0

echo "Configuration:"
echo "  NATS Server: ${NATS_SERVER}"
echo "  Max Episode Steps: ${MAX_EPISODE_STEPS}"

echo ""
echo "⚠ WARNING: Rover will drive AUTONOMOUSLY!"
echo "  - No teleoperation required"
echo "  - Rover will explore on its own"
echo "  - Early episodes may result in collisions"
echo "  - Keep emergency stop ready"
echo ""
read -p "Press Enter to continue or Ctrl+C to abort..."

# Build workspace
echo "Building workspace..."
colcon build --symlink-install --packages-select tractor_bringup tractor_control tractor_sensors --cmake-args -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
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

# Launch
echo "Launching ES-SAC autonomous training..."
mkdir -p log
LOG_FILE="log/es_sac_rover_$(date +%Y%m%d_%H%M%S).log"

ros2 launch tractor_bringup es_sac_autonomous.launch.py \
  nats_server:=${NATS_SERVER} \
  max_episode_steps:=${MAX_EPISODE_STEPS} \
  2>&1 | tee "$LOG_FILE" &

LAUNCH_PID=$!

trap 'echo; echo "🛑 Stopping autonomous episodes..."; kill $LAUNCH_PID 2>/dev/null; sleep 2; echo "✅ Stopped"; exit 0' SIGINT SIGTERM

echo ""
echo "ES-SAC autonomous training running (PID: $LAUNCH_PID)"
echo "Log file: $LOG_FILE"
echo ""
echo "What's happening:"
echo "  1. Rover requests actions from remote server via NATS"
echo "  2. Server runs inference on V620 GPU (population-based)"
echo "  3. Rover executes actions and collects experience"
echo "  4. Episode results sent to server for ES-SAC training"
echo ""
echo "Monitor:"
echo "  - ros2 topic echo /cmd_vel_ai"
echo "  - ros2 topic echo /scan"
echo "  - ros2 topic echo /safety_monitor_status"
echo ""
echo "Press Ctrl+C to stop"

wait $LAUNCH_PID
