#!/bin/bash

# PPO Autonomous Rover Startup Script
# Runs continuous PPO inference and data collection

echo "=================================================="
echo "ROS2 Rover - PPO Autonomous Training"
echo "=================================================="

if [ ! -f "install/setup.bash" ]; then
  echo "Error: Please run this script from the ros2-rover directory"
  exit 1
fi

# Configuration
SERVER_ADDR=${1:-"tcp://10.0.0.200:5556"}
MAX_SPEED=${2:-"0.18"}

echo "Configuration:"
echo "  V620 Server: ${SERVER_ADDR}"
echo "  Max Speed: ${MAX_SPEED} m/s"

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

# Launch
echo "Launching PPO autonomous training..."
mkdir -p log
LOG_FILE="log/ppo_rover_$(date +%Y%m%d_%H%M%S).log"

ros2 launch tractor_bringup ppo_autonomous.launch.py \
  server_addr:=${SERVER_ADDR} \
  max_speed:=${MAX_SPEED} \
  2>&1 | tee "$LOG_FILE" &

LAUNCH_PID=$!

trap 'echo; echo "🛑 Stopping autonomous episodes..."; kill $LAUNCH_PID 2>/dev/null; sleep 2; echo "✅ Stopped"; exit 0' SIGINT SIGTERM

echo ""
echo "PPO autonomous training running (PID: $LAUNCH_PID)"
echo "Log file: $LOG_FILE"
echo ""
echo "What's happening:"
echo "  1. Rover runs PPO policy on NPU (30Hz)"
echo "  2. Collects experience (RGB, Depth, Actions, Rewards)"
echo "  3. Sends batches to V620 server asynchronously"
echo "  4. Downloads updated model weights periodically"
echo ""
echo "Monitor:"
echo "  - ros2 topic echo /cmd_vel_ai"
echo "  - ros2 topic echo /min_forward_distance"
echo "  - ros2 topic echo /safety_monitor_status"
echo ""
echo "Press Ctrl+C to stop"

wait $LAUNCH_PID
