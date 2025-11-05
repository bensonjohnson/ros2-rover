#!/bin/bash

# Evolution Training Autonomous Episode Runner for Rover
# Runs autonomous episodes and sends results to V620 server

echo "=================================================="
echo "ROS2 Rover - Evolution Training Episodes"
echo "=================================================="

if [ ! -f "install/setup.bash" ]; then
  echo "Error: Please run this script from the ros2-rover directory"
  exit 1
fi

# Configuration
SERVER_ADDR=${1:-"tcp://10.0.0.200:5556"}
EPISODE_DURATION=${2:-"60.0"}
MAX_SPEED=${3:-"0.18"}
COLLISION_DIST=${4:-"0.12"}

echo "Configuration:"
echo "  V620 Server: ${SERVER_ADDR}"
echo "  Episode Duration: ${EPISODE_DURATION}s"
echo "  Max Speed: ${MAX_SPEED} m/s"
echo "  Collision Distance: ${COLLISION_DIST} m"

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
echo "Launching evolution training autonomous episodes..."
mkdir -p log
LOG_FILE="log/evolution_rover_$(date +%Y%m%d_%H%M%S).log"

ros2 launch tractor_bringup map_elites_autonomous.launch.py \
  server_addr:=${SERVER_ADDR} \
  episode_duration:=${EPISODE_DURATION} \
  max_speed:=${MAX_SPEED} \
  collision_distance:=${COLLISION_DIST} \
  2>&1 | tee "$LOG_FILE" &

LAUNCH_PID=$!

trap 'echo; echo "🛑 Stopping autonomous episodes..."; kill $LAUNCH_PID 2>/dev/null; sleep 2; echo "✅ Stopped"; exit 0' SIGINT SIGTERM

echo ""
echo "Evolution training autonomous episodes running (PID: $LAUNCH_PID)"
echo "Log file: $LOG_FILE"
echo ""
echo "What's happening:"
echo "  1. Rover runs ${EPISODE_DURATION}s episodes autonomously"
echo "  2. Episode results sent to V620 server"
echo "  3. V620 evolves a single adaptive driving policy"
echo "  4. Rover receives improved model for next episode"
echo "  5. Model learns to adapt speed based on obstacles"
echo ""
echo "Monitor:"
echo "  - ros2 topic echo /cmd_vel_ai"
echo "  - ros2 topic echo /min_forward_distance"
echo "  - ros2 topic echo /safety_monitor_status"
echo ""
echo "Press Ctrl+C to stop"

wait $LAUNCH_PID
