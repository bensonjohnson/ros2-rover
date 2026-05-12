#!/bin/bash

# RLPD + HIL-SERL TELEOP COLLECTION Startup Script
# Drives the rover via Xbox controller and ships RLPD chunks (rgb + bev +
# proprio + teleop-action) to the GPU server. Every chunk is marked
# `is_demo=True` so the trainer appends to its persistent demo replay
# (checkpoints_rlpd/demos.npz) for HIL-SERL warm-start.
#
# After enough collection (typically 30-60 minutes), switch to
# start_rlpd_remote_rover.sh for autonomous training with live interventions.

echo "=================================================="
echo "ROS2 Rover - RLPD + HIL-SERL TELEOP COLLECTION"
echo "=================================================="

if [ ! -f "install/setup.bash" ]; then
  echo "Error: Please run this script from the ros2-rover directory"
  exit 1
fi

# Configuration
SERVER_ADDR=${1:-"192.168.1.100"}
MAX_SPEED=${2:-"0.18"}
CHUNK_LEN=${3:-"64"}
FRAME_STACK=${4:-"4"}

echo "Configuration:"
echo "  Server Address: ${SERVER_ADDR}"
echo "  Max Speed: ${MAX_SPEED} m/s"
echo "  Chunk Length: ${CHUNK_LEN} steps"
echo "  Frame Stack: ${FRAME_STACK} frames"
echo "  Architecture: RLPD (SAC + critic ensemble + frozen ResNet18)"
echo "  Mode: TELEOP collection (RKNN inference disabled, all chunks marked is_demo=True)"
echo ""

echo "WARNING: Rover will move under YOUR controller commands!"
echo "  - Hold the deadman button (RB) and steer with the left stick"
echo "  - Twist on /cmd_vel_teleop is converted to track actions and recorded"
echo "  - Chunks ship to ${SERVER_ADDR} via ZMQ; server appends to demos.npz"
echo "  - When done, Ctrl+C, then run start_rlpd_remote_rover.sh for autonomous"
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

echo "Launching RLPD teleop collection..."
mkdir -p log
LOG_FILE="log/rlpd_teleop_collection_$(date +%Y%m%d_%H%M%S).log"

ros2 launch tractor_bringup rlpd_remote_teleop.launch.py \
  server_addr:=${SERVER_ADDR} \
  max_speed:=${MAX_SPEED} \
  chunk_len:=${CHUNK_LEN} \
  frame_stack:=${FRAME_STACK} \
  2>&1 | tee "$LOG_FILE" &

LAUNCH_PID=$!

trap 'echo; echo "Stopping RLPD teleop collection..."; kill $LAUNCH_PID 2>/dev/null; sleep 2; echo "Stopped"; exit 0' SIGINT SIGTERM

echo ""
echo "RLPD teleop collection running (PID: $LAUNCH_PID)"
echo "Log file: $LOG_FILE"
echo ""
echo "What's happening:"
echo "  1. You drive the rover with the Xbox controller (hold RB, left stick to steer)"
echo "  2. Twist on /cmd_vel_teleop is converted to track actions and published on /track_cmd_ai"
echo "  3. Sensors + your actions + rewards + is_demo=True flags get packaged into ${CHUNK_LEN}-step chunks"
echo "  4. Chunks ship to the GPU server via ZMQ; server appends to checkpoints_rlpd/demos.npz"
echo "  5. RLPD trainer begins SAC updates once min_online_size transitions are accumulated"
echo ""
echo "When done collecting:"
echo "  - Ctrl+C here to stop teleop"
echo "  - Run ./start_rlpd_remote_rover.sh ${SERVER_ADDR} for autonomous training"
echo "  - The server keeps running across the swap; demos.npz + checkpoints persist"
echo ""
echo "Monitor:"
echo "  - Dashboard: http://$(hostname -I | awk '{print $1}'):8081"
echo "  - ros2 topic echo /cmd_vel_teleop"
echo "  - ros2 topic echo /track_cmd_ai"
echo "  - ros2 topic echo /emergency_stop"
echo ""
echo "Press Ctrl+C to stop"

wait $LAUNCH_PID
