#!/bin/bash

# Remote Training Data Collection Script
# Streams RGB-D + proprioception data to V620 training server

echo "=================================================="
echo "ROS2 Rover - Remote Training Data Collection"
echo "=================================================="

if [ ! -f "install/setup.bash" ]; then
  echo "Error: Please run this script from the ros2-rover directory"
  exit 1
fi

# Configuration
V620_SERVER=${1:-"tcp://192.168.1.100:5555"}
COLLECTION_RATE=${2:-10.0}
MAX_SPEED=${3:-0.18}

echo "Configuration:"
echo "  V620 Server: ${V620_SERVER}"
echo "  Collection Rate: ${COLLECTION_RATE} Hz"
echo "  Max Speed: ${MAX_SPEED} m/s"

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

# Check ZeroMQ
echo "Checking ZeroMQ installation..."
python3 -c "import zmq; print(f'✓ ZeroMQ {zmq.zmq_version()} installed')" 2>/dev/null
if [ $? -ne 0 ]; then
  echo "⚠ ZeroMQ not installed. Installing..."
  pip3 install pyzmq
fi

# Test connection to V620
echo "Testing connection to V620 server..."
timeout 2 bash -c "python3 -c 'import zmq; ctx=zmq.Context(); s=ctx.socket(zmq.PUSH); s.connect(\"${V620_SERVER}\"); print(\"✓ Connected to V620\")'" 2>/dev/null
if [ $? -ne 0 ]; then
  echo "⚠ Could not connect to V620 server at ${V620_SERVER}"
  echo "Make sure the V620 training server is running!"
  read -p "Continue anyway? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi
fi

# Launch
echo "Launching data collection..."
mkdir -p log
LOG_FILE="log/remote_collection_$(date +%Y%m%d_%H%M%S).log"

ros2 launch tractor_bringup remote_training_collection.launch.py \
  server_address:=${V620_SERVER} \
  collection_rate_hz:=${COLLECTION_RATE} \
  max_speed:=${MAX_SPEED} \
  2>&1 | tee "$LOG_FILE" &

LAUNCH_PID=$!

trap 'echo; echo "🛑 Stopping data collection..."; kill $LAUNCH_PID 2>/dev/null; sleep 2; echo "✅ Stopped"; exit 0' SIGINT SIGTERM

echo "Data collection running (PID: $LAUNCH_PID)"
echo "Log file: $LOG_FILE"
echo ""
echo "Controls:"
echo "  - Use teleop to manually drive the rover"
echo "  - ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=cmd_vel_teleop"
echo ""
echo "Press Ctrl+C to stop"

wait $LAUNCH_PID
