#!/bin/bash

# NPU BEV Exploration with Live PPO Training (bounded updates)

echo "=================================================="
echo "ROS2 Tractor - NPU BEV Exploration (PPO Live)"
echo "=================================================="

if [ ! -f "install/setup.bash" ]; then
  echo "Error: Please run this script from the ros2-rover directory"
  echo "Current directory: $(pwd)"
  exit 1
fi

MODE="inference"
MAX_SPEED=${2:-0.15}
EXPLORATION_TIME=${3:-300}
SAFETY_DISTANCE=${4:-0.2}

echo "Building minimal workspace..."
colcon build --packages-select tractor_bringup tractor_control --cmake-args -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
if [ $? -ne 0 ]; then
  echo "❌ Build failed!"
  exit 1
fi
echo "✓ Build complete"

echo "Sourcing ROS2 environment..."
source /opt/ros/jazzy/setup.bash
source install/setup.bash

echo "Configuring USB power management for RealSense..."
USB_DEVICE_PATH=""
for device in /sys/bus/usb/devices/*/idProduct; do
  if [ -f "$device" ] && [ "$(cat $device 2>/dev/null)" = "0b3a" ]; then
    USB_DEVICE_PATH=$(dirname $device)
    echo "✓ Found D435i at USB path: $USB_DEVICE_PATH"
    break
  fi
done
if [ -z "$USB_DEVICE_PATH" ]; then
  for path in "/sys/bus/usb/devices/8-1" "/sys/bus/usb/devices/2-1" "/sys/bus/usb/devices/1-1"; do
    if [ -d "$path" ]; then
      USB_DEVICE_PATH="$path"
      echo "✓ Using fallback USB path: $USB_DEVICE_PATH"
      break
    fi
  done
fi
if [ -n "$USB_DEVICE_PATH" ]; then
  echo "on" | sudo tee $USB_DEVICE_PATH/power/control > /dev/null 2>&1
  echo "-1" | sudo tee $USB_DEVICE_PATH/power/autosuspend > /dev/null 2>&1
  echo "✓ USB power management configured"
else
  echo "⚠ USB device path not found, continuing anyway"
fi

echo "Checking RealSense D435i..."
if command -v rs-enumerate-devices &> /dev/null; then
  timeout 5s rs-enumerate-devices | grep -q "D435I"
  if [ $? -eq 0 ]; then
    echo "✓ RealSense D435i detected"
    if [ -n "$USB_DEVICE_PATH" ] && [ -w "$USB_DEVICE_PATH" ]; then
      echo "Resetting USB device..."
      echo "0" | sudo tee $USB_DEVICE_PATH/authorized > /dev/null 2>&1
      sleep 1
      echo "1" | sudo tee $USB_DEVICE_PATH/authorized > /dev/null 2>&1
      echo "✓ USB device reset"
    fi
  else
    echo "⚠ RealSense D435i not detected - proceeding and letting rs_launch initialize"
  fi
else
  echo "⚠ rs-enumerate-devices not found - skipping camera pre-check"
fi

echo "Launching PPO exploration stack..."
# Prepare logging
mkdir -p log
LOG_FILE="log/ppo_exploration_$(date +%Y%m%d_%H%M%S).log"
{
  echo "[INFO] $(date) Starting launch with max_speed=${MAX_SPEED} safety_distance=${SAFETY_DISTANCE}"
  ros2 launch tractor_bringup npu_exploration_ppo.launch.py \
    max_speed:=${MAX_SPEED} \
    safety_distance:=${SAFETY_DISTANCE}
} | tee -a "$LOG_FILE" &

LAUNCH_PID=$!

trap 'echo; echo "🛑 Stopping PPO exploration..."; kill $LAUNCH_PID 2>/dev/null; sleep 2; kill -9 $LAUNCH_PID 2>/dev/null; echo "✅ Stopped"; exit 0' SIGINT SIGTERM

echo "PPO exploration running (PID: $LAUNCH_PID)"
echo "Log file: $LOG_FILE"
echo "- Monitor: ros2 topic echo /ppo_status"
echo "- Safety: ros2 topic echo /safety_monitor_status"
echo "- Min distance: ros2 topic echo /min_forward_distance"

# Background diagnostics snapshot after startup
(
  sleep 12
  {
    echo "[DIAG] $(date) Topics present:";
    ros2 topic list;
    echo "[DIAG] Nodes:";
    ros2 node list;
    echo "[DIAG] /bev/image rate (5s):";
    timeout 5s ros2 topic hz /bev/image || true;
    echo "[DIAG] /cmd_vel_ai rate (5s):";
    timeout 5s ros2 topic hz /cmd_vel_ai || true;
    echo "[DIAG] /min_forward_distance samples (5s):";
    timeout 5s ros2 topic echo /min_forward_distance || true;
    echo "[DIAG] /ppo_status samples (5s):";
    timeout 5s ros2 topic echo /ppo_status || true;
    echo "[HINT] grep -E 'PPO updated|RKNN model saved|RKNN reloaded' $LOG_FILE"
  } | tee -a "$LOG_FILE"
) &
wait $LAUNCH_PID
