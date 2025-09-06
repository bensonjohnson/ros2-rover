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

echo "Launching PPO exploration stack..."
ros2 launch tractor_bringup npu_exploration_ppo.launch.py \
  max_speed:=${MAX_SPEED} \
  safety_distance:=${SAFETY_DISTANCE} \
  > log/ppo_exploration_$(date +%Y%m%d_%H%M%S).log 2>&1 &

LAUNCH_PID=$!

trap 'echo; echo "🛑 Stopping PPO exploration..."; kill $LAUNCH_PID 2>/dev/null; sleep 2; kill -9 $LAUNCH_PID 2>/dev/null; echo "✅ Stopped"; exit 0' SIGINT SIGTERM

echo "PPO exploration running (PID: $LAUNCH_PID)"
echo "- Monitor: ros2 topic echo /ppo_status"
echo "- Safety: ros2 topic echo /safety_monitor_status"
echo "- Min distance: ros2 topic echo /min_forward_distance"
wait $LAUNCH_PID

