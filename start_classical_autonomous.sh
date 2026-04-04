#!/bin/bash

# Classical Autonomous Rover Startup Script (Tier 1)
# Launches the classical Nav2 + SLAM Toolbox + RF2O + EKF stack
# for agricultural / outdoor autonomous mapping and exploration.

echo "=================================================="
echo "ROS2 Rover - Classical Autonomous (Tier 1)"
echo "=================================================="

if [ ! -f "install/setup.bash" ]; then
  echo "Error: Please run this script from the ros2-rover directory"
  exit 1
fi

# Configuration
EXPLORATION_RADIUS=${1:-"10.0"}
WITH_FRONTIER=${2:-"false"}   # Start disabled; enable after first manual teleop verification
WITH_REALSENSE=${3:-"false"}  # Off by default for Tier 1 (LiDAR is authoritative)

echo "Configuration:"
echo "  Exploration Radius: ${EXPLORATION_RADIUS} m"
echo "  Frontier Explorer:  ${WITH_FRONTIER}"
echo "  RealSense D435i:    ${WITH_REALSENSE}"
echo "  Stack: Nav2 + SLAM Toolbox + RF2O + robot_localization EKF + STL-19p LiDAR + LSM9DS1 IMU"
echo ""

echo "WARNING: Rover may drive AUTONOMOUSLY if frontier explorer is enabled."
echo "  - Keep Xbox controller / kill switch ready"
echo "  - Verify TF tree and sensors before goal-giving"
echo ""
read -p "Press Enter to continue or Ctrl+C to abort..."

# Build workspace
echo "Building workspace..."
colcon build --packages-select tractor_bringup tractor_control tractor_sensors rf2o_laser_odometry --cmake-args -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
if [ $? -ne 0 ]; then
  echo "Build failed! Re-run with full output: colcon build --packages-select tractor_bringup tractor_control tractor_sensors rf2o_laser_odometry"
  exit 1
fi
echo "Build complete"

# Source environment
echo "Sourcing ROS2 environment..."
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# Configure RealSense USB power management (only relevant when RealSense is enabled,
# but harmless when it's off; mirrors start_ppo_remote_rover.sh).
if [ "${WITH_REALSENSE}" = "true" ]; then
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
fi

# Launch
echo "Launching classical autonomous stack..."
mkdir -p log
LOG_FILE="log/classical_autonomous_$(date +%Y%m%d_%H%M%S).log"

ros2 launch tractor_bringup classical_autonomous.launch.py \
  exploration_radius:=${EXPLORATION_RADIUS} \
  with_frontier_explorer:=${WITH_FRONTIER} \
  with_realsense:=${WITH_REALSENSE} \
  2>&1 | tee "$LOG_FILE" &

LAUNCH_PID=$!

trap 'echo; echo "Stopping classical autonomous stack..."; kill $LAUNCH_PID 2>/dev/null; sleep 2; echo "Stopped"; exit 0' SIGINT SIGTERM

echo ""
echo "Classical autonomous running (PID: $LAUNCH_PID)"
echo "Log file: $LOG_FILE"
echo ""
echo "Verification commands (run in another terminal):"
echo "  ros2 topic list | grep -E '/scan|/odom|/odom_rf2o|/imu/data|/odometry/filtered|/map|/cmd_vel'"
echo "  ros2 topic hz /scan                 # expect ~10 Hz"
echo "  ros2 topic hz /odometry/filtered    # expect ~30 Hz"
echo "  ros2 run tf2_tools view_frames      # TF tree sanity check"
echo "  ros2 lifecycle get /slam_toolbox    # expect: active"
echo "  ros2 lifecycle get /controller_server"
echo ""
echo "Give a goal: RViz2 '2D Nav Goal'"
echo "Press Ctrl+C to stop"

wait $LAUNCH_PID
