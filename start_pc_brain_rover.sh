#!/bin/bash

# Predictive-Coding Active-Inference Rover Brain — Startup Script
# Builds the workspace and launches the from-scratch "brain": a temporal
# predictive-coding world model that learns purely online on the rover CPU
# (PyTorch, local updates — no backprop) from lidar alone, and explores by
# maximizing expected information gain (pure epistemic). The lidar safety
# monitor hard-stops the tracks near obstacles so early erratic behavior is
# physically bounded.

echo "=================================================="
echo "ROS2 Rover - Predictive-Coding Active-Inference Brain"
echo "=================================================="

if [ ! -f "install/setup.bash" ] && [ ! -d "src" ]; then
  echo "Error: Please run this script from the ros2-rover directory"
  exit 1
fi

# Configuration (all optional positional args)
ACTION_SCALE=${1:-"0.6"}     # scales track output [-1,1] (gentler while young)
CONTROL_RATE=${2:-"15.0"}    # brain inference/control rate (Hz) — match lidar scan rate
LIDAR_PORT=${3:-"/dev/ttyUSB0"}
DASHBOARD_PORT=${4:-"8082"}

echo "Configuration:"
echo "  Action Scale:   ${ACTION_SCALE}"
echo "  Control Rate:   ${CONTROL_RATE} Hz"
echo "  Lidar Port:     ${LIDAR_PORT}"
echo "  Dashboard Port: ${DASHBOARD_PORT}"
echo "  Model:          temporal predictive coding (local updates, no backprop)"
echo "  Drive:          pure epistemic (ensemble-disagreement info gain)"
echo "  Input:          lidar only  ->  Output: per-track [L,R] in [-1,1]"
echo ""

echo "WARNING: Rover will drive AUTONOMOUSLY and learn online!"
echo "  - Behavior WILL be erratic until the world model settles"
echo "  - Lidar safety monitor hard-stops the tracks near obstacles"
echo "  - Brain weights persist to ~/.ros/pnn_brain.pt (saved periodically)"
echo "  - Keep emergency stop ready"
echo ""
read -p "Press Enter to continue or Ctrl+C to abort..."

echo "Building workspace..."
colcon build --packages-select tractor_bringup tractor_control tractor_sensors \
  --cmake-args -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
if [ $? -ne 0 ]; then
  echo "Build failed! Re-running with output:"
  colcon build --packages-select tractor_bringup tractor_control tractor_sensors \
    --cmake-args -DCMAKE_BUILD_TYPE=Release
  exit 1
fi
echo "Build complete"

echo "Sourcing ROS2 environment..."
source /opt/ros/jazzy/setup.bash
source install/setup.bash

echo "Launching predictive-coding active-inference brain..."
mkdir -p log
LOG_FILE="log/pc_brain_rover_$(date +%Y%m%d_%H%M%S).log"

ros2 launch tractor_bringup pc_active_inference.launch.py \
  action_scale:=${ACTION_SCALE} \
  control_rate_hz:=${CONTROL_RATE} \
  lidar_port:=${LIDAR_PORT} \
  2>&1 | tee "$LOG_FILE" &

LAUNCH_PID=$!

trap 'echo; echo "Stopping PC brain..."; kill $LAUNCH_PID 2>/dev/null; sleep 2; echo "Stopped"; exit 0' SIGINT SIGTERM

echo ""
echo "PC brain running (PID: $LAUNCH_PID)"
echo "Log file: $LOG_FILE"
echo ""
echo "What's happening:"
echo "  1. Lidar -> 72-bin openness vector"
echo "  2. PC world model settles a latent state (free-energy minimization)"
echo "  3. Local PC update nudges weights every tick (no backprop)"
echo "  4. Actor picks the most informative track command (max info gain)"
echo "  5. /track_cmd_ai -> lidar_safety_monitor -> /track_cmd -> motors"
echo ""
echo "Monitor:"
echo "  - Dashboard: http://$(hostname -I | awk '{print $1}'):${DASHBOARD_PORT}"
echo "  - ros2 topic echo /pnn/diagnostics   # [F, obs_err, epi, epi_max, L, R]"
echo "  - ros2 topic echo /track_cmd_ai"
echo "  - ros2 topic echo /emergency_stop"
echo ""
echo "Press Ctrl+C to stop"

wait $LAUNCH_PID
