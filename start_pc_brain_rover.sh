#!/bin/bash

# Predictive-Coding Active-Inference Rover Brain — Startup Script
# Builds the workspace and launches the from-scratch "brain": a temporal
# predictive-coding world model that learns purely online on the rover CPU
# (PyTorch, local updates — no backprop) from lidar alone, and explores by
# maximizing expected information gain (pure epistemic). The lidar safety
# monitor hard-stops the tracks near obstacles so early erratic behavior is
# physically bounded.
#
# Usage:
#   ./start_pc_brain_rover.sh [awake|sleep] [options]
# Options (named flags; legacy positional args still accepted in this order:
# action_scale, control_rate, lidar_port, dashboard_port):
#   --scale <f>        action scale (default 0.6)
#   --rate <hz>        control rate (default 15.0)
#   --lidar-port <dev> lidar serial port (default /dev/ttyUSB0)
#   --port <n>         web dashboard port (default 8082)

echo "=================================================="
echo "ROS2 Rover - Predictive-Coding Active-Inference Brain"
echo "=================================================="

if [ ! -f "install/setup.bash" ] && [ ! -d "src" ]; then
  echo "Error: Please run this script from the ros2-rover directory"
  exit 1
fi

# Parse Mode (awake or sleep)
MODE=$1
if [ -z "$MODE" ]; then
  echo "Select Rover Brain Mode:"
  echo "  1) Awake (Active Inference & Exploratory Driving)"
  echo "  2) Sleep (Offline Memory Consolidation & Dreaming)"
  read -p "Choose mode [1]: " choice
  if [ "$choice" = "2" ] || [ "$choice" = "sleep" ]; then
    MODE="sleep"
  else
    MODE="awake"
  fi
else
  if [ "$MODE" = "sleep" ] || [ "$MODE" = "awake" ]; then
    shift
  else
    # Default to awake if first arg is not sleep or awake
    MODE="awake"
  fi
fi

# Configuration: defaults, then named flags, then legacy positional args.
ACTION_SCALE="0.6"     # scales track output [-1,1] (gentler while young)
CONTROL_RATE="15.0"    # brain inference/control rate (Hz) — match lidar scan rate
LIDAR_PORT="/dev/ttyUSB0"
DASHBOARD_PORT="8082"

POSITIONAL=()
while [ $# -gt 0 ]; do
  case "$1" in
    --scale)      ACTION_SCALE="$2";   shift 2 ;;
    --rate)       CONTROL_RATE="$2";   shift 2 ;;
    --lidar-port) LIDAR_PORT="$2";     shift 2 ;;
    --port)       DASHBOARD_PORT="$2"; shift 2 ;;
    *)            POSITIONAL+=("$1");  shift ;;
  esac
done
[ -n "${POSITIONAL[0]}" ] && ACTION_SCALE="${POSITIONAL[0]}"
[ -n "${POSITIONAL[1]}" ] && CONTROL_RATE="${POSITIONAL[1]}"
[ -n "${POSITIONAL[2]}" ] && LIDAR_PORT="${POSITIONAL[2]}"
[ -n "${POSITIONAL[3]}" ] && DASHBOARD_PORT="${POSITIONAL[3]}"

echo "Configuration:"
echo "  Mode:           ${MODE}"
if [ "$MODE" = "awake" ]; then
  echo "  Action Scale:   ${ACTION_SCALE}"
  echo "  Control Rate:   ${CONTROL_RATE} Hz"
  echo "  Lidar Port:     ${LIDAR_PORT}"
fi
echo "  Dashboard Port: ${DASHBOARD_PORT}"
echo "  Model:          temporal predictive coding (local Hebbian updates, no backprop)"
if [ "$MODE" = "awake" ]; then
  echo "  Drive:          pure epistemic (ensemble-disagreement info gain)"
  echo "  Input:          lidar only  ->  Output: per-track [L,R] in [-1,1]"
else
  echo "  Cycle:          Slow-Wave Sleep (replay) + REM Sleep (dreaming) + Homeostasis Pruning"
fi
echo ""

if [ "$MODE" = "awake" ]; then
  echo "WARNING: Rover will drive AUTONOMOUSLY and learn online!"
  echo "  - Behavior WILL be erratic until the world model settles"
  echo "  - Lidar safety monitor hard-stops the tracks near obstacles"
  echo "  - Brain weights persist to ~/.ros/pnn_brain.pt (saved periodically)"
  echo "  - Keep emergency stop ready"
else
  echo "INFO: Rover is entering Sleep Mode (Memory Consolidation)."
  echo "  - The rover will remain stationary"
  echo "  - Will replay experience log from ~/.ros/pnn_experience.jsonl"
  echo "  - Consolidates weights back to ~/.ros/pnn_brain.pt"
fi
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

if [ "$MODE" = "awake" ]; then
  echo "Launching predictive-coding active-inference brain..."
  mkdir -p log
  LOG_FILE="log/pc_brain_rover_$(date +%Y%m%d_%H%M%S).log"

  # Process substitution (not a pipeline) so $! is the ros2 launch process
  # itself and the trap below actually stops it.
  ros2 launch tractor_bringup pc_active_inference.launch.py \
    action_scale:=${ACTION_SCALE} \
    control_rate_hz:=${CONTROL_RATE} \
    lidar_port:=${LIDAR_PORT} \
    dashboard_port:=${DASHBOARD_PORT} \
    > >(tee "$LOG_FILE") 2>&1 &

  LAUNCH_PID=$!
  trap 'echo; echo "Stopping PC brain..."; kill $LAUNCH_PID 2>/dev/null; sleep 2; echo "Stopped"; exit 0' SIGINT SIGTERM

  echo ""
  echo "PC brain running (PID: $LAUNCH_PID)"
  echo "Log file: $LOG_FILE"
  echo ""
  echo "Monitor:"
  echo "  - Dashboard: http://$(hostname -I | awk '{print $1}'):${DASHBOARD_PORT}"
  echo "  - ros2 topic echo /pnn/diagnostics"
  echo "  - ros2 topic echo /emergency_stop"
  echo ""
  echo "Press Ctrl+C to stop"
  wait $LAUNCH_PID
else
  echo "Launching sleep consolidator and dreaming server..."

  ros2 run tractor_bringup sleep_consolidator \
    --model_path ~/.ros/pnn_brain.pt \
    --experience_log_path ~/.ros/pnn_experience.jsonl \
    --dashboard_port ${DASHBOARD_PORT} \
    --visualize &

  CONSOLIDATOR_PID=$!
  trap 'echo; echo "Terminating sleep consolidator..."; kill $CONSOLIDATOR_PID 2>/dev/null; sleep 2; echo "Stopped"; exit 0' SIGINT SIGTERM

  echo ""
  echo "Sleep consolidator running (PID: $CONSOLIDATOR_PID)"
  echo "Monitor:"
  echo "  - Dream Dashboard: http://$(hostname -I | awk '{print $1}'):${DASHBOARD_PORT}"
  echo ""
  echo "Press Ctrl+C to stop sleep consolidator early"
  wait $CONSOLIDATOR_PID
fi
