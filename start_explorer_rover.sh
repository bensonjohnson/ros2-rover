#!/bin/bash
# ============================================================================
# Deep Exploration Network — Autonomous Home Mapping Startup Script
# ============================================================================
# NO CAMERA REQUIRED. Works with JUST LiDAR + IMU (the default).
#   - slam_toolbox builds a metric occupancy grid from LiDAR alone.
#   - The NN fuses LiDAR + occupancy map + IMU + wheel odometry.
#   - Add --depth to enable RealSense D435i (RGB-D RTAB-Map + NN depth stream).
#
# Sensor stack (default, no camera):
#   LiDAR (STL19P) + IMU (BNO085) + wheel encoders
#   slam_toolbox LiDAR SLAM → /map   ← metric occupancy grid
#   RF2O scan matching → /odom_rf2o   ← drift-free odometry
#   EKF fusion → /odometry/filtered   ← smoothed pose
#
# Usage:
#   ./start_explorer_rover.sh [mode] [options]
#
# Modes:
#   auto        Full autonomous exploration + mapping (default)
#   explore     Frontier-driven exploration only (no completion detection)
#   collect     Human teleop data collection (for remote training)
#
# Options:
#   --rate <hz>         Control rate (default 15.0)
#   --scale <f>         Action scale (default 0.6)
#   --noise <f>         Exploration noise stddev (default 0.05)
#   --lidar-port <dev>  LiDAR serial port (default /dev/ttyUSB0)
#   --depth             Enable RealSense D435i depth + RTAB-Map RGB-D SLAM
#   --learn             Enable online learning (PyTorch only, not RKNN)
#   --port <n>          Dashboard port (default 8083)
#   --server <addr>     Remote training server ZMQ address
#
# Examples:
#   ./start_explorer_rover.sh auto                    # LiDAR-only (default)
#   ./start_explorer_rover.sh auto --depth            # with RealSense
#   ./start_explorer_rover.sh collect --server tcp://192.168.1.100:5557
#   ./start_explorer_rover.sh auto --rate 20.0
# ============================================================================

echo "=================================================="
echo "ROS2 Rover - Deep Exploration Network"
echo "Autonomous Home Mapping with NPU Neural Network"
echo "=================================================="

if [ ! -f "install/setup.bash" ] && [ ! -d "src" ]; then
  echo "Error: Please run this script from the ros2-rover directory"
  exit 1
fi

# ---- Parse mode ----
MODE="${1:-auto}"
if [[ "$MODE" =~ ^(auto|explore|collect|sleep)$ ]]; then
  shift
else
  MODE="auto"
fi

# ---- Defaults ----
CONTROL_RATE="15.0"
ACTION_SCALE="0.6"
EXPLORATION_NOISE="0.05"
LIDAR_PORT="/dev/ttyUSB0"
USE_DEPTH="false"
LEARN="false"
DASHBOARD_PORT="8083"
SERVER_ADDR=""

# ---- Parse options ----
while [ $# -gt 0 ]; do
  case "$1" in
    --rate)       CONTROL_RATE="$2";       shift 2 ;;
    --scale)      ACTION_SCALE="$2";       shift 2 ;;
    --noise)      EXPLORATION_NOISE="$2";  shift 2 ;;
    --lidar-port) LIDAR_PORT="$2";         shift 2 ;;
    --depth)      USE_DEPTH="true";        shift ;;
    --learn)      LEARN="true";            shift ;;
    --port)       DASHBOARD_PORT="$2";     shift 2 ;;
    --server)     SERVER_ADDR="$2";        shift 2 ;;
    *)            echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo ""
echo "Configuration:"
echo "  Mode:               ${MODE}"
echo "  Control Rate:       ${CONTROL_RATE} Hz"
echo "  Action Scale:       ${ACTION_SCALE}"
echo "  Exploration Noise:  ${EXPLORATION_NOISE}"
echo "  LiDAR Port:         ${LIDAR_PORT}"
echo "  Use Depth Camera:   ${USE_DEPTH}"
echo "  Online Learning:    ${LEARN}"
echo "  Dashboard Port:     ${DASHBOARD_PORT}"
echo "  Remote Server:      ${SERVER_ADDR:-<none> (local only)}"
if [ "$USE_DEPTH" = "true" ]; then
  echo "  SLAM:               RTAB-Map RGB-D (RealSense D435i)"
else
  echo "  SLAM:               slam_toolbox LiDAR-only (no camera needed)"
fi
echo "  Neural Network:     $([ -f ~/.ros/explorer_brain.rknn ] && echo 'RKNN NPU' || echo 'PyTorch CPU fallback')"
echo "  Architecture:       LiDAR($([ "$USE_DEPTH" = "true" ] && echo '72 bins + 32x32 Depth' || echo '72 bins')) + OccMap(64x64) + IMU+Wheels + PlaceNovelty"
echo ""

if [ "$MODE" = "auto" ] || [ "$MODE" = "explore" ]; then
  echo "WARNING: Rover will drive AUTONOMOUSLY to map the environment!"
  echo "  - NN inference at ${CONTROL_RATE} Hz"
  echo "  - RTAB-Map builds a metric occupancy grid in real-time"
  echo "  - Frontier exploration drives toward unmapped areas"
  echo "  - Safety monitor hard-stops near obstacles"
  echo "  - Hold RB on Xbox controller for shadow teleop"
  echo "  - Model weights persist to ~/.ros/explorer_brain.pt"
  echo "  - Keep emergency stop ready"
elif [ "$MODE" = "collect" ]; then
  echo "INFO: Data collection mode — rover will NOT drive autonomously."
  echo "  - Hold RB on Xbox controller to drive manually"
  echo "  - All sensor data logged for remote GPU training"
  if [ -n "$SERVER_ADDR" ]; then
    echo "  - Chunks streamed via ZMQ to ${SERVER_ADDR}"
  else
    echo "  - Chunks saved locally to ~/.ros/explorer_chunks/"
  fi
else
  echo "INFO: Sleep mode — memory consolidation and dreaming"
fi

echo ""
read -p "Press Enter to continue or Ctrl+C to abort..."

# ---- Build ----
echo "Building workspace..."
colcon build --packages-select tractor_bringup tractor_control tractor_sensors tractor_explorer \
  --cmake-args -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
if [ $? -ne 0 ]; then
  echo "Build failed! Re-running with output:"
  colcon build --packages-select tractor_bringup tractor_control tractor_sensors tractor_explorer \
    --cmake-args -DCMAKE_BUILD_TYPE=Release
  exit 1
fi
echo "Build complete"

# ---- Source ----
echo "Sourcing ROS2 environment..."
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# ---- USB power config for RealSense ----
if [ "$USE_DEPTH" = "true" ]; then
  echo "Configuring USB power for RealSense..."
  for device in /sys/bus/usb/devices/*/idProduct; do
    if [ -f "$device" ] && [ "$(cat $device 2>/dev/null)" = "0b3a" ]; then
      USB_PATH=$(dirname $device)
      echo "on" | sudo tee $USB_PATH/power/control > /dev/null 2>&1
      echo "-1" | sudo tee $USB_PATH/power/autosuspend > /dev/null 2>&1
      break
    fi
  done
fi

# ---- Launch ----
echo "Launching Deep Explorer Network..."
mkdir -p log
LOG_FILE="log/explorer_rover_$(date +%Y%m%d_%H%M%S).log"

LAUNCH_ARGS=(
  "control_rate_hz:=${CONTROL_RATE}"
  "action_scale:=${ACTION_SCALE}"
  "exploration_noise:=${EXPLORATION_NOISE}"
  "lidar_port:=${LIDAR_PORT}"
  "use_depth:=${USE_DEPTH}"
  "mode:=${MODE}"
  "learn:=${LEARN}"
  "dashboard_port:=${DASHBOARD_PORT}"
  "model_path:=${HOME}/.ros/explorer_brain.pt"
  "rknn_model_path:=${HOME}/.ros/explorer_brain.rknn"
)

ros2 launch tractor_explorer explorer_nn.launch.py \
  "${LAUNCH_ARGS[@]}" \
  > >(tee "$LOG_FILE") 2>&1 &

LAUNCH_PID=$!
trap 'echo; echo "Stopping explorer..."; kill $LAUNCH_PID 2>/dev/null; sleep 2; echo "Stopped"; exit 0' SIGINT SIGTERM

echo ""
echo "Deep Explorer Network running (PID: $LAUNCH_PID)"
echo "Log file: $LOG_FILE"
echo ""
echo "Monitor:"
echo "  - Dashboard:  http://$(hostname -I | awk '{print $1}'):${DASHBOARD_PORT}"
echo "  - Map:        http://$(hostname -I | awk '{print $1}'):8080 (RTAB-Map viz)"
echo "  - Status:     ros2 topic echo /explorer/status"
echo "  - Goals:      ros2 topic echo /explorer/goal"
echo "  - Diagnostics: ros2 topic echo /explorer/diagnostics"
echo "  - Safety:     ros2 topic echo /emergency_stop"
echo "  - Tracks:     ros2 topic echo /track_cmd_ai"
echo ""
echo "Services:"
echo "  - Start exploration: ros2 service call /explore_manager/start_exploration std_srvs/srv/Trigger"
echo "  - Stop exploration:  ros2 service call /explore_manager/stop_exploration std_srvs/srv/Trigger"
echo ""

# In collect mode, print extra info
if [ "$MODE" = "collect" ]; then
  echo "DATA COLLECTION MODE"
  echo "  Drive the rover with Xbox controller (hold RB)"
  echo "  Data saved to ~/.ros/explorer_chunks/"
  echo "  To ship to training server: rsync -avz ~/.ros/explorer_chunks/ user@server:~/explorer_data/"
  echo ""
fi

echo "Press Ctrl+C to stop"
wait $LAUNCH_PID
