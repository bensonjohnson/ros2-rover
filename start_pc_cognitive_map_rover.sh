#!/bin/bash

# Cognitive-Map Active-Inference Rover Brain — Startup Script
# Builds the workspace and launches the growing-latent-field brain: lidar gives
# the local view, rf2o gives the pose, and the rover accumulates a sparse latent
# MAP that fills in unseen cells via a learned spatial prior. The actor steers
# toward frontier / low-confidence space (pure epistemic over the map). All
# learning is online predictive coding (local rules, no backprop, no pretrain).

echo "=================================================="
echo "ROS2 Rover - Cognitive-Map Active-Inference Brain"
echo "=================================================="

if [ ! -f "install/setup.bash" ] && [ ! -d "src" ]; then
  echo "Error: Please run this script from the ros2-rover directory"
  exit 1
fi

ACTION_SCALE=${1:-"0.6"}
CONTROL_RATE=${2:-"15.0"}
LIDAR_PORT=${3:-"/dev/ttyUSB0"}
CELL_SIZE=${4:-"0.5"}

echo "Configuration:"
echo "  Action Scale: ${ACTION_SCALE}   Control Rate: ${CONTROL_RATE} Hz"
echo "  Lidar Port:   ${LIDAR_PORT}   Cell Size: ${CELL_SIZE} m"
echo "  Pose:  rf2o lidar odometry (/odom_rf2o)"
echo "  Map:   growing latent field (online PC, no pretrain) -> fills in + frontiers"
echo "  Drive: pure epistemic over the map (steer toward unknown)"
echo ""

echo "WARNING: Rover drives AUTONOMOUSLY and learns/maps online!"
echo "  - Behavior is erratic until the map and decoder settle"
echo "  - Lidar safety monitor hard-stops the tracks near obstacles"
echo "  - Map persists to ~/.ros/pnn_cogmap.pt (saved periodically + on stop)"
echo "  - Keep emergency stop ready"
echo ""
read -p "Press Enter to continue or Ctrl+C to abort..."

echo "Building workspace..."
colcon build --packages-select tractor_bringup tractor_control tractor_sensors \
  rf2o_laser_odometry ldlidar_stl_ros2 --cmake-args -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
if [ $? -ne 0 ]; then
  echo "Build failed! Re-running with output:"
  colcon build --packages-select tractor_bringup tractor_control tractor_sensors \
    rf2o_laser_odometry ldlidar_stl_ros2 --cmake-args -DCMAKE_BUILD_TYPE=Release
  exit 1
fi
echo "Build complete"

echo "Sourcing ROS2 environment..."
source /opt/ros/jazzy/setup.bash
source install/setup.bash

echo "Launching cognitive-map brain..."
mkdir -p log
LOG_FILE="log/pc_cogmap_rover_$(date +%Y%m%d_%H%M%S).log"

ros2 launch tractor_bringup pc_cognitive_map.launch.py \
  action_scale:=${ACTION_SCALE} \
  control_rate_hz:=${CONTROL_RATE} \
  lidar_port:=${LIDAR_PORT} \
  cell_size:=${CELL_SIZE} \
  2>&1 | tee "$LOG_FILE" &

LAUNCH_PID=$!
trap 'echo; echo "Stopping cognitive-map brain..."; kill $LAUNCH_PID 2>/dev/null; sleep 2; echo "Stopped"; exit 0' SIGINT SIGTERM

echo ""
echo "Cognitive-map brain running (PID: $LAUNCH_PID)"
echo "Log file: $LOG_FILE"
echo ""
echo "Monitor:"
echo "  - Map dashboard: http://$(hostname -I | awk '{print $1}'):8083"
echo "  - ros2 topic echo /pnn/map_diagnostics   # [decode_err, novelty, n_cells, L, R, done]"
echo "  - ros2 topic echo /odom_rf2o"
echo "  - ros2 topic echo /emergency_stop"
echo ""
echo "Press Ctrl+C to stop"
wait $LAUNCH_PID
