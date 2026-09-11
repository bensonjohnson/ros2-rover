#!/bin/bash
#
# Spatial Explorer — rover startup.
#
# Builds a local occupancy grid in the odom frame from lidar + EKF pose, finds
# FRONTIERS (free cells bordering unknown = what lies beyond a doorway), and
# pure-pursuits the nearest one with a committed escape pivot when the safety
# gate blocks. Validated in sim against ground-truth map rooms: ~1.40
# rooms/house vs 1.00 and ZERO doorway crossings for every reactive controller.
#
# --driver pnn switches to the research path (predictive-coding map + distilled
# delta-rule policy). No checkpoint passes its acceptance gate yet, so that path
# is opt-in and gate-checked before it is allowed to launch.
#
# Usage:
#   ./start_spatial_explorer.sh [options]
#     --driver <d>      frontier (default) or pnn
#     --policy <path>   driver=pnn only (default ~/.ros/spatial_policy.pt)
#     --scale <f>       action scale (default 1.0 = the envelope it was
#                       distilled under; the safety monitor bounds risk)
#     --rate <hz>       control rate (default 15.0, matches the lidar)
#     --lidar-port <d>  lidar serial port (default: the LD19's stable
#                       by-id path; ttyUSB numbering shifts when the GPS
#                       is plugged/unplugged)
#     --imu <type>      bno085 (default) or lsm9ds1
#     --slam            also run slam_toolbox for a human-viewable map
#     --no-build        skip colcon build

set -e

DRIVER=frontier
POLICY="$HOME/.ros/spatial_policy.pt"
SCALE=1.0
RATE=15.0
LIDAR_PORT=/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0
IMU=bno085
SLAM=false
BUILD=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --driver)     DRIVER="$2"; shift 2 ;;
    --policy)     POLICY="$2"; shift 2 ;;
    --scale)      SCALE="$2"; shift 2 ;;
    --rate)       RATE="$2"; shift 2 ;;
    --lidar-port) LIDAR_PORT="$2"; shift 2 ;;
    --imu)        IMU="$2"; shift 2 ;;
    --slam)       SLAM=true; shift ;;
    --no-build)   BUILD=false; shift ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done

echo "=================================================="
echo "ROS2 Rover — Spatial Explorer (driver: $DRIVER)"
echo "=================================================="

if [ ! -d "src" ]; then
  echo "Error: run this from the ros2-rover directory"
  exit 1
fi

if [ "$DRIVER" = "pnn" ]; then
  if [ ! -f "$POLICY" ]; then
    echo "Error: no policy checkpoint at $POLICY"
    echo "  Train one on the dev machine:"
    echo "    python3 -m pnn_sim.spatial.distill_policy --save spatial_policy.pt"
    echo "  then copy it to the rover's ~/.ros/."
    exit 1
  fi
  # A policy that ignores the map still scores respectably on coverage — the
  # first one shipped did exactly that and drove in a constant slow arc.
  # Refuse to launch one that cannot steer by the map.
  echo "--- verifying the policy reads the map ---"
  if ! python3 -m pnn_sim.spatial.check_policy "$POLICY"; then
    echo
    echo "REFUSING TO LAUNCH: $POLICY does not steer by the map."
    echo "Use the default --driver frontier instead."
    exit 1
  fi
  echo
fi

if [ "$BUILD" = true ]; then
  echo "--- building ---"
  colcon build --packages-select tractor_bringup tractor_control tractor_sensors \
    --symlink-install
fi

source install/setup.bash

echo "--- launching ---"
echo "  driver      : $DRIVER"
echo "  action_scale: $SCALE"
echo "  control rate: $RATE Hz"
echo "  IMU         : $IMU"
echo "  slam_toolbox: $SLAM"
echo
echo "  watch the brain:  ros2 topic echo /pnn/spatial_diag"
echo "    [target_bearing, L, R, map_err, frontier_cells, seen_cells, blocked, tick_ms, exploring, odom_jump]"
echo

exec ros2 launch tractor_bringup spatial_explorer.launch.py \
  driver:="$DRIVER" \
  policy_path:="$POLICY" \
  action_scale:="$SCALE" \
  control_rate_hz:="$RATE" \
  lidar_port:="$LIDAR_PORT" \
  imu_type:="$IMU" \
  slam:="$SLAM"
