#!/usr/bin/env bash
set -e

# Simple starter for RTAB-Map + Nav2
# Usage:
#   ./start_rtabmap_nav2.sh mapping        # builds map, global frame=odom
#   ./start_rtabmap_nav2.sh localization   # uses saved map, global frame=map

MODE=${1:-mapping}
WITH_TELEOP=false
WITH_AUTO=true

for arg in "$@"; do
  case "$arg" in
    --teleop)
      WITH_TELEOP=true ;;
    --no-autonomous)
      WITH_AUTO=false ;;
  esac
done

if [[ "$MODE" == "mapping" ]]; then
  echo "Launching RTAB-Map mapping mode (global_frame=odom)..."
  ros2 launch tractor_bringup rtabmap_nav2.launch.py \
    nav2_params_file:=`ros2 pkg prefix tractor_bringup`/share/tractor_bringup/config/nav2_params.yaml \
    with_teleop:=${WITH_TELEOP} with_autonomous_mapper:=${WITH_AUTO}
elif [[ "$MODE" == "localization" ]]; then
  echo "Launching RTAB-Map localization mode (global_frame=map + static layer)..."
  ros2 launch tractor_bringup rtabmap_nav2.launch.py \
    nav2_params_file:=`ros2 pkg prefix tractor_bringup`/share/tractor_bringup/config/nav2_params_localization.yaml \
    with_teleop:=${WITH_TELEOP} with_autonomous_mapper:=${WITH_AUTO}
else
  echo "Unknown mode: $MODE (use 'mapping' or 'localization')" >&2
  exit 1
fi
