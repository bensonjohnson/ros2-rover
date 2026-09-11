#!/bin/bash
# Install the rover control service (start/stop/HARDSTOP dashboard on :8090).
# Run from the ros2-rover workspace root ON THE ROVER:
#   ./deploy/install_rover_control.sh
#
# Replaces the legacy pc-brain-supervisor service (disabled here; the old
# BrainSupervisor class stays in the tree — rover_control composes it).

set -e

if [ ! -d "src" ] || [ ! -d ".git" ]; then
  echo "Error: run this from the ros2-rover workspace root"
  exit 1
fi

WORKSPACE=$(pwd)
SERVICE_NAME="rover-control"
TEMPLATE="deploy/${SERVICE_NAME}.service"

# The service must run as the workspace owner, never root — even if this
# script itself was invoked with sudo (SUDO_USER preserves the real user).
RUN_USER="${SUDO_USER:-$USER}"
if [ "$RUN_USER" = "root" ]; then
  echo "Error: could not determine a non-root user to run the service as."
  exit 1
fi

echo "Building workspace..."
source /opt/ros/jazzy/setup.bash
colcon build --packages-select tractor_bringup tractor_control tractor_sensors \
  --cmake-args -DCMAKE_BUILD_TYPE=Release

# The two services must not coexist (old supervisor binds :8082, which the
# brain dashboard needs when this service spawns AWAKE).
if systemctl list-unit-files "${LEGACY_NAME:-pc-brain-supervisor}.service" \
     --no-legend 2>/dev/null | grep -q pc-brain-supervisor; then
  echo "Stopping + disabling legacy pc-brain-supervisor..."
  sudo systemctl stop pc-brain-supervisor || true
  sudo systemctl disable pc-brain-supervisor || true
fi

echo "Installing ${SERVICE_NAME}.service (user=$RUN_USER, workspace=$WORKSPACE)"
sed -e "s|@USER@|$RUN_USER|g" -e "s|@WORKSPACE@|$WORKSPACE|g" "$TEMPLATE" | \
  sudo tee "/etc/systemd/system/${SERVICE_NAME}.service" > /dev/null

sudo systemctl daemon-reload
sudo systemctl enable --now "$SERVICE_NAME"

echo ""
echo "Done. Control page: http://$(hostname -I | awk '{print $1}'):8090"
echo "  (brain dashboard appears on :8082 while AWAKE)"
echo ""
echo "  status:  systemctl status $SERVICE_NAME"
echo "  logs:    journalctl -u $SERVICE_NAME -f"
echo "  stop:    sudo systemctl stop $SERVICE_NAME"
