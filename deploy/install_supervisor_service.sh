#!/bin/bash
# Install the PC brain supervisor as a systemd service.
# Run from the ros2-rover workspace root on the rover:
#   ./deploy/install_supervisor_service.sh

set -e

if [ ! -d "src" ] || [ ! -d ".git" ]; then
  echo "Error: run this from the ros2-rover workspace root"
  exit 1
fi

WORKSPACE=$(pwd)
SERVICE_NAME="pc-brain-supervisor"
TEMPLATE="deploy/${SERVICE_NAME}.service"

# The service must run as the workspace owner, never root — even if this
# script itself was invoked with sudo (SUDO_USER preserves the real user).
RUN_USER="${SUDO_USER:-$USER}"
if [ "$RUN_USER" = "root" ]; then
  echo "Error: could not determine a non-root user to run the service as."
  echo "Run this script as the rover user (it will sudo where needed)."
  exit 1
fi

# Always rebuild so freshly added executables (entry points) exist before
# the service tries to run them.
echo "Building workspace..."
source /opt/ros/jazzy/setup.bash
colcon build --packages-select tractor_bringup tractor_control tractor_sensors \
  --cmake-args -DCMAKE_BUILD_TYPE=Release

echo "Installing ${SERVICE_NAME}.service (user=$RUN_USER, workspace=$WORKSPACE)"
sed -e "s|@USER@|$RUN_USER|g" -e "s|@WORKSPACE@|$WORKSPACE|g" "$TEMPLATE" | \
  sudo tee "/etc/systemd/system/${SERVICE_NAME}.service" > /dev/null

sudo systemctl daemon-reload
sudo systemctl enable --now "$SERVICE_NAME"

echo ""
echo "Done. The supervisor starts at boot, pulls + rebuilds the code, and"
echo "serves the dashboard at http://$(hostname -I | awk '{print $1}'):8082"
echo ""
echo "  status:  systemctl status $SERVICE_NAME"
echo "  logs:    journalctl -u $SERVICE_NAME -f"
echo "  stop:    sudo systemctl stop $SERVICE_NAME"
