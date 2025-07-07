#!/bin/bash

# Script to build the ROS 2 workspace

# Exit immediately if a command exits with a non-zero status.
set -e

# Source ROS 2 environment - MODIFY IF YOUR SETUP IS DIFFERENT
# For example, if using Humble:
ROS2_SETUP_SCRIPT="/opt/ros/jazzy/setup.bash"
# Or, if you have an overlay workspace, source that instead:
# ROS2_SETUP_SCRIPT="$HOME/my_ros2_ws/install/setup.bash"

if [ -f "$ROS2_SETUP_SCRIPT" ]; then
  echo "Sourcing ROS 2 environment from $ROS2_SETUP_SCRIPT"
  # shellcheck disable=SC1090
  source "$ROS2_SETUP_SCRIPT"
else
  echo "ROS 2 setup script not found at $ROS2_SETUP_SCRIPT."
  echo "Please modify this script to point to your ROS 2 installation."
  exit 1
fi

# Navigate to the root of your ROS 2 workspace (assuming this script is in the root)
WORKSPACE_ROOT=$(dirname "$(readlink -f "$0")")
cd "$WORKSPACE_ROOT"
echo "Building workspace at $WORKSPACE_ROOT"

# Build the workspace
# --symlink-install is recommended for development to avoid re-copying files
# --event-handlers console_direct+ shows output directly in the console
colcon build --symlink-install --event-handlers console_direct+ "$@"

echo "Build complete."
