#!/bin/bash
# Start ES-SAC Rover

set -e  # Exit on error

# Parse arguments
NATS_SERVER=${1:-"nats://nats.gokickrocks.org:4222"}

echo "=============================================="
echo "Starting ES-SAC Rover Runner"
echo "=============================================="
echo "NATS Server: ${NATS_SERVER}"
echo ""

# Verify ROS workspace
if [ ! -f "install/setup.bash" ]; then
    echo "Error: ROS workspace not found. Please build the workspace first."
    exit 1
fi

# Source ROS workspace
source install/setup.bash

# Set ROS domain
export ROS_DOMAIN_ID=0

# Verify the es_episode_runner is available
if ! command -v es_episode_runner.py &> /dev/null; then
    echo "Error: es_episode_runner.py not found. Please rebuild tractor_bringup package."
    exit 1
fi

echo "Starting ES-SAC Rover Runner..."
ros2 run tractor_bringup es_episode_runner.py --ros-args -p nats_server:="${NATS_SERVER}"
