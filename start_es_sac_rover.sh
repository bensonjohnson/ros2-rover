#!/bin/bash
# Start ES-SAC Rover
set -e # Exit on error

# Configuration
NATS_SERVER=${1:-"nats://nats.gokickrocks.org:4222"}
export ROS_DOMAIN_ID=0

echo "=============================================="
echo "Starting ES-SAC Rover Runner"
echo "=============================================="
echo "NATS Server: ${NATS_SERVER}"

# 1. Build the package to ensure the new node is installed
echo "🔧 Building tractor_bringup..."
# We use --symlink-install for faster development (so python changes reflect immediately)
colcon build --symlink-install --packages-select tractor_bringup

# 2. Source the workspace
if [ -f "install/setup.bash" ]; then
    echo "🔧 Sourcing workspace..."
    source install/setup.bash
else
    echo "❌ Error: install/setup.bash not found. Build failed?"
    exit 1
fi

# 3. Check if executable exists (debug check)
# ros2 run searches specifically, but we can verify via ros2 pkg executables
if ! ros2 pkg executables tractor_bringup | grep -q "es_episode_runner.py"; then
    echo "❌ Error: es_episode_runner.py not registered with ROS 2."
    echo "   Ensure it is in setup.py entry_points."
    exit 1
fi

# 4. Run
echo "🚀 Running ES-SAC Episode Runner..."
ros2 run tractor_bringup es_episode_runner.py --ros-args -p nats_server:=${NATS_SERVER}
