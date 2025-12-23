#!/bin/bash
set -e

# Source ROS 2
source /opt/ros/jazzy/setup.bash

# Source workspace if built
if [ -f /ros2_ws/install/setup.bash ]; then
    source /ros2_ws/install/setup.bash
fi

# Configure ROS 2 DDS for communication with host
# Use same domain ID as host (default 0)
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}

# For network host mode, use simple participant discovery
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE=/ros2_ws/fastrtps_profile.xml

exec "$@"
