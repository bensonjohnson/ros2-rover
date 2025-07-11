#!/bin/bash

# Autonomous Mapping Startup Script for ROS2 Tractor
# This script starts the complete autonomous mapping system with RealSense camera

echo "=================================================="
echo "ROS2 Tractor Autonomous Mapping System"
echo "=================================================="
echo "This will start:"
echo "  - Robot control and odometry"
echo "  - GPS and IMU sensors"
echo "  - RealSense D435i camera"
echo "  - SLAM mapping"
echo "  - Nav2 navigation"
echo "  - Autonomous exploration"
echo "  - Safety monitoring"
echo "  - Foxglove Bridge (port 8765)"
echo ""

# Check if we're in the right directory
if [ ! -f "install/setup.bash" ]; then
    echo "Error: Please run this script from the ros2-rover directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Source ROS2 workspace
echo "Sourcing ROS2 workspace..."
source install/setup.bash

echo "Ensuring RealSense device is ready..."
# Wait for RealSense device to be fully initialized
sleep 2

# Ensure proper USB power management
if [ -d "/sys/bus/usb/devices/8-1" ]; then
    echo "on" | sudo tee /sys/bus/usb/devices/8-1/power/control > /dev/null 2>&1
fi

# Test RealSense connectivity
echo "Testing RealSense connection..."
timeout 5s rs-enumerate-devices > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ RealSense device detected and ready"
else
    echo "⚠ RealSense device not ready - continuing anyway"
fi

# Python scripts are now installed as executables via setup.py

# Set mapping parameters
MAPPING_DURATION=${1:-600}  # Default 10 minutes (600 seconds)
MAX_SPEED=${2:-0.3}         # Default 0.3 m/s
SAFETY_DISTANCE=${3:-0.8}   # Default 0.8 meters

echo ""
echo "Configuration:"
echo "  Mapping Duration: ${MAPPING_DURATION} seconds"
echo "  Maximum Speed: ${MAX_SPEED} m/s"
echo "  Safety Distance: ${SAFETY_DISTANCE} m"
echo ""

# Countdown before starting
echo "Starting in:"
for i in {5..1}; do
    echo "  $i..."
    sleep 1
done
echo "  GO!"
echo ""

# Create directory for saved maps
mkdir -p maps/autonomous

# Launch the autonomous mapping system
echo "Launching autonomous mapping system..."
echo "Press Ctrl+C to stop safely"
echo ""
echo "🦊 Foxglove Bridge will be available at: ws://$(hostname -I | awk '{print $1}'):8765"
echo "    - Point Cloud: /camera/camera/depth/color/points"
echo "    - Camera: /camera/camera/color/image_raw"
echo "    - Map: /map"
echo "    - Robot Path: /navigate_to_pose/_action/feedback"
echo ""

ros2 launch tractor_bringup autonomous_mapping.launch.py \
    mapping_duration:=${MAPPING_DURATION} \
    max_speed:=${MAX_SPEED} \
    safety_distance:=${SAFETY_DISTANCE} \
    use_sim_time:=false

echo ""
echo "Autonomous mapping completed!"
echo ""

# Check if map was created
if [ -f "/home/ubuntu/ros2-rover/autonomous_map.pgm" ]; then
    echo "Map saved successfully:"
    echo "  - autonomous_map.pgm"
    echo "  - autonomous_map.yaml"
    
    # Move to maps directory with timestamp
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    mv autonomous_map.pgm "maps/autonomous/map_${TIMESTAMP}.pgm"
    mv autonomous_map.yaml "maps/autonomous/map_${TIMESTAMP}.yaml"
    
    echo "  - Moved to maps/autonomous/map_${TIMESTAMP}.*"
else
    echo "Warning: No map file found. Check logs for errors."
fi

echo ""
echo "Autonomous mapping session complete!"
echo "=================================================="