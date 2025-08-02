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

# Build and source ROS2 workspace  
echo "Building workspace with minimal launch file..."
colcon build --packages-select tractor_bringup --cmake-args -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
echo "✓ Workspace built successfully"
echo "Sourcing ROS2 system installation..."
source /opt/ros/jazzy/setup.bash
echo "Sourcing local workspace..."
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
SAFETY_DISTANCE=${3:-0.5}   # Default 0.5 meters (obstacle detection)

echo ""
echo "Configuration:"
echo "  Mapping Duration: ${MAPPING_DURATION} seconds"
echo "  Maximum Speed: ${MAX_SPEED} m/s"
echo "  Safety Distance: ${SAFETY_DISTANCE} m (obstacle detection: 1 inch emergency stop)"
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
echo "    - Point Cloud (Direct to Costmap): /camera/camera/depth/color/points"
echo "    - Laser Scan (SLAM only): /scan"
echo "    - Camera: /camera/camera/color/image_raw"
echo "    - Map: /map"
echo "    - Local Costmap: /local_costmap/costmap"
echo "    - Global Costmap: /global_costmap/costmap"
echo "    - Voxel Grid: /local_costmap/voxel_grid"
echo ""

ros2 launch tractor_bringup autonomous_mapping_minimal.launch.py \
    mapping_duration:=${MAPPING_DURATION} \
    max_speed:=${MAX_SPEED} \
    safety_distance:=${SAFETY_DISTANCE} \
    exploration_distance:=${EXPLORATION_DISTANCE:-2.5} \
    use_sim_time:=false &

# Store the launch process ID
LAUNCH_PID=$!

# Wait for nodes to initialize
echo "Waiting for nodes to initialize..."
sleep 8

# Costmaps are now integrated with Nav2 lifecycle manager
echo "Costmaps will be auto-activated by Nav2 lifecycle manager..."
echo "✓ Voxel layer costmaps configured for direct point cloud processing"

# Nav2 lifecycle manager will automatically activate all nodes including collision monitor
echo "Waiting for Nav2 lifecycle manager to activate all nodes..."
sleep 15  # Wait for all nodes to be fully ready and activated

# Robot will now start with initial sweep to populate costmap
echo "Robot will start with initial 180-degree sweep to populate costmap..."
echo "  - Initial sweep phase: 180-degree rotation to scan environment"
echo "  - Costmap population: Camera will detect obstacles during sweep"
echo "  - Forward exploration: After sweep, robot begins mapping motion"
echo "  - Automatic obstacle avoidance and navigation"
echo "✓ Initial sweep and forward exploration mapping ready"

# Wait for the launch process to complete
wait $LAUNCH_PID

echo ""
echo "Autonomous mapping completed!"
echo ""

# Check if map was created (default SLAM toolbox location)
if [ -f "/home/ubuntu/ros2-rover/maps/yard_slam_map.pgm" ]; then
    echo "Map saved successfully:"
    echo "  - yard_slam_map.pgm"
    echo "  - yard_slam_map.yaml"
    
    # Move to maps directory with timestamp
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    cp /home/ubuntu/ros2-rover/maps/yard_slam_map.pgm "maps/autonomous/map_${TIMESTAMP}.pgm"
    cp /home/ubuntu/ros2-rover/maps/yard_slam_map.yaml "maps/autonomous/map_${TIMESTAMP}.yaml"
    
    echo "  - Copied to maps/autonomous/map_${TIMESTAMP}.*"
else
    echo "Warning: No map file found at expected location. Check logs for errors."
    echo "Expected location: /home/ubuntu/ros2-rover/maps/yard_slam_map.*"
fi

echo ""
echo "Autonomous mapping session complete!"
echo "=================================================="
