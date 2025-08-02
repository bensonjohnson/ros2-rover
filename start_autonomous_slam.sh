#!/bin/bash

# Simplified Autonomous SLAM Mapping Script for ROS2 Tractor
# Uses RealSense D435i (RGB-D + IMU) + wheel encoders for complete area mapping

echo "=================================================="
echo "ROS2 Tractor - Autonomous SLAM Mapping System"
echo "=================================================="
echo "This simplified system will:"
echo "  ✓ Use RealSense D435i camera (RGB-D + IMU)"
echo "  ✓ Use Hiwonder motor wheel encoders"
echo "  ✓ Run SLAM Toolbox for mapping & localization"
echo "  ✓ Use frontier-based exploration for complete coverage"
echo "  ✓ Avoid dynamic obstacles (people)"
echo "  ✓ Create high-quality maps of entire area"
echo ""

# Check if we're in the right directory
if [ ! -f "install/setup.bash" ]; then
    echo "Error: Please run this script from the ros2-rover directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Build the workspace
echo "Building ROS2 workspace..."
colcon build --packages-select tractor_bringup --cmake-args -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Build failed! Please check for compilation errors."
    exit 1
fi
echo "✓ Workspace built successfully"

# Source ROS2 environment
echo "Sourcing ROS2 environment..."
source /opt/ros/jazzy/setup.bash
source install/setup.bash
echo "✓ ROS2 environment sourced"

# Check RealSense device
echo "Checking RealSense D435i connection..."
timeout 5s rs-enumerate-devices > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ RealSense D435i detected and ready"
else
    echo "⚠ RealSense device not ready - will attempt to continue"
fi

# Ensure proper USB power management for RealSense
if [ -d "/sys/bus/usb/devices/8-1" ]; then
    echo "on" | sudo tee /sys/bus/usb/devices/8-1/power/control > /dev/null 2>&1
fi

# Configuration parameters
MAX_SPEED=${1:-0.2}          # Maximum exploration speed (m/s)
EXPLORATION_RADIUS=${2:-10.0} # Maximum distance from start (meters)

echo ""
echo "Configuration:"
echo "  Maximum Speed: ${MAX_SPEED} m/s"
echo "  Exploration Radius: ${EXPLORATION_RADIUS} m"
echo "  SLAM Algorithm: SLAM Toolbox (async)"
echo "  Exploration: Frontier-based (complete coverage)"
echo "  Dynamic Obstacles: Collision monitor active"
echo ""

# Create maps directory
mkdir -p maps/slam_maps

# Countdown
echo "Starting autonomous SLAM mapping in:"
for i in {5..1}; do
    echo "  $i..."
    sleep 1
done
echo "  🚀 LAUNCHING!"
echo ""

# Launch the system
echo "Launching autonomous SLAM mapping system..."
echo "Press Ctrl+C to stop and save the map"
echo ""
echo "📊 System Status:"
echo "  - SLAM Toolbox: Creating map in real-time"
echo "  - Frontier Explorer: Finding unexplored areas"
echo "  - Collision Monitor: Avoiding dynamic obstacles"
echo "  - Map will be saved to: maps/slam_maps/"
echo ""

# Run the launch file
ros2 launch tractor_bringup autonomous_slam_mapping.launch.py \
    max_speed:=${MAX_SPEED} \
    exploration_radius:=${EXPLORATION_RADIUS} \
    use_sim_time:=false &

# Store the launch process ID
LAUNCH_PID=$!

# Function to save map on exit
save_map_on_exit() {
    echo ""
    echo "🛑 Stopping autonomous mapping..."
    
    # Kill the launch process
    kill $LAUNCH_PID 2>/dev/null
    
    # Wait a moment for nodes to shutdown gracefully
    sleep 3
    
    # Save the map
    echo "💾 Saving final map..."
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    MAP_NAME="autonomous_slam_${TIMESTAMP}"
    
    # Use map_saver service to save the map
    timeout 10s ros2 service call /map_saver/save_map nav2_msgs/srv/SaveMap \
        "{map_topic: '/map', map_url: 'maps/slam_maps/${MAP_NAME}', image_format: 'pgm', map_mode: 'trinary', free_thresh: 0.25, occupied_thresh: 0.65}" > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✅ Map saved successfully:"
        echo "   📁 maps/slam_maps/${MAP_NAME}.pgm"
        echo "   📁 maps/slam_maps/${MAP_NAME}.yaml"
        
        # Also try to save SLAM Toolbox's serialized map
        timeout 5s ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph \
            "{filename: 'maps/slam_maps/${MAP_NAME}_serialized'}" > /dev/null 2>&1
        
        if [ $? -eq 0 ]; then
            echo "   📁 maps/slam_maps/${MAP_NAME}_serialized.posegraph"
            echo "   (SLAM Toolbox serialized map for later use)"
        fi
        
        # Display map info
        if [ -f "maps/slam_maps/${MAP_NAME}.yaml" ]; then
            echo ""
            echo "📋 Map Information:"
            grep -E "(resolution|origin)" "maps/slam_maps/${MAP_NAME}.yaml"
        fi
        
    else
        echo "⚠ Failed to save map via map_saver service"
        echo "   The SLAM Toolbox may have auto-saved to its default location"
        
        # Check for SLAM Toolbox auto-saved maps
        if [ -f "/home/ubuntu/ros2-rover/maps/autonomous_slam_map.pgm" ]; then
            echo "   Found SLAM Toolbox auto-saved map, copying..."
            cp "/home/ubuntu/ros2-rover/maps/autonomous_slam_map.pgm" "maps/slam_maps/${MAP_NAME}.pgm"
            cp "/home/ubuntu/ros2-rover/maps/autonomous_slam_map.yaml" "maps/slam_maps/${MAP_NAME}.yaml"
            echo "✅ Map copied to maps/slam_maps/${MAP_NAME}.*"
        fi
    fi
    
    echo ""
    echo "🏁 Autonomous SLAM mapping session complete!"
    echo "=================================================="
}

# Set up signal handlers
trap save_map_on_exit SIGINT SIGTERM

# Wait for the launch process and monitor
echo "🔄 Autonomous mapping in progress..."
echo "   To monitor progress, you can run in another terminal:"
echo "   - 'ros2 topic echo /exploration_status' (exploration status)"
echo "   - 'ros2 topic echo /mapping_status' (SLAM status)"
echo "   - 'rviz2' (visualization)"
echo ""

# Keep the script running and wait for user interrupt
wait $LAUNCH_PID

# If we get here, the launch process ended naturally (shouldn't happen normally)
save_map_on_exit