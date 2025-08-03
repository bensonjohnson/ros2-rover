#!/bin/bash

# Minimal NPU Point Cloud Exploration Script
# Clean architecture: Hardware + AI only, no SLAM/Nav2 complexity

echo "=================================================="
echo "ROS2 Tractor - NPU Point Cloud Exploration"
echo "=================================================="
echo "This minimal system uses:"
echo "  ✓ Hiwonder motor control"
echo "  ✓ RealSense D435i (point clouds only)"
echo "  ✓ NPU-based exploration AI"
echo "  ✓ Direct safety monitoring"
echo "  ✓ No SLAM/Nav2 complexity"
echo ""

# Check if we're in the right directory
if [ ! -f "install/setup.bash" ]; then
    echo "Error: Please run this script from the ros2-rover directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Build only what we need
echo "Building minimal workspace..."
colcon build --packages-select tractor_bringup tractor_control --cmake-args -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Build failed! Please check for compilation errors."
    exit 1
fi
echo "✓ Workspace built successfully"

# Source environment
echo "Sourcing ROS2 environment..."
source /opt/ros/jazzy/setup.bash
source install/setup.bash
echo "✓ ROS2 environment sourced"

# USB power management for RealSense
echo "Configuring USB power management..."
if [ -d "/sys/bus/usb/devices/8-1" ]; then
    echo "on" | sudo tee /sys/bus/usb/devices/8-1/power/control > /dev/null 2>&1
    echo "✓ USB power management configured"
else
    echo "⚠ USB device path not found, continuing anyway"
fi

# Check RealSense
echo "Checking RealSense D435i..."
timeout 3s rs-enumerate-devices > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ RealSense D435i detected"
else
    echo "⚠ RealSense not detected - will attempt to continue"
fi

# Configuration
MAX_SPEED=${1:-0.15}        # Conservative speed for AI learning
EXPLORATION_TIME=${2:-300}  # 5 minutes default
SAFETY_DISTANCE=${3:-0.2}   # 20cm safety bubble

echo ""
echo "Configuration:"
echo "  Maximum Speed: ${MAX_SPEED} m/s"
echo "  Exploration Time: ${EXPLORATION_TIME} seconds"
echo "  Safety Distance: ${SAFETY_DISTANCE} m"
echo ""

# Countdown
echo "Starting NPU exploration in:"
for i in {3..1}; do
    echo "  $i..."
    sleep 1
done
echo "  🚀 LAUNCHING!"
echo ""

# Launch the minimal system
echo "Launching NPU exploration system..."
echo "Press Ctrl+C to stop safely"
echo ""

ros2 launch tractor_bringup npu_exploration.launch.py \
    max_speed:=${MAX_SPEED} \
    exploration_time:=${EXPLORATION_TIME} \
    safety_distance:=${SAFETY_DISTANCE} \
    use_sim_time:=false &

LAUNCH_PID=$!

# Simple shutdown handler
shutdown_handler() {
    echo ""
    echo "🛑 Stopping NPU exploration..."
    kill $LAUNCH_PID 2>/dev/null
    sleep 2
    echo "✅ NPU exploration stopped safely"
    echo "=================================================="
}

trap shutdown_handler SIGINT SIGTERM

echo "🤖 NPU exploration active..."
echo "   System is learning to navigate autonomously"
echo "   Monitor via: ros2 topic echo /npu_exploration_status"
echo ""

# Wait for completion
wait $LAUNCH_PID

shutdown_handler