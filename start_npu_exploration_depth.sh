#!/bin/bash

# Minimal NPU Depth Image Exploration Script
# Clean architecture: Hardware + AI only, no SLAM/Nav2 complexity

echo "=================================================="
echo "ROS2 Tractor - NPU Depth Image Exploration"
echo "=================================================="
echo "This minimal system uses:"
echo "  ✓ Hiwonder motor control"
echo "  ✓ RealSense D435i (depth images only, IMU disabled, USB optimized)"
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
# Check if packages need to be built
if [ ! -d "build/tractor_bringup" ] || [ ! -d "build/tractor_control" ] || [ ! -d "install/tractor_bringup" ] || [ ! -d "install/tractor_control" ]; then
    echo "Building packages..."
    colcon build --packages-select tractor_bringup tractor_control --cmake-args -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "❌ Build failed! Please check for compilation errors."
        exit 1
    fi
    echo "✓ Workspace built successfully"
else
    echo "✓ Packages already built"
fi

# Source environment
echo "Sourcing ROS2 environment..."
source /opt/ros/jazzy/setup.bash
source install/setup.bash
echo "✓ ROS2 environment sourced"

# USB power management for RealSense
echo "Configuring USB power management..."
USB_DEVICE_PATH=""
# Find D435i device by Product ID (0B3A)
for device in /sys/bus/usb/devices/*/idProduct; do
    if [ -f "$device" ] && [ "$(cat $device 2>/dev/null)" = "0b3a" ]; then
        USB_DEVICE_PATH=$(dirname $device)
        echo "✓ Found D435i at USB path: $USB_DEVICE_PATH"
        break
    fi
done

# Fallback to common paths if Product ID detection fails
if [ -z "$USB_DEVICE_PATH" ]; then
    for path in "/sys/bus/usb/devices/8-1" "/sys/bus/usb/devices/2-1" "/sys/bus/usb/devices/1-1"; do
        if [ -d "$path" ]; then
            USB_DEVICE_PATH="$path"
            echo "✓ Using fallback USB path: $USB_DEVICE_PATH"
            break
        fi
    done
fi

if [ -n "$USB_DEVICE_PATH" ]; then
    # Disable autosuspend for the device
    echo "on" | sudo tee $USB_DEVICE_PATH/power/control > /dev/null 2>&1
    echo "-1" | sudo tee $USB_DEVICE_PATH/power/autosuspend > /dev/null 2>&1
    echo "✓ USB power management configured for $USB_DEVICE_PATH"
else
    echo "⚠ USB device path not found, continuing anyway"
fi

# Check RealSense D435i specifically
echo "Checking RealSense D435i..."
if command -v rs-enumerate-devices &> /dev/null; then
    timeout 5s rs-enumerate-devices | grep -q "D435I"
    if [ $? -eq 0 ]; then
        echo "✓ RealSense D435i detected"
        # Reset the USB device to clear any error states
        if [ -n "$USB_DEVICE_PATH" ] && [ -w "$USB_DEVICE_PATH" ]; then
            echo "Resetting USB device..."
            echo "0" | sudo tee $USB_DEVICE_PATH/authorized > /dev/null 2>&1
            sleep 1
            echo "1" | sudo tee $USB_DEVICE_PATH/authorized > /dev/null 2>&1
            echo "✓ USB device reset"
        fi
    else
        echo "⚠ RealSense D435i not detected - will attempt to continue"
    fi
else
    echo "⚠ rs-enumerate-devices command not found - skipping RealSense check"
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
echo "  IMU Status: Disabled (to reduce USB errors)"
echo "  USB Mode: Optimized for stability"
echo ""

# Countdown
echo "Starting NPU depth exploration in:"
for i in {3..1}; do
    echo "  $i..."
    sleep 1
done
echo "  🚀 LAUNCHING!"
echo ""

# Launch the minimal system
echo "Launching NPU depth exploration system..."
echo "Press Ctrl+C to stop safely"
echo ""

ros2 launch tractor_bringup npu_exploration_depth.launch.py \
    max_speed:=${MAX_SPEED} \
    exploration_time:=${EXPLORATION_TIME} \
    safety_distance:=${SAFETY_DISTANCE} \
    use_sim_time:=false &

LAUNCH_PID=$!

# Simple shutdown handler
shutdown_handler() {
    echo ""
    echo "🛑 Stopping NPU depth exploration..."
    if ps -p $LAUNCH_PID > /dev/null; then
        kill $LAUNCH_PID 2>/dev/null
        sleep 2
        # Force kill if still running
        if ps -p $LAUNCH_PID > /dev/null; then
            kill -9 $LAUNCH_PID 2>/dev/null
        fi
    fi
    echo "✅ NPU depth exploration stopped safely"
    echo "=================================================="
    exit 0
}

trap shutdown_handler SIGINT SIGTERM

echo "🤖 NPU depth exploration active..."
echo "   System is learning to navigate autonomously"
echo "   Monitor via: ros2 topic echo /npu_exploration_status"
echo ""

# Wait for completion
wait $LAUNCH_PID

shutdown_handler
