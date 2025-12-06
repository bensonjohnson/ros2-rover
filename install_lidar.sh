#!/bin/bash
set -e

echo "=================================================="
echo "STL-19p LiDAR Installer for ROS 2 Jazzy"
echo "=================================================="

WORKSPACE_ROOT=$(dirname "$(readlink -f "$0")")
SRC_DIR="$WORKSPACE_ROOT/src"

cd "$WORKSPACE_ROOT"

# 1. Clone Repositories
echo "Checking repositories..."

# LDROBOT LiDAR Driver
if [ ! -d "$SRC_DIR/ldlidar_stl_ros2" ]; then
    echo "⬇ Cloning ldlidar_stl_ros2..."
    git clone https://github.com/ldrobot-sensor/ldlidar_stl_ros2.git "$SRC_DIR/ldlidar_stl_ros2"
else
    echo "✓ ldlidar_stl_ros2 already exists"
fi

# RF2O Laser Odometry
if [ ! -d "$SRC_DIR/rf2o_laser_odometry" ]; then
    echo "⬇ Cloning rf2o_laser_odometry..."
    git clone -b ros2 https://github.com/MAPIRlab/rf2o_laser_odometry.git "$SRC_DIR/rf2o_laser_odometry"
else
    echo "✓ rf2o_laser_odometry already exists"
fi

# 2. Install Dependencies
echo "Installing dependencies..."
if command -v rosdep &> /dev/null; then
    rosdep update
    rosdep install --from-paths src --ignore-src -r -y
else
    echo "⚠ rosdep not found, skipping dependency check via rosdep."
fi

# 3. Setup Udev Rules (USB Permissions)
echo "Setting up USB permissions..."
# Standard rule for USB serial devices (allows non-root access)
# This is valid for CP210x (used by LD19)
RULE_FILE="/etc/udev/rules.d/99-lidar-usb.rules"
if [ ! -f "$RULE_FILE" ]; then
    echo "Creating udev rule for LiDAR..."
    # Match generic USB serial or specific vendor if known. 
    # LD19 often uses CP2102 (Silicon Labs). 
    # For now, broad permission for ttyUSB* is easiest for user dev environment.
    echo 'KERNEL=="ttyUSB*", MODE="0666"' | sudo tee $RULE_FILE > /dev/null
    
    echo "Reloading udev rules..."
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    echo "✓ Udev rules configured"
else
    echo "✓ Udev rules already exist"
fi

# 4. Build
echo "Building LiDAR packages..."
colcon build --packages-select ldlidar_stl_ros2 rf2o_laser_odometry tractor_sensors tractor_bringup --cmake-args -DCMAKE_BUILD_TYPE=Release --symlink-install

echo "=================================================="
echo "✅ LiDAR Install Complete"
echo "To explore: ros2 launch tractor_sensors stl19p_lidar.launch.py"
echo "=================================================="
