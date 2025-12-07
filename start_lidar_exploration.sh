#!/bin/bash
# Start LiDAR-Based Frontier Exploration
#
# This script launches autonomous frontier exploration using:
# - LD19 LiDAR for mapping and odometry (rf2o)
# - RealSense D435i for 3D obstacle detection
# - LSM9DS1 IMU for sensor fusion
# - robot_localization EKF (rf2o + IMU)
# - SLAM Toolbox for mapping
# - Nav2 for navigation
# - Wavefront Frontier Explorer for autonomous exploration

set -e

cd "$(dirname "$0")"

echo "======================================="
echo " LiDAR Frontier Exploration System    "
echo "======================================="
echo ""
echo "Sensors:"
echo "  - LD19 LiDAR (mapping + odometry)"
echo "  - RealSense D435i (3D obstacles)"
echo "  - LSM9DS1 IMU (sensor fusion)"
echo ""
echo "Starting in 3 seconds..."
sleep 3

source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 launch tractor_bringup lidar_frontier_exploration.launch.py
