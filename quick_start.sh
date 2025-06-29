#!/bin/bash

# Quick start script for ROS2 Tractor System
# This is a simplified launcher that calls the main startup script

echo "🚜 Quick Starting ROS2 Tractor System..."
echo "   - GPS & Compass on /dev/ttyS6 & I2C-5"
echo "   - Robot Localization with GPS fusion"  
echo "   - Foxglove Bridge for visualization"
echo ""

exec "$(dirname "$0")/start_tractor_system.sh" "$@"