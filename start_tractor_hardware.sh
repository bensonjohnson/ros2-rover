#!/bin/bash

# Tractor Hardware Startup Script with Xbox Controller
# This script starts the tractor with Xbox controller for testing
# Long-term goal is autonomous operation

echo "=================================================="
echo "        TRACTOR HARDWARE STARTUP (TESTING)"
echo "=================================================="
echo "Starting tractor with Xbox controller for testing..."
echo "Long-term goal: Autonomous operation"
echo ""

# Check if we're in the correct workspace
if [ ! -f "install/setup.bash" ]; then
    echo "❌ Error: Not in tractor workspace or workspace not built"
    echo "Please run from /home/benson/tractor_ws and ensure 'colcon build' has been run"
    exit 1
fi

# Source the workspace
echo "🔧 Sourcing ROS2 workspace..."
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# Check if Xbox controller is connected
echo "🎮 Checking for Xbox controller..."
if ! lsusb | grep -i xbox > /dev/null && ! ls /dev/input/js* > /dev/null 2>&1; then
    echo "⚠️  Warning: No Xbox controller detected"
    echo "   Connect Xbox controller and press any button to activate"
    echo "   Continuing anyway - you can connect controller later"
    echo ""
fi

# Check I2C connection
echo "🔌 Checking I2C connection to motor controller..."
if command -v i2cdetect > /dev/null; then
    # Check i2c-5 first (where motor controller was found)
    if i2cdetect -y 5 2>/dev/null | grep -q "34"; then
        echo "✅ Motor controller detected at address 0x34 on i2c-5"
        I2C_BUS=5
    elif i2cdetect -y 6 2>/dev/null | grep -q "34"; then
        echo "✅ Motor controller detected at address 0x34 on i2c-6"
        I2C_BUS=6
    elif i2cdetect -y 7 2>/dev/null | grep -q "34"; then
        echo "✅ Motor controller detected at address 0x34 on i2c-7"
        I2C_BUS=7
    elif i2cdetect -y 0 2>/dev/null | grep -q "34"; then
        echo "✅ Motor controller detected at address 0x34 on i2c-0"
        I2C_BUS=0
    else
        echo "⚠️  Warning: Motor controller not detected at 0x34"
        echo "   Make sure the motor controller is connected and powered"
        echo "   Trying default i2c-5"
        I2C_BUS=5
    fi
else
    echo "⚠️  i2c-tools not found - install with: sudo apt install i2c-tools"
    echo "   Using default i2c-5"
    I2C_BUS=5
fi

echo ""
echo "🚜 Starting tractor hardware control system..."
echo ""

# Function to check battery voltage
check_battery_voltage() {
    echo "🔋 Checking battery voltage..."
    timeout 5 ros2 topic echo /battery_voltage --once 2>/dev/null | grep "data:" | awk '{print "   Battery: " $2 "V"}' || echo "   Battery: Unable to read (will show after startup)"
}

# Function to monitor system in background
monitor_system() {
    sleep 10  # Wait for system to start
    while true; do
        echo ""
        echo "=== SYSTEM STATUS $(date +%H:%M:%S) ==="
        check_battery_voltage
        echo ""
        sleep 30  # Update every 30 seconds
    done
}

# Start monitoring in background
monitor_system &
MONITOR_PID=$!
echo "=== XBOX CONTROLLER CONTROLS ==="
echo "Tank Drive Mode (Default):"
echo "  • Left stick Y:  Left motor speed"
echo "  • Right stick Y: Right motor speed"
echo "  • Y button:      Emergency stop toggle"
echo "  • A button:      Resume from emergency stop"
echo ""
echo "=== PUBLISHED TOPICS ==="
echo "  • /cmd_vel         - Robot movement commands"
echo "  • /motor_speeds    - Current motor speeds"
echo "  • /battery_voltage - Battery voltage (V)"
echo "  • /joint_states    - Wheel joint states"
echo "  • /emergency_stop  - Emergency stop status"
echo ""
echo "=== MONITORING COMMANDS ==="
echo "  • Watch motor speeds:   ros2 topic echo /motor_speeds"
echo "  • Watch battery:        ros2 topic echo /battery_voltage"
echo "  • Watch joint states:   ros2 topic echo /joint_states"
echo "  • Manual control:       ros2 topic pub /cmd_vel geometry_msgs/Twist ..."
echo ""
echo "Press Ctrl+C to stop all nodes"
echo "=================================================="
echo ""

# Start the hardware control system
ros2 launch tractor_bringup hiwonder_control.launch.py i2c_bus:=${I2C_BUS}

# Cleanup monitoring when main process stops
kill $MONITOR_PID 2>/dev/null

echo ""
echo "🛑 Tractor hardware system stopped"
echo "All motors should be stopped automatically"