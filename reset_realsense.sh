#!/bin/bash

echo "Resetting RealSense D435i..."

# Kill any existing RealSense processes
pkill -f realsense
pkill -f rs-

# Reset USB device power
echo "Resetting USB power..."
if [ -d "/sys/bus/usb/devices/8-1" ]; then
    echo "auto" | sudo tee /sys/bus/usb/devices/8-1/power/control > /dev/null
    sleep 1
    echo "on" | sudo tee /sys/bus/usb/devices/8-1/power/control > /dev/null
fi

# Wait for device to reinitialize
echo "Waiting for device to reinitialize..."
sleep 3

# Test connection
echo "Testing connection..."
timeout 5s rs-enumerate-devices > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ RealSense device reset successfully"
else
    echo "✗ RealSense device still not responding"
    echo "Try unplugging and reconnecting the USB-C cable"
fi