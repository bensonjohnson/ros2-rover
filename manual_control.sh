#!/bin/bash

# Simple manual control script for direct motor testing
# Starts just the motor driver and provides manual controls

set -e

# Configuration
WORKSPACE_DIR="/home/ubuntu/ros2-rover"
ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}

echo "=================================="
echo "    MANUAL MOTOR CONTROL"
echo "=================================="
echo "Starting hiwonder motor driver..."

cd "$WORKSPACE_DIR"

# Source ROS2
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# Export ROS domain
export ROS_DOMAIN_ID

# Start motor driver in background (corrected JGB3865-520R45-12 parameters)
ros2 run tractor_control hiwonder_motor_driver --ros-args \
    -p i2c_bus:=5 \
    -p motor_controller_address:=0x34 \
    -p use_pwm_control:=true \
    -p motor_type:=3 \
    -p encoder_ppr:=1980 \
    -p publish_rate:=100.0 \
    -p max_motor_speed:=100 \
    -p wheel_separation:=0.5 \
    -p wheel_radius:=0.15 \
    -p min_samples_for_estimation:=10 \
    -p max_history_minutes:=60 &

MOTOR_PID=$!
sleep 3

echo ""
echo "Motor driver started! Available topics:"
ros2 topic list | grep -E "(cmd_vel|battery|motor|odom)"

echo ""
echo "Battery status:"
ros2 topic echo /battery_voltage --once 2>/dev/null && \
ros2 topic echo /battery_percentage --once 2>/dev/null && \
ros2 topic echo /battery_runtime --once 2>/dev/null

echo ""
echo "Manual controls:"
echo "  w - Forward"
echo "  s - Backward" 
echo "  a - Turn left"
echo "  d - Turn right"
echo "  x - Stop"
echo "  q - Quit"
echo ""

# Function to send velocity commands
send_cmd() {
    local linear=$1
    local angular=$2
    ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
        "{linear: {x: $linear, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: $angular}}" > /dev/null
}

# Interactive control loop
while true; do
    echo -n "control> "
    read -n 1 -s key
    echo ""
    
    case $key in
        w)
            echo "Forward"
            send_cmd 0.2 0.0
            ;;
        s)
            echo "Backward"
            send_cmd -0.2 0.0
            ;;
        a)
            echo "Turn left"
            send_cmd 0.0 0.5
            ;;
        d)
            echo "Turn right"
            send_cmd 0.0 -0.5
            ;;
        x)
            echo "Stop"
            send_cmd 0.0 0.0
            ;;
        q)
            echo "Stopping motor driver..."
            send_cmd 0.0 0.0
            kill $MOTOR_PID 2>/dev/null || true
            echo "Done!"
            exit 0
            ;;
        *)
            echo "Unknown key: $key"
            ;;
    esac
done