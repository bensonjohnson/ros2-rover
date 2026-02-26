#!/bin/bash
# Motor test script - brings up motor driver and tests both tracks
# Usage: ./test_motors.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARAMS="$SCRIPT_DIR/src/tractor_bringup/config/hiwonder_motor_params.yaml"

echo "=== Motor Test Script ==="
echo ""

# Source ROS2
source /opt/ros/humble/setup.bash 2>/dev/null || source /opt/ros/jazzy/setup.bash 2>/dev/null
if [ -f "$SCRIPT_DIR/install/setup.bash" ]; then
    source "$SCRIPT_DIR/install/setup.bash"
fi

# Kill any existing motor driver
echo "[1/8] Cleaning up existing nodes..."
pkill -f hiwonder_motor_driver 2>/dev/null || true
sleep 1

# Launch motor driver in background
echo "[2/8] Starting motor driver..."
ros2 run tractor_control hiwonder_motor_driver --ros-args --params-file "$PARAMS" -p publish_tf:=false &
DRIVER_PID=$!
sleep 3

# Check it's running
if ! kill -0 $DRIVER_PID 2>/dev/null; then
    echo "ERROR: Motor driver failed to start"
    exit 1
fi
echo "       Motor driver running (PID $DRIVER_PID)"

# Helper: publish cmd_vel at 10Hz for a duration, then stop
send_cmd() {
    local desc="$1" lx="$2" az="$3" duration="$4"
    echo "       -> $desc (linear=$lx, angular=$az) for ${duration}s"
    # Publish at 10Hz to keep watchdog happy
    timeout "$duration" ros2 topic pub --rate 10 /cmd_vel geometry_msgs/msg/Twist \
        "{linear: {x: $lx}, angular: {z: $az}}" > /dev/null 2>&1 || true
    # Stop
    ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
        "{linear: {x: 0.0}, angular: {z: 0.0}}" > /dev/null 2>&1
    sleep 1
}

echo ""
echo "[3/8] Test: Both tracks FORWARD (linear=0.5)"
send_cmd "both forward" 0.5 0.0 3

echo "[4/8] Test: Both tracks REVERSE (linear=-0.5)"
send_cmd "both reverse" -0.5 0.0 3

echo "[5/8] Test: LEFT track only (angular=-2.0 -> L fwd, R rev)"
send_cmd "left fwd, right rev" 0.0 -2.0 3

echo "[6/8] Test: RIGHT track only (angular=2.0 -> R fwd, L rev)"
send_cmd "right fwd, left rev" 0.0 2.0 3

echo "[7/8] Test: Zero turn CW (angular=-4.0 -> strong spin)"
send_cmd "zero turn CW" 0.0 -4.0 3

echo "[8/8] Test: Zero turn CCW (angular=4.0 -> strong spin)"
send_cmd "zero turn CCW" 0.0 4.0 3

# Cleanup
echo ""
echo "=== Tests complete. Stopping motor driver ==="
kill $DRIVER_PID 2>/dev/null || true
wait $DRIVER_PID 2>/dev/null || true
echo "Done."
