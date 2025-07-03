#!/bin/bash

# ===================================================================
# Simple ROS2 Tractor Startup Script (No Navigation)
# Launches GPS, motor control, and Foxglove for manual control
# ===================================================================

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
WORKSPACE_DIR="/home/ubuntu/ros2-rover"
LOG_DIR="$WORKSPACE_DIR/logs"
ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
FOXGLOVE_PORT=${FOXGLOVE_PORT:-8765}

# Create log directory
mkdir -p "$LOG_DIR"

# Function to print colored messages
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

print_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ✗ $1${NC}"
}

# Function to check if a process is running
check_process() {
    if pgrep -f "$1" > /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to wait for topics to be available
wait_for_topic() {
    local topic=$1
    local timeout=${2:-15}
    local count=0
    
    print_status "Waiting for topic $topic..."
    while [ $count -lt $timeout ]; do
        if ros2 topic list | grep -q "^$topic$"; then
            print_success "Topic $topic is available"
            return 0
        fi
        sleep 1
        ((count++))
    done
    print_error "Timeout waiting for topic $topic"
    return 1
}

# Function to cleanup on exit
cleanup() {
    print_warning "Shutting down simple tractor system..."
    
    # Kill background processes
    pkill -f "ros2 launch" || true
    pkill -f "ros2 run" || true
    pkill -f "foxglove_bridge" || true
    pkill -f "hglrc_m100_5883" || true
    pkill -f "xbox_controller_teleop" || true
    
    print_success "System shutdown complete"
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Main startup function
main() {
    print_status "🚜 Starting Simple ROS2 Tractor System (Manual Control)"
    print_status "Workspace: $WORKSPACE_DIR"
    print_status "ROS Domain ID: $ROS_DOMAIN_ID"
    print_status "Foxglove Port: $FOXGLOVE_PORT"
    
    # Export ROS domain
    export ROS_DOMAIN_ID
    
    # Check if in correct directory
    if [ ! -f "$WORKSPACE_DIR/src/tractor_bringup/package.xml" ]; then
        print_error "Not in ROS2 workspace directory!"
        exit 1
    fi
    
    cd "$WORKSPACE_DIR"
    
    # Source ROS2
    print_status "Sourcing ROS2 environment..."
    source /opt/ros/jazzy/setup.bash
    
    # Source workspace
    if [ -f "$WORKSPACE_DIR/install/setup.bash" ]; then
        source "$WORKSPACE_DIR/install/setup.bash"
        print_success "ROS2 workspace sourced"
    else
        print_error "Workspace not built! Run 'colcon build' first"
        exit 1
    fi
    
    # Check hardware connections
    print_status "Checking hardware connections..."
    
    # Check GPS device
    if [ -e "/dev/ttyS6" ]; then
        print_success "GPS device found at /dev/ttyS6"
    else
        print_warning "GPS device not found at /dev/ttyS6 - GPS will not work"
    fi
    
    # Check I2C for motor controller and compass
    if [ -e "/dev/i2c-5" ]; then
        print_success "I2C-5 found for motor controller and compass"
    else
        print_warning "I2C-5 not found - motor control may not work"
    fi
    
    print_status "=== Starting Core Systems ==="
    
    # Start robot description (URDF) and state publisher
    print_status "Launching robot description and state publisher..."
    ros2 launch tractor_bringup robot_description.launch.py > "$LOG_DIR/robot_description.log" 2>&1 &
    ROBOT_DESC_PID=$!
    sleep 2
    
    # Verify robot description is running
    if ! wait_for_topic "/robot_description" 5; then
        print_error "Robot description failed to start"
        exit 1
    fi
    
    print_success "Robot description started successfully"
    
    # Start GPS and compass (if available)
    if [ -e "/dev/ttyS6" ]; then
        print_status "Launching GPS and compass sensors..."
        ros2 launch tractor_sensors hglrc_m100_5883.launch.py > "$LOG_DIR/gps_compass.log" 2>&1 &
        GPS_PID=$!
        sleep 3
        
        # Verify GPS topics
        if ! wait_for_topic "/hglrc_gps/fix" 10; then
            print_warning "GPS topics not available - continuing without GPS"
        fi
    else
        print_warning "Skipping GPS/compass - device not found"
    fi
    
    # Start motor driver with battery monitoring (using corrected JGB3865-520R45-12 parameters)
    print_status "Launching motor driver with battery monitoring..."
    ros2 run tractor_control hiwonder_motor_driver --ros-args \
        -p i2c_bus:=5 \
        -p motor_controller_address:=0x34 \
        -p use_pwm_control:=false \
        -p motor_type:=3 \
        -p encoder_ppr:=135 \
        -p publish_rate:=5.0 \
        -p max_motor_speed:=50 \
        -p wheel_separation:=0.5 \
        -p wheel_radius:=0.15 \
        -p min_samples_for_estimation:=10 \
        -p max_history_minutes:=60 \
        > "$LOG_DIR/motor_control.log" 2>&1 &
    MOTOR_PID=$!
    sleep 3
    
    # Verify motor topics
    if ! wait_for_topic "/cmd_vel" 10; then
        print_error "Motor driver failed to start"
        exit 1
    fi
    
    print_success "Motor driver started successfully"
    
    # Start Xbox controller teleop (direct control)
    print_status "Launching Xbox controller teleop (direct control)..."
    export SDL_JOYSTICK_DEVICE=/dev/input/js0
    ros2 run tractor_control xbox_controller_teleop --ros-args \
        -p max_linear_speed:=4.0 \
        -p max_angular_speed:=2.0 \
        -p deadzone:=0.15 \
        -p tank_drive_mode:=true \
        -p controller_index:=0 \
        -p use_feedback_control:=false \
        > "$LOG_DIR/xbox_controller.log" 2>&1 &
    XBOX_PID=$!
    sleep 2
    
    # Check if Xbox controller started (don't fail if no controller connected)
    if check_process "xbox_controller_teleop"; then
        print_success "Xbox controller teleop started"
    else
        print_warning "Xbox controller teleop failed to start (controller may not be connected)"
    fi
    
    print_status "=== Starting Foxglove Bridge ==="
    
    # Start Foxglove bridge
    print_status "Launching Foxglove bridge on port $FOXGLOVE_PORT..."
    ros2 run foxglove_bridge foxglove_bridge --ros-args \
        -p port:=$FOXGLOVE_PORT \
        -p address:=0.0.0.0 \
        -p tls:=false \
        > "$LOG_DIR/foxglove_bridge.log" 2>&1 &
    FOXGLOVE_PID=$!
    sleep 3
    
    # Check if Foxglove bridge is running
    if check_process "foxglove_bridge"; then
        print_success "Foxglove bridge started on port $FOXGLOVE_PORT"
    else
        print_error "Failed to start Foxglove bridge"
        exit 1
    fi
    
    print_status "=== System Status Summary ==="
    
    # Display available topics
    print_status "Available ROS2 topics:"
    ros2 topic list | sort | while read topic; do
        echo "  - $topic"
    done
    
    # Show battery status
    echo ""
    print_status "Battery Status:"
    if ros2 topic echo /battery_voltage --once 2>/dev/null; then
        echo -n "  Percentage: "; ros2 topic echo /battery_percentage --once 2>/dev/null | grep data | cut -d' ' -f2
        echo -n "  Runtime: "; ros2 topic echo /battery_runtime --once 2>/dev/null | grep data | cut -d' ' -f2; echo " hours"
    else
        print_warning "Battery monitoring not available"
    fi
    
    # Display system information
    echo ""
    print_success "🚜 Simple Tractor System Successfully Started!"
    echo ""
    echo -e "${GREEN}Core Systems:${NC}"
    echo "  - Motor Control: /cmd_vel (hiwonder driver with 100Hz encoder feedback)"
    echo "  - Direct Control: Xbox controller → /cmd_vel (open-loop)"
    echo "  - Battery Monitoring: /battery_voltage, /battery_percentage, /battery_runtime"
    echo "  - Odometry: /odom (from motor encoders at 100Hz)"
    if [ -e "/dev/ttyS6" ]; then
        echo "  - GPS/Compass: /hglrc_gps/fix, /hglrc_gps/imu"
    fi
    echo ""
    echo -e "${GREEN}Manual Control (Direct):${NC}"
    echo "  - Xbox Controller: Connect via Bluetooth and use joysticks to drive"
    echo "    - Tank Drive Mode: Left stick = left motor, Right stick = right motor"
    echo "    - Y button: Emergency stop toggle, A button: Resume"
    echo "    - Direct PWM control (no encoder feedback)"
    echo "  - Use Foxglove Studio Teleop panel with /cmd_vel topic"
    echo "  - Or run: ./manual_control.sh"
    echo ""
    echo -e "${GREEN}Foxglove Studio:${NC}"
    echo "  - WebSocket URL: ws://$(hostname -I | awk '{print $1}'):$FOXGLOVE_PORT"
    echo "  - Dashboard Layout: Import foxglove_tractor_dashboard.json for complete setup"
    echo "  - Manual Teleop: Topic = /cmd_vel, Type = geometry_msgs/Twist"
    echo ""
    echo -e "${GREEN}Process IDs:${NC}"
    echo "  - Robot Description: $ROBOT_DESC_PID"
    if [ ! -z ${GPS_PID+x} ]; then
        echo "  - GPS/Compass: $GPS_PID"
    fi
    echo "  - Motor Driver: $MOTOR_PID"
    if [ ! -z ${XBOX_PID+x} ]; then
        echo "  - Xbox Controller: $XBOX_PID"
    fi
    echo "  - Foxglove Bridge: $FOXGLOVE_PID"
    echo ""
    echo -e "${YELLOW}Logs available in: $LOG_DIR${NC}"
    echo ""
    print_status "Press Ctrl+C to shutdown the system"
    
    # Keep script running and monitor processes
    while true; do
        # Check if critical processes are still running
        if ! kill -0 $ROBOT_DESC_PID 2>/dev/null; then
            print_error "Robot description stopped unexpectedly"
            break
        fi
        
        if ! check_process "foxglove_bridge"; then
            print_error "Foxglove bridge stopped unexpectedly"
            break
        fi
        
        if ! kill -0 $MOTOR_PID 2>/dev/null; then
            print_error "Motor driver stopped unexpectedly"
            break
        fi
        
        # Xbox controller is optional, just warn if it stops
        if [ ! -z ${XBOX_PID+x} ] && ! kill -0 $XBOX_PID 2>/dev/null; then
            print_warning "Xbox controller teleop stopped (controller may have been disconnected)"
            unset XBOX_PID
        fi
        
        sleep 5
    done
}

# Display usage information
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Simple ROS2 Tractor System (Manual Control)"
    echo "Starts: Motor control, battery monitoring, GPS (optional), Xbox controller, and Foxglove"
    echo ""
    echo "Options:"
    echo "  -h, --help           Show this help message"
    echo "  -p, --port PORT      Set Foxglove bridge port (default: 8765)"
    echo "  -d, --domain ID      Set ROS domain ID (default: 42)"
    echo ""
    echo "Example:"
    echo "  $0 --port 8888 --domain 10"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -p|--port)
            FOXGLOVE_PORT="$2"
            shift 2
            ;;
        -d|--domain)
            ROS_DOMAIN_ID="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Run main function
main "$@"