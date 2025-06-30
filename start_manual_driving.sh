#!/bin/bash

# ===================================================================
# ROS2 Tractor Manual Driving Script
# Xbox controller + sensors + Foxglove monitoring (no navigation)
# Perfect for testing and playing around with the robot!
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
CONTROLLER_DEVICE=${CONTROLLER_DEVICE:-"/dev/input/js0"}

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
    local timeout=${2:-30}
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
    print_warning "Topic $topic not available (continuing anyway)"
    return 1
}

# Function to check controller connection
check_controller() {
    if [ -e "$CONTROLLER_DEVICE" ]; then
        print_success "Xbox controller found at $CONTROLLER_DEVICE"
        return 0
    else
        print_warning "Xbox controller not found at $CONTROLLER_DEVICE"
        echo "Available input devices:"
        ls -la /dev/input/js* 2>/dev/null || echo "  No joystick devices found"
        echo ""
        echo "Please connect your Xbox controller and run again, or specify device with:"
        echo "  CONTROLLER_DEVICE=/dev/input/js1 $0"
        return 1
    fi
}

# Function to cleanup on exit
cleanup() {
    print_warning "Shutting down manual driving system..."
    
    # Kill background processes
    pkill -f "ros2 launch" || true
    pkill -f "foxglove_bridge" || true
    pkill -f "xbox_controller_teleop" || true
    
    print_success "Manual driving system shutdown complete"
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Main function
main() {
    print_status "🎮 Starting ROS2 Tractor Manual Driving System"
    print_status "Workspace: $WORKSPACE_DIR"
    print_status "ROS Domain ID: $ROS_DOMAIN_ID"
    print_status "Foxglove Port: $FOXGLOVE_PORT"
    print_status "Controller Device: $CONTROLLER_DEVICE"
    
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
    
    # Check Xbox controller
    print_status "Checking Xbox controller..."
    if ! check_controller; then
        echo ""
        echo -e "${YELLOW}You can still start the system without a controller for sensor monitoring.${NC}"
        echo -e "${YELLOW}Continue anyway? (y/N)${NC}"
        read -n 1 continue_choice
        echo ""
        if [[ ! $continue_choice =~ ^[Yy]$ ]]; then
            exit 1
        fi
        CONTROLLER_AVAILABLE=false
    else
        CONTROLLER_AVAILABLE=true
    fi
    
    # Check hardware connections
    print_status "Checking hardware connections..."
    
    # Check GPS device
    if [ -e "/dev/ttyS6" ]; then
        print_success "GPS device found at /dev/ttyS6"
        GPS_AVAILABLE=true
    else
        print_warning "GPS device not found at /dev/ttyS6"
        GPS_AVAILABLE=false
    fi
    
    # Check I2C for compass
    if [ -e "/dev/i2c-5" ]; then
        print_success "I2C-5 found for compass"
        I2C_AVAILABLE=true
    else
        print_warning "I2C-5 not found - compass may not work"
        I2C_AVAILABLE=false
    fi
    
    # Check for motor driver GPIO access
    if [ -w "/sys/class/gpio" ] || [ -e "/dev/gpiochip0" ]; then
        print_success "GPIO access available for motor control"
        GPIO_AVAILABLE=true
    else
        print_warning "GPIO access limited - motor control may not work"
        GPIO_AVAILABLE=false
    fi
    
    print_status "=== Starting Manual Driving System ==="
    
    # Start robot description and basic transforms
    print_status "Launching robot description..."
    ros2 launch tractor_bringup robot_description.launch.py > "$LOG_DIR/manual_description.log" 2>&1 &
    DESCRIPTION_PID=$!
    sleep 3
    
    # Start sensors (GPS, compass, encoders)
    print_status "Launching sensors..."
    ros2 launch tractor_bringup sensors.launch.py > "$LOG_DIR/manual_sensors.log" 2>&1 &
    SENSORS_PID=$!
    sleep 5
    
    # Start robot localization (GPS fusion) if GPS available
    if [ "$GPS_AVAILABLE" = true ]; then
        print_status "Launching robot localization (GPS + odometry fusion)..."
        ros2 launch tractor_bringup robot_localization.launch.py > "$LOG_DIR/manual_localization.log" 2>&1 &
        LOCALIZATION_PID=$!
        sleep 5
    else
        print_warning "Skipping robot localization (no GPS)"
        LOCALIZATION_PID=""
    fi
    
    # Start motor control
    if [ "$GPIO_AVAILABLE" = true ]; then
        print_status "Launching motor control..."
        ros2 launch tractor_bringup control.launch.py > "$LOG_DIR/manual_control.log" 2>&1 &
        CONTROL_PID=$!
        sleep 3
    else
        print_warning "Skipping motor control (no GPIO access)"
        CONTROL_PID=""
    fi
    
    # Start Xbox controller teleop if controller available
    if [ "$CONTROLLER_AVAILABLE" = true ]; then
        print_status "Launching Xbox controller teleop..."
        ros2 run tractor_control xbox_controller_teleop --ros-args \
            -p device:=$CONTROLLER_DEVICE \
            -p max_linear_speed:=1.0 \
            -p max_angular_speed:=2.0 \
            > "$LOG_DIR/manual_xbox.log" 2>&1 &
        XBOX_PID=$!
        sleep 3
        
        if check_process "xbox_controller_teleop"; then
            print_success "Xbox controller teleop started"
        else
            print_warning "Xbox controller teleop failed to start - check controller connection"
        fi
    else
        print_warning "Skipping Xbox controller (not available)"
        XBOX_PID=""
    fi
    
    # Start vision processing if available
    print_status "Launching vision processing..."
    ros2 launch tractor_bringup vision.launch.py > "$LOG_DIR/manual_vision.log" 2>&1 &
    VISION_PID=$!
    sleep 3
    
    print_status "=== Starting Foxglove Bridge ==="
    
    # Start Foxglove bridge
    print_status "Launching Foxglove bridge on port $FOXGLOVE_PORT..."
    ros2 run foxglove_bridge foxglove_bridge --ros-args \
        -p port:=$FOXGLOVE_PORT \
        -p address:=0.0.0.0 \
        -p tls:=false \
        > "$LOG_DIR/manual_foxglove.log" 2>&1 &
    FOXGLOVE_PID=$!
    sleep 3
    
    # Check if Foxglove bridge is running
    if check_process "foxglove_bridge"; then
        print_success "Foxglove bridge started on port $FOXGLOVE_PORT"
    else
        print_error "Failed to start Foxglove bridge"
        exit 1
    fi
    
    # Wait for critical topics
    print_status "Verifying system startup..."
    
    # Check for transforms
    if ! wait_for_topic "/tf" 10; then
        print_warning "Transform system not fully active"
    fi
    
    # Check for GPS topics if available
    if [ "$GPS_AVAILABLE" = true ]; then
        wait_for_topic "/hglrc_gps/fix" 10
        wait_for_topic "/odometry/filtered" 10
    fi
    
    # Check for motor command topic
    wait_for_topic "/cmd_vel" 5
    
    print_status "=== Manual Driving System Ready ==="
    
    # Display system information
    echo ""
    print_success "🎮 Manual Driving System Successfully Started!"
    echo ""
    echo -e "${GREEN}Foxglove Studio Connection:${NC}"
    echo "  - WebSocket URL: ws://$(hostname -I | awk '{print $1}'):$FOXGLOVE_PORT"
    echo "  - Load config: foxglove_manual_driving_config.json"
    echo ""
    echo -e "${GREEN}Available Sensors:${NC}"
    if [ "$GPS_AVAILABLE" = true ]; then
        echo "  ✓ GPS Fix: /hglrc_gps/fix"
        echo "  ✓ Compass: /hglrc_gps/imu"
        echo "  ✓ Filtered Odometry: /odometry/filtered"
    else
        echo "  ✗ GPS (not connected)"
    fi
    echo "  ✓ Wheel Encoders: /wheel_odom"
    echo "  ✓ Robot Description: /robot_description"
    echo "  ✓ Transforms: /tf, /tf_static"
    echo ""
    echo -e "${GREEN}Control:${NC}"
    if [ "$CONTROLLER_AVAILABLE" = true ]; then
        echo "  ✓ Xbox Controller: Connected ($CONTROLLER_DEVICE)"
        echo "    - Left stick: Forward/Backward"
        echo "    - Right stick: Turn Left/Right"
        echo "    - Triggers: Speed control"
    else
        echo "  ✗ Xbox Controller (not connected)"
        echo "  ℹ You can still publish to /cmd_vel manually"
    fi
    if [ "$GPIO_AVAILABLE" = true ]; then
        echo "  ✓ Motor Control: /cmd_vel → motors"
    else
        echo "  ✗ Motor Control (no GPIO access)"
    fi
    echo ""
    echo -e "${GREEN}Process Status:${NC}"
    echo "  - Robot Description: $DESCRIPTION_PID"
    echo "  - Sensors: $SENSORS_PID"
    [ -n "$LOCALIZATION_PID" ] && echo "  - Localization: $LOCALIZATION_PID"
    [ -n "$CONTROL_PID" ] && echo "  - Control: $CONTROL_PID"
    [ -n "$XBOX_PID" ] && echo "  - Xbox Controller: $XBOX_PID"
    echo "  - Vision: $VISION_PID"
    echo "  - Foxglove Bridge: $FOXGLOVE_PID"
    echo ""
    echo -e "${YELLOW}Usage:${NC}"
    if [ "$CONTROLLER_AVAILABLE" = true ]; then
        echo "  🎮 Drive around with your Xbox controller!"
        echo "  📊 Monitor sensors in real-time via Foxglove"
        echo "  🔧 Test all systems before autonomous operation"
    else
        echo "  📊 Monitor sensors in real-time via Foxglove"
        echo "  🔧 Connect Xbox controller for manual driving"
        echo "  💻 Publish to /cmd_vel for manual control"
    fi
    echo ""
    echo -e "${YELLOW}Logs available in: $LOG_DIR${NC}"
    echo ""
    print_status "Press Ctrl+C to shutdown the system"
    
    # Keep script running and monitor processes
    while true; do
        # Check if critical processes are still running
        if ! check_process "foxglove_bridge"; then
            print_error "Foxglove bridge stopped unexpectedly"
            break
        fi
        
        if ! kill -0 $DESCRIPTION_PID 2>/dev/null; then
            print_error "Robot description stopped unexpectedly"
            break
        fi
        
        # Optional: Check controller connection periodically
        if [ "$CONTROLLER_AVAILABLE" = true ] && [ -n "$XBOX_PID" ]; then
            if ! kill -0 $XBOX_PID 2>/dev/null; then
                print_warning "Xbox controller disconnected - you can reconnect it"
                CONTROLLER_AVAILABLE=false
                XBOX_PID=""
            fi
        fi
        
        sleep 10
    done
}

# Display usage information
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "ROS2 Tractor Manual Driving System"
    echo "Xbox controller + sensors + Foxglove monitoring (no nav/SLAM)"
    echo ""
    echo "Options:"
    echo "  -h, --help           Show this help message"
    echo "  -p, --port PORT      Set Foxglove bridge port (default: 8765)"
    echo "  -d, --domain ID      Set ROS domain ID (default: 42)"
    echo "  -c, --controller DEV Set controller device (default: /dev/input/js0)"
    echo ""
    echo "Example:"
    echo "  $0                           # Basic startup"
    echo "  $0 --port 8888              # Custom Foxglove port"
    echo "  $0 --controller /dev/input/js1  # Use different controller"
    echo ""
    echo "Environment variables:"
    echo "  CONTROLLER_DEVICE    Xbox controller device path"
    echo "  FOXGLOVE_PORT        Foxglove WebSocket port"
    echo "  ROS_DOMAIN_ID        ROS domain ID"
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
        -c|--controller)
            CONTROLLER_DEVICE="$2"
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