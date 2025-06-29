#!/bin/bash

# ===================================================================
# ROS2 Tractor System Startup Script with Foxglove Bridge
# Initializes GPS, compass, localization, navigation, and visualization
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
ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}  # Set unique domain ID
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
    print_error "Timeout waiting for topic $topic"
    return 1
}

# Function to cleanup on exit
cleanup() {
    print_warning "Shutting down tractor system..."
    
    # Kill background processes
    pkill -f "ros2 launch" || true
    pkill -f "foxglove_bridge" || true
    pkill -f "hglrc_m100_5883" || true
    
    print_success "System shutdown complete"
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Main startup function
main() {
    print_status "Starting ROS2 Tractor System with GPS and Foxglove Bridge"
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
        print_error "GPS device not found at /dev/ttyS6"
        exit 1
    fi
    
    # Check I2C for compass
    if [ -e "/dev/i2c-5" ]; then
        print_success "I2C-5 found for compass"
    else
        print_warning "I2C-5 not found - compass may not work"
    fi
    
    print_status "=== Starting GPS and Compass System ==="
    
    # Start GPS and compass
    print_status "Launching GPS and compass sensors..."
    ros2 launch tractor_sensors hglrc_m100_5883.launch.py > "$LOG_DIR/gps_compass.log" 2>&1 &
    GPS_PID=$!
    sleep 5
    
    # Verify GPS topics
    if ! wait_for_topic "/hglrc_gps/fix" 15; then
        print_error "GPS topics not available"
        exit 1
    fi
    
    if ! wait_for_topic "/hglrc_gps/imu" 5; then
        print_error "Compass topics not available"  
        exit 1
    fi
    
    print_success "GPS and compass system started"
    
    # Start sensors (encoders)
    print_status "Launching encoder sensors..."
    ros2 launch tractor_bringup sensors.launch.py > "$LOG_DIR/sensors.log" 2>&1 &
    SENSORS_PID=$!
    sleep 3
    
    print_status "=== Starting Robot Localization ==="
    
    # Start robot localization (GPS fusion)
    print_status "Launching robot localization (GPS + odometry fusion)..."
    ros2 launch tractor_bringup robot_localization.launch.py > "$LOG_DIR/localization.log" 2>&1 &
    LOCALIZATION_PID=$!
    sleep 5
    
    # Verify localization topics
    if ! wait_for_topic "/odometry/filtered" 15; then
        print_warning "Filtered odometry not available - continuing anyway"
    else
        print_success "Robot localization system started"
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
    
    # Display system information
    echo ""
    print_success "🚜 Tractor System Successfully Started!"
    echo ""
    echo -e "${GREEN}GPS & Compass:${NC}"
    echo "  - GPS Fix: /hglrc_gps/fix"
    echo "  - Compass: /hglrc_gps/imu"
    echo "  - Magnetic Field: /hglrc_gps/magnetic_field"
    echo ""
    echo -e "${GREEN}Localization:${NC}"
    echo "  - Filtered Odometry: /odometry/filtered"
    echo "  - GPS Odometry: /odometry/gps"
    echo ""
    echo -e "${GREEN}Foxglove Studio:${NC}"
    echo "  - WebSocket URL: ws://$(hostname -I | awk '{print $1}'):$FOXGLOVE_PORT"
    echo "  - Connect Foxglove Studio to this URL for visualization"
    echo ""
    echo -e "${GREEN}Process IDs:${NC}"
    echo "  - GPS/Compass: $GPS_PID"
    echo "  - Sensors: $SENSORS_PID"
    echo "  - Localization: $LOCALIZATION_PID"
    echo "  - Foxglove Bridge: $FOXGLOVE_PID"
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
        
        if ! check_process "hglrc_m100_5883"; then
            print_error "GPS/compass system stopped unexpectedly"
            break
        fi
        
        sleep 5
    done
}

# Display usage information
usage() {
    echo "Usage: $0 [options]"
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