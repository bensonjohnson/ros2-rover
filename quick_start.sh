#!/bin/bash

# ===================================================================
# ROS2 Tractor Complete System Startup Script
# Launches GPS, localization, navigation, control, and visualization
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
USE_SIM_TIME=${USE_SIM_TIME:-false}

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
    print_warning "Shutting down complete tractor system..."
    
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
    print_status "🚜 Starting Complete ROS2 Tractor System"
    print_status "Workspace: $WORKSPACE_DIR"
    print_status "ROS Domain ID: $ROS_DOMAIN_ID"
    print_status "Foxglove Port: $FOXGLOVE_PORT"
    print_status "Use Sim Time: $USE_SIM_TIME"
    
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
        print_warning "GPS device not found at /dev/ttyS6 - will use defaults"
    fi
    
    # Check I2C for compass
    if [ -e "/dev/i2c-5" ]; then
        print_success "I2C-5 found for compass"
    else
        print_warning "I2C-5 not found - compass may not work"
    fi
    
    print_status "=== Starting Complete Tractor System ==="
    
    # Start complete tractor bringup (includes sensors, localization, navigation, control)
    print_status "Launching complete tractor system..."
    ros2 launch tractor_bringup tractor_bringup.launch.py \
        use_sim_time:=$USE_SIM_TIME > "$LOG_DIR/tractor_system.log" 2>&1 &
    TRACTOR_PID=$!
    sleep 10
    
    # Wait for critical topics
    print_status "Verifying system startup..."
    
    # Check for GPS topics (if available)
    if [ -e "/dev/ttyS6" ]; then
        if ! wait_for_topic "/hglrc_gps/fix" 20; then
            print_warning "GPS topics not available - continuing anyway"
        fi
    fi
    
    # Check for filtered odometry
    if ! wait_for_topic "/odometry/filtered" 20; then
        print_warning "Filtered odometry not available - continuing anyway"
    fi
    
    # Check for transforms
    sleep 5
    if ros2 topic list | grep -q "/tf"; then
        print_success "Transform system active"
    else
        print_warning "Transform system not fully active"
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
    print_success "🚜 Complete Tractor System Successfully Started!"
    echo ""
    echo -e "${GREEN}Sensors & Localization:${NC}"
    echo "  - GPS Fix: /hglrc_gps/fix"
    echo "  - Compass: /hglrc_gps/imu"
    echo "  - Wheel Encoders: /wheel_odom"
    echo "  - Filtered Odometry: /odometry/filtered"
    echo ""
    echo -e "${GREEN}Navigation:${NC}"
    echo "  - Map Server: /map"
    echo "  - Costmaps: /local_costmap/costmap, /global_costmap/costmap"
    echo "  - Navigation Actions: /navigate_to_pose"
    echo ""
    echo -e "${GREEN}Control:${NC}"
    echo "  - Motor Commands: /cmd_vel"
    echo "  - Motor Status: /motor_status"
    echo ""
    echo -e "${GREEN}Visualization:${NC}"
    echo "  - Foxglove Studio: ws://$(hostname -I | awk '{print $1}'):$FOXGLOVE_PORT"
    echo ""
    echo -e "${GREEN}Process IDs:${NC}"
    echo "  - Tractor System: $TRACTOR_PID"
    echo "  - Foxglove Bridge: $FOXGLOVE_PID"
    echo ""
    echo -e "${YELLOW}Usage:${NC}"
    echo "  - Send navigation goals via Foxglove Studio"
    echo "  - Monitor system status in real-time"
    echo "  - Logs available in: $LOG_DIR"
    echo ""
    print_status "Press Ctrl+C to shutdown the complete system"
    
    # Keep script running and monitor processes
    while true; do
        # Check if critical processes are still running
        if ! check_process "foxglove_bridge"; then
            print_error "Foxglove bridge stopped unexpectedly"
            break
        fi
        
        if ! kill -0 $TRACTOR_PID 2>/dev/null; then
            print_error "Tractor system stopped unexpectedly"
            break
        fi
        
        sleep 10
    done
}

# Display usage information
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Complete ROS2 Tractor System Launcher"
    echo "Starts: GPS, localization, navigation, control, and visualization"
    echo ""
    echo "Options:"
    echo "  -h, --help           Show this help message"
    echo "  -p, --port PORT      Set Foxglove bridge port (default: 8765)"
    echo "  -d, --domain ID      Set ROS domain ID (default: 42)"
    echo "  -s, --sim            Use simulation time (default: false)"
    echo ""
    echo "Example:"
    echo "  $0 --port 8888 --domain 10"
    echo "  $0 --sim  # For simulation mode"
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
        -s|--sim)
            USE_SIM_TIME="true"
            shift
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