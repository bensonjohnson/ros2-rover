#!/bin/bash

# ===================================================================
# ROS2 Tractor SLAM Mapping Script
# Creates maps of your yard using RealSense camera and GPS data
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
MAPS_DIR="$WORKSPACE_DIR/maps"
ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
FOXGLOVE_PORT=${FOXGLOVE_PORT:-8765}

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$MAPS_DIR"

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

# Function to save current map
save_map() {
    local map_name=${1:-"yard_map_$(date +%Y%m%d_%H%M%S)"}
    print_status "Saving map as: $map_name"
    
    ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap "{name: {data: '$MAPS_DIR/$map_name'}}"
    
    if [ $? -eq 0 ]; then
        print_success "Map saved successfully to $MAPS_DIR/$map_name"
        echo ""
        echo -e "${GREEN}Map files created:${NC}"
        echo "  - $MAPS_DIR/$map_name.yaml (map metadata)"
        echo "  - $MAPS_DIR/$map_name.pgm (map image)"
    else
        print_error "Failed to save map"
    fi
}

# Function to cleanup on exit
cleanup() {
    print_warning "Shutting down mapping system..."
    
    # Ask if user wants to save the map before exit
    echo ""
    echo -e "${YELLOW}Would you like to save the current map before exiting? (y/N)${NC}"
    read -t 10 -n 1 save_choice
    echo ""
    
    if [[ $save_choice =~ ^[Yy]$ ]]; then
        save_map "final_yard_map_$(date +%Y%m%d_%H%M%S)"
    fi
    
    # Kill background processes
    pkill -f "ros2 launch.*slam" || true
    pkill -f "rviz2" || true
    
    print_success "Mapping system shutdown complete"
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Main mapping function
main() {
    print_status "🗺️  Starting ROS2 Tractor SLAM Mapping System"
    print_status "Workspace: $WORKSPACE_DIR"
    print_status "Maps Directory: $MAPS_DIR"
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
    
    # Check for RealSense camera
    print_status "Checking for RealSense camera..."
    if lsusb | grep -q "Intel Corp"; then
        print_success "RealSense camera detected"
    else
        print_warning "RealSense camera not detected - mapping may not work properly"
    fi
    
    print_status "=== Starting SLAM Mapping System ==="
    
    # Start sensors and basic robot setup
    print_status "Launching robot sensors and description..."
    ros2 launch tractor_bringup sensors.launch.py > "$LOG_DIR/mapping_sensors.log" 2>&1 &
    SENSORS_PID=$!
    sleep 5
    
    # Start SLAM mapping with Foxglove
    print_status "Launching SLAM mapping system with Foxglove..."
    ros2 launch tractor_bringup slam_mapping.launch.py \
        foxglove_port:=$FOXGLOVE_PORT > "$LOG_DIR/mapping_slam.log" 2>&1 &
    SLAM_PID=$!
    sleep 10
    
    # Wait for critical topics
    print_status "Verifying SLAM system startup..."
    
    # Check for depth camera topics
    if ! wait_for_topic "/realsense_435i/depth/image_rect_raw" 20; then
        print_warning "RealSense depth topics not available - check camera connection"
    fi
    
    # Check for laser scan (converted from depth)
    if ! wait_for_topic "/realsense_435i/scan" 15; then
        print_warning "Laser scan from depth image not available"
    fi
    
    # Check for SLAM topics
    if ! wait_for_topic "/map" 15; then
        print_warning "SLAM map topic not available"
    else
        print_success "SLAM mapping system active"
    fi
    
    print_status "=== SLAM Mapping Ready ==="
    
    # Display system information
    echo ""
    print_success "🗺️  SLAM Mapping System Successfully Started!"
    echo ""
    echo -e "${GREEN}Mapping Topics:${NC}"
    echo "  - RealSense Depth: /realsense_435i/depth/image_rect_raw"
    echo "  - Laser Scan: /realsense_435i/scan"
    echo "  - Live Map: /map"
    echo "  - Robot Pose: /pose"
    echo ""
    echo -e "${GREEN}GPS Integration:${NC}"
    echo "  - GPS Fix: /hglrc_gps/fix"
    echo "  - Filtered Odometry: /odometry/filtered"
    echo ""
    echo -e "${GREEN}Commands:${NC}"
    echo "  - Drive around your yard to build the map"
    echo "  - Type 'save' + Enter to save current map"
    echo "  - Press Ctrl+C to exit and optionally save final map"
    echo -e "${GREEN}Foxglove Visualization:${NC}"
    echo "  - Foxglove Studio: ws://$(hostname -I | awk '{print $1}'):$FOXGLOVE_PORT"
    echo "  - Open Foxglove Studio and connect to see live mapping"
    echo "  - View: Map, Robot Pose, Laser Scan, Camera feeds"
    echo ""
    echo -e "${YELLOW}Mapping Tips:${NC}"
    echo "  - Drive slowly for better mapping quality"
    echo "  - Overlap your paths to improve loop closure"
    echo "  - Map both front and back yard in one session if possible"
    echo "  - Avoid direct sunlight on the camera for best results"
    echo ""
    print_status "Start driving around your yard to begin mapping..."
    
    # Interactive command loop
    while true; do
        echo -e "${BLUE}Commands: [save] [status] [quit]${NC}"
        read -p "> " command
        
        case $command in
            save|s)
                echo "Enter map name (or press Enter for auto-generated name):"
                read map_name
                if [ -z "$map_name" ]; then
                    save_map
                else
                    save_map "$map_name"
                fi
                ;;
            status|st)
                print_status "System Status:"
                echo "  - SLAM Process: $(if kill -0 $SLAM_PID 2>/dev/null; then echo "Running"; else echo "Stopped"; fi)"
                echo "  - Available topics: $(ros2 topic list | wc -l)"
                echo "  - Maps directory: $MAPS_DIR"
                echo "  - Saved maps: $(ls -1 $MAPS_DIR/*.yaml 2>/dev/null | wc -l)"
                ;;
            quit|q|exit)
                print_status "Exiting mapping mode..."
                break
                ;;
            *)
                echo "Unknown command. Available: save, status, quit"
                ;;
        esac
    done
}

# Display usage information
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "ROS2 Tractor SLAM Mapping System"
    echo "Creates maps of your yard using RealSense camera and GPS data"
    echo ""
    echo "Options:"
    echo "  -h, --help           Show this help message"
    echo "  -p, --port PORT      Set Foxglove bridge port (default: 8765)"
    echo "  -d, --domain ID      Set ROS domain ID (default: 42)"
    echo ""
    echo "Example:"
    echo "  $0 --port 8888      # Start mapping with custom Foxglove port"
    echo "  $0 --domain 10      # Use custom ROS domain"
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