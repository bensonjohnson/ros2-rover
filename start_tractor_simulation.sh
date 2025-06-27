#!/bin/bash

# Tractor Simulation Startup Script
# This script starts all components needed for tractor simulation and remote visualization

set -e  # Exit on any error

echo "🚜 Starting Tractor Simulation System..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to cleanup on exit
cleanup() {
    print_warning "Shutting down tractor simulation..."
    pkill -f "gz sim" 2>/dev/null || true
    pkill -f "ros2 launch" 2>/dev/null || true
    pkill -f "foxglove" 2>/dev/null || true
    pkill -f "coverage" 2>/dev/null || true
    sleep 2
    print_status "Cleanup complete"
}

# Trap cleanup on script exit
trap cleanup EXIT INT TERM

# Check if we're in the right directory
if [ ! -f "src/tractor_bringup/package.xml" ]; then
    print_error "Please run this script from the tractor_ws root directory"
    exit 1
fi

# Source ROS2 and workspace
print_status "Sourcing ROS2 environment..."
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# Kill any existing processes
print_status "Cleaning up existing processes..."
pkill -f "gz sim" 2>/dev/null || true
pkill -f "robot_state_publisher" 2>/dev/null || true
pkill -f "foxglove" 2>/dev/null || true
pkill -f "coverage" 2>/dev/null || true
sleep 2

# Start Gazebo simulation
print_status "Starting Gazebo simulation..."
ros2 launch tractor_bringup tractor_gazebo.launch.py headless:=true &
GAZEBO_PID=$!
sleep 5

# Check if Gazebo started successfully
if ! ps -p $GAZEBO_PID > /dev/null; then
    print_error "Gazebo failed to start"
    exit 1
fi

# Start robot state publisher for simulation
print_status "Starting robot state publisher..."
ros2 launch tractor_bringup tractor_sim.launch.py &
RSP_PID=$!
sleep 3

# Start navigation system
print_status "Starting navigation system..."
ros2 launch tractor_bringup navigation.launch.py map:=maps/yard_map.yaml use_sim_time:=true &
NAV_PID=$!
sleep 5

# Start coverage system
print_status "Starting coverage planning system..."
ros2 launch tractor_coverage coverage_system.launch.py &
COVERAGE_PID=$!
sleep 3

# Start Foxglove bridge
print_status "Starting Foxglove bridge..."
ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765 &
FOXGLOVE_PID=$!
sleep 2

print_status "System startup complete!"
echo ""
echo -e "${BLUE}🚜 Tractor Simulation Running${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✓ Gazebo:${NC}     Headless simulation running"
echo -e "${GREEN}✓ Robot:${NC}      State publisher active"
echo -e "${GREEN}✓ Navigation:${NC} Nav2 stack with yard map loaded"
echo -e "${GREEN}✓ Coverage:${NC}   Planning system ready"
echo -e "${GREEN}✓ Foxglove:${NC}   Bridge running on port 8765"
echo ""
echo -e "${BLUE}📱 Remote Access:${NC}"
echo "  • Foxglove Studio: Connect to ws://$(hostname -I | awk '{print $1}'):8765"
echo "  • SSH: ssh benson@$(hostname -I | awk '{print $1}')"
echo ""
echo -e "${BLUE}🎮 Usage:${NC}"
echo "  • Test coverage: ros2 run tractor_coverage coverage_client demo"
echo "  • Manual control: ros2 topic pub /cmd_vel geometry_msgs/msg/Twist ..."
echo "  • Check status: ros2 topic echo /coverage_status"
echo ""
echo -e "${YELLOW}Press Ctrl+C to shutdown all processes${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Wait for processes and monitor health
while true; do
    # Check if critical processes are still running
    if ! ps -p $GAZEBO_PID > /dev/null; then
        print_error "Gazebo process died unexpectedly"
        break
    fi
    
    if ! ps -p $FOXGLOVE_PID > /dev/null; then
        print_warning "Foxglove bridge died, restarting..."
        ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765 &
        FOXGLOVE_PID=$!
    fi
    
    sleep 5
done

print_warning "System shutting down..."