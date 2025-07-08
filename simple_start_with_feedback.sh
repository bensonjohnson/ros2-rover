#!/bin/bash

# ============================================================================
# Simple ROS2 Tractor Startup Script (WITH Velocity Feedback Control)
# Launches core systems via control_with_feedback.launch.py (includes
# hiwonder_motor_driver and velocity_feedback_controller) and adds
# Xbox controller teleop.
# ============================================================================

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
WORKSPACE_DIR="/home/ubuntu/ros2-rover" # Assuming this is the correct path
LOG_DIR="$WORKSPACE_DIR/logs"
ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
FOXGLOVE_PORT=${FOXGLOVE_PORT:-8765} # Foxglove might be launched separately or added if needed

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

# Function to wait for topics to be available (optional, good for robustness)
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
    print_warning "Shutting down feedback-controlled tractor system..."

    # Kill background processes
    # This will kill the launch command and any nodes it started, plus xbox_controller
    pkill -f "ros2 launch tractor_bringup control_with_feedback.launch.py" || true
    pkill -f "ros2 run tractor_control xbox_controller_teleop" || true
    pkill -f "foxglove_bridge" || true # If Foxglove is started by this script

    print_success "System shutdown complete"
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Main startup function
main() {
    print_status "🚜 Starting ROS2 Tractor System with Velocity Feedback Control"
    print_status "Workspace: $WORKSPACE_DIR"
    print_status "ROS Domain ID: $ROS_DOMAIN_ID"

    # Export ROS domain
    export ROS_DOMAIN_ID

    # Check if in correct directory (optional, depends on execution context)
    # if [ ! -f "$WORKSPACE_DIR/src/tractor_bringup/package.xml" ]; then
    #     print_error "Not in ROS2 workspace directory or workspace not found at $WORKSPACE_DIR!"
    #     exit 1
    # fi

    cd "$WORKSPACE_DIR"

    # Source ROS2
    print_status "Sourcing ROS2 environment..."
    if [ -f "/opt/ros/jazzy/setup.bash" ]; then
        source /opt/ros/jazzy/setup.bash
    else
        print_error "ROS2 Jazzy setup.bash not found!"
        exit 1
    fi

    # Source workspace
    if [ -f "$WORKSPACE_DIR/install/setup.bash" ]; then
        source "$WORKSPACE_DIR/install/setup.bash"
        print_success "ROS2 workspace sourced from $WORKSPACE_DIR/install/setup.bash"
    else
        print_error "Workspace not built or setup.bash not found in $WORKSPACE_DIR/install!"
        print_warning "Attempting to build the workspace... This may take a few minutes."
        ./build.sh # Assuming build.sh is in the WORKSPACE_DIR
        if [ -f "$WORKSPACE_DIR/install/setup.bash" ]; then
            source "$WORKSPACE_DIR/install/setup.bash"
            print_success "ROS2 workspace successfully built and sourced."
        else
            print_error "Workspace build failed or setup.bash still not found. Please build manually."
            exit 1
        fi
    fi

    # Check hardware connections (can be simplified if launch file handles this)
    print_status "Checking hardware connections..."
    if [ -e "/dev/i2c-5" ]; then
        print_success "I2C-5 found for motor controller"
    else
        print_warning "I2C-5 not found - motor control may not work"
    fi

    print_status "=== Starting Core Control System (with Feedback) ==="

    # Launch the main control system including hiwonder_motor_driver and velocity_feedback_controller
    print_status "Launching control_with_feedback.launch.py..."
    ros2 launch tractor_bringup control_with_feedback.launch.py > "$LOG_DIR/control_with_feedback.log" 2>&1 &
    CONTROL_LAUNCH_PID=$!
    sleep 5 # Give time for nodes to start

    # Verify essential topics from the launch file
    if ! wait_for_topic "/odom" 10 || ! wait_for_topic "/cmd_vel" 10 || ! wait_for_topic "/cmd_vel_raw" 10; then
        print_error "Core control system via launch file failed to start crucial topics."
        print_error "Check $LOG_DIR/control_with_feedback.log for details."
        exit 1
    fi
    print_success "Core control system (control_with_feedback.launch.py) started."

    # Start Xbox controller teleop, remapping its output to /cmd_vel_desired
    # which is then remapped to /cmd_vel_raw by control_with_feedback.launch.py
    print_status "Launching Xbox controller teleop..."
    export SDL_JOYSTICK_DEVICE=${SDL_JOYSTICK_DEVICE:-/dev/input/js0}
    if [ ! -e "$SDL_JOYSTICK_DEVICE" ]; then
        print_warning "Xbox controller device $SDL_JOYSTICK_DEVICE not found. Teleop will likely fail."
    fi

    ros2 run tractor_control xbox_controller_teleop --ros-args \
        -p max_linear_speed:=1.0 \
        -p max_angular_speed:=1.0 \
        -p deadzone:=0.15 \
        -p tank_drive_mode:=true \
        -p controller_index:=0 \
        -p use_feedback_control:=false \
        --remap cmd_vel:=cmd_vel_desired \
        > "$LOG_DIR/xbox_controller_feedback_mode.log" 2>&1 &
    XBOX_PID=$!
    sleep 2

    if check_process "xbox_controller_teleop"; then
        print_success "Xbox controller teleop started, publishing to /cmd_vel_desired."
    else
        print_warning "Xbox controller teleop failed to start (controller may not be connected or SDL_JOYSTICK_DEVICE is wrong)."
    fi

    # Optionally, start Foxglove bridge if not handled elsewhere
    # print_status "Launching Foxglove bridge on port $FOXGLOVE_PORT..."
    # ros2 run foxglove_bridge foxglove_bridge --ros-args \
    #     -p port:=$FOXGLOVE_PORT \
    #     -p address:=0.0.0.0 \
    #     -p tls:=false \
    #     > "$LOG_DIR/foxglove_bridge_feedback.log" 2>&1 &
    # FOXGLOVE_PID=$!
    # sleep 3
    # if check_process "foxglove_bridge"; then
    #     print_success "Foxglove bridge started on port $FOXGLOVE_PORT"
    # else
    #     print_warning "Failed to start Foxglove bridge"
    # fi

    echo ""
    print_success "🚜 Tractor System with Velocity Feedback Successfully Started!"
    echo ""
    echo -e "${GREEN}Control Flow:${NC}"
    echo "  - Xbox Controller → /cmd_vel_desired"
    echo "  - control_with_feedback.launch.py:"
    echo "    - Remaps /cmd_vel_desired → /cmd_vel_raw (input to velocity_feedback_controller)"
    echo "    - velocity_feedback_controller:"
    echo "      - Subscribes: /cmd_vel_raw, /odom, /joint_states"
    echo "      - Publishes corrected speeds → /cmd_vel"
    echo "    - hiwonder_motor_driver:"
    echo "      - Subscribes: /cmd_vel"
    echo "      - Publishes: /odom, /joint_states, /battery_voltage, etc."
    echo ""
    echo -e "${YELLOW}To tune drift:${NC}"
    echo "  1. Stop this script (Ctrl+C)."
    echo "  2. Edit 'src/tractor_bringup/launch/control_with_feedback.launch.py'."
    echo "  3. Modify the 'drift_correction_gain' parameter."
    echo "     (If drifting left, try increasing the gain from its current value of 0.0001)."
    echo "  4. Save the launch file."
    echo "  5. Re-run this script: ./simple_start_with_feedback.sh"
    echo ""
    echo -e "${YELLOW}Logs available in: $LOG_DIR${NC}"
    echo ""
    print_status "Press Ctrl+C to shutdown the system"

    # Keep script running and monitor processes
    while true; do
        if ! kill -0 $CONTROL_LAUNCH_PID 2>/dev/null; then
            print_error "Core control system (control_with_feedback.launch.py) stopped unexpectedly."
            break
        fi

        # Xbox controller is optional, just warn if it stops
        if [ ! -z ${XBOX_PID+x} ] && ! kill -0 $XBOX_PID 2>/dev/null && check_process "xbox_controller_teleop"; then
            # Process might have exited cleanly if controller disconnected
            if ! check_process "xbox_controller_teleop"; then
                 print_warning "Xbox controller teleop stopped (controller may have been disconnected or exited)."
                 unset XBOX_PID # Avoid further checks if it's meant to be off
            fi
        fi

        sleep 5
    done
}

# Display usage information
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Simple ROS2 Tractor System (WITH Velocity Feedback Control)"
    echo "Uses control_with_feedback.launch.py for core motor control and feedback."
    echo "Starts Xbox controller for teleoperation."
    echo ""
    echo "Options:"
    echo "  -h, --help           Show this help message"
    # echo "  -p, --port PORT      Set Foxglove bridge port (default: 8765)" # If enabling foxglove here
    echo "  -d, --domain ID      Set ROS domain ID (default: 42)"
    echo ""
    echo "Example:"
    echo "  $0 --domain 10"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        # -p|--port) # If enabling foxglove here
        #     FOXGLOVE_PORT="$2"
        #     shift 2
        #     ;;
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
