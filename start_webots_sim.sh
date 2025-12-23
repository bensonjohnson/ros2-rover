#!/bin/bash

# Start Webots Simulation for SAC Rover Training
# Usage: ./start_webots_sim.sh [--docker] [--training]
#
# Options:
#   --docker    Run ROS 2 nodes in Docker container (recommended)
#   --training  Also start SAC episode runner for training

set -e

echo "=================================================="
echo "Webots Simulation - SAC Rover Training"
echo "=================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
USE_DOCKER=false
START_TRAINING=false

for arg in "$@"; do
    case $arg in
        --docker)
            USE_DOCKER=true
            ;;
        --training)
            START_TRAINING=true
            ;;
        --help|-h)
            echo "Usage: $0 [--docker] [--training]"
            echo ""
            echo "Options:"
            echo "  --docker    Run ROS 2 nodes in Docker container"
            echo "  --training  Also start SAC episode runner"
            exit 0
            ;;
    esac
done

# Check if Webots is installed
if ! command -v webots &> /dev/null; then
    echo "❌ Webots not found. Install with: sudo snap install webots"
    exit 1
fi

echo "Configuration:"
echo "  Webots: Running on host (native Wayland)"
if [ "$USE_DOCKER" = true ]; then
    echo "  ROS 2:  Docker container (ROS 2 Jazzy)"
    echo "  Controller: Extern mode - runs in Docker, connects to Webots on port 1234"
else
    echo "  ROS 2:  Host (requires ROS 2 Jazzy)"
    echo "  Controller: Runs on host in Webots"
fi
echo "  Training: $([ "$START_TRAINING" = true ] && echo "Enabled" || echo "Disabled")"
echo ""

# Only warn about missing ROS 2 in host mode (not Docker mode)
if [ "$USE_DOCKER" = false ] && [ ! -f /opt/ros/jazzy/setup.bash ]; then
    echo "⚠ WARNING: ROS 2 Jazzy not found on host!"
    echo "  Use --docker flag or install ROS 2 Jazzy"
    echo ""
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping simulation..."
    
    # Stop Docker containers if running
    if [ "$USE_DOCKER" = true ]; then
        docker compose -f sim/docker/docker-compose.yml down 2>/dev/null || true
    fi
    
    # Kill Webots
    pkill -f "webots.*training_arena" 2>/dev/null || true
    
    # Kill any background processes we started
    jobs -p | xargs -r kill 2>/dev/null || true
    
    echo "✅ Stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM

if [ "$USE_DOCKER" = true ]; then
    # === Docker Mode (Recommended for Fedora/non-Ubuntu) ===
    echo "🐳 Docker mode: Webots on host, ROS 2 + controller in container"
    echo ""
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker not found. Install Docker first."
        exit 1
    fi
    
    # Start Webots on host FIRST (extern controller waits for connection)
    echo "Starting Webots on host..."
    webots --mode=realtime "${SCRIPT_DIR}/sim/worlds/training_arena.wbt" &
    WEBOTS_PID=$!
    
    # Wait for Webots to start
    sleep 3
    
    # Build container if needed
    if ! docker images | grep -q ros2_webots_sim; then
        echo "Building Docker image (first time, ~5 min)..."
        docker compose -f sim/docker/docker-compose.yml build
    fi
    
    # Start ROS 2 container with extern controller
    echo "Starting ROS 2 container with extern controller..."
    if [ "$START_TRAINING" = true ]; then
        docker compose -f sim/docker/docker-compose.yml --profile training up -d
    else
        docker compose -f sim/docker/docker-compose.yml up -d ros2_sim
    fi
    
    echo ""
    echo "=================================================="
    echo "✅ Simulation Running"
    echo "=================================================="
    echo ""
    echo "Webots PID: $WEBOTS_PID"
    echo ""
    echo "Container logs:"
    echo "  docker logs -f ros2_webots_sim"
    echo ""
    echo "Interactive shell:"
    echo "  docker exec -it ros2_webots_sim bash"
    echo ""
    echo "Test topics:"
    echo "  docker exec ros2_webots_sim ros2 topic list"
    echo ""
    echo "Press Ctrl+C to stop"
    
    # Wait for Webots to exit
    wait $WEBOTS_PID
    
else
    # === Host Mode ===
    echo "Setting up ROS 2 on host..."
    
    # Source ROS 2
    if [ -f /opt/ros/jazzy/setup.bash ]; then
        source /opt/ros/jazzy/setup.bash
    else
        echo "❌ ROS 2 Jazzy not found. Use --docker flag or install ROS 2."
        exit 1
    fi
    
    # Build if needed
    if [ ! -d "install/webots_sim" ]; then
        echo "Building webots_sim package..."
        colcon build --packages-select webots_sim tractor_bringup
    fi
    
    source install/setup.bash
    
    # Start robot state publisher in background
    echo "Starting robot state publisher..."
    ros2 launch tractor_bringup robot_description.launch.py use_sim_time:=true &
    
    # Start Webots on host
    echo ""
    echo "Starting Webots..."
    webots --mode=realtime "${SCRIPT_DIR}/sim/worlds/training_arena.wbt" &
    WEBOTS_PID=$!

    # Wait for Webots to initialize
    sleep 3

    echo ""
    echo "=================================================="
    echo "✅ Simulation Running (Host Mode)"
    echo "=================================================="
    echo ""
    echo "Webots PID: $WEBOTS_PID"
    echo ""
    echo "Test commands (run in another terminal):"
    echo "  source install/setup.bash"
    echo "  ros2 topic list"
    echo "  ros2 topic echo /scan --once"
    echo ""
    echo "Drive the rover:"
    echo "  ros2 topic pub /cmd_vel_ai geometry_msgs/Twist \"{linear: {x: 0.1}}\" -r 10"
    echo ""
    echo "Press Ctrl+C to stop"

    # Wait for Webots to exit
    wait $WEBOTS_PID
fi
