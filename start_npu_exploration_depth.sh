#!/bin/bash

# Minimal NPU Depth Image Exploration Script
# Clean architecture: Hardware + AI only, no SLAM/Nav2 complexity

echo "=================================================="
echo "ROS2 Tractor - NPU Depth Image Exploration"
echo "=================================================="
echo "This minimal system uses:"
echo "  ✓ Hiwonder motor control"
echo "  ✓ RealSense D435i (depth images only, IMU disabled, USB optimized)"
echo "  ✓ NPU-based exploration AI"
echo "  ✓ Direct safety monitoring"
echo "  ✓ No SLAM/Nav2 complexity"
echo ""

# Check if we're in the right directory
if [ ! -f "install/setup.bash" ]; then
    echo "Error: Please run this script from the ros2-rover directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Build only what we need
echo "Building minimal workspace..."
# Check if packages need to be built
if [ ! -d "build/tractor_bringup" ] || [ ! -d "build/tractor_control" ] || [ ! -d "install/tractor_bringup" ] || [ ! -d "install/tractor_control" ]; then
    echo "Building packages..."
    colcon build --packages-select tractor_bringup tractor_control --cmake-args -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "❌ Build failed! Please check for compilation errors."
        exit 1
    fi
    echo "✓ Workspace built successfully"
else
    echo "✓ Packages already built"
fi

# Source environment
echo "Sourcing ROS2 environment..."
source /opt/ros/jazzy/setup.bash
source install/setup.bash
echo "✓ ROS2 environment sourced"

# Initial RKNN conversion step
echo "Checking/performing initial RKNN conversion..."
python3 - <<'PYCODE'
try:
    from tractor_bringup.rknn_trainer_depth import RKNNTrainerDepth, RKNN_AVAILABLE
    trainer = RKNNTrainerDepth(model_dir="models", enable_debug=True)
    if RKNN_AVAILABLE:
        trainer.convert_to_rknn()
    else:
        print("RKNN toolkit not available - skipping initial conversion")
except Exception as e:
    print(f"Initial RKNN conversion failed: {e}")
PYCODE

echo "Initial RKNN conversion step complete"

# USB power management for RealSense
echo "Configuring USB power management..."
USB_DEVICE_PATH=""
# Find D435i device by Product ID (0B3A)
for device in /sys/bus/usb/devices/*/idProduct; do
    if [ -f "$device" ] && [ "$(cat $device 2>/dev/null)" = "0b3a" ]; then
        USB_DEVICE_PATH=$(dirname $device)
        echo "✓ Found D435i at USB path: $USB_DEVICE_PATH"
        break
    fi
done

# Fallback to common paths if Product ID detection fails
if [ -z "$USB_DEVICE_PATH" ]; then
    for path in "/sys/bus/usb/devices/8-1" "/sys/bus/usb/devices/2-1" "/sys/bus/usb/devices/1-1"; do
        if [ -d "$path" ]; then
            USB_DEVICE_PATH="$path"
            echo "✓ Using fallback USB path: $USB_DEVICE_PATH"
            break
        fi
    done
fi

if [ -n "$USB_DEVICE_PATH" ]; then
    # Disable autosuspend for the device
    echo "on" | sudo tee $USB_DEVICE_PATH/power/control > /dev/null 2>&1
    echo "-1" | sudo tee $USB_DEVICE_PATH/power/autosuspend > /dev/null 2>&1
    echo "✓ USB power management configured for $USB_DEVICE_PATH"
else
    echo "⚠ USB device path not found, continuing anyway"
fi

# Check RealSense D435i specifically
echo "Checking RealSense D435i..."
if command -v rs-enumerate-devices &> /dev/null; then
    timeout 5s rs-enumerate-devices | grep -q "D435I"
    if [ $? -eq 0 ]; then
        echo "✓ RealSense D435i detected"
        # Reset the USB device to clear any error states
        if [ -n "$USB_DEVICE_PATH" ] && [ -w "$USB_DEVICE_PATH" ]; then
            echo "Resetting USB device..."
            echo "0" | sudo tee $USB_DEVICE_PATH/authorized > /dev/null 2>&1
            sleep 1
            echo "1" | sudo tee $USB_DEVICE_PATH/authorized > /dev/null 2>&1
            echo "✓ USB device reset"
        fi
    else
        echo "⚠ RealSense D435i not detected - will attempt to continue"
    fi
else
    echo "⚠ rs-enumerate-devices command not found - skipping RealSense check"
fi

# Configuration (interactive if no args supplied)
# Existing positional usage still works:
#   ./start_npu_exploration_depth.sh <mode> <max_speed> <exploration_time> <safety_distance>
# If no args -> interactive menu with arrow keys.

DEFAULT_MODE="cpu_training"
DEFAULT_MAX_SPEED="0.15"
DEFAULT_EXPLORATION_TIME="300"
DEFAULT_SAFETY_DISTANCE="0.2"

# Fallback-friendly interactive helpers
supports_arrows() {
  # Disable with env var or if terminal looks incompatible
  if [[ -n "$DISABLE_INTERACTIVE_ARROWS" ]]; then return 1; fi
  if [[ ! -t 0 ]]; then return 1; fi
  if [[ "$TERM" == "dumb" || "$TERM" == "" ]]; then return 1; fi
  command -v tput >/dev/null 2>&1 || return 1
  # Ensure cursor up works
  tput cuu 0 >/dev/null 2>&1 || return 1
  return 0
}

numeric_mode_menu() {
  echo "Select Operation Mode:";
  echo "  1) cpu_training (PyTorch training + periodic export)";
  echo "  2) hybrid       (RKNN inference + ongoing training)";
  echo "  3) inference    (Pure RKNN inference, no training)";
  read -p "Enter choice [1-3] (default 1): " choice
  case "$choice" in
    2) MODE="hybrid";;
    3) MODE="inference";;
    1|"" ) MODE="cpu_training";;
    *) echo "Invalid choice, defaulting to cpu_training"; MODE="cpu_training";;
  esac
  echo "Selected: $MODE"
}

choose_mode() {
  # If arrows unsupported, fall back immediately
  if ! supports_arrows; then
    numeric_mode_menu
    return 0
  fi
  local options=("cpu_training" "hybrid" "inference")
  local index=0
  local key
  echo "(Use ↑/↓ then Enter, or press Enter now for default: ${options[0]})"
  while true; do
    for i in "${!options[@]}"; do
      if [ $i -eq $index ]; then
        printf "  > %s\n" "${options[$i]}"
      else
        printf "    %s\n" "${options[$i]}"
      fi
    done
    IFS= read -rsn1 key 2>/dev/null || true
    # Empty input (user just pressed Enter but read may have timed out)
    if [[ -z "$key" ]]; then
      MODE="${options[$index]}"; echo "Selected: $MODE"; return 0
    fi
    if [[ $key == $'\n' || $key == $'\r' ]]; then
      MODE="${options[$index]}"; echo "Selected: $MODE"; return 0
    elif [[ $key == $'\x1b' ]]; then
      read -rsn2 -t 0.002 key_rest || true
      key+="$key_rest"
      case "$key" in
        $'\x1b[A') ((index--)); (( index < 0 )) && index=$(( ${#options[@]} - 1 )) ;;
        $'\x1b[B') ((index++)); (( index >= ${#options[@]} )) && index=0 ;;
      esac
    fi
    # Redraw: move cursor up number of option lines
    local lines=${#options[@]}
    tput cuu $lines 2>/dev/null || clear
  done
}

if [ $# -eq 0 ] && [ -t 0 ]; then
  echo "No arguments supplied - entering interactive setup..."
  choose_mode
  read -p "Maximum Speed [${DEFAULT_MAX_SPEED}]: " INPUT_MAX_SPEED
  read -p "Exploration Time seconds [${DEFAULT_EXPLORATION_TIME}]: " INPUT_EXPLORATION_TIME
  read -p "Safety Distance meters [${DEFAULT_SAFETY_DISTANCE}]: " INPUT_SAFETY_DISTANCE
  MAX_SPEED=${INPUT_MAX_SPEED:-$DEFAULT_MAX_SPEED}
  EXPLORATION_TIME=${INPUT_EXPLORATION_TIME:-$DEFAULT_EXPLORATION_TIME}
  SAFETY_DISTANCE=${INPUT_SAFETY_DISTANCE:-$DEFAULT_SAFETY_DISTANCE}
else
  MODE=${1:-$DEFAULT_MODE}
  shift || true
  MAX_SPEED=${1:-$DEFAULT_MAX_SPEED}
  EXPLORATION_TIME=${2:-$DEFAULT_EXPLORATION_TIME}
  SAFETY_DISTANCE=${3:-$DEFAULT_SAFETY_DISTANCE}
fi

if [[ "$MODE" != "cpu_training" && "$MODE" != "hybrid" && "$MODE" != "inference" ]]; then
  echo "Invalid mode '$MODE'. Valid: cpu_training | hybrid | inference"
  exit 1
fi

echo ""
echo "Configuration:"
echo "  Operation Mode: ${MODE}"
echo "  Maximum Speed: ${MAX_SPEED} m/s"
echo "  Exploration Time: ${EXPLORATION_TIME} seconds"
echo "  Safety Distance: ${SAFETY_DISTANCE} m"
echo "  IMU Status: Disabled (to reduce USB errors)"
echo "  USB Mode: Optimized for stability"
echo ""

# Countdown
echo "Starting NPU depth exploration in:"
for i in {3..1}; do
    echo "  $i..."
    sleep 1
done
echo "  🚀 LAUNCHING!"
echo ""

# Launch the minimal system
echo "Launching NPU depth exploration system..."
echo "Press Ctrl+C to stop safely"
echo ""

# Launch Foxglove bridge for visualization
echo "Launching Foxglove bridge..."
ros2 launch foxglove_bridge foxglove_bridge_launch.py &
FOXGLOVE_PID=$!
echo "✓ Foxglove bridge launched with PID: $FOXGLOVE_PID"
echo ""

ros2 launch tractor_bringup npu_exploration_depth.launch.py \
    operation_mode:=${MODE} \
    max_speed:=${MAX_SPEED} \
    exploration_time:=${EXPLORATION_TIME} \
    safety_distance:=${SAFETY_DISTANCE} \
    use_sim_time:=false &

LAUNCH_PID=$!

# Simple shutdown handler
shutdown_handler() {
    echo ""
    echo "🛑 Stopping NPU depth exploration..."
    if ps -p $LAUNCH_PID > /dev/null; then
        kill $LAUNCH_PID 2>/dev/null
        sleep 2
        # Force kill if still running
        if ps -p $LAUNCH_PID > /dev/null; then
            kill -9 $LAUNCH_PID 2>/dev/null
        fi
    fi
    if ps -p $FOXGLOVE_PID > /dev/null; then
        kill $FOXGLOVE_PID 2>/dev/null
    fi
    echo "✅ NPU depth exploration stopped safely"
    echo "=================================================="
    exit 0
}

trap shutdown_handler SIGINT SIGTERM

echo "🤖 NPU depth exploration active..."
echo "   System is learning to navigate autonomously"
echo "   Monitor via: ros2 topic echo /npu_exploration_status"
echo ""

# Wait for completion
wait $LAUNCH_PID

shutdown_handler
