#!/bin/bash

# NoMaD Autonomous Rover Startup Script
# Runs the pretrained NoMaD visual-navigation policy on the RKNN NPU.
# RGB context frames -> vision_encoder -> 10-step diffusion via noise_pred_net
# -> 8 waypoints -> pure-pursuit -> [left_track, right_track]. The Xbox
# controller stays live so holding RB suppresses autonomy.

echo "=================================================="
echo "ROS2 Rover - NoMaD Autonomous (pretrained)"
echo "=================================================="

if [ ! -f "install/setup.bash" ]; then
  echo "Error: Please run this script from the ros2-rover directory"
  exit 1
fi

GOAL_MODE=${1:-"exploration"}
NOMINAL_SPEED=${2:-"0.20"}
INFERENCE_RATE=${3:-"7.0"}
GOAL_IMAGE_PATH=${4:-""}

echo "Configuration:"
echo "  Goal mode:        ${GOAL_MODE}"
echo "  Nominal speed:    ${NOMINAL_SPEED} m/s"
echo "  Inference rate:   ${INFERENCE_RATE} Hz"
echo "  Goal image path:  ${GOAL_IMAGE_PATH:-<none>}"
echo "  Models dir:       $(pwd)/models/nomad/"
echo ""

echo "WARNING: Rover will drive AUTONOMOUSLY!"
echo "  - Pretrained NoMaD policy (no online learning)"
echo "  - Exploration mode: NoMaD samples exploratory waypoints"
echo "  - Image-goal mode: NoMaD navigates toward the provided RGB image"
echo "  - Hold RB on Xbox controller to suppress the policy"
echo "  - lidar_safety_monitor gates dangerous commands"
echo "  - Keep emergency stop ready"
echo ""
read -p "Press Enter to continue or Ctrl+C to abort..."

echo "Building workspace..."
colcon build --packages-select tractor_bringup tractor_control tractor_sensors --cmake-args -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
if [ $? -ne 0 ]; then
  echo "Build failed!"
  exit 1
fi
echo "Build complete"

echo "Sourcing ROS2 environment..."
source /opt/ros/jazzy/setup.bash
source install/setup.bash

echo "Configuring USB power management for RealSense..."
USB_DEVICE_PATH=""
for device in /sys/bus/usb/devices/*/idProduct; do
  if [ -f "$device" ] && [ "$(cat $device 2>/dev/null)" = "0b3a" ]; then
    USB_DEVICE_PATH=$(dirname $device)
    echo "Found D435i at USB path: $USB_DEVICE_PATH"
    break
  fi
done

if [ -n "$USB_DEVICE_PATH" ]; then
  echo "on" | sudo tee $USB_DEVICE_PATH/power/control > /dev/null 2>&1
  echo "-1" | sudo tee $USB_DEVICE_PATH/power/autosuspend > /dev/null 2>&1
  echo "USB power management configured"
fi

if [ ! -f "models/nomad/vision_encoder.rknn" ] || [ ! -f "models/nomad/noise_pred_net.rknn" ]; then
  echo "Warning: RKNN model files not found under models/nomad/"
  echo "Deploy them with remote_training_server/deploy_nomad_model.sh from the DGX."
fi

echo "Launching NoMaD autonomous..."
mkdir -p log
LOG_FILE="log/nomad_rover_$(date +%Y%m%d_%H%M%S).log"

# Resolve model paths against the actual checkout location so this works
# regardless of where ros2-rover lives.
MODELS_DIR="$(pwd)/models/nomad"

LAUNCH_ARGS=(
  "goal_mode:=${GOAL_MODE}"
  "nominal_speed:=${NOMINAL_SPEED}"
  "inference_rate_hz:=${INFERENCE_RATE}"
  "vision_encoder_rknn:=${MODELS_DIR}/vision_encoder.rknn"
  "noise_pred_net_rknn:=${MODELS_DIR}/noise_pred_net.rknn"
)
if [ -n "$GOAL_IMAGE_PATH" ]; then
  LAUNCH_ARGS+=("goal_image_path:=${GOAL_IMAGE_PATH}")
fi

ros2 launch tractor_bringup nomad_rknn_autonomous.launch.py \
  "${LAUNCH_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE" &

LAUNCH_PID=$!

trap 'echo; echo "Stopping NoMaD..."; kill $LAUNCH_PID 2>/dev/null; sleep 2; echo "Stopped"; exit 0' SIGINT SIGTERM

echo ""
echo "NoMaD running (PID: $LAUNCH_PID)"
echo "Log file: $LOG_FILE"
echo ""
echo "Monitor:"
echo "  - Live path view: http://$(hostname -I | awk '{print $1}'):8081"
echo "  - ros2 topic echo /track_cmd_ai"
echo "  - ros2 topic echo /emergency_stop"
echo "  - ros2 topic echo /joy"
echo "  - ros2 topic hz /camera/camera/color/image_raw"
echo ""
echo "Press Ctrl+C to stop"

wait $LAUNCH_PID
