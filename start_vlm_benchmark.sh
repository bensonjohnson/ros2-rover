#!/usr/bin/env bash
set -e

# VLM Benchmark Startup Script
# Tests VLM inference performance at different resolutions to optimize real-time control

# Ensure ROS 2 + workspace environment is sourced
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
  source /opt/ros/jazzy/setup.bash
fi

if [ -f "install/setup.bash" ]; then
  source install/setup.bash
else
  echo "Workspace not built yet; building tractor_bringup..."
  colcon build --packages-select tractor_bringup --cmake-args -DCMAKE_BUILD_TYPE=Release
  source install/setup.bash
fi

# Default parameters
TEST_DURATION=30
TARGET_FPS=1.0
MODEL_PATH="/home/ubuntu/models/Qwen2.5-VL-7B-Instruct-rk3588-1.2.1.rkllm"
CAMERA_ONLY=false

# Parse command line arguments
for arg in "$@"; do
  case "$arg" in
    --duration=*)
      TEST_DURATION="${arg#*=}" ;;
    --fps=*)
      TARGET_FPS="${arg#*=}" ;;
    --model-path=*)
      MODEL_PATH="${arg#*=}" ;;
    --camera-only)
      CAMERA_ONLY=true ;;
    --help)
      echo "VLM Benchmark Options:"
      echo "  --duration=30          Test duration in seconds (default: 30)"
      echo "  --fps=1.0             Target FPS to test (default: 1.0)"
      echo "  --model-path=PATH     Path to RKLLM model file"
      echo "  --camera-only         Only start camera, no benchmark"
      echo "  --help                Show this help"
      exit 0 ;;
  esac
done

echo "VLM Benchmark Configuration:"
echo "  Test duration: ${TEST_DURATION}s"
echo "  Target FPS: ${TARGET_FPS}"
echo "  Model path: ${MODEL_PATH}"
echo "  Camera only: ${CAMERA_ONLY}"

if [ "$CAMERA_ONLY" = true ]; then
  echo "Starting camera only for manual testing..."
  
  # Start just the camera and basic nodes
  ros2 launch realsense2_camera rs_launch.py \
    camera_name:=camera \
    camera_namespace:=camera \
    enable_color:=true \
    enable_depth:=false \
    enable_pointcloud:=false \
    rgb_camera.color_profile:=640x480x30 \
    enable_imu:=false \
    enable_gyro:=false \
    enable_accel:=false &
  
  CAMERA_PID=$!
  
  echo "Camera started (PID: $CAMERA_PID). Press Ctrl+C to stop."
  
  # Wait for interrupt
  trap "echo 'Stopping camera...'; kill $CAMERA_PID 2>/dev/null; exit 0" INT
  wait $CAMERA_PID
  
else
  echo "Starting full VLM benchmark..."
  
  # Start camera first
  echo "Starting RealSense camera..."
  ros2 launch realsense2_camera rs_launch.py \
    camera_name:=camera \
    camera_namespace:=camera \
    enable_color:=true \
    enable_depth:=false \
    enable_pointcloud:=false \
    rgb_camera.color_profile:=640x480x30 \
    enable_imu:=false \
    enable_gyro:=false \
    enable_accel:=false &
  
  CAMERA_PID=$!
  
  # Wait for camera to initialize
  sleep 3
  
  # Start benchmark
  echo "Starting VLM benchmark..."
  ros2 run tractor_bringup vlm_benchmark.py \
    --ros-args \
    -p vlm_model_path:=${MODEL_PATH} \
    -p test_duration:=${TEST_DURATION}.0 \
    -p target_fps:=${TARGET_FPS} \
    -p test_resolutions:="[240, 320, 480, 640, 800]"
  
  # Clean up
  echo "Benchmark complete, stopping camera..."
  kill $CAMERA_PID 2>/dev/null || true
  wait $CAMERA_PID 2>/dev/null || true
fi

echo "VLM benchmark session complete."