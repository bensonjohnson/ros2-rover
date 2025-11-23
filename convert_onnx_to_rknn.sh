#!/bin/bash

# ONNX to RKNN Conversion Script (Run on Rover)
# Converts ONNX models to RKNN format using RK3588's RKNN-Toolkit-Lite2

set -e

echo "=================================================="
echo "ONNX → RKNN Converter (RK3588)"
echo "=================================================="

ONNX_PATH=${1:-""}
CALIBRATION_DIR=${2:-""} # Default to empty (disable auto-quantization)

if [ -z "$ONNX_PATH" ]; then
  echo "Usage: $0 <onnx_file> [calibration_dir]"
  echo ""
  echo "Examples:"
  echo "  $0 models/ppo_v620_update_50.onnx"
  echo "  $0 models/ppo_v620_update_50.onnx calibration_data  # With INT8 quantization"
  echo ""
  echo "Arguments:"
  echo "  onnx_file        Path to ONNX model"
  echo "  calibration_dir  Directory with .npz calibration files (optional, for INT8 quantization)"
  exit 1
fi

if [ ! -f "$ONNX_PATH" ]; then
  echo "❌ ONNX file not found: ${ONNX_PATH}"
  echo ""
  echo "Available models:"
  ls -lh models/*.onnx 2>/dev/null || echo "  (no ONNX models found)"
  exit 1
fi

# Determine output path
RKNN_PATH="${ONNX_PATH%.onnx}.rknn"

echo "Input:  ${ONNX_PATH}"
echo "Output: ${RKNN_PATH}"
echo ""

# Check if RKNN-Toolkit-Lite2 is installed
echo "Checking RKNN-Toolkit-Lite2..."
if ! python3 -c "from rknnlite.api import RKNNLite" 2>/dev/null; then
  echo "❌ RKNN-Toolkit-Lite2 not installed!"
  echo ""
  echo "Install from: https://github.com/rockchip-linux/rknn-toolkit2"
  echo ""
  echo "Quick install:"
  echo "  git clone https://github.com/rockchip-linux/rknn-toolkit2"
  echo "  cd rknn-toolkit2/rknn-toolkit-lite2/packages"
  echo "  pip3 install rknn_toolkit_lite2-*-cp310-cp310-linux_aarch64.whl"
  exit 1
fi

echo "✓ RKNN-Toolkit-Lite2 installed"

# Check platform
echo ""
echo "Checking platform..."
PLATFORM=$(uname -m)
if [ "$PLATFORM" != "aarch64" ]; then
  echo "⚠ Warning: Not running on aarch64 (detected: ${PLATFORM})"
  echo "RKNN conversion is designed for RK3588 (aarch64)"
fi

# Check NPU
if [ -f "/sys/kernel/debug/rknpu/version" ]; then
  NPU_VERSION=$(cat /sys/kernel/debug/rknpu/version 2>/dev/null || echo "unknown")
  echo "✓ NPU detected: ${NPU_VERSION}"
else
  echo "⚠ NPU version not found (may not be RK3588)"
fi

# Check for calibration data
USE_QUANTIZATION=false
if [ -d "$CALIBRATION_DIR" ]; then
  NUM_SAMPLES=$(find "$CALIBRATION_DIR" -name "*.npz" 2>/dev/null | wc -l)
  if [ "$NUM_SAMPLES" -gt 0 ]; then
    echo "✓ Found calibration data: $NUM_SAMPLES samples in $CALIBRATION_DIR"
    USE_QUANTIZATION=true
  else
    echo "⚠ No calibration samples found in $CALIBRATION_DIR"
  fi
else
  echo "⚠ Calibration directory not found: $CALIBRATION_DIR"
fi

# Convert using Python script
echo ""
echo "Converting ONNX → RKNN..."
if [ "$USE_QUANTIZATION" = true ]; then
  echo "Mode: INT8 quantization with calibration data"
  python3 src/tractor_bringup/tractor_bringup/convert_onnx_to_rknn.py \
    "${ONNX_PATH}" \
    --output "${RKNN_PATH}" \
    --target rk3588 \
    --quantize \
    --calibration-dir "${CALIBRATION_DIR}"
else
  echo "Mode: Float16 (no quantization)"
  python3 src/tractor_bringup/tractor_bringup/convert_onnx_to_rknn.py \
    "${ONNX_PATH}" \
    --output "${RKNN_PATH}" \
    --target rk3588
fi

if [ $? -ne 0 ]; then
  echo ""
  echo "❌ Conversion failed"
  exit 1
fi

# Create symlink to latest model
echo ""
echo "Creating symlink to latest model..."
MODELS_DIR=$(dirname "${RKNN_PATH}")
cd "${MODELS_DIR}"
ln -sf $(basename "${RKNN_PATH}") remote_trained.rknn
echo "✓ Symlink created: remote_trained.rknn → $(basename ${RKNN_PATH})"

cd - > /dev/null

echo ""
echo "=================================================="
echo "✅ Conversion Complete!"
echo "=================================================="
echo ""
echo "RKNN model: ${RKNN_PATH}"
echo "Symlink:    ${MODELS_DIR}/remote_trained.rknn"
echo ""
echo "Model size:"
ls -lh "${RKNN_PATH}"
echo ""
echo "Test inference with:"
echo "  ./start_remote_trained_inference.sh"
echo ""
echo "Or reload in running system:"
echo "  ros2 service call /reload_remote_model std_srvs/srv/Trigger"
