#!/bin/bash

# NoMaD ONNX deployment + on-rover RKNN conversion.
# Mirrors deploy_model.sh, but ships TWO ONNX files (vision_encoder,
# noise_pred_net) and runs the rover's existing convert_onnx_to_rknn.sh
# on each. The diffusion loop itself lives in nomad_rknn_runner.py — only
# the two static-shape submodels need RKNN compilation.

set -e

echo "=================================================="
echo "NoMaD ONNX -> Rover -> RKNN"
echo "=================================================="

ROVER_IP=${1:-"192.168.1.50"}
ROVER_USER=${2:-"benson"}
ONNX_DIR=${3:-"./nomad_export/onnx"}
REMOTE_DIR=${4:-"~/ros2-rover/models/nomad"}

VISION_ENCODER="${ONNX_DIR}/vision_encoder.onnx"
NOISE_PRED_NET="${ONNX_DIR}/noise_pred_net.onnx"

if [ ! -f "$VISION_ENCODER" ] || [ ! -f "$NOISE_PRED_NET" ]; then
  echo "ONNX files missing under ${ONNX_DIR}."
  echo "Run: python nomad_export/export_nomad_onnx.py --checkpoint ... --vint_repo ..."
  exit 1
fi

echo "Rover:     ${ROVER_USER}@${ROVER_IP}"
echo "Remote dir: ${REMOTE_DIR}"
echo "ONNX files:"
echo "  $(du -h ${VISION_ENCODER} | cut -f1)  ${VISION_ENCODER}"
echo "  $(du -h ${NOISE_PRED_NET} | cut -f1)  ${NOISE_PRED_NET}"
echo ""

echo "Testing SSH..."
if ! ssh -o ConnectTimeout=5 ${ROVER_USER}@${ROVER_IP} "echo connected" 2>/dev/null; then
  echo "SSH to ${ROVER_USER}@${ROVER_IP} failed."
  echo ""
  echo "Manual deployment:"
  echo "  scp ${VISION_ENCODER} ${NOISE_PRED_NET} ${ROVER_USER}@${ROVER_IP}:${REMOTE_DIR}/"
  echo "  ssh ${ROVER_USER}@${ROVER_IP}"
  echo "  cd ~/ros2-rover"
  echo "  ./convert_onnx_to_rknn.sh models/nomad/vision_encoder.onnx"
  echo "  ./convert_onnx_to_rknn.sh models/nomad/noise_pred_net.onnx"
  exit 1
fi

ssh ${ROVER_USER}@${ROVER_IP} "mkdir -p ${REMOTE_DIR}"

echo "Copying ONNX files..."
scp "${VISION_ENCODER}" "${NOISE_PRED_NET}" "${ROVER_USER}@${ROVER_IP}:${REMOTE_DIR}/"

echo ""
echo "Converting ONNX -> RKNN on rover (this uses the existing"
echo "convert_onnx_to_rknn.sh wrapper around RKNN-Toolkit2 on RK3588)..."

ssh ${ROVER_USER}@${ROVER_IP} bash <<EOF
set -e
cd ~/ros2-rover
./convert_onnx_to_rknn.sh ${REMOTE_DIR}/vision_encoder.onnx
./convert_onnx_to_rknn.sh ${REMOTE_DIR}/noise_pred_net.onnx
ls -lh ${REMOTE_DIR}/
EOF

echo ""
echo "=================================================="
echo "Deployment complete."
echo "=================================================="
echo ""
echo "On the rover:"
echo "  cd ~/ros2-rover"
echo "  ./start_nomad_rover.sh"
