#!/bin/bash

# V620 Training Server Setup Script
# Run this script once to set up the V620 machine for remote training

set -e

echo "=================================================="
echo "V620 Training Server Setup"
echo "=================================================="
echo ""
echo "This script will install all dependencies for remote training."
echo "Platform: x86_64 Linux (Ubuntu 22.04/24.04 recommended)"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  exit 0
fi

# Detect OS
if [ -f /etc/os-release ]; then
  . /etc/os-release
  echo ""
  echo "Detected OS: $NAME $VERSION"
else
  echo "⚠ Cannot detect OS version"
fi

# Check architecture
ARCH=$(uname -m)
echo "Architecture: $ARCH"

if [ "$ARCH" != "x86_64" ]; then
  echo "❌ Error: This setup requires x86_64 architecture"
  echo "RKNN-Toolkit2 conversion tools only work on x86_64"
  exit 1
fi

# Step 1: Update system
echo ""
echo "=================================================="
echo "Step 1: Updating system packages"
echo "=================================================="
sudo apt update
sudo apt upgrade -y

# Step 2: Install system dependencies
echo ""
echo "=================================================="
echo "Step 2: Installing system dependencies"
echo "=================================================="
sudo apt install -y \
  python3 \
  python3-pip \
  python3-venv \
  git \
  wget \
  curl \
  build-essential \
  cmake \
  libopencv-dev \
  net-tools \
  lsof

# Step 3: Check ROCm installation
echo ""
echo "=================================================="
echo "Step 3: Checking ROCm installation"
echo "=================================================="

if command -v rocm-smi &> /dev/null; then
  echo "✓ ROCm already installed"
  rocm-smi --showproductname | head -5
else
  echo "❌ ROCm not installed!"
  echo ""
  echo "Please install ROCm manually:"
  echo "Ubuntu 22.04:"
  echo "  wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/jammy/amdgpu-install_6.0.60000-1_all.deb"
  echo "  sudo apt install ./amdgpu-install_6.0.60000-1_all.deb"
  echo "  sudo amdgpu-install --usecase=rocm"
  echo ""
  echo "Ubuntu 24.04:"
  echo "  wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/noble/amdgpu-install_6.0.60000-1_all.deb"
  echo "  sudo apt install ./amdgpu-install_6.0.60000-1_all.deb"
  echo "  sudo amdgpu-install --usecase=rocm"
  echo ""
  echo "Then re-run this script."
  exit 1
fi

# Step 4: Install Python packages
echo ""
echo "=================================================="
echo "Step 4: Installing Python packages"
echo "=================================================="

# Upgrade pip
python3 -m pip install --upgrade pip

# Install PyTorch with ROCm
echo ""
echo "Installing PyTorch with ROCm support..."
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

# Install other requirements
echo ""
echo "Installing other Python packages..."
python3 -m pip install -r requirements.txt

# Verify PyTorch + ROCm
echo ""
echo "Verifying PyTorch + ROCm installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU count: {torch.cuda.device_count()}')
else:
    print('WARNING: GPU not detected!')
"

# Step 5: Install RKNN-Toolkit2
echo ""
echo "=================================================="
echo "Step 5: Installing RKNN-Toolkit2"
echo "=================================================="

RKNN_DIR="/tmp/rknn-toolkit2"

if python3 -c "from rknn.api import RKNN" 2>/dev/null; then
  echo "✓ RKNN-Toolkit2 already installed"
else
  echo "Downloading RKNN-Toolkit2..."

  if [ -d "$RKNN_DIR" ]; then
    rm -rf "$RKNN_DIR"
  fi

  git clone https://github.com/rockchip-linux/rknn-toolkit2 "$RKNN_DIR"

  echo "Installing RKNN-Toolkit2..."
  cd "$RKNN_DIR/rknn-toolkit2/packages"

  # Find the correct wheel for Python version
  WHEEL=$(ls rknn_toolkit2-*-cp310-cp310-linux_x86_64.whl 2>/dev/null | head -1)

  if [ -z "$WHEEL" ]; then
    echo "⚠ Warning: Could not find RKNN-Toolkit2 wheel for Python 3.10"
    echo "Available wheels:"
    ls rknn_toolkit2-*.whl
    echo ""
    echo "Install manually with:"
    echo "  cd $RKNN_DIR/rknn-toolkit2/packages"
    echo "  pip3 install rknn_toolkit2-<version>-cp3XX-cp3XX-linux_x86_64.whl"
  else
    python3 -m pip install "$WHEEL"
    echo "✓ RKNN-Toolkit2 installed"
  fi

  cd -
fi

# Step 6: Configure firewall
echo ""
echo "=================================================="
echo "Step 6: Configuring firewall"
echo "=================================================="

if command -v ufw &> /dev/null; then
  echo "Configuring ufw firewall..."
  sudo ufw allow 5555/tcp comment 'ZeroMQ training data'
  sudo ufw allow 6006/tcp comment 'TensorBoard'
  sudo ufw allow 22/tcp comment 'SSH'
  echo "✓ Firewall configured"
else
  echo "⚠ ufw not installed, skipping firewall configuration"
  echo "Make sure ports 5555 (ZeroMQ) and 6006 (TensorBoard) are open"
fi

# Step 7: Create directory structure
echo ""
echo "=================================================="
echo "Step 7: Creating directory structure"
echo "=================================================="

mkdir -p checkpoints export runs logs
echo "✓ Directories created:"
echo "  - checkpoints/ (saved PyTorch models)"
echo "  - export/      (ONNX and RKNN models)"
echo "  - runs/        (TensorBoard logs)"
echo "  - logs/        (training logs)"

# Step 8: Test installation
echo ""
echo "=================================================="
echo "Step 8: Testing installation"
echo "=================================================="

echo ""
echo "Testing imports..."
python3 << EOF
import sys

errors = []

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    if not torch.cuda.is_available():
        print("  ⚠ GPU not detected")
        errors.append("GPU")
except ImportError as e:
    print(f"✗ PyTorch: {e}")
    errors.append("torch")

try:
    import zmq
    print(f"✓ ZeroMQ {zmq.zmq_version()}")
except ImportError as e:
    print(f"✗ ZeroMQ: {e}")
    errors.append("zmq")

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy: {e}")
    errors.append("numpy")

try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__}")
except ImportError as e:
    print(f"✗ OpenCV: {e}")
    errors.append("opencv")

try:
    import tensorboard
    print(f"✓ TensorBoard {tensorboard.__version__}")
except ImportError as e:
    print(f"✗ TensorBoard: {e}")
    errors.append("tensorboard")

try:
    from rknn.api import RKNN
    print(f"✓ RKNN-Toolkit2")
except ImportError as e:
    print(f"✗ RKNN-Toolkit2: {e}")
    errors.append("rknn-toolkit2")

if errors:
    print(f"\n❌ Missing packages: {', '.join(errors)}")
    sys.exit(1)
else:
    print("\n✅ All packages installed successfully!")
EOF

if [ $? -ne 0 ]; then
  echo ""
  echo "❌ Installation test failed"
  echo "Please check error messages above and install missing packages"
  exit 1
fi

# Final summary
echo ""
echo "=================================================="
echo "✅ V620 Setup Complete!"
echo "=================================================="
echo ""
echo "Installation summary:"
echo "  ✓ System packages"
echo "  ✓ ROCm (GPU driver)"
echo "  ✓ PyTorch with ROCm"
echo "  ✓ ZeroMQ, NumPy, OpenCV, TensorBoard"
echo "  ✓ RKNN-Toolkit2"
echo "  ✓ Firewall configured"
echo "  ✓ Directory structure created"
echo ""
echo "Next steps:"
echo "  1. Copy training code to this machine"
echo "  2. Start training server: ./start_v620_server.sh"
echo "  3. Connect rover for data collection"
echo ""
echo "Quick start:"
echo "  ./start_v620_server.sh          # Start server on port 5555"
echo "  ./start_v620_server.sh 5556     # Use custom port"
echo ""
echo "Verify GPU:"
echo "  rocm-smi"
echo "  python3 -c 'import torch; print(torch.cuda.get_device_name(0))'"
echo ""
echo "Access TensorBoard:"
echo "  http://$(hostname -I | awk '{print $1}'):6006"
echo ""
