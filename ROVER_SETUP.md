# Rover Setup Guide

Complete setup instructions for the ROS2 rover to enable remote training.

## Prerequisites

- ROS2 Jazzy installed
- RK3588 platform (for NPU support)
- Network connectivity to V620 server

## Installation Steps

### 1. Install ZeroMQ (Python)

ZeroMQ is needed for streaming data to the V620 training server.

```bash
# Simple installation via pip
pip3 install pyzmq

# Or with apt (includes C library)
sudo apt update
sudo apt install python3-zmq

# Verify installation
python3 -c "import zmq; print(f'ZeroMQ {zmq.zmq_version()} installed')"
```

### 2. Install RKNN-Toolkit-Lite2 (for model conversion)

RKNN-Toolkit-Lite2 is needed to convert ONNX models to RKNN format on the rover.

```bash
# Clone RKNN toolkit repository
cd ~
git clone https://github.com/rockchip-linux/rknn-toolkit2

# Install RKNNLite runtime library
cd rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/aarch64
sudo cp librknnrt.so /usr/lib/

# Install RKNN-Toolkit-Lite2 Python package
cd ~/rknn-toolkit2/rknn-toolkit-lite2/packages
pip3 install rknn_toolkit_lite2-*-cp310-cp310-linux_aarch64.whl

# Verify installation
python3 -c "from rknnlite.api import RKNNLite; print('RKNNLite installed')"
```

**Note:** The wheel filename may vary based on your Python version. Use `python3 --version` to check.

For Python 3.11:
```bash
pip3 install rknn_toolkit_lite2-*-cp311-cp311-linux_aarch64.whl
```

### 3. Install Additional Dependencies

```bash
# OpenCV (for image processing)
pip3 install opencv-python

# NumPy (for array operations)
pip3 install numpy

# ROS2 cv_bridge (if not already installed)
sudo apt install ros-jazzy-cv-bridge

# Verify
python3 -c "import cv2; import numpy as np; print('Dependencies OK')"
```

### 4. Build ROS2 Workspace

```bash
cd ~/Documents/ros2-rover

# Build required packages
colcon build --packages-select tractor_bringup tractor_control tractor_sensors

# Source workspace
source install/setup.bash
```

### 5. Verify NPU (Optional)

Check that the RK3588 NPU is accessible:

```bash
# Check NPU version
cat /sys/kernel/debug/rknpu/version

# Check NPU load (should show 0% if idle)
cat /sys/kernel/debug/rknpu/load

# Example output:
# NPU load:  Core0:  0%, Core1:  0%, Core2:  0%
```

## Quick Installation Script

Or use this one-liner to install everything:

```bash
# Install all dependencies
sudo apt update && \
sudo apt install -y python3-zmq && \
pip3 install pyzmq opencv-python numpy && \
cd ~ && \
git clone https://github.com/rockchip-linux/rknn-toolkit2 && \
cd rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/aarch64 && \
sudo cp librknnrt.so /usr/lib/ && \
cd ~/rknn-toolkit2/rknn-toolkit-lite2/packages && \
pip3 install rknn_toolkit_lite2-*-cp3*-cp3*-linux_aarch64.whl && \
cd ~/Documents/ros2-rover && \
colcon build --packages-select tractor_bringup tractor_control tractor_sensors && \
source install/setup.bash && \
echo "✅ Rover setup complete!"
```

## Verify Installation

Run this verification script to check all components:

```bash
python3 << 'EOF'
import sys

print("Checking rover dependencies...")
print()

errors = []

# Check ZeroMQ
try:
    import zmq
    print(f"✓ ZeroMQ {zmq.zmq_version()}")
except ImportError as e:
    print(f"✗ ZeroMQ not installed")
    errors.append("pyzmq")

# Check RKNN
try:
    from rknnlite.api import RKNNLite
    print(f"✓ RKNNLite")
except ImportError as e:
    print(f"✗ RKNNLite not installed")
    errors.append("rknn-toolkit-lite2")

# Check OpenCV
try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__}")
except ImportError as e:
    print(f"✗ OpenCV not installed")
    errors.append("opencv-python")

# Check NumPy
try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy not installed")
    errors.append("numpy")

# Check cv_bridge (ROS2)
try:
    from cv_bridge import CvBridge
    print(f"✓ cv_bridge (ROS2)")
except ImportError as e:
    print(f"✗ cv_bridge not installed")
    errors.append("ros-jazzy-cv-bridge")

# Check ROS2
try:
    import rclpy
    print(f"✓ rclpy (ROS2)")
except ImportError as e:
    print(f"✗ ROS2 not sourced or installed")
    errors.append("ROS2")

print()
if errors:
    print(f"❌ Missing packages: {', '.join(errors)}")
    print()
    print("Install with:")
    if 'pyzmq' in errors:
        print("  pip3 install pyzmq")
    if 'opencv-python' in errors:
        print("  pip3 install opencv-python")
    if 'numpy' in errors:
        print("  pip3 install numpy")
    if 'rknn-toolkit-lite2' in errors:
        print("  See RKNN-Toolkit-Lite2 installation instructions above")
    if 'ros-jazzy-cv-bridge' in errors:
        print("  sudo apt install ros-jazzy-cv-bridge")
    if 'ROS2' in errors:
        print("  source /opt/ros/jazzy/setup.bash")
    sys.exit(1)
else:
    print("✅ All dependencies installed!")
    print()
    print("Ready to:")
    print("  - Collect data: ./start_remote_training_collection.sh tcp://V620_IP:5555")
    print("  - Run inference: ./start_remote_trained_inference.sh")
EOF
```

## Network Configuration

### Test Connection to V620

```bash
# Test network connectivity
ping -c 3 V620_IP

# Test ZeroMQ connection
python3 << EOF
import zmq
import sys

try:
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUSH)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect('tcp://V620_IP:5555')
    print('✓ Connected to V620 server')
    socket.close()
    ctx.term()
except Exception as e:
    print(f'✗ Failed to connect: {e}')
    sys.exit(1)
EOF
```

Replace `V620_IP` with your actual V620 server IP address.

### Firewall (if needed)

```bash
# Allow outgoing connections (usually not needed)
# But if you have strict firewall rules:
sudo ufw allow out to V620_IP port 5555
```

## File Structure After Setup

```
~/Documents/ros2-rover/
├── src/
│   ├── tractor_bringup/
│   ├── tractor_control/
│   └── tractor_sensors/
├── build/
├── install/
├── calibration_data/              # Created during data collection
├── models/                        # Created for RKNN models
├── logs/                          # Created for logs
├── convert_onnx_to_rknn.sh       # Conversion script
├── start_remote_training_collection.sh
└── start_remote_trained_inference.sh

~/rknn-toolkit2/                   # RKNN toolkit (installed separately)
```

## Common Issues

### "pip3 install pyzmq fails"

If pip installation fails, try with apt:
```bash
sudo apt install python3-zmq
```

### "RKNNLite wheel not found"

Check your Python version:
```bash
python3 --version
# Python 3.10.x → use cp310 wheel
# Python 3.11.x → use cp311 wheel
```

List available wheels:
```bash
cd ~/rknn-toolkit2/rknn-toolkit-lite2/packages
ls -lh *.whl
```

### "librknnrt.so not found"

```bash
# Copy runtime library
cd ~/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/aarch64
sudo cp librknnrt.so /usr/lib/
sudo ldconfig
```

### "cv_bridge not found"

```bash
# Install ROS2 cv_bridge
sudo apt install ros-jazzy-cv-bridge

# Source ROS2
source /opt/ros/jazzy/setup.bash
source ~/Documents/ros2-rover/install/setup.bash
```

## Next Steps

After setup is complete:

1. **Test data collection**:
   ```bash
   ./start_remote_training_collection.sh tcp://V620_IP:5555
   ```

2. **Check calibration data** (after 10-20 minutes):
   ```bash
   ls -lh calibration_data/
   ```

3. **Convert a model** (after receiving ONNX from V620):
   ```bash
   ./convert_onnx_to_rknn.sh models/model.onnx
   ```

4. **Run inference**:
   ```bash
   ./start_remote_trained_inference.sh
   ```

## Summary

**Essential packages:**
- ✅ `pyzmq` - For streaming data to V620
- ✅ `rknn-toolkit-lite2` - For ONNX→RKNN conversion
- ✅ `opencv-python` - For image processing
- ✅ `numpy` - For array operations
- ✅ `ros-jazzy-cv-bridge` - For ROS2 image conversion

**Installation time:** ~5-10 minutes

**Disk space:** ~500 MB (mostly RKNN toolkit)
