# Corrected Remote Training Workflow

## Important Clarification

**RKNN-Toolkit2 is ARM-only** and designed to run on the RK3588, **not on x86_64**.

The correct workflow is:

```
V620 (x86_64)          →  Rover (RK3588 ARM)
─────────────────────     ────────────────────────
1. Train PyTorch model
2. Export to ONNX      →  3. Receive ONNX via SCP
                          4. Convert ONNX → RKNN
                          5. Load RKNN on NPU
                          6. Run inference
```

## Updated Architecture

### V620 Server (x86_64)
- **Role**: Training only
- **Output**: ONNX models
- **Tools**: PyTorch, ROCm, ZeroMQ, TensorBoard
- **NOT needed**: RKNN-Toolkit2 (ARM-only)

### Rover (RK3588 ARM)
- **Role**: Data collection, conversion, inference
- **Input**: ONNX from V620
- **Output**: Velocity commands
- **Tools**: RKNN-Toolkit-Lite2, RKNNLite, ROS2

## Deployment Workflow

### Step 1: Train on V620

```bash
# On V620
cd ~/remote_training_server
./start_v620_server.sh
```

Training server will:
- Receive data from rover via ZeroMQ
- Train PPO model with ROCm
- Auto-export ONNX every N checkpoints
- Log to TensorBoard

### Step 2: Deploy ONNX to Rover

```bash
# On V620
./deploy_model.sh \
  checkpoints/ppo_v620_update_50.pt \
  ROVER_IP \
  ROVER_USER
```

This script will:
1. Export PyTorch → ONNX on V620
2. SCP ONNX file to rover
3. SSH to rover and run conversion script
4. Convert ONNX → RKNN on rover
5. Create symlink to latest model

### Step 3: Run Inference on Rover

```bash
# On rover (automatic after deployment)
# Or manually:
cd ~/Documents/ros2-rover
./start_remote_trained_inference.sh
```

## Manual Conversion (If Needed)

If automatic deployment fails, manually convert on rover:

```bash
# On rover
cd ~/Documents/ros2-rover

# Convert ONNX → RKNN
./convert_onnx_to_rknn.sh models/ppo_v620_update_50.onnx

# This creates:
#   models/ppo_v620_update_50.rknn
#   models/remote_trained.rknn (symlink)
```

## Why RKNN Conversion Must Be On Rover

1. **RKNN-Toolkit2 is ARM-only**
   - Requires aarch64 (ARM64) architecture
   - Does not have x86_64 wheels
   - Trying to install on x86_64 gives: "not a supported wheel on this platform"

2. **Platform-specific compilation**
   - RKNN models are compiled for specific NPU hardware
   - RK3588 NPU driver must be present during conversion
   - On-device conversion ensures compatibility

3. **Available tools per platform**
   ```
   x86_64 (V620):     PyTorch, ONNX export
   ARM (RK3588):      RKNN-Toolkit-Lite2, RKNN conversion, NPU inference
   ```

## Model Formats

| Format   | Where Created | Tool                  | Purpose             |
|----------|---------------|-----------------------|---------------------|
| .pt      | V620          | PyTorch               | Training checkpoint |
| .onnx    | V620          | torch.onnx.export     | Portable format     |
| .rknn    | Rover         | RKNN-Toolkit-Lite2    | NPU deployment      |

## Files Updated

### V620 Server
- ✅ `setup_v620.sh` - Removed RKNN-Toolkit2 installation
- ✅ `deploy_model.sh` - Now deploys ONNX + triggers conversion on rover
- ✅ `start_v620_server.sh` - No changes needed
- ✅ `v620_ppo_trainer.py` - Exports ONNX only

### Rover
- ✅ `convert_onnx_to_rknn.sh` - Shell wrapper for conversion
- ✅ `convert_onnx_to_rknn.py` - Python conversion script using RKNNLite
- ✅ `remote_trained_inference.py` - Loads .rknn and runs on NPU

## Troubleshooting

### V620: "RKNN not installed"
✅ **This is correct!** RKNN tools are not needed on V620.

### Rover: "RKNN-Toolkit-Lite2 not installed"
```bash
# Install on rover
git clone https://github.com/rockchip-linux/rknn-toolkit2
cd rknn-toolkit2/rknn-toolkit-lite2/packages
pip3 install rknn_toolkit_lite2-*-cp310-cp310-linux_aarch64.whl
```

### Conversion fails with "NPU not detected"
```bash
# Check NPU
cat /sys/kernel/debug/rknpu/version

# Verify platform
uname -m  # Should show: aarch64
```

## Performance Comparison

### ONNX vs RKNN

| Metric           | ONNX (CPU)  | RKNN (NPU)  |
|------------------|-------------|-------------|
| Inference time   | 200-500ms   | 30-100ms    |
| Power usage      | ~5W         | ~3W         |
| FPS              | 2-5         | 10-30       |
| Recommended      | ❌ No       | ✅ Yes      |

**Always convert to RKNN for deployment!**

## Quick Reference

### V620 Commands
```bash
# Setup (one-time)
./setup_v620.sh

# Start training
./start_v620_server.sh

# Deploy model
./deploy_model.sh checkpoints/model.pt ROVER_IP USER
```

### Rover Commands
```bash
# Convert ONNX → RKNN
./convert_onnx_to_rknn.sh models/model.onnx

# Run inference
./start_remote_trained_inference.sh

# Reload model
ros2 service call /reload_remote_model std_srvs/srv/Trigger
```

## Summary

✅ **Correct workflow:**
- V620: PyTorch training → ONNX export
- Deploy: SCP ONNX to rover
- Rover: ONNX → RKNN conversion → NPU inference

❌ **Incorrect (won't work):**
- V620: RKNN-Toolkit2 installation (ARM-only, not x86_64)
- V620: RKNN conversion (requires RK3588 NPU)

The setup has been corrected to reflect this architecture!
