# Remote Training Quick Start Guide

This guide will help you get started training neural networks on your V620 GPU server and deploying them to your rover's RK3588 NPU.

## Overview

**What you're building:**
- Train a vision-based navigation policy on a powerful V620 GPU
- Use RGB-D camera + IMU + wheel encoders as inputs
- Train faster than on-device training (10-100x speedup)
- Deploy optimized models to rover's NPU for real-time inference

**Key Decision: RGB-D vs Depth-Only**

✅ **Recommended: RGB-D (Color + Depth)**
- Richer semantic information (colors, textures, landmarks)
- Better navigation performance
- V620 has plenty of compute power
- Only slightly more data to transmit (JPEG compression helps)

❌ **Alternative: Depth-Only (Grayscale)**
- Loses semantic information
- Harder to distinguish similar objects
- Saves ~60% bandwidth
- Consider if network is very constrained

**Current implementation uses RGB-D** for best performance.

---

## Prerequisites

### V620 Server Requirements
- Ubuntu 22.04 or 24.04 (x86_64)
- AMD Radeon Pro V620 GPU
- ROCm 5.7+ installed
- Python 3.10
- Network access to rover

### Rover Requirements
- ROS2 Jazzy
- RK3588 SoC with NPU
- RKNNLite runtime installed
- Network access to V620 server
- RealSense D435i camera
- IMU (LSM9DS1)
- Wheel encoders

---

## Installation

### 1. V620 Server Setup

**Automated setup (recommended):**
```bash
# Copy remote_training_server directory to V620 machine
cd remote_training_server

# Run setup script (installs everything)
./setup_v620.sh
```

**Manual setup:**
```bash
# Install ROCm
sudo apt update
sudo apt install rocm-hip-sdk rocm-libs

# Install Python packages
pip3 install -r requirements.txt
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

# Install RKNN-Toolkit2 (for model conversion)
git clone https://github.com/rockchip-linux/rknn-toolkit2
cd rknn-toolkit2/rknn-toolkit2/packages
pip3 install rknn_toolkit2-*-cp310-cp310-linux_x86_64.whl

# Verify GPU
rocm-smi
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### 2. Rover Setup

```bash
# Install RKNNLite
git clone https://github.com/rockchip-linux/rknn-toolkit2
cd rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/aarch64
sudo cp librknnrt.so /usr/lib/
cd ../../../rknn-toolkit-lite2/packages
pip3 install rknn_toolkit_lite2-*-cp310-cp310-linux_aarch64.whl

# Install ZeroMQ
pip3 install pyzmq

# Build ROS2 workspace
cd ~/Documents/ros2-rover
colcon build --packages-select tractor_bringup tractor_control tractor_sensors
```

---

## Quick Start (5 Steps)

### Step 1: Start V620 Training Server

On your V620 server:

```bash
cd ~/remote_training_server

# Start training server (handles TensorBoard automatically)
./start_v620_server.sh

# Or with custom settings:
./start_v620_server.sh 5555 8192 10 6006
# Args: ZMQ_PORT UPDATE_INTERVAL CHECKPOINT_INTERVAL TENSORBOARD_PORT
```

The script will:
- Check all dependencies
- Verify GPU is accessible
- Start TensorBoard automatically
- Display access URLs

Access TensorBoard: `http://V620_IP:6006`

### Step 2: Start Rover Data Collection

On your rover:

```bash
cd ~/Documents/ros2-rover

# Update server IP in the script or pass as argument
./start_remote_training_collection.sh tcp://V620_IP:5555
```

### Step 3: Drive Rover to Collect Data

In another terminal on rover:

```bash
# Use keyboard teleop to manually drive
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args -r cmd_vel:=cmd_vel_teleop
```

Drive the rover around for 10-30 minutes to collect diverse data.

### Step 4: Monitor Training

Check TensorBoard for:
- Episode rewards (should increase over time)
- Policy loss (should decrease)
- Value loss (should stabilize)

Training updates happen every 8192 samples (~13 minutes at 10Hz).

### Step 5: Deploy & Test Model

After 30-50 training updates:

```bash
# On V620 server: Deploy model
cd ~/ros2-rover/remote_training_server
./deploy_model.sh \
  ./checkpoints/ppo_v620_update_50.pt \
  ROVER_IP \
  ROVER_USERNAME

# On rover: Run inference
cd ~/Documents/ros2-rover
./start_remote_trained_inference.sh
```

Watch your rover navigate autonomously!

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Start Training Server (V620)                             │
│    python3 v620_ppo_trainer.py --port 5555                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Start Data Collection (Rover)                            │
│    ./start_remote_training_collection.sh tcp://V620_IP:5555 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Drive Rover with Teleop (Rover)                          │
│    - Collect 10-30 minutes of data                          │
│    - Vary terrain, obstacles, speeds                        │
│    - Data streams to V620 automatically                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. V620 Trains Model Automatically                          │
│    - Updates every 8192 samples                             │
│    - Saves checkpoints every 10 updates                     │
│    - Exports ONNX models                                    │
│    - Monitor via TensorBoard                                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Deploy Model (V620 → Rover)                              │
│    ./deploy_model.sh checkpoint.pt ROVER_IP USER            │
│    - Converts PyTorch → ONNX → RKNN                         │
│    - SCPs to rover                                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Run Autonomous Navigation (Rover)                        │
│    ./start_remote_trained_inference.sh                      │
│    - Loads RKNN model on NPU                                │
│    - 10-30 FPS inference                                    │
│    - Drives autonomously!                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Network Configuration

### Firewall Setup (V620 Server)

```bash
sudo ufw allow 5555/tcp  # ZeroMQ port
sudo ufw allow 6006/tcp  # TensorBoard
sudo ufw allow 22/tcp    # SSH (for deployment)
```

### Test Network Connection

```bash
# From rover to V620
ping V620_IP

# Test ZeroMQ connection
python3 -c "
import zmq
ctx = zmq.Context()
s = ctx.socket(zmq.PUSH)
s.connect('tcp://V620_IP:5555')
print('Connected!')
"
```

---

## Monitoring & Debugging

### Monitor Data Collection (Rover)

```bash
# Check data stream rate
ros2 topic hz /camera/camera/color/image_raw
ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw

# Check velocities
ros2 topic echo /cmd_vel_ai

# Check safety
ros2 topic echo /min_forward_distance
```

### Monitor Training (V620)

TensorBoard metrics:
- `train/episode_reward`: Higher is better
- `train/policy_loss`: Should decrease
- `train/value_loss`: Should stabilize
- `train/entropy`: Should slowly decrease

Console output:
```
Episode finished | Reward: 15.23 | Avg (100): 12.45
Update 10: Policy Loss=0.0234, Value Loss=1.2345, Entropy=0.5678
```

### Monitor Inference (Rover)

```bash
# Check inference rate
ros2 topic hz /cmd_vel_ai

# Service to reload model
ros2 service call /reload_remote_model std_srvs/srv/Trigger

# Check NPU usage (if available)
cat /sys/kernel/debug/rknpu/load
```

---

## Tuning Tips

### If Training is Unstable:
1. Reduce learning rate: `--lr 1e-4`
2. Increase PPO clip: `--clip-epsilon 0.3`
3. Reduce entropy coefficient: `--entropy-coef 0.005`

### If Rover Behavior is Jerky:
1. Increase inference rate: `inference_rate_hz:=30.0`
2. Smooth actions in post-processing
3. Check safety monitor is not over-triggering

### If Training is Too Slow:
1. Increase batch size: `--minibatch-size 512`
2. Reduce update interval: `--update-interval 4096`
3. Use multiple rover instances (advanced)

### If Model is Too Large:
1. Use aggressive quantization (already INT8)
2. Reduce network size (modify encoder layers)
3. Use knowledge distillation (advanced)

---

## File Structure

```
ros2-rover/
├── remote_training_server/           # V620 server code
│   ├── v620_ppo_trainer.py          # Main training server
│   ├── export_to_rknn.py            # ONNX → RKNN converter
│   ├── deploy_model.sh              # Deployment script
│   ├── README.md                    # Detailed documentation
│   ├── checkpoints/                 # Saved models (.pt)
│   ├── export/                      # Exported models (.onnx, .rknn)
│   └── runs/                        # TensorBoard logs
│
├── src/tractor_bringup/tractor_bringup/
│   ├── remote_training_collector.py # Data collection node
│   └── remote_trained_inference.py  # NPU inference node
│
├── src/tractor_bringup/launch/
│   ├── remote_training_collection.launch.py
│   └── remote_trained_inference.launch.py
│
├── models/                          # Deployed RKNN models
│   ├── remote_trained.rknn         # Symlink to latest
│   └── ppo_v620_update_XX.rknn     # Versioned models
│
├── start_remote_training_collection.sh
├── start_remote_trained_inference.sh
└── REMOTE_TRAINING_QUICKSTART.md   # This file
```

---

## Troubleshooting

### "ZeroMQ not installed"
```bash
pip3 install pyzmq
```

### "ROCm GPU not detected"
```bash
rocm-smi
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # If needed
```

### "RKNN model not found"
```bash
# Make sure deployment script ran successfully
ls -lh models/
# Check symlink
ls -l models/remote_trained.rknn
```

### "NPU inference failed"
```bash
# Check RKNN runtime
python3 -c "from rknnlite.api import RKNNLite; print('OK')"

# Try CPU fallback
./start_remote_trained_inference.sh model.rknn 0.18 0.25 10.0 false
```

### "Cannot connect to V620 server"
```bash
# Check firewall
sudo ufw status
sudo ufw allow 5555

# Check network
ping V620_IP
telnet V620_IP 5555
```

---

## Next Steps

Once your basic setup works:

1. **Collect More Data**: Drive in diverse environments
2. **Tune Rewards**: Modify reward function for better behavior
3. **Experiment with Inputs**: Try different observation configurations
4. **Add Exploration Bonus**: Encourage visiting new areas
5. **Multi-Task Learning**: Train for multiple objectives
6. **Sim-to-Real**: Pre-train in simulator, fine-tune on real robot

---

## Support & Documentation

- **Detailed README**: `remote_training_server/README.md`
- **Architecture Details**: See main README sections on RGB-D encoder, PPO, etc.
- **ROCm Docs**: https://rocm.docs.amd.com/
- **RKNN Toolkit**: https://github.com/rockchip-linux/rknn-toolkit2

---

## Summary

You now have a complete remote training pipeline:

✅ Data collection from rover (RGB-D + proprioception)
✅ GPU-accelerated training on V620 (10-100x faster)
✅ Model export and quantization (PyTorch → ONNX → RKNN)
✅ NPU inference on rover (10-30 FPS)
✅ Monitoring via TensorBoard
✅ Hot-reload capability for rapid iteration

**Time to first autonomous drive: ~2-4 hours**
- 30 min setup
- 30 min data collection
- 1-2 hours training (30-50 updates)
- 10 min deployment
- Test!

Happy training! 🚀
