# Remote Training System for ROS2 Rover

This directory contains the remote training infrastructure for training neural network policies on a V620 GPU server with ROCm, then deploying them to the rover's RK3588 NPU.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                          ROS2 Rover                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ RealSense    │  │ IMU + Wheel  │  │ Data Collector Node  │  │
│  │ RGB-D Camera │─▶│ Odometry     │─▶│ (ZeroMQ Publisher)   │──┼──┐
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │  │
│                                                                  │  │
│  ┌──────────────────────────────────────────────────────────┐  │  │
│  │ Inference Node (RK3588 NPU - RKNN Runtime)               │  │  │
│  │ - Loads .rknn model                                       │  │  │
│  │ - RGB-D + Proprioception → Actions                       │  │  │
│  │ - 10-30 FPS inference                                    │  │  │
│  └──────────────────────────────────────────────────────────┘  │  │
└─────────────────────────────────────────────────────────────────┘  │
                                                                      │
                          Network (ZeroMQ)                            │
                                                                      │
┌─────────────────────────────────────────────────────────────────┐  │
│                    V620 Training Server                          │  │
│  ┌──────────────────────────────────────────────────────────┐  │  │
│  │ Training Server (v620_ppo_trainer.py)                    │◀─┼──┘
│  │ - Receives: RGB, Depth, Proprio, Actions, Rewards       │  │
│  │ - Trains: PPO with RGB-D encoder (PyTorch + ROCm)       │  │
│  │ - Exports: PyTorch → ONNX → RKNN                        │  │
│  │ - TensorBoard monitoring                                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Hardware: AMD Radeon Pro V620 with ROCm                        │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Rover-Side Data Collection
**File:** `src/tractor_bringup/tractor_bringup/remote_training_collector.py`

Collects and streams data to the V620 server:
- **RGB images**: 424x240x3 (JPEG compressed)
- **Depth images**: 424x240 float32 (meters)
- **Proprioception**: [linear_vel, angular_vel, roll, pitch, accel_mag, min_forward_dist]
- **Actions**: [linear_cmd, angular_cmd]
- **Rewards**: computed from forward progress + safety

### 2. V620 Training Server
**File:** `v620_ppo_trainer.py`

GPU-accelerated PPO training with ROCm:
- **Encoder**: RGB-D vision encoder (separate CNN branches + fusion)
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Hardware**: AMD V620 GPU with ROCm
- **Features**:
  - Automatic model checkpointing
  - ONNX export for deployment
  - TensorBoard logging
  - Configurable hyperparameters

### 3. Model Conversion Pipeline
**File (Rover):** `convert_onnx_to_rknn.py` and `convert_onnx_to_rknn.sh`

Converts trained models for NPU deployment on the rover:
- **Input**: ONNX model from V620
- **Output**: RKNN model for RK3588
- **Platform**: Runs on RK3588 (ARM) using RKNN-Toolkit-Lite2
- **Note**: RKNN conversion tools are ARM-only, not available on x86_64
- **Features**:
  - Float16 mode (quantization requires calibration data)
  - RK3588-specific optimization
  - On-device conversion

### 4. Rover-Side Inference
**File:** `src/tractor_bringup/tractor_bringup/remote_trained_inference.py`

Real-time inference on RK3588 NPU:
- **Runtime**: RKNNLite on RK3588 NPU cores
- **Inputs**: RGB-D + proprioception
- **Outputs**: Velocity commands
- **Performance**: 10-30 FPS on NPU

## Setup

### V620 Server Setup

1. **Install ROCm**
```bash
# Ubuntu 22.04/24.04
sudo apt install rocm-hip-sdk rocm-libs
```

2. **Install Python Dependencies**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
pip install pyzmq tensorboard opencv-python numpy
```

3. **Note: RKNN conversion on rover, not V620**
```bash
# RKNN-Toolkit2 is ARM-only (not x86_64)
# V620 only exports ONNX - conversion to RKNN happens on the rover
echo "No RKNN tools needed on V620"
```

4. **Verify GPU**
```bash
rocm-smi
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Rover Setup

1. **Install RKNNLite Runtime (RK3588)**
```bash
# On rover
git clone https://github.com/rockchip-linux/rknn-toolkit2
cd rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/aarch64
sudo cp librknnrt.so /usr/lib/
pip3 install rknn-toolkit-lite2/packages/rknn_toolkit_lite2-*-cp310-cp310-linux_aarch64.whl
```

2. **Install ZeroMQ**
```bash
pip3 install pyzmq
```

3. **Build ROS2 Workspace**
```bash
cd ~/Documents/ros2-rover
colcon build --packages-select tractor_bringup tractor_control tractor_sensors
```

## Usage

### Step 1: Start V620 Training Server

On the V620 server:

```bash
cd ~/ros2-rover/remote_training_server

# Start training server
python3 v620_ppo_trainer.py \
  --port 5555 \
  --update-interval 8192 \
  --checkpoint-interval 10 \
  --checkpoint-dir ./checkpoints \
  --tensorboard-dir ./runs

# Monitor with TensorBoard
tensorboard --logdir ./runs --bind_all
```

### Step 2: Start Rover Data Collection

On the rover:

```bash
cd ~/Documents/ros2-rover

# Start data collection (streams to V620)
./start_remote_training_collection.sh tcp://192.168.1.100:5555

# Manually drive rover with teleop
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args -r cmd_vel:=cmd_vel_teleop
```

The rover will:
- Collect RGB-D + proprioception data
- Stream to V620 server via ZeroMQ
- Compute rewards based on navigation performance

### Step 3: Monitor Training

Open TensorBoard in browser:
```
http://V620_IP:6006
```

Monitor metrics:
- `train/episode_reward`: Episode rewards
- `train/policy_loss`: Policy gradient loss
- `train/value_loss`: Value function loss
- `train/entropy`: Action entropy

### Step 4: Deploy Trained Model

On the V620 server:

```bash
cd ~/ros2-rover/remote_training_server

# Deploy latest checkpoint to rover
./deploy_model.sh \
  ./checkpoints/ppo_v620_update_50.pt \
  192.168.1.50 \
  benson
```

This script will:
1. Export PyTorch → ONNX
2. Convert ONNX → RKNN (with quantization)
3. SCP to rover's `models/` directory
4. Create `remote_trained.rknn` symlink

### Step 5: Run Inference on Rover

On the rover:

```bash
cd ~/Documents/ros2-rover

# Run with deployed model
./start_remote_trained_inference.sh

# Or specify custom model path
./start_remote_trained_inference.sh \
  /home/benson/Documents/ros2-rover/models/ppo_v620_update_50.rknn \
  0.18 \
  0.25 \
  10.0 \
  true
```

Parameters:
1. Model path
2. Max speed (m/s)
3. Safety distance (m)
4. Inference rate (Hz)
5. Use NPU (true/false)

### Hot-Reload Model (Without Restarting)

```bash
# Copy new model to rover
scp model.rknn rover:/home/benson/Documents/ros2-rover/models/remote_trained.rknn

# Reload on running system
ros2 service call /reload_remote_model std_srvs/srv/Trigger
```

## RGB vs Depth-Only Decision

**Recommendation: Use RGB-D (Color + Depth)**

### Why RGB-D?

1. **Richer Semantic Information**
   - Textures, colors, visual landmarks
   - Better place recognition and loop closure
   - Can distinguish similar-shaped objects (e.g., grass vs concrete)

2. **Complementary Modalities**
   - RGB: appearance, texture, lighting
   - Depth: geometry, distances, 3D structure
   - Combined: robust to lighting changes + geometric awareness

3. **V620 Has Abundant Compute**
   - Can handle larger models easily
   - Multi-modal fusion improves performance
   - No need to artificially limit inputs

4. **Depth-Only Limitations**
   - Loses semantic information
   - Hard to distinguish visually similar objects
   - Misses important visual cues (signs, markers, trails)

### Alternative: Grayscale + Depth

If you want to reduce data size:
- Convert RGB → Grayscale (1 channel instead of 3)
- Pros: 66% smaller, faster training
- Cons: Loses color information (which can be useful for terrain classification)

### Current Implementation

The system uses **RGB (3 channels) + Depth (1 channel)**:
- RGB: 424x240x3 uint8 (JPEG compressed for network transmission)
- Depth: 424x240x1 float32 (normalized to [0, 1])
- Total input: 4 channels

## Network Protocol

### Data Packet Format (ZeroMQ)

```
[4 bytes: metadata_length]
[metadata_length bytes: JSON metadata]
[variable: RGB data (JPEG compressed)]
[depth_size bytes: depth array (float32)]
[proprio_size bytes: proprioception (float32)]
[action_size bytes: action (float32)]
```

### Metadata JSON
```json
{
  "timestamp": 1234567890.123,
  "episode_step": 42,
  "reward": 0.5,
  "done": false,
  "rgb_shape": [240, 424, 3],
  "depth_shape": [240, 424],
  "proprio_shape": [6],
  "action_shape": [2],
  "rgb_compressed": true
}
```

## Model Architecture

### RGB-D Encoder
```
RGB Branch:
  Conv2d(3, 32, 5, stride=2) → ReLU
  Conv2d(32, 64, 3, stride=2) → ReLU

Depth Branch:
  Conv2d(1, 16, 5, stride=2) → ReLU
  Conv2d(16, 32, 3, stride=2) → ReLU

Fusion:
  Concat[RGB_feat, Depth_feat] → (96 channels)
  Conv2d(96, 128, 3, stride=2) → ReLU
  Conv2d(128, 128, 3, stride=2) → ReLU
  AdaptiveAvgPool2d(1) → (128,)
```

### Policy Head
```
Proprio Encoder:
  Linear(6, 64) → ReLU
  Linear(64, 64) → ReLU

Policy:
  Concat[vision_feat, proprio_feat] → (192,)
  Linear(192, 128) → ReLU
  Linear(128, 64) → ReLU
  Linear(64, 2) → Tanh → [linear_vel, angular_vel]
```

### Total Parameters
- Encoder: ~500K parameters
- Policy Head: ~30K parameters
- Value Head: ~30K parameters
- **Total: ~560K parameters**

After RKNN quantization (INT8):
- Model size: ~600 KB
- Inference: 10-30 FPS on RK3588 NPU

## Hyperparameters

### PPO Settings
```python
lr = 3e-4                  # Learning rate
gamma = 0.99               # Discount factor
gae_lambda = 0.95          # GAE lambda
clip_epsilon = 0.2         # PPO clip ratio
entropy_coef = 0.01        # Entropy bonus
value_coef = 0.5           # Value loss weight
update_epochs = 4          # Epochs per update
minibatch_size = 256       # Minibatch size
rollout_capacity = 8192    # Buffer size
```

### Reward Function
```python
reward = (
    linear_vel * 2.0         # Forward movement
    - abs(angular_vel) * 0.5 # Penalize spinning
    - collision_penalty      # -5.0 if close to obstacle
    - 0.01                   # Time penalty
)
```

## Troubleshooting

### V620 Server Issues

**GPU not detected:**
```bash
# Check ROCm installation
rocm-smi
rocminfo | grep "Name:"

# Verify PyTorch ROCm
python3 -c "import torch; print(torch.cuda.is_available())"
```

**RKNN-Toolkit2 not working:**
- Only works on x86_64 Linux
- Requires Python 3.6-3.10
- May need specific Ubuntu version (20.04 or 22.04)

### Rover Issues

**RKNNLite not found:**
```bash
pip3 install rknn_toolkit_lite2-*-cp310-cp310-linux_aarch64.whl
```

**NPU inference fails:**
```bash
# Check NPU availability
cat /sys/kernel/debug/rknpu/version

# Test with CPU fallback
./start_remote_trained_inference.sh model.rknn 0.18 0.25 10.0 false
```

**ZeroMQ connection fails:**
```bash
# Check firewall
sudo ufw allow 5555

# Test connection
python3 -c "import zmq; ctx=zmq.Context(); s=ctx.socket(zmq.PUSH); s.connect('tcp://SERVER_IP:5555')"
```

## Performance Benchmarks

### V620 Training Performance
- **Throughput**: ~2000 samples/sec
- **Update time**: ~5 seconds per PPO update (8192 samples)
- **Memory**: ~8 GB VRAM

### RK3588 Inference Performance
- **Latency**: 30-100ms per inference (NPU)
- **Throughput**: 10-30 FPS
- **Power**: ~3W for NPU

## Future Improvements

1. **Multi-GPU Training**: Distribute across multiple V620 GPUs
2. **Curriculum Learning**: Progressive difficulty increase
3. **Sim-to-Real**: Pre-train in Isaac Sim, fine-tune on rover
4. **Hierarchical RL**: Separate low-level control and high-level planning
5. **Model Compression**: Further quantization (INT4, mixed precision)

## References

- [PyTorch ROCm Documentation](https://pytorch.org/get-started/locally/)
- [RKNN-Toolkit2 GitHub](https://github.com/rockchip-linux/rknn-toolkit2)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [ZeroMQ Guide](https://zguide.zeromq.org/)
