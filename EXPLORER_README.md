# Deep Explorer Network — Autonomous Home Mapping with Neural Network

## What This Is

A complete ROS2-based autonomous exploration and mapping system for a skid-steer rover. It fuses **all rover sensors** into a single lightweight neural network that runs on the **RK3588 NPU** (Rockchip) in real-time, learning to explore unknown houses and build metric SLAM maps automatically.

## Architecture

```
                    ┌──────────────────────────────┐
                    │     Deep Explorer Network     │
                    │  (RKNN NPU inference 15Hz)   │
                    └──────┬───────────┬───────────┘
                           │           │
              ┌────────────┘           └────────────┐
              ▼                                      ▼
    ┌──────────────────┐                 ┌──────────────────────┐
    │  Explore Manager  │                 │     RTAB-Map SLAM    │
    │ (frontier-driven) │◄──────────────►│  (metric occupancy   │
    │                  │  /map topic    │       grid  + tf)     │
    └──────────────────┘                 └──────────────────────┘
              │                                      │
              ▼                                      ▼
    ┌──────────────────┐                 ┌──────────────────────┐
    │  Safety Monitor   │                 │   Data Collector     │
    │ (lidar gate)     │                 │ (chunks → ZMQ/disk)  │
    └──────────────────┘                 └──────────┬───────────┘
                                                    │
                                                    ▼
                                         ┌──────────────────────┐
                                         │  Remote GPU Trainer   │
                                         │  (NVIDIA, TD-MPC2)    │
                                         │  PyTorch → ONNX→ RKNN │
                                         └──────────────────────┘
```

## Sensor Fusion

The neural network receives **all of these** every control tick:

| Sensor | Input | Shape | What it encodes |
|--------|-------|-------|-----------------|
| LiDAR | Binned ranges [0,1] | 72 | Openness around the robot (360°) |
| Occupancy Grid | Local crop from **slam_toolbox** (LiDAR SLAM) | 64×64 | Metric map: free/occupied/unknown |
| Depth Camera (optional) | Downsampled depth (RealSense D435i) | 32×32 | Fine obstacle geometry (use `--depth`) |
| Wheel Odometry | vx, vyaw | 2 | Self-motion from encoders |
| IMU | Yaw rate | 1 | Angular velocity (heading change) |
| Place Novelty | Interoceptive novelty | 1 | "Have I been here?" (from place memory) |
| Safety Hold | Gate clamp state | 1 | "Is the safety monitor stopping me?" |

**Output:** `[left_track, right_track] ∈ [-1, 1]` — smooth tank-steering commands.

## Neural Network Architecture

**DeepExplorerNetwork** — a dual-head actor-critic with CNN + MLP encoders:

```
LiDAR(72) ──▶ 1D CNN(8) ──▶ MLP ──▶ 32
OccMap(64²)─▶ 2D CNN(16) ──▶ MLP ──▶ 64        ┌──▶ Actor(head) ──▶ [L,R]
Proprio(5) ─▶ MLP ──▶ 16          ──▶ Fusion ──┤
Depth(32²) ─▶ 2D CNN(8) ──▶ 32        (128)    └──▶ Value(head) ──▶ V(s)
```

- **210K parameters** — fits easily on RK3588 NPU with INT8 quantization
- **Static ONNX graph** — all ops supported by NPU hardware (Conv2D, Gemm, ReLU, Tanh)
- **Dual head** — actor for control, value for TD-learning during training

## Getting Started

### On the Rover (RK3588 with ROS2 Jazzy)

```bash
# 1. Build
cd ~/Documents/ros2-rover
colcon build --packages-select tractor_explorer

# 2. Source
source install/setup.bash

# 3. Start autonomous mapping
./start_explorer_rover.sh auto

# Or data collection mode (for remote training):
./start_explorer_rover.sh collect --server tcp://192.168.1.100:5557
```

### On the Training Server (NVIDIA GPU)

```bash
# Start server with live rover connection
cd remote_training_server
./start_explorer_server.sh --port 5557 --export

# Or train from pre-collected data
./start_explorer_server.sh --data-dir ~/.ros/explorer_chunks/ --export
```

### Deploy Trained Model to Rover

```bash
# On the training server, after training:
# The model is exported to: checkpoints_explorer/explorer_model.rknn

# Copy to rover:
scp checkpoints_explorer/explorer_model.rknn rover:~/.ros/explorer_brain.rknn

# On the rover, the runner auto-detects the RKNN file and loads it on the NPU
```

## How It Maps a House

The system works in phases:

### Phase 1: Exploration (frontier-driven)
1. RTAB-Map SLAM builds a metric occupancy grid from LiDAR + depth
2. The Explore Manager detects frontiers (boundaries between known-free and unknown space)
3. The Deep Explorer Network receives the local occupancy crop + LiDAR + IMU
4. The NN outputs track commands that drive toward frontiers
5. Safety monitor prevents collisions during exploration

### Phase 2: Coverage Completion
1. When no more frontiers exist, the system switches to coverage mode
2. Detects unmapped areas using map entropy and coverage statistics
3. Drives to fill gaps in the known map
4. Reports completion at 95%+ coverage

### Phase 3: Training Iteration
1. All experience is logged (chunks of 64 steps)
2. Shipped to GPU server for training (ZMQ live or rsync)
3. Server trains TD-MPC2-style world model + actor-critic
4. Exports ONNX → RKNN → deployed back to rover
5. Rover swaps model in-place without restart

## Training Pipeline

### Rewards (5-channel)

The reward signal is multi-channel, learned by the world model:

| Channel | Description | Weight |
|---------|-------------|--------|
| Progress | Forward velocity | + |
| Smoothness | -‖a_t - a_{t-1}‖² | small |
| Turn Penalty | -|angular_vel| | small |
| Collision | -1 when safety gate fires | high |
| Coverage Gain | Δ(map_coverage) | high |

### Off-Policy Training

The server uses **TD-MPC2** (temporal difference learning with model-based policy optimization):
1. **World Model**: predicts next latent state, reward, and continuation
2. **Actor**: maximizes Q-value from critic ensemble
3. **Critic (Ensemble of 2)**: TD(λ) learning with target networks and Clipped Double-Q
4. **Target Networks**: EMA update for stable training

## File Structure

```
ros2-rover/
├── start_explorer_rover.sh                    # Rover startup
├── src/tractor_explorer/                      # New ROS2 package
│   ├── setup.py / package.xml / CMakeLists.txt
│   ├── launch/explorer_nn.launch.py
│   ├── config/explorer_params.yaml
│   └── tractor_explorer/
│       ├── deep_explorer_network.py           # Core NN architecture
│       ├── explorer_runner.py                 # ROS2 runner node
│       ├── explore_manager.py                 # High-level mapping coordinator
│       ├── map_integrator.py                  # RTAB-Map bridge
│       ├── data_collector.py                  # Experience logging
│       └── convert_to_rknn.py                 # ONNX → RKNN export
│
├── remote_training_server/
│   ├── start_explorer_server.sh               # Training server startup
│   ├── explorer_trainer.py                    # GPU training server
│   └── requirements.txt                       # Server dependencies
│
└── models/                                    # Deployed RKNN models
```

## Comparison to Prior Approaches

| Approach | Sensors | Learning | Mapping | NPU? |
|----------|---------|----------|---------|------|
| PC Brain (existing) | LiDAR + IMU only | Online PC | Topological (room memory) | No (CPU) |
| PPO (existing) | BEV + Depth | Remote PPO | No | Yes |
| DreamerV3 (existing) | RGB-D + BEV | Remote Dreamer | No | Yes |
| **Deep Explorer (NEW)** | **LiDAR + IMU + Wheels + OccMap + Depth** | **Remote TD-MPC2** | **Automatic SLAM mapping** | **Yes** |

## Requirements

### Rover
- ROS2 Jazzy
- RK3588 SoC (Orange Pi 5 / Radxa Rock 5)
- RKNNLite runtime (`rknnlite.api`)
- PyTorch (for training — optional on rover if using RKNN)
- RealSense D435i (optional, for depth-enhanced mapping)
- STL19P lidar, LSM9DS1/BNO085 IMU, HiWonder motor driver

### Training Server
- NVIDIA GPU (CUDA) or AMD GPU (ROCm)
- PyTorch 2.0+
- rknn-toolkit2 (for RKNN conversion)
- Python 3.10+

## Quick Test

```bash
# 1. On the rover, start in explore mode:
./start_explorer_rover.sh explore --no-slam  # lidar-only, no metric map

# 2. The NN runs random exploration (untrained model)
#    and logs experience to ~/.ros/explorer_chunks/

# 3. Transfer data to server:
rsync -avz ~/.ros/explorer_chunks/ server:~/explorer_data/

# 4. On server, train:
./start_explorer_server.sh --data-dir ~/explorer_data/ --export

# 5. The server exports explorer_model.rknn — copy back to rover
#    On next start, the rover uses the trained model
```
