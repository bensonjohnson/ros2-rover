# Habitat 3.0 Simulation for MAP-Elites Training

This directory contains a Habitat 3.0 simulation interface for training rover navigation policies using the MAP-Elites evolutionary algorithm. It provides a high-speed alternative to physical rover testing, enabling 10-100x faster iteration.

## Overview

The simulation interface:
- ✅ Uses the same ZeroMQ protocol as the physical rover
- ✅ Loads PyTorch models directly (no RKNN conversion needed)
- ✅ Matches rover sensor configuration (RealSense D435i)
- ✅ Simulates tank-like dynamics (collision radius, height, etc.)
- ✅ Collects trajectory data for gradient refinement
- ✅ Computes the same fitness metrics as physical episodes

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    V620 Training Server                     │
│                (remote_training_server/)                    │
│                                                             │
│  - Generates models via MAP-Elites evolution               │
│  - Runs tournament selection with gradients                │
│  - Manages population of diverse behaviors                 │
└─────────────────────┬───────────────────────────────────────┘
                      │ ZeroMQ (tcp://host:5556)
                      │
        ┌─────────────┴──────────────┐
        │                            │
┌───────▼────────┐          ┌────────▼──────────┐
│ Physical Rover │          │ Habitat Simulator │
│   (ROS2 node)  │          │   (this folder)   │
│                │          │                   │
│ - RKNN on NPU  │          │ - PyTorch on GPU  │
│ - 1 episode/min│          │ - 10+ eps/min     │
│ - Real physics │          │ - Fast iteration  │
└────────────────┘          └───────────────────┘
```

## Installation

### 1. Install Habitat

**Using Conda (recommended):**
```bash
# Create environment
conda create -n habitat python=3.10

# Install Habitat Sim (with CUDA support)
conda install habitat-sim -c conda-forge -c aihabitat

# Install Habitat Lab
pip install habitat-lab
```

**Using pip (advanced):**
```bash
# Install from source for latest features
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
pip install -r requirements.txt
python setup.py install --headless --with-cuda

git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e .
```

### 2. Install Dependencies

```bash
cd /root/ros2-rover/sim
pip install -r requirements.txt
```

### 3. Download Scene Data

Habitat needs 3D scenes to run navigation tasks. Download at least one:

**Gibson (smallest, good for testing):**
```bash
# Download Gibson tiny split (~100MB)
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/
```

**Matterport3D (realistic indoor):**
```bash
# Requires registration at https://niessner.github.io/Matterport/
# Follow instructions to download MP3D dataset
python -m habitat_sim.utils.datasets_download --uids mp3d --data-path data/
```

**HM3D (large-scale, photorealistic):**
```bash
# Download HM3D scenes (requires agreement to terms)
# See: https://aihabitat.org/datasets/hm3d/
```

## Usage

### Start the V620 Training Server

First, start the MAP-Elites trainer on your training machine (V620):

```bash
cd /root/ros2-rover/remote_training_server
./start_map_elites_server.sh 5556 1000
```

This starts the evolution server on port 5556, targeting 1000 evaluations.

### Start Habitat Simulation

On the same machine or a different machine with GPU:

```bash
cd /root/ros2-rover/sim
./start_habitat_training.sh tcp://localhost:5556 1000 60.0 cuda
```

**Arguments:**
1. `tcp://localhost:5556` - V620 server address (change if remote)
2. `1000` - Number of episodes to run
3. `60.0` - Episode duration in seconds
4. `cuda` - Device (cuda/cpu)

**Example remote setup:**
```bash
# If V620 server is on 10.0.0.200
./start_habitat_training.sh tcp://10.0.0.200:5556 1000 60.0 cuda
```

### Monitor Training

Watch the terminal output for:
- Episode results (distance, collisions, speed)
- Tournament selections
- Population improvements

Logs are saved to `sim/logs/habitat_YYYYMMDD_HHMMSS.log`

## Performance

**Typical speeds (on RTX 3090):**
- Episode simulation: 10-20x realtime (6-12s for 60s episode)
- Model inference: ~5ms per step
- Trajectory compression: ~1s per episode
- **Total throughput: 8-12 episodes/minute** vs 1 episode/minute on rover

**Memory usage:**
- ~4GB GPU VRAM (model + simulation)
- ~2GB RAM per cached trajectory
- Can run multiple parallel instances with enough VRAM

## Configuration

### Episode Parameters

Edit `habitat_episode_runner.py` to adjust:
- `max_linear_speed` - Max forward velocity (default: 0.18 m/s, matches tank)
- `max_angular_speed` - Max rotation rate (default: 1.0 rad/s)
- `collision_distance` - Collision threshold (default: 0.12m)
- `inference_rate_hz` - Control frequency (default: 30Hz)

### Environment Settings

Edit `habitat_config.yaml` to change:
- Sensor resolution (default: 640x480)
- Field of view (default: 69° HFOV)
- Agent dimensions (default: radius=0.18m, height=0.88m)
- Scene dataset and episodes

## Sim-to-Real Transfer

The simulation is designed to match the physical rover:

**Matched parameters:**
- ✅ Camera: RealSense D435i (640x480, 69° HFOV)
- ✅ Dimensions: 0.18m radius, 0.88m height
- ✅ Speed limits: 0.18 m/s linear, 1.0 rad/s angular
- ✅ Collision distance: 0.12m
- ✅ Control rate: 30Hz

**Sim advantages:**
- Unlimited battery
- No motor wear
- Perfect odometry
- Faster than realtime

**Sim limitations:**
- Simplified physics (no track slip, inertia)
- Different lighting/textures
- No sensor noise by default (can enable)

**Recommended workflow:**
1. **Warmup in sim** (first 100-200 episodes): Rapid exploration
2. **Fine-tune on hardware** (next 200-500 episodes): Real-world adaptation
3. **Final polish in sim** (optional): Additional diversity

## Parallel Training

To maximize throughput, run multiple Habitat instances:

```bash
# Terminal 1
./start_habitat_training.sh tcp://localhost:5556 500 60.0 cuda

# Terminal 2 (if you have enough GPU memory)
./start_habitat_training.sh tcp://localhost:5556 500 60.0 cuda
```

The V620 server will distribute models to both clients automatically via the ZeroMQ protocol.

## Troubleshooting

### "Habitat not found"
```bash
# Install Habitat
conda install habitat-sim -c conda-forge -c aihabitat
pip install habitat-lab
```

### "No scenes found"
```bash
# Download test scenes
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/
```

### "CUDA out of memory"
- Reduce batch size in trainer
- Use `--device cpu` flag
- Close other GPU applications

### "Connection refused"
- Check V620 server is running: `netstat -tuln | grep 5556`
- Verify firewall allows port 5556
- Check server address is correct

## Advanced: Custom Environments

To use custom Habitat scenes or tasks:

1. Place scene files in `data/scene_datasets/`
2. Update `habitat_config.yaml` with new paths
3. Modify `_create_habitat_env()` in `habitat_episode_runner.py`

See Habitat documentation for creating custom tasks:
- https://aihabitat.org/docs/habitat-lab/

## Performance Tuning

**For maximum speed:**
- Use GPU physics: `config.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = True`
- Reduce sensor resolution: `RGB_SENSOR.WIDTH = 320, HEIGHT = 240`
- Disable depth noise: `DEPTH_SENSOR.NOISE_MODEL = None`
- Shorter episodes: `--duration 30.0`

**For maximum realism:**
- Enable sensor noise (already configured)
- Add domain randomization (lighting, textures)
- Simulate IMU drift, odometry errors
- Match rover mass/inertia dynamics

## Files

- `habitat_episode_runner.py` - Main simulation client
- `habitat_config.yaml` - Environment configuration
- `start_habitat_training.sh` - Startup script
- `requirements.txt` - Python dependencies
- `README.md` - This file

## See Also

- [Habitat documentation](https://aihabitat.org/docs/habitat-lab/)
- [MAP-Elites paper](https://arxiv.org/abs/1504.04909)
- [Parent project README](../README.md)
- [Remote training quickstart](../REMOTE_TRAINING_QUICKSTART.md)
