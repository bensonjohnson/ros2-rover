# Webots Simulation for SAC Rover

This package provides a Webots simulation environment for training the SAC rover policy without physical hardware.

## Prerequisites

1. **Webots R2023b+** (installed via snap)
   ```bash
   sudo snap install webots
   ```

2. **webots_ros2** for ROS 2 Jazzy (build from source recommended)
   ```bash
   cd ~/ros2_ws/src
   git clone https://github.com/cyberbotics/webots_ros2.git
   cd ..
   rosdep install --from-paths src --ignore-src -r -y
   colcon build --packages-select webots_ros2
   ```

## Quick Start

### Option 1: Docker (Recommended)

Webots runs on host (native Wayland), ROS 2 runs in Docker container:

```bash
cd /home/benson/Documents/ros2-rover
./start_webots_sim.sh --docker
```

First run will build the Docker image (~2-3 minutes).

### Option 2: Native ROS 2 Jazzy

If you have ROS 2 Jazzy installed locally:

```bash
cd /home/benson/Documents/ros2-rover
colcon build --packages-select webots_sim tractor_bringup
source install/setup.bash
./start_webots_sim.sh
```

### Verify Topics

In another terminal (or inside Docker container):
```bash
# If using Docker:
docker exec -it ros2_webots_sim bash

# Then:
ros2 topic list
ros2 topic echo /scan --once
```

### Test Driving
```bash
ros2 topic pub /cmd_vel_ai geometry_msgs/Twist "{linear: {x: 0.1}}" -r 10
```

## Architecture

```
sim/
├── protos/TractorRover.proto   # Robot model (D435i depth, LD19 LiDAR, IMU)
├── worlds/training_arena.wbt   # 5×5m arena with obstacles
├── controllers/ros2_driver/    # ROS 2 ↔ Webots bridge
└── launch/webots_sim.launch.py # ROS 2 launch file
```

## Topic Mapping

| Topic | Type | Description |
|-------|------|-------------|
| `/cmd_vel_ai` | Twist | Velocity commands (subscribed) |
| `/scan` | LaserScan | 360° LiDAR (360 samples, 12m range) |
| `/camera/camera/depth/image_rect_raw` | Image (16UC1) | Depth camera (848×100, uint16 mm) |
| `/imu/data` | Imu | IMU with orientation, gyro, accel |
| `/odometry/filtered` | Odometry | Ground truth odometry (GPS-based) |
| `/joint_states` | JointState | Wheel positions and velocities |
| `/safety_monitor_status` | Bool | Always False in sim |

## Integration with SAC Training

The simulation publishes identical topics to the real rover, so `sac_episode_runner.py` works with minimal changes:

1. **For local testing** (no NATS):
   - Modify `sac_episode_runner.py` to skip NATS initialization
   - Or create a mock NATS server

2. **For parallel training**:
   - Run multiple Webots instances with different ROS namespaces
   - All instances can send data to the same V620 training server

## Customization

### Adding obstacles
Edit `worlds/training_arena.wbt` to add more obstacles. Supported types:
- Box (`geometry Box { size x y z }`)
- Cylinder (`geometry Cylinder { height h radius r }`)

### Domain randomization (future)
- Randomize obstacle positions at episode reset
- Vary floor textures and lighting
- Add physics perturbations (friction, mass)

## Troubleshooting

**Webots snap permissions**
```bash
sudo snap connect webots:home
```

**Controller not found**
Ensure the controller path matches:
```
sim/controllers/ros2_driver/ros2_driver.py
```

**ROS 2 topics not appearing**
Check that the Webots controller is running in the simulation window (Webots → Show → Console).
