# NPU Bird's Eye View Exploration System

This document describes the Bird's Eye View (BEV) exploration system for the tank-style steer rover using a RealSense D435i camera.

## Overview

The BEV exploration system replaces the depth image-based neural network with a more sophisticated approach that uses point cloud data to generate bird's eye view maps. This provides the neural network with a richer understanding of the environment's vertical structure, enabling better path planning and obstacle avoidance.

## Key Features

### 1. Bird's Eye View Generation
The system converts point cloud data from the RealSense D435i into multi-channel BEV maps:

- **Maximum Height Channel**: Shows the z-coordinate of the highest point in each cell, excellent for identifying obstacles and required clearance
- **Point Density Channel**: Represents the number of points in each cell, helping distinguish between solid walls and sparse objects
- **Height Slice Channels**: Multiple channels representing specific height ranges (e.g., 0-20cm, 20-100cm, 100+cm) for detailed vertical structure
- **Ground Plane Removal**: Uses RANSAC algorithm to remove ground points, focusing the BEV on actual obstacles

### 2. Neural Network Architecture
The neural network has been modified to accept multi-channel BEV inputs:

- **BEV Processing Branch**: CNN layers process the multi-channel BEV image
- **Sensor Fusion**: Combines BEV features with proprioceptive data (velocity, wheel speeds, etc.)
- **Action Output**: Produces linear velocity, angular velocity, and confidence values

### 3. Training System
The system maintains all the advanced training features of the original system:

- **Reinforcement Learning**: Standard policy gradient training
- **Evolutionary Strategies**: Population-based training with Bayesian optimization
- **Anti-Overtraining Protection**: Behavioral diversity tracking and curriculum learning
- **Multi-Objective Optimization**: Balances performance, safety, efficiency, and robustness

## System Components

### 1. BEV Generator (`bev_generator.py`)
Converts point cloud data into multi-channel bird's eye view maps:
- Point cloud to grid mapping
- Ground plane segmentation using RANSAC
- Multi-channel BEV generation (max height, density, height slices)

### 2. Neural Network Trainer (`rknn_trainer_bev.py`)
Handles training and inference for the BEV-based neural network:
- Modified CNN architecture for BEV inputs
- Experience replay with prioritized sampling
- RKNN conversion for NPU deployment

### 3. Exploration Node (`npu_exploration_bev.py`)
Main ROS2 node that orchestrates the exploration system:
- Point cloud processing and BEV generation
- Neural network inference and training
- Control command generation
- Safety monitoring integration

### 4. Launch File (`npu_exploration_bev.launch.py`)
Configures and launches all system components:
- RealSense camera with point cloud enabled
- BEV exploration node with configurable parameters
- Safety monitoring and training components

### 5. Start Script (`start_npu_exploration_bev.sh`)
User-friendly script to launch the BEV exploration system:
- Interactive mode selection
- Automatic environment setup
- USB power management for RealSense
- Graceful shutdown handling

## Usage

### Starting the System
```bash
./start_npu_exploration_bev.sh
```

### Available Modes
- `cpu_training`: Standard PyTorch RL training
- `hybrid`: RKNN inference + RL training
- `inference`: Pure RKNN inference
- `safe_training`: Anti-overtraining RL protection
- `es_training`: Evolutionary Strategy training
- `es_hybrid`: RKNN inference + ES training
- `es_inference`: Pure RKNN inference with ES model
- `safe_es_training`: Anti-overtraining ES protection

### Configuration Parameters
- `bev_size`: BEV image dimensions [height, width] in pixels (default: [200, 200])
- `bev_range`: BEV coverage area [x_range, y_range] in meters (default: [10.0, 10.0])
- `bev_height_channels`: Height thresholds for channel generation (default: [0.2, 1.0])
- `enable_ground_removal`: Enable/disable ground plane removal (default: true)

## Advantages Over Depth Image Approach

1. **Better Obstacle Understanding**: Multi-channel BEV provides detailed vertical structure information
2. **Improved Path Planning**: Clear distinction between different obstacle types and heights
3. **Reduced Ambiguity**: BEV representation is less ambiguous than depth images for navigation
4. **Enhanced Safety**: Better awareness of clearance requirements for the robot
5. **Robust Ground Handling**: Ground plane removal focuses the system on actual obstacles

## Technical Details

### BEV Generation Process
1. Point cloud data is received from the RealSense D435i
2. Ground plane is segmented and removed using RANSAC
3. Points are mapped to a 2D grid based on x,y coordinates
4. Multiple channels are generated:
   - Maximum height in each cell
   - Point density in each cell
   - Height slice information
5. BEV map is passed to the neural network for processing

### Neural Network Input
- Multi-channel BEV image (typically 4 channels)
- Proprioceptive data including:
  - Current linear and angular velocity
  - Wheel velocities
  - Previous action
  - Distance metrics
  - Emergency status

### Training Process
1. Experience is collected using BEV observations and actions
2. Rewards are calculated based on progress, safety, and exploration
3. Experience is stored in a prioritized replay buffer
4. Training occurs using either RL or ES methods
5. Model is periodically converted to RKNN for NPU deployment

## Monitoring and Debugging

### Status Information
- Current operation mode
- Battery level
- Training progress
- Buffer status
- Average rewards

### Visualization
- Foxglove Bridge integration for real-time monitoring
- BEV map visualization
- Training metrics and performance data

## Troubleshooting

### Common Issues
1. **RealSense Not Detected**: Check USB connections and power management
2. **Point Cloud Quality**: Ensure proper camera positioning and lighting
3. **Training Performance**: Monitor rewards and adjust hyperparameters
4. **NPU Inference**: Verify RKNN model conversion and runtime availability

### Logs and Debugging
- Training logs are saved in the `logs/` directory
- Anti-overtraining logs for safe modes
- Detailed console output for system status
- ROS2 topic monitoring for real-time data
