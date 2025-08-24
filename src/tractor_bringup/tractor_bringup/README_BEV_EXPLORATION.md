# Bird's Eye View (BEV) Exploration System

## Overview

This document describes the Bird's Eye View (BEV) based exploration system for the ROS2 rover. This system replaces the previous depth-image-based approach with a more sophisticated point cloud processing pipeline that generates multi-channel BEV maps for neural network input.

## Key Features

### 1. Point Cloud to BEV Conversion
- Converts 3D point clouds from the RealSense D435i to 2D Bird's Eye View maps
- Implements ground plane removal using RANSAC algorithm
- Generates multi-channel BEV maps with:
  - Height slices (multiple elevation layers)
  - Maximum height channel
  - Point density channel

### 2. Multi-Channel BEV Maps
The system generates multi-channel BEV images that provide rich spatial information:

1. **Height Slices**: Multiple channels representing different height ranges
   - Channel 1 (0-20 cm): Low-lying obstacles like curbs or debris
   - Channel 2 (20-100 cm): Medium-height obstacles like benches or fire hydrants
   - Channel 3 (100+ cm): Tall obstacles like walls or people

2. **Maximum Height Channel**: Shows the highest point in each cell for obstacle clearance

3. **Point Density Channel**: Represents the number of points in each cell to distinguish between solid and sparse obstacles

### 3. Ground Plane Removal
- Uses RANSAC algorithm to identify and remove ground plane points
- Improves obstacle detection by focusing on non-ground objects
- Configurable parameters for RANSAC iterations and distance threshold

## System Architecture

### Components
1. **BEV Generator** (`bev_generator.py`)
   - Point cloud preprocessing and filtering
   - Ground plane segmentation with RANSAC
   - BEV map generation with configurable parameters

2. **BEV Neural Network** (`rknn_trainer_bev.py`)
   - CNN architecture optimized for multi-channel BEV input
   - Experience replay with prioritized sampling
   - RKNN conversion for NPU acceleration

3. **Exploration Controller** (`npu_exploration_bev.py`)
   - Main ROS2 node for BEV-based exploration
   - Integration with motor control and odometry
   - Safety monitoring and recovery behaviors

4. **Launch System** (`npu_exploration_bev.launch.py`)
   - Configurable launch file with BEV-specific parameters
   - Integration with existing hardware components

### Data Flow
1. RealSense D435i captures point cloud data
2. BEV Generator processes point cloud into multi-channel BEV maps
3. Neural network uses BEV maps and proprioceptive data to generate control commands
4. Control commands are sent to motor controller with safety monitoring
5. Experience is collected for training and model improvement

## Configuration Parameters

### BEV Generation
- `bev_size`: [height, width] in pixels (default: [200, 200])
- `bev_range`: [x_range, y_range] in meters (default: [10.0, 10.0])
- `bev_height_channels`: Height thresholds for channels (default: [0.2, 1.0])
- `enable_ground_removal`: Enable ground plane removal (default: true)
- `ground_ransac_iterations`: RANSAC iterations (default: 100)
- `ground_ransac_threshold`: Distance threshold for ground points (default: 0.05)

### Neural Network
- Input: Multi-channel BEV maps + proprioceptive data
- Output: Linear velocity, angular velocity, confidence
- Training: Reinforcement Learning with prioritized experience replay

## Usage

### Starting the System
```bash
./start_npu_exploration_bev.sh [mode] [max_speed] [exploration_time] [safety_distance]
```

### Available Modes
- `cpu_training`: Standard PyTorch training
- `hybrid`: RKNN inference + RL training
- `inference`: Pure RKNN inference
- `safe_training`: Anti-overtraining RL protection
- `es_training`: Evolutionary Strategy training
- `es_hybrid`: RKNN inference + ES training
- `es_inference`: Pure RKNN inference with ES model
- `safe_es_training`: Anti-overtraining ES protection

## Advantages Over Depth-Based System

1. **Better Spatial Understanding**: BEV maps provide a comprehensive view of the environment
2. **Improved Obstacle Detection**: Height slices allow for better classification of obstacles
3. **Ground Plane Removal**: Eliminates ground points that can interfere with navigation
4. **Enhanced Path Planning**: Multi-channel input provides richer information for decision making
5. **Robustness**: Less susceptible to depth sensor noise and artifacts

## Future Improvements

1. **Dynamic BEV Resolution**: Adjust BEV resolution based on speed and environment complexity
2. **Temporal Integration**: Incorporate temporal information from consecutive BEV frames
3. **Semantic Segmentation**: Add semantic information to BEV channels
4. **Adaptive Thresholds**: Dynamically adjust height thresholds based on environment
5. **Multi-Sensor Fusion**: Integrate additional sensors for enhanced perception
