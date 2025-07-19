# Autonomous Mapping System for ROS2 Tractor

## Overview

This autonomous mapping system enables your ROS2 tractor to drive around independently while avoiding obstacles and creating a detailed map using the RealSense D435i camera. The system integrates SLAM mapping, Nav2 navigation, and intelligent exploration algorithms.

## Features

### 🗺️ **Autonomous Exploration**
- **Waypoint-based exploration**: Systematic coverage of the area
- **Intelligent path planning**: Uses Nav2 for optimal routes
- **Dynamic obstacle avoidance**: Real-time RealSense point cloud detection
- **Adaptive exploration**: Expands search radius as areas are mapped

### 🛡️ **Multi-layer Safety System**
- **Emergency stop**: Immediate halt when obstacles within 0.3m
- **Speed limiting**: Dynamic speed reduction near obstacles
- **Heartbeat monitoring**: Stops if main system fails
- **Redundant sensors**: Both laser scan and RealSense monitoring
- **Safe command filtering**: All movement commands safety-checked

### 📊 **Advanced Mapping**
- **SLAM integration**: Real-time simultaneous localization and mapping
- **Sensor fusion**: GPS, IMU, wheel odometry, and visual data
- **Point cloud mapping**: 3D obstacle detection and mapping
- **Periodic map saving**: Automatic map backups during exploration

### 🎯 **Intelligent Navigation**
- **Nav2 integration**: Professional-grade path planning
- **Dynamic costmaps**: Real-time obstacle map updates
- **Recovery behaviors**: Automatic recovery from navigation failures
- **Goal timeout handling**: Moves to new areas if stuck

## System Components

### 1. Autonomous Mapper (`autonomous_mapping.py`)
**Core exploration and mapping logic**

**Features**:
- Generates exploration waypoints in grid pattern
- Sends navigation goals to Nav2
- Monitors goal completion and timeouts
- Tracks visited areas to avoid repetition
- Publishes status updates

**Key Parameters**:
- `mapping_duration`: Total exploration time (default: 600s)
- `exploration_radius`: Initial search radius (default: 5.0m)
- `max_speed`: Maximum travel speed (default: 0.3 m/s)
- `safety_distance`: Minimum obstacle distance (default: 0.8m)
- `waypoint_timeout`: Time limit per waypoint (default: 45s)

### 2. Safety Monitor (`safety_monitor.py`)
**Multi-layer safety protection system**

**Features**:
- Monitors all sensor inputs for obstacles
- Filters command velocities for safety
- Implements emergency stop functionality
- Provides heartbeat monitoring
- Speed limiting based on obstacle proximity

**Safety Zones**:
- **Emergency Zone** (< 0.3m): Immediate stop
- **Warning Zone** (0.3-0.8m): Speed reduction
- **Safe Zone** (> 0.8m): Normal operation

### 3. Launch System (`autonomous_mapping.launch.py`)
**Coordinated system startup**

**Initialization Sequence**:
1. Robot description and control (0s)
2. Sensor systems (0s)
3. RealSense camera (3s delay)
4. SLAM mapping (5s delay)
5. Nav2 navigation (8s delay)
6. Safety monitor (2s delay)
7. Autonomous mapper (15s delay)
8. Map saver (20s delay)

## Quick Start

### 1. Basic Operation
```bash
# Navigate to the ros2-rover directory
cd /home/ubuntu/ros2-rover

# Start autonomous mapping (10 minutes, 0.3 m/s max speed)
./start_autonomous_mapping.sh

# Custom duration and speed
./start_autonomous_mapping.sh 1200 0.2 0.6  # 20 min, 0.2 m/s, 0.6m safety
```

### 2. Manual Launch
```bash
# Source workspace
source install/setup.bash

# Launch complete system
ros2 launch tractor_bringup autonomous_mapping.launch.py \
    mapping_duration:=600 \
    max_speed:=0.3 \
    safety_distance:=0.8
```

### 3. Monitor Status
```bash
# Watch mapping progress
ros2 topic echo /mapping_status

# Monitor safety status
ros2 topic echo /safety_status

# View emergency stop status
ros2 topic echo /emergency_stop

# Check navigation goals
ros2 topic echo /navigate_to_pose/_action/goal
```

## System Configuration

### Navigation Parameters (`nav2_params.yaml`)
```yaml
# Point cloud obstacle detection
observation_sources: pointcloud
pointcloud:
  topic: /realsense/depth/points
  max_obstacle_height: 2.0
  obstacle_max_range: 2.5
  raytrace_max_range: 3.0
```

### SLAM Configuration (`slam_toolbox_params.yaml`)
- Real-time mapping with loop closure detection
- Optimized for outdoor agricultural environments
- Handles dynamic obstacles and terrain variations

### Robot Localization (`robot_localization.yaml`)
```yaml
# Multi-sensor fusion
odom0: /odom                # Wheel odometry
imu0: /hglrc_gps/imu       # GPS compass
imu1: /realsense/imu       # RealSense IMU
```

## Operation Modes

### 1. Standard Exploration
- **Duration**: 10 minutes (configurable)
- **Coverage**: ~25m radius area
- **Speed**: 0.3 m/s maximum
- **Pattern**: Grid-based with randomization

### 2. Extended Mapping
```bash
./start_autonomous_mapping.sh 1800 0.2  # 30 minutes, slower speed
```

### 3. High-Speed Survey
```bash
./start_autonomous_mapping.sh 600 0.5 1.0  # Fast mapping, larger safety margin
```

### 4. Precision Mapping
```bash
./start_autonomous_mapping.sh 900 0.15 0.5  # Slow, detailed mapping
```

## Safety Features

### Emergency Stop Conditions
1. **Obstacle Detection**: Any obstacle within 0.3m
2. **Sensor Failure**: Loss of critical sensor data
3. **Communication Loss**: Mapping node heartbeat timeout
4. **Manual Override**: User-initiated emergency stop

### Recovery Procedures
1. **Obstacle Avoidance**: Automatic path replanning
2. **Goal Timeout**: Move to alternative waypoint
3. **Navigation Failure**: Reset and try new approach
4. **Emergency Recovery**: Full system restart if needed

### Speed Limiting
- **Dynamic adjustment**: Speed reduces near obstacles
- **Safety margins**: Larger buffers at higher speeds
- **Terrain adaptation**: Slower on rough terrain

## Monitoring and Diagnostics

### Real-time Status
```bash
# System overview
ros2 topic list | grep -E "(status|emergency|mapping)"

# Detailed diagnostics
ros2 run rqt_robot_monitor rqt_robot_monitor
```

### Log Analysis
```bash
# View recent logs
ros2 bag record -o mapping_session /tf /map /odom /scan /realsense/depth/points

# Analyze mapping quality
ros2 run nav2_map_server map_server --ros-args -p yaml_filename:=autonomous_map.yaml
```

### Performance Metrics
- **Coverage Rate**: Area mapped per minute
- **Obstacle Detection**: Successful avoidance events
- **Navigation Efficiency**: Path planning success rate
- **Map Quality**: Loop closure accuracy

## Troubleshooting

### Common Issues

#### 1. Camera Not Working
```bash
# Check camera detection
v4l2-ctl --list-devices

# Test with fallback
ros2 launch tractor_bringup realsense_opencv.launch.py
```

#### 2. Navigation Not Starting
```bash
# Check Nav2 status
ros2 service call /is_path_valid nav2_msgs/srv/IsPathValid

# Verify transforms
ros2 run tf2_tools view_frames
```

#### 3. Poor Mapping Quality
- **Slower speed**: Reduce max_speed parameter
- **Better lighting**: Ensure adequate lighting for cameras
- **Sensor cleaning**: Clean RealSense lens regularly

#### 4. Emergency Stops
```bash
# Check obstacle distances
ros2 topic echo /scan | grep range

# Monitor safety status
ros2 topic echo /safety_status
```

### Performance Optimization

#### For Large Areas
```bash
# Extend exploration radius and duration
ros2 launch tractor_bringup autonomous_mapping.launch.py \
    mapping_duration:=2400 \
    max_speed:=0.4
```

#### For Detailed Mapping
```bash
# Slower, more precise mapping
ros2 launch tractor_bringup autonomous_mapping.launch.py \
    mapping_duration:=1800 \
    max_speed:=0.15 \
    safety_distance:=0.5
```

## Map Output

### Generated Files
- **autonomous_map.pgm**: Map image (occupancy grid)
- **autonomous_map.yaml**: Map metadata and parameters
- **Timestamped copies**: Saved in `maps/autonomous/`

### Map Usage
```bash
# View map
ros2 run nav2_map_server map_server --ros-args -p yaml_filename:=autonomous_map.yaml

# Use for navigation
ros2 launch nav2_bringup localization_launch.py map:=autonomous_map.yaml
```

## Future Enhancements

### Planned Features
1. **Coverage optimization**: Better exploration algorithms
2. **Crop detection**: AI-based crop row recognition
3. **Multi-session mapping**: Merge multiple mapping runs
4. **Weather adaptation**: Adjust behavior for conditions
5. **Remote monitoring**: Web interface for status

### Integration Options
1. **Precision agriculture**: Integrate with farming tools
2. **Fleet coordination**: Multiple robot coordination
3. **Cloud mapping**: Upload maps to farm management systems
4. **AI enhancement**: Machine learning for better navigation

## Conclusion

This autonomous mapping system provides a robust, safe, and intelligent solution for creating detailed maps of agricultural environments. The multi-layer safety system ensures safe operation while the advanced navigation and mapping capabilities deliver high-quality results.

The system is designed to be:
- **Safe**: Multiple emergency stop systems
- **Reliable**: Redundant sensors and robust error handling
- **Efficient**: Intelligent exploration patterns
- **Maintainable**: Clear documentation and modular design
- **Extensible**: Easy to add new features and capabilities

Perfect for mapping fields, orchards, vineyards, and other agricultural areas with autonomous precision!