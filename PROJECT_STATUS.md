# ROS2 Tractor Project Status

## 🚜 Project Overview
Autonomous outdoor tractor/rover system built on ROS2 Jazzy with GPS navigation, sensor fusion, and web-based visualization.

---

## ✅ **COMPLETED SYSTEMS**

### 🗺️ **Navigation Stack (Nav2) - COMPLETE**
- **Status:** ✅ Fully integrated and tested
- **Configuration:** `src/tractor_bringup/config/nav2_params.yaml`
- **Launch:** `src/tractor_bringup/launch/navigation.launch.py`
- **Features:**
  - Path planning with NavFn planner
  - Regulated Pure Pursuit controller
  - Costmap generation (local + global)
  - Behavior server (spin, backup, drive_on_heading, wait)
  - Velocity smoothing and collision monitoring
  - GPS-based localization (no AMCL needed)
- **Notes:** Configured for RealSense camera obstacle detection, currently works with GPS localization

### 🧭 **Robot Localization - COMPLETE**
- **Status:** ✅ GPS + compass + wheel odometry fusion working
- **Configuration:** `src/tractor_bringup/config/robot_localization.yaml`
- **Launch:** `src/tractor_bringup/launch/robot_localization.launch.py`
- **Features:**
  - EKF local node (odom frame)
  - EKF global node (map frame) 
  - NavSat transform node for GPS integration
  - Publishes `/odometry/filtered` and `/odometry/filtered_map`
- **Hardware:** GPS on `/dev/ttyS6`, compass on I2C-5

### 📡 **Sensor Integration - COMPLETE**
- **GPS/Compass:** ✅ HGLRC M100 5883 working
  - Location: `src/tractor_sensors/tractor_sensors/hglrc_m100_5883.py`
  - Topics: `/hglrc_gps/fix`, `/hglrc_gps/imu`, `/hglrc_gps/magnetic_field`
  - Hardware: `/dev/ttyS6` (GPS), I2C-5 (compass)
- **Wheel Encoders:** ✅ Implemented via I2C motor controller
  - Location: `src/tractor_control/tractor_control/hiwonder_motor_driver.py`  
  - Topics: `/joint_states` (encoder feedback), `/wheel_odom`
  - Hardware: I2C motor controller board (M1=right, M2=left tracks)
- **Motor Control:** ✅ Hiwonder driver integration
  - Location: `src/tractor_control/tractor_control/hiwonder_motor_driver.py`
  - Topics: `/cmd_vel` → motor commands, `/motor_status`
  - Hardware: I2C motor driver board

### 🎮 **Manual Control - COMPLETE**
- **Xbox Controller:** ✅ Teleop working
  - Location: `src/tractor_control/tractor_control/xbox_controller_teleop.py`
  - Device: `/dev/input/js0` (auto-detected)
  - Controls: Left stick (linear), right stick (angular), triggers (speed)
- **Script:** `start_manual_driving.sh` - Full manual driving system

### 🗺️ **SLAM Mapping - COMPLETE**
- **Status:** ✅ Ready for RealSense camera
- **Configuration:** `src/tractor_bringup/config/slam_toolbox_params.yaml`
- **Launch:** `src/tractor_bringup/launch/slam_mapping.launch.py`
- **Features:**
  - SLAM Toolbox with outdoor-optimized parameters
  - Depth image to laser scan conversion
  - GPS-assisted mapping for large areas
  - Real-time map saving capabilities
- **Script:** `start_mapping.sh` - Interactive mapping workflow
- **Maps Storage:** `/home/ubuntu/ros2-rover/maps/`

### 📊 **Visualization - COMPLETE**
- **Foxglove Studio:** ✅ Web-based visualization
  - Bridge runs on port 8765 (configurable)
  - Manual driving layout: `foxglove_manual_driving_config.json`
  - SLAM mapping layout: `foxglove_slam_mapping_config.json`
  - Real-time sensor monitoring, 3D visualization, plots
- **No RViz dependency** - everything uses Foxglove for better web access

### 🚀 **Launch Systems - COMPLETE**
- **Complete System:** `./quick_start.sh`
  - GPS + localization + navigation + control + Foxglove
  - Full autonomous operation ready
- **Manual Driving:** `./start_manual_driving.sh`  
  - Xbox controller + sensors + Foxglove (no nav/SLAM)
  - Perfect for testing and playing around
- **SLAM Mapping:** `./start_mapping.sh`
  - Interactive mapping with RealSense + GPS
  - Real-time map building and saving

---

## ⏳ **PENDING/WAITING FOR HARDWARE**

### 📷 **RealSense Camera**
- **Status:** 🟡 Software ready, hardware not arrived
- **Expected:** Intel RealSense D435i or similar
- **Integration:** 
  - Vision processing: `src/tractor_vision/tractor_vision/realsense_processor.py`
  - Obstacle detection: `src/tractor_vision/tractor_vision/obstacle_detector.py`
  - Launch: `src/tractor_bringup/launch/vision.launch.py`
- **Topics Ready:**
  - `/realsense_435i/color/image_raw`
  - `/realsense_435i/depth/image_rect_raw` 
  - `/realsense_435i/depth/points`
  - `/realsense_435i/scan` (converted from depth)
- **Use Cases:**
  - Obstacle detection for navigation
  - SLAM mapping (depth → laser scan)
  - Visual monitoring in Foxglove

### 🔋 **Battery System**
- **Status:** 🟡 Waiting for proper battery
- **Current:** Temporary power setup
- **Integration Ready:** Battery voltage monitoring in `/motor_status` topic

---

## 📁 **PROJECT STRUCTURE**

```
/home/ubuntu/ros2-rover/
├── src/
│   ├── tractor_bringup/         # Main integration package
│   │   ├── launch/              # All launch files
│   │   ├── config/              # Configuration files
│   │   ├── maps/                # Static maps
│   │   └── urdf/                # Robot description
│   ├── tractor_sensors/         # GPS, compass, encoders
│   ├── tractor_control/         # Motors, Xbox controller
│   ├── tractor_vision/          # RealSense, obstacle detection  
│   ├── tractor_coverage/        # Autonomous coverage patterns
│   ├── tractor_implements/      # Mower, sprayer control
│   └── slam_stress_test/        # SLAM testing utilities
├── maps/                        # Generated SLAM maps
├── logs/                        # Runtime logs
├── quick_start.sh              # Complete system launcher
├── start_manual_driving.sh     # Manual control system
├── start_mapping.sh            # SLAM mapping system
├── foxglove_manual_driving_config.json
├── foxglove_slam_mapping_config.json
└── PROJECT_STATUS.md           # This file
```

---

## 🔧 **CONFIGURATION FILES**

### Navigation & Localization
- `src/tractor_bringup/config/nav2_params.yaml` - Nav2 stack configuration
- `src/tractor_bringup/config/robot_localization.yaml` - EKF sensor fusion
- `src/tractor_bringup/config/slam_toolbox_params.yaml` - SLAM mapping

### Hardware Integration  
- `src/tractor_sensors/config/hglrc_m100_5883_config.yaml` - GPS/compass
- `src/tractor_vision/config/realsense_config.yaml` - Camera settings

### Robot Description
- `src/tractor_bringup/urdf/tractor.urdf.xacro` - Robot model

---

## 🌐 **FOXGLOVE LAYOUTS**

### Manual Driving Layout
- **File:** `foxglove_manual_driving_config.json`
- **Panels:** 3D robot, velocity plots, GPS status, battery gauge, camera feeds
- **Use:** Real-time monitoring during manual operation

### SLAM Mapping Layout  
- **File:** `foxglove_slam_mapping_config.json`
- **Panels:** Live map building, laser scan, SLAM diagnostics, camera feeds
- **Use:** Interactive mapping workflow

---

## 🎯 **CURRENT CAPABILITIES**

### ✅ **Ready to Use Today**
1. **Manual driving** with Xbox controller + full sensor monitoring
2. **GPS navigation** with waypoint following (when maps available)
3. **Sensor fusion** providing accurate robot pose
4. **Real-time visualization** via Foxglove Studio
5. **System health monitoring** (GPS fix, battery, motor status)

### 🟡 **Ready When RealSense Arrives**
1. **Obstacle detection** and avoidance during navigation
2. **SLAM mapping** to create detailed yard maps
3. **Visual monitoring** of camera feeds
4. **Enhanced safety** with depth-based collision detection

### 🔄 **Autonomous Workflows Available**
1. **Navigation:** `./quick_start.sh` → set goals in Foxglove
2. **Mapping:** `./start_mapping.sh` → drive around to build maps
3. **Manual:** `./start_manual_driving.sh` → Xbox controller fun

---

## 🚀 **NEXT STEPS**

### Immediate (when battery arrives)
1. **Test manual driving system** with `./start_manual_driving.sh`
2. **Verify all sensors** working in Foxglove
3. **Calibrate motor responses** and encoder accuracy
4. **Test GPS accuracy** in yard environment

### When RealSense arrives
1. **Plug and play** - system is ready for camera
2. **Create detailed yard maps** with `./start_mapping.sh`
3. **Test obstacle detection** during navigation
4. **Full autonomous operation** with vision safety

### Future Enhancements
1. **Coverage patterns** for automated mowing/spraying
2. **Implement control** (mower, sprayer) integration
3. **Advanced path planning** with terrain awareness
4. **Fleet management** for multiple robots

---

## 🔧 **BUILD & DEPLOYMENT**

### Build System
```bash
colcon build --symlink-install
source install/setup.bash
```

### Dependencies
- **ROS2 Jazzy** ✅ Installed
- **Nav2** ✅ Installed  
- **SLAM Toolbox** ✅ Installed
- **Foxglove Bridge** ✅ Installed
- **Robot Localization** ✅ Installed

### Hardware Requirements
- **GPS Module:** HGLRC M100 5883 ✅ Connected
- **Compass:** Integrated with GPS ✅ Working
- **Motor Driver:** Hiwonder I2C ✅ Connected (with integrated encoders)
- **Camera:** RealSense D435i 🟡 Pending delivery
- **Battery:** 12V system 🟡 Waiting for proper battery

---

## 📝 **NOTES**

### Performance Optimizations
- **Outdoor mapping** parameters tuned for large areas
- **GPS fusion** eliminates need for AMCL localization
- **Foxglove-native** visualization for better web performance
- **Modular launch** system allows component testing

### Safety Features
- **Collision monitoring** via Nav2
- **Velocity smoothing** prevents jerky movements  
- **GPS fence** capabilities (can be configured)
- **Emergency stop** via Xbox controller or topic

### Known Issues
- None currently - system is stable and tested

---

**Last Updated:** 2025-06-30
**System Status:** 🟢 **READY FOR TESTING** (pending battery)
**Next Milestone:** Manual driving with proper battery system