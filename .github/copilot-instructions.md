# ROS2 Autonomous Tractor/Rover Development Instructions

Always follow these instructions first and fallback to additional search and context gathering only if the information here is incomplete or found to be in error.

## Working Effectively

### Essential Build and Setup Commands
- Install ROS2 Jazzy: `sudo apt install ros-jazzy-desktop-full ros-jazzy-nav2-bringup ros-jazzy-slam-toolbox ros-jazzy-robot-localization ros-jazzy-foxglove-bridge`
- Install Python dependencies: `pip install RPi.GPIO smbus2 pyserial pynmea2 pyrealsense2`
- Bootstrap and build the repository:
  - `source /opt/ros/jazzy/setup.bash`
  - `./build.sh` -- takes 5-8 minutes to complete. NEVER CANCEL. Set timeout to 15+ minutes.
  - `source install/setup.bash`

### Testing Commands
- Run basic tests: `colcon test --packages-select tractor_bringup tractor_control tractor_sensors` -- takes 2-3 minutes. NEVER CANCEL. Set timeout to 10+ minutes.
- Run style checks: `colcon test --pytest-args -v` -- takes 1-2 minutes for linting.
- Check for build issues: `colcon list --packages-up-to tractor_bringup`

### System Operation Scripts
- **Complete autonomous mapping**: `./start_autonomous_mapping.sh`
  - Full system with RealSense camera, SLAM, Nav2, and safety monitoring
  - Takes 60+ seconds to fully initialize. NEVER CANCEL during startup.
  - Runs until manually stopped with Ctrl+C
- **SLAM-based mapping**: `./start_autonomous_slam.sh`
  - Advanced SLAM mapping with frontier exploration
  - Takes 45+ seconds to initialize. NEVER CANCEL during startup.
- **NPU exploration**: `./start_npu_exploration.sh` and `./start_npu_exploration_depth.sh`
  - Neural processing unit-based exploration variants
- **Manual Nav2 activation**: `./activate_nav2.sh`
  - Use when lifecycle nodes need manual activation
  - Takes 10-15 seconds to complete
- **Test motor outputs**: `./test_motor_outputs.sh`
- **Test SLAM activation**: `./test_slam_activation.sh`

## Validation and Testing Requirements

### CRITICAL: Manual Validation After Changes
- ALWAYS test actual robot functionality, not just node startup
- **Complete system test**: Start `./start_autonomous_mapping.sh`, verify Foxglove connection at `ws://[ROBOT_IP]:8765`, check sensor data flows
- **Navigation test**: Send navigation goals in Foxglove 3D panel, verify robot responds and avoids obstacles
- **Safety test**: Place obstacle in robot path, verify emergency stop within 0.3m (emergency zone)
- **Manual control test**: Use Xbox controller via `/cmd_vel` topic, verify tank steering response
- **Sensor validation**: Monitor all sensor topics in Foxglove: GPS (`/hglrc_gps/fix`), camera (`/realsense_435i/color/image_raw`), depth (`/realsense_435i/depth/image_rect_raw`), laser scan (`/scan`)

### Build Time Expectations and Timeouts
- **NEVER CANCEL**: colcon build takes 5-8 minutes clean, 2-3 minutes incremental
- **NEVER CANCEL**: System startup scripts take 45-90 seconds to fully initialize
- **NEVER CANCEL**: Test suites take 2-5 minutes total
- Set minimum timeouts: build=15min, tests=10min, startup scripts=5min

### Required System Health Checks
- Verify ROS2 Jazzy installation: `ros2 --version`
- Check workspace build status: `colcon list --packages-up-to tractor_bringup`
- Test core dependencies: `ros2 pkg list | grep -E "(nav2|slam_toolbox|robot_localization|foxglove)"`
- Validate hardware connections: `ls /dev/ttyS6` (GPS), `rs-enumerate-devices` (RealSense), `i2cdetect -y 5` (motor controller)

## Project Structure and Key Locations

### Package Organization
```
src/
├── tractor_bringup/         # Main integration package - START HERE
│   ├── launch/              # All system launch files
│   ├── config/              # Navigation, SLAM, sensor configs
│   ├── tractor_bringup/     # Python nodes (autonomous mapping, safety monitor)
│   └── urdf/                # Robot description URDF files
├── tractor_control/         # Motor control and Xbox teleop
├── tractor_sensors/         # GPS (HGLRC M100 5883), compass, encoders
├── tractor_vision/          # Intel RealSense D435i camera integration
├── tractor_coverage/        # Autonomous mowing/spraying coverage patterns
├── tractor_implements/      # Mower and sprayer control systems
└── slam_stress_test/        # SLAM testing and performance utilities
```

### Critical Configuration Files
- `src/tractor_bringup/config/nav2_params.yaml` - Nav2 navigation stack parameters
- `src/tractor_bringup/config/slam_toolbox_params.yaml` - SLAM mapping configuration
- `src/tractor_bringup/config/hiwonder_motor_params.yaml` - Motor controller settings
- `src/tractor_sensors/config/hglrc_m100_5883_config.yaml` - GPS/compass configuration
- Multiple RealSense configs: `realsense_config.yaml`, `realsense_usb_stable.yaml`, `realsense_low_bandwidth.yaml`

### Launch Files and System Integration
- `autonomous_mapping_minimal.launch.py` - Core autonomous mapping system
- `autonomous_slam_mapping.launch.py` - Advanced SLAM mapping
- `robot_description.launch.py` - Robot URDF and transforms
- `npu_exploration.launch.py` / `npu_exploration_depth.launch.py` - NPU-based exploration

### Maps and Data Storage
- Generated SLAM maps: `maps/` directory
- Foxglove configurations: `foxglove_*.json` files in repository root
- Runtime logs: `log/` directory

## Hardware Integration and Requirements

### Required Hardware Components
- **Computing**: Raspberry Pi 4 or equivalent SBC with Ubuntu 22.04
- **Navigation**: HGLRC M100 5883 GPS/compass module (connected via /dev/ttyS6 and I2C-5)
- **Vision**: Intel RealSense D435i camera (USB 3.0 connection required)
- **Locomotion**: Hiwonder I2C motor controller (address 0x60) with integrated encoders
- **Control**: Xbox controller for manual teleop (auto-detected at /dev/input/js0)
- **Power**: 12V battery system with voltage monitoring

### Sensor Integration Status
- **GPS/Compass**: ✅ Fully working - topics: `/hglrc_gps/fix`, `/hglrc_gps/imu`, `/hglrc_gps/magnetic_field`
- **Motor Control**: ✅ Tank steering working - topics: `/cmd_vel` (input), `/joint_states` (encoders), `/wheel_odom`
- **RealSense Camera**: ⚠️ Software ready, hardware may not be present - topics: `/realsense_435i/color/image_raw`, `/realsense_435i/depth/image_rect_raw`, `/realsense_435i/depth/points`
- **Xbox Controller**: ✅ Teleop working - device: `/dev/input/js0`

## Common Development Tasks

### Making Changes to Navigation
- Edit `src/tractor_bringup/config/nav2_params.yaml` for path planning parameters
- Test changes: `./start_autonomous_mapping.sh` then send navigation goals via Foxglove
- Always verify obstacle avoidance with physical or simulated obstacles

### Modifying SLAM Configuration
- Edit `src/tractor_bringup/config/slam_toolbox_params.yaml` for mapping parameters
- Test with: `./start_autonomous_slam.sh` and monitor map quality in Foxglove
- Check map saving: maps are automatically saved to `maps/` directory

### Sensor Integration Changes
- GPS/Compass: modify `src/tractor_sensors/tractor_sensors/hglrc_m100_5883.py`
- RealSense: modify `src/tractor_vision/tractor_vision/realsense_processor.py`
- Motor control: modify `src/tractor_control/tractor_control/hiwonder_motor_driver.py`
- Always test individual sensors before full system integration

### Safety System Modifications
- Safety monitor: `src/tractor_bringup/tractor_bringup/safety_monitor.py`
- Emergency stop zones: Emergency (<0.3m), Warning (0.3-0.8m), Safe (>0.8m)
- Test safety: place obstacle in robot path and verify immediate stop

## Visualization and Monitoring

### Foxglove Studio Integration
- **Connection**: WebSocket at `ws://[ROBOT_IP]:8765` (port 8765)
- **Layouts**: `foxglove_manual_driving_config.json`, `foxglove_slam_mapping_config.json`, `foxglove_autonomous_mapping.json`
- **Key panels**: 3D robot view, map display, sensor data plots, battery monitoring
- Always use Foxglove for real-time monitoring - no RViz dependency

### System Monitoring Commands
- Check system health: `ros2 topic echo /system_status`
- Monitor GPS: `ros2 topic echo /hglrc_gps/fix`
- Watch navigation status: `ros2 topic echo /navigation_status`
- View safety status: `ros2 topic echo /safety_status`
- Monitor battery: `ros2 topic echo /motor_status` (includes voltage)

## Troubleshooting Common Issues

### Build Failures
- Missing ROS2 Jazzy: Ensure `/opt/ros/jazzy/setup.bash` exists and is sourced
- Missing dependencies: Run `rosdep install --from-paths src --ignore-src -r -y`
- Clean build: `rm -rf build/ install/ log/` then `./build.sh`

### Runtime Issues
- "No transform" errors: Check TF tree with `ros2 run tf2_tools view_frames.py`
- GPS not working: Verify `/dev/ttyS6` exists and has correct permissions
- Motors not responding: Check I2C bus 5 with `i2cdetect -y 5`
- Camera not detected: Run `rs-enumerate-devices` to check RealSense connection
- Nav2 nodes inactive: Run `./activate_nav2.sh` to manually activate lifecycle nodes

### Performance Optimization
- High CPU usage: Check for infinite loops in custom nodes
- Poor mapping quality: Adjust SLAM parameters in `slam_toolbox_params.yaml`
- Navigation failures: Tune Nav2 parameters in `nav2_params.yaml`
- Slow startup: RealSense initialization can take 30+ seconds, this is normal

## Development Workflow

### Standard Development Process
1. Make targeted changes to specific packages
2. Build incrementally: `colcon build --packages-select [package_name]`
3. Source workspace: `source install/setup.bash`
4. Test individual components before full system testing
5. Run full system validation with hardware
6. Always monitor safety systems during testing

### Before Committing Changes
- Run style checks: `colcon test --pytest-args -v`
- Verify build succeeds: `./build.sh`
- Test core functionality with actual hardware
- Check that safety systems remain functional
- Update documentation if configuration changes are made

### System Integration Notes
- Robot uses tank steering (differential drive) with left/right track control
- EKF sensor fusion combines GPS, compass, and wheel odometry
- Maps are saved automatically during autonomous operations
- Foxglove Bridge starts automatically with all launch scripts
- Emergency stop can be triggered via Xbox controller or `/emergency_stop` topic

This autonomous tractor system is designed for agricultural applications with robust safety systems, comprehensive sensor integration, and professional-grade navigation capabilities.