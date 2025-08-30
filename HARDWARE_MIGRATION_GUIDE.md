# Hardware Migration Guide: HGLRC M100-5883 to LC29H RTK + BNO055

This guide covers the complete migration from the combined HGLRC M100-5883 GPS/compass module to separate LC29H RTK GPS and BNO055 sensor fusion IMU.

## Overview of Changes

### Old Hardware (HGLRC M100-5883)
- **GPS**: M10 chip with basic GNSS (GPS/GLONASS/Galileo/BDS)
- **Compass**: QMC5883 magnetometer
- **Accuracy**: ~2.5m GPS positioning
- **Interfaces**: UART (GPS) + I2C (compass)
- **Limitations**: No RTK, basic magnetometer only

### New Hardware 
- **LC29H RTK GPS**: Dual-frequency L1+L5 GNSS with RTK capability
- **BNO055 IMU**: 9-DOF sensor with hardware sensor fusion
- **Accuracy**: cm-level RTK positioning, ±1° orientation
- **Interfaces**: UART (GPS) + I2C (IMU)
- **Benefits**: RTK precision, hardware-accelerated sensor fusion

## Hardware Connections

### LC29H RTK GPS
- **Interface**: UART via GPIO pins
- **Default Port**: `/dev/ttyS6` (UART6 overlay, same as previous HGLRC)
- **Baudrate**: 460800 bps (high-speed for RTK data)
- **Power**: 5V via GPIO or external supply
- **Physical Connection**: GPIO pins 8 (TX) and 10 (RX)
- **Antenna**: External GNSS antenna required

**Verify Serial Connection**:
```bash
# Check if the serial port exists
ls -l /dev/ttyS*

# Test communication (should show NMEA sentences)
sudo cat /dev/ttyS6

# Check current baudrate setting
stty -F /dev/ttyS6
```

**UART Configuration**: The system should already have the UART6 overlay configured from the previous HGLRC setup. Verify in `/boot/config.txt`:
```
dtoverlay=uart6
```

### BNO055 IMU
- **Interface**: I2C
- **Default Address**: 0x28 (can be 0x29 with ADR pin)
- **Default Bus**: I2C bus 1 (`/dev/i2c-1`)
- **Power**: 3.3V or 5V
- **Orientation**: Mount according to robot coordinate system

## RTK Base Station Setup

### Option 1: NTRIP Caster (Recommended)
Configure your LC29H as an RTK rover using an NTRIP correction service:

```yaml
# In lc29h_rtk_config.yaml
rtk_mode: 'rover'
ntrip_host: 'your.ntrip.provider.com'
ntrip_port: 2101
ntrip_mountpoint: 'NEAR_REF_STATION'
ntrip_username: 'your_username'
ntrip_password: 'your_password'
```

### Option 2: Local Base Station
Set up a second LC29H module as a base station:

```yaml
# Base station configuration
rtk_mode: 'base'
base_latitude: 40.7128    # Known base position
base_longitude: -74.0060  # (decimal degrees)
base_altitude: 10.0       # meters above sea level
```

## Software Configuration

### 1. Update Configuration Files

#### LC29H RTK GPS (`lc29h_rtk_config.yaml`):
```yaml
lc29h_rtk_gps_publisher:
  ros__parameters:
    gps_port: '/dev/ttyUSB0'
    gps_baudrate: 460800
    rtk_mode: 'rover'  # or 'base' or 'disabled'
    
    # NTRIP settings (for rover mode)
    ntrip_host: 'your.rtk.provider.com'
    ntrip_port: 2101
    ntrip_mountpoint: 'REF_STATION'
    ntrip_username: 'username'
    ntrip_password: 'password'
```

#### BNO055 IMU (`bno055_config.yaml`):
```yaml
bno055_imu_publisher:
  ros__parameters:
    i2c_bus: 1
    i2c_address: 0x28
    operation_mode: 'NDOF'  # 9DOF fusion with magnetometer
    update_rate: 100.0      # Hz
    auto_calibrate: true
```

#### Robot Localization (`robot_localization.yaml`):
The system now includes proper EKF sensor fusion:
- **Local EKF**: Fuses wheel odometry + IMU
- **Global EKF**: Fuses GPS + local EKF output
- **NavSat Transform**: Converts GPS to local coordinates

### 2. Launch File Updates

The main launch file now includes:
```python
# New sensor launches
lc29h_gps_launch = IncludeLaunchDescription(...)
bno055_imu_launch = IncludeLaunchDescription(...)
robot_localization_launch = IncludeLaunchDescription(...)
```

## Calibration Procedures

### BNO055 IMU Calibration

The BNO055 requires calibration for optimal performance:

1. **System Calibration**: Move the robot in a figure-8 pattern
2. **Accelerometer**: Place robot in 6 different orientations (±X, ±Y, ±Z)
3. **Magnetometer**: Rotate robot around all axes away from magnetic interference
4. **Gyroscope**: Keep robot stationary (auto-calibrates)

**Monitor calibration status**:
```bash
ros2 topic echo /imu/calibration_status
ros2 topic echo /imu/status
```

Calibration levels: 0 (uncalibrated) to 3 (fully calibrated)

### RTK GPS Setup

1. **Initial GPS Fix**: Allow 2-5 minutes for initial satellite acquisition
2. **RTK Convergence**: With RTCM corrections, expect <30 second convergence
3. **Base Station Survey**: For base mode, allow 5+ minutes for position survey

**Monitor GPS status**:
```bash
ros2 topic echo /gps/fix
ros2 topic echo /gps/velocity
```

RTK Status indicators:
- **Status 1**: Standard GPS (~3m accuracy)
- **Status 2**: DGPS (~1m accuracy)  
- **Status 4**: RTK Float (~0.5m accuracy)
- **Status 5**: RTK Fixed (~0.01m accuracy)

## Testing and Validation

### 1. Individual Sensor Tests

**Test LC29H GPS**:
```bash
ros2 launch tractor_sensors lc29h_rtk.launch.py
ros2 topic hz /gps/fix  # Should show ~5-10 Hz
```

**Test BNO055 IMU**:
```bash
ros2 launch tractor_sensors bno055_imu.launch.py
ros2 topic hz /imu/data  # Should show ~100 Hz
```

### 2. Sensor Fusion Tests

**Test Robot Localization**:
```bash
ros2 launch tractor_bringup robot_localization.launch.py
ros2 topic echo /odometry/filtered  # Local EKF output
ros2 topic echo /odometry/filtered_map  # Global EKF with GPS
```

### 3. Full System Test

**Launch complete system**:
```bash
ros2 launch tractor_bringup autonomous_mapping_minimal.launch.py
```

**Verify topics**:
```bash
ros2 topic list | grep -E "(gps|imu|odometry)"
```

## Performance Improvements

### Expected Improvements

1. **Positioning Accuracy**:
   - Old: ~2.5m GPS accuracy
   - New: ~0.01m RTK fixed accuracy (100x improvement)

2. **Orientation Accuracy**:
   - Old: ~5° magnetometer-based heading
   - New: ~1° hardware-fused orientation

3. **Update Rates**:
   - GPS: 5-10 Hz (vs. 1 Hz previously)
   - IMU: 100 Hz (vs. 10 Hz compass)

4. **System Reliability**:
   - Hardware sensor fusion reduces computational load
   - RTK provides robust positioning in challenging environments
   - Proper EKF sensor fusion eliminates single points of failure

### Sensor Fusion Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Wheel Odom    │────│   Local EKF      │────│  /odometry/     │
└─────────────────┘    │  (30 Hz)         │    │  filtered       │
                       │                  │    └─────────────────┘
┌─────────────────┐    │                  │           │
│   BNO055 IMU    │────│                  │           │
│   (100 Hz)      │    └──────────────────┘           │
└─────────────────┘                                   │
                                                      │
┌─────────────────┐    ┌──────────────────┐           │
│   LC29H RTK     │────│   Global EKF     │───────────┘
│   GPS (10 Hz)   │    │   (10 Hz)        │
└─────────────────┘    │                  │    ┌─────────────────┐
                       │                  │────│  /odometry/     │
┌─────────────────┐    │                  │    │  filtered_map   │
│  NavSat Transform│────│                  │    └─────────────────┘
└─────────────────┘    └──────────────────┘
```

## Troubleshooting

### Common Issues

1. **LC29H not detected**: Check USB connection and permissions
   ```bash
   sudo usermod -a -G dialout $USER
   ls -l /dev/ttyUSB*
   ```

2. **BNO055 I2C errors**: Verify I2C bus and address
   ```bash
   i2cdetect -y 1  # Should show device at 0x28
   ```

3. **RTK not converging**: Check NTRIP credentials and base station distance
   ```bash
   # Monitor RTCM corrections
   ros2 topic echo /gps/fix --field status.status
   ```

4. **Poor sensor fusion**: Ensure proper calibration and topic remapping

### Performance Monitoring

Use these commands to monitor system health:
```bash
# Check sensor update rates
ros2 topic hz /gps/fix
ros2 topic hz /imu/data  
ros2 topic hz /odometry/filtered

# Monitor calibration status
ros2 topic echo /imu/calibration_status

# Check RTK status
ros2 topic echo /gps/fix --field status.status

# Verify transforms
ros2 run tf2_tools view_frames.py
```

## Next Steps

After successful hardware migration:

1. **Fine-tune sensor fusion parameters** based on your specific environment
2. **Set up persistent RTK base station** for consistent cm-level accuracy
3. **Implement waypoint navigation** taking advantage of RTK precision
4. **Add sensor redundancy** with multiple IMU units for critical applications

The new sensor setup provides a solid foundation for precision autonomous navigation with professional-grade accuracy and reliability.