# Foxglove Studio Dashboard Setup

## Quick Setup

1. **Start the tractor system:**
   ```bash
   ./simple_start.sh
   ```

2. **Connect Foxglove Studio:**
   - Open Foxglove Studio
   - Connect to WebSocket: `ws://YOUR_ROBOT_IP:8765`
   - Import the dashboard: `foxglove_tractor_dashboard.json`

## Dashboard Layout

The custom dashboard includes:

### Main View (Left Side - 75%)
- **3D Robot View (60%)**: Live robot visualization with tank tracks
- **Map View (40%)**: Navigation map display
- **GPS Status**: Latitude, longitude, GPS fix status
- **Motor Status**: Left/right motor command values

### Control Panel (Right Side - 25%)
- **Teleop Controls**: WASD-style robot control
- **Battery Voltage Gauge**: 9.9V - 12.6V range
- **Battery Percentage Gauge**: 0% - 100%
- **Runtime Gauge**: Estimated hours remaining
- **Speed Gauge**: Current robot velocity

## Manual Import Instructions

If auto-import doesn't work:

1. **Open Foxglove Studio**
2. **Connect to robot**: `ws://YOUR_ROBOT_IP:8765`
3. **Import layout**:
   - Click "Layout" menu → "Import from file..."
   - Select `foxglove_tractor_dashboard.json`
   - Click "Import"

## Manual Panel Setup

If you prefer to set up manually:

### 3D Panel
- **Panel Type**: 3D
- **Follow Mode**: Follow pose
- **Follow Frame**: `base_link`
- **Topics**: Enable `/robot_description`, `/tf`, `/tf_static`, `/odom`

### Map Panel  
- **Panel Type**: Map
- **Topic**: `/map`
- **Follow Frame**: `base_link`

### Teleop Panel
- **Panel Type**: Teleop  
- **Topic**: `/cmd_vel`
- **Message Type**: `geometry_msgs/Twist`
- **Publish Rate**: 10 Hz
- **Controls**:
  - Up: `linear.x = 0.3`
  - Down: `linear.x = -0.3` 
  - Left: `angular.z = 0.5`
  - Right: `angular.z = -0.5`

### Gauge Panels
- **Battery Voltage**: `/battery_voltage.data` (9.9-12.6V)
- **Battery Percentage**: `/battery_percentage.data` (0-100%)
- **Runtime**: `/battery_runtime.data` (0-8 hours)
- **Speed**: `/odom.twist.twist.linear.x` (-1.0 to 1.0 m/s)

### Raw Messages Panels
- **GPS Status**: `/hglrc_gps/fix` 
- **Motor Commands**: `/motor_speeds`

## Troubleshooting

### Dashboard doesn't load
- Make sure robot system is running first
- Check WebSocket connection is active
- Verify topics are being published: `ros2 topic list`

### Missing topics
- Check if simple_start.sh completed successfully
- Verify battery topics: `ros2 topic echo /battery_voltage --once`
- Check motor topics: `ros2 topic echo /cmd_vel --once`

### Teleop not working
- Verify `/cmd_vel` topic exists
- Check motor driver is running: `ros2 node list`
- Test manual command: `ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.1}}"`

### 3D view issues
- Check `/robot_description` is published
- Verify `/tf` and `/tf_static` topics
- Ensure tank tracks are fixed (not spinning)

## Available Topics

- `/battery_voltage` - Battery voltage (Float32)
- `/battery_percentage` - Battery percentage (Float32) 
- `/battery_runtime` - Runtime remaining (Float32)
- `/cmd_vel` - Motor commands (geometry_msgs/Twist)
- `/motor_speeds` - Motor speed feedback (Float32MultiArray)
- `/odom` - Wheel odometry (nav_msgs/Odometry)
- `/hglrc_gps/fix` - GPS fix data (sensor_msgs/NavSatFix)
- `/hglrc_gps/imu` - Compass/IMU data (sensor_msgs/Imu)
- `/joint_states` - Robot joint positions (sensor_msgs/JointState)
- `/tf` - Transform data (tf2_msgs/TFMessage)
- `/robot_description` - Robot URDF (std_msgs/String)

## Tips

- Use the 3D view to monitor robot orientation and movement
- Watch battery gauges to avoid running out of power
- GPS status shows fix quality (0=no fix, 1=fix, 2=DGPS fix)
- Runtime gauge shows -1.0 until enough data is collected for estimation
- Tank tracks should remain fixed to the chassis (no pinwheel spinning!)