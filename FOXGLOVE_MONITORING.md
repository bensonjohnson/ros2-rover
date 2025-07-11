# Foxglove Monitoring for Autonomous Mapping

## Overview
Foxglove Bridge is now integrated into the autonomous mapping system, providing real-time visualization of the RealSense point cloud, camera feeds, mapping progress, and system status.

## Connection Details
- **URL**: `ws://ROBOT_IP:8765`
- **Protocol**: WebSocket
- **Port**: 8765 (default)
- **Auto-discovery**: Available if on same network

## Available Data Streams

### 🎯 **3D Visualization**
- **Point Cloud**: `/realsense/depth/points` - Real-time 3D obstacle detection
- **Laser Scan**: `/scan` - 2D laser scan data
- **Map**: `/map` - SLAM-generated occupancy grid
- **Robot Model**: `/robot_description` - 3D robot visualization
- **Transforms**: `/tf` & `/tf_static` - Coordinate frames

### 📸 **Camera Feeds**
- **Color Image**: `/realsense/color/image_raw` - RGB camera feed
- **Depth Image**: `/realsense/depth/image_raw` - Depth visualization
- **Camera Info**: `/realsense/color/camera_info` - Camera calibration

### 🗺️ **Navigation Data**
- **Odometry**: `/odom` & `/odometry/filtered` - Robot position
- **Navigation Goals**: `/navigate_to_pose/_action/goal` - Target waypoints
- **Path Feedback**: `/navigate_to_pose/_action/feedback` - Navigation progress
- **Command Velocity**: `/cmd_vel` - Robot movement commands

### 🛡️ **Safety Monitoring**
- **Emergency Stop**: `/emergency_stop` - Emergency status
- **Safety Status**: `/safety_status` - Safety system health
- **Mapping Status**: `/mapping_status` - Exploration progress

## Quick Start

### 1. Start Autonomous Mapping
```bash
cd /home/ubuntu/ros2-rover
./start_autonomous_mapping.sh
```

The system will display:
```
🦊 Foxglove Bridge will be available at: ws://192.168.1.100:8765
    - Point Cloud: /realsense/depth/points
    - Camera: /realsense/color/image_raw
    - Map: /map
    - Robot Path: /navigate_to_pose/_action/feedback
```

### 2. Connect with Foxglove Studio

#### Option A: Web Browser
1. Go to [foxglove.dev/studio](https://foxglove.dev/studio)
2. Click "Open connection"
3. Select "Foxglove WebSocket"
4. Enter: `ws://YOUR_ROBOT_IP:8765`
5. Click "Open"

#### Option B: Desktop App
1. Download [Foxglove Studio](https://foxglove.dev/download)
2. Open Foxglove Studio
3. Click "Open connection"
4. Select "Foxglove WebSocket"
5. Enter: `ws://YOUR_ROBOT_IP:8765`
6. Click "Open"

### 3. Load Mapping Layout
1. In Foxglove Studio, click "Layout" → "Import"
2. Select the file: `/home/ubuntu/ros2-rover/foxglove_autonomous_mapping.json`
3. The layout will automatically configure all panels

## Monitoring Features

### 🌐 **3D Scene View**
- **Point Cloud Visualization**: Real-time 3D obstacles
- **Robot Following**: Camera follows robot movement
- **Map Overlay**: Shows explored areas
- **Grid Reference**: Coordinate system visualization

**Key Settings**:
- Point cloud colored by height (rainbow colormap)
- Robot model visible with coordinate frames
- Map overlay with transparency
- Follow mode tracks robot position

### 📊 **Real-time Plots**
- **Speed Monitoring**: Robot linear/angular velocity
- **Status Tracking**: Mapping and safety status over time
- **Emergency Events**: Emergency stop triggers
- **Exploration Progress**: Waypoints visited

### 🎛️ **Control Panels**
- **Emergency Stop Indicator**: Visual warning for safety events
- **Speed Gauge**: Current robot speed
- **Status Messages**: Raw system messages
- **State Transitions**: System state changes

## Point Cloud Monitoring

### **What You'll See**:
- **Red/Orange Points**: Close obstacles (< 1m)
- **Yellow Points**: Medium distance obstacles (1-3m)
- **Green/Blue Points**: Far obstacles (3-10m)
- **Moving Points**: Dynamic obstacles being detected

### **Key Indicators**:
- **Density**: More points = more detailed obstacles
- **Color Changes**: Height-based coloring shows terrain
- **Movement**: Points moving = robot navigation
- **Gaps**: Areas being explored

### **Troubleshooting**:
- **No Points**: Check `/realsense/depth/points` topic
- **Sparse Points**: Camera may be too far from obstacles
- **Flickering**: Normal for dynamic point cloud data

## Camera Feed Monitoring

### **Color Camera**:
- **Topic**: `/realsense/color/image_raw`
- **Resolution**: 640x480 (default)
- **Frame Rate**: 30 FPS
- **Use**: Visual confirmation of robot view

### **Depth Camera**:
- **Topic**: `/realsense/depth/image_raw` 
- **Visualization**: Rainbow colormap (blue=close, red=far)
- **Range**: 0.1m to 10m
- **Use**: Verify depth sensing accuracy

## Navigation Monitoring

### **Waypoint Tracking**:
- **Goal Markers**: Green arrows showing target locations
- **Path Planning**: Blue line showing planned route
- **Robot Position**: Red arrow showing current pose
- **Visited Areas**: Darker regions on map

### **Real-time Feedback**:
- **Speed**: Current linear/angular velocity
- **Distance to Goal**: Remaining distance to waypoint
- **Navigation Status**: Active/completed/failed goals
- **Obstacle Avoidance**: Path modifications around obstacles

## Safety Monitoring

### **Emergency Stop Indicators**:
- **Visual Alert**: Red warning when emergency active
- **Audio Alert**: Browser notification (if enabled)
- **Status Message**: Reason for emergency stop
- **Recovery**: Green when normal operation resumes

### **Safety Zones**:
- **Red Zone**: Emergency stop (< 0.3m)
- **Yellow Zone**: Speed reduction (0.3-0.8m)
- **Green Zone**: Normal operation (> 0.8m)

## Custom Visualizations

### **Add New Panels**:
1. Right-click in layout → "Add Panel"
2. Select panel type (Plot, Image, 3D, etc.)
3. Configure topic and settings
4. Save layout for future use

### **Useful Additional Topics**:
- `/tf_static`: Static coordinate frames
- `/scan`: 2D laser scan overlay
- `/local_costmap/costmap`: Navigation obstacles
- `/global_costmap/costmap`: Global path planning

## Performance Tips

### **Optimize Performance**:
- **Reduce Point Cloud Density**: Lower publish rate if needed
- **Limit Camera Resolution**: Use smaller image size
- **Filter Topics**: Only subscribe to needed data
- **Close Unused Panels**: Reduce CPU usage

### **Network Considerations**:
- **Local Network**: Best performance on same WiFi
- **Remote Access**: Use VPN for external monitoring
- **Bandwidth**: Point cloud data is high-bandwidth
- **Latency**: Some delay is normal for real-time data

## Troubleshooting

### **Connection Issues**:
```bash
# Check Foxglove Bridge status
ros2 node info /foxglove_bridge

# Verify port is open
netstat -ln | grep 8765

# Test connection
curl -I ws://localhost:8765
```

### **Missing Data**:
```bash
# Check topic availability
ros2 topic list | grep realsense

# Verify point cloud data
ros2 topic hz /realsense/depth/points

# Check camera feed
ros2 topic hz /realsense/color/image_raw
```

### **Performance Issues**:
- Reduce point cloud publish rate
- Close unnecessary Foxglove panels
- Use wired network connection
- Lower camera resolution

## Advanced Features

### **Data Recording**:
- Record sessions directly in Foxglove
- Export data for analysis
- Create custom visualizations
- Share layouts with team

### **Remote Monitoring**:
- Access from multiple devices
- Real-time collaboration
- Mobile monitoring (tablet/phone)
- Cloud-based visualization

## Conclusion

Foxglove provides comprehensive real-time monitoring of your autonomous mapping system. The point cloud visualization lets you see exactly what obstacles the RealSense camera is detecting, while the multi-panel layout gives you complete situational awareness of the robot's exploration progress.

**Perfect for**:
- 🔍 **Debugging**: Visual confirmation of sensor data
- 📊 **Monitoring**: Real-time system status
- 🎯 **Analysis**: Understanding robot behavior
- 🛡️ **Safety**: Early warning of issues
- 📈 **Optimization**: Performance tuning

The integrated Foxglove Bridge makes your autonomous mapping system fully observable and debuggable in real-time!