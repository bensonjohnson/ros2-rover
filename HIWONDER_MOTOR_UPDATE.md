# Hiwonder Motor Controller Update

## Overview
This update implements the corrected I2C addresses and communication protocol for the Hiwonder motor controller based on successful ESP32 testing.

## Key Changes

### 1. Corrected I2C Addresses
- **Motor Controller Address**: Changed from `0x60` to `0x34`
- **Motor Type Address**: Changed from `0x20` to `0x14`
- **Encoder Polarity Address**: Changed from `0x21` to `0x15`
- **Motor Speed Control Address**: Changed from `0x31` to `0x33`
- **Encoder Total Address**: Changed from `0x60` to `0x3C`
- **Battery Voltage Address**: `0x00` (unchanged)

### 2. New Components

#### Hiwonder Motor Driver (`hiwonder_motor_driver.py`)
- Implements corrected I2C communication protocol
- Battery voltage monitoring
- Encoder reading with proper data unpacking
- Motor speed control with JGB37 motor type initialization
- Publishes joint states and motor feedback

#### Xbox Controller Teleop (`xbox_controller_teleop.py`)
- Tank drive mode (default) - left stick controls left motor, right stick controls right motor
- Arcade drive mode - single stick for forward/backward and turning
- Emergency stop functionality (Y button)
- Configurable deadzone and speed limits
- Uses pygame for controller input

#### Launch File (`hiwonder_control.launch.py`)
- Starts both motor driver and Xbox controller nodes
- Configurable parameters for I2C settings and motor limits
- Can disable Xbox controller if needed

### 3. Configuration Changes
- **Maximum Motor Speed**: Reduced to 50 (from ESP32 testing)
- **Motor Type**: Set to JGB37_520_12V (type 3)
- **Encoder Polarity**: Set to 0 (from official sample)
- **Deadzone**: Increased to 0.15 for better control

## Usage

### Start the Updated Motor Control System
```bash
# Source the workspace
source install/setup.bash

# Launch with Xbox controller (default)
ros2 launch tractor_bringup hiwonder_control.launch.py

# Launch without Xbox controller
ros2 launch tractor_bringup hiwonder_control.launch.py use_xbox_controller:=false

# Launch with custom motor speed limit
ros2 launch tractor_bringup hiwonder_control.launch.py max_motor_speed:=30
```

### Xbox Controller Controls
- **Tank Drive Mode (default)**:
  - Left stick Y: Left motor speed
  - Right stick Y: Right motor speed
  - Y button: Emergency stop toggle
  - A button: Resume from emergency stop

- **Arcade Drive Mode**:
  - Left stick Y: Forward/backward
  - Left stick X: Left/right turning
  - Y button: Emergency stop toggle
  - A button: Resume from emergency stop

### Published Topics
- `/cmd_vel` - Twist messages for robot movement
- `/motor_speeds` - Current motor speeds
- `/battery_voltage` - Battery voltage in volts
- `/joint_states` - Wheel joint states
- `/emergency_stop` - Emergency stop status

### Subscribed Topics
- `/cmd_vel` - Twist commands for robot movement

## Dependencies
- `smbus2` - I2C communication
- `pygame` - Xbox controller input
- Standard ROS2 packages (geometry_msgs, std_msgs, sensor_msgs)

## Installation Notes
Make sure pygame is installed for Xbox controller support:
```bash
pip3 install pygame
```

## Testing
The motor controller has been tested with:
- ESP32 development board
- Xbox controller via Bluetooth
- JGB37-520 12V motors
- Hiwonder motor driver board

All I2C addresses and communication protocols have been verified to work correctly.