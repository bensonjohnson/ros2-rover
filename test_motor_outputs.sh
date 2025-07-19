#!/bin/bash

# Test all motor output combinations to find which ones are connected

MOTOR_ADDR=0x34
I2C_BUS=5

echo "=================================================="
echo "    MOTOR OUTPUT TESTING TOOL"
echo "=================================================="
echo "This will test all motor output combinations"
echo "Watch/listen for motor movement during each test"
echo ""

# Kill any running processes first
pkill -f "hiwonder_motor_driver\|xbox_controller_teleop" 2>/dev/null || true
sleep 2

# Initialize motor controller
echo "Initializing motor controller..."
i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 0x14 3 2>/dev/null || echo "Init failed, continuing..."
sleep 0.1
i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 0x15 0 2>/dev/null || echo "Polarity failed, continuing..."
sleep 0.1

echo ""
echo "Testing motor outputs with speed 75..."
echo "Press Ctrl+C to stop if you see movement"
echo ""

# Test M1 only
echo "Test 1: Motor 1 only (M1=75, M2=0, M3=0, M4=0)"
i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 0x33 && i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 75 && i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 0 && i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 0 && i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 0
echo "Listening for 3 seconds..."
sleep 3

# Test M2 only  
echo "Test 2: Motor 2 only (M1=0, M2=75, M3=0, M4=0)"
i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 0x33 && i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 0 && i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 75 && i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 0 && i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 0
echo "Listening for 3 seconds..."
sleep 3

# Test M3 only
echo "Test 3: Motor 3 only (M1=0, M2=0, M3=75, M4=0)"
i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 0x33 && i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 0 && i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 0 && i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 75 && i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 0
echo "Listening for 3 seconds..."
sleep 3

# Test M4 only
echo "Test 4: Motor 4 only (M1=0, M2=0, M3=0, M4=75)"
i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 0x33 && i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 0 && i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 0 && i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 0 && i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 75
echo "Listening for 3 seconds..."
sleep 3

# Test M1+M2 (what ROS is doing)
echo "Test 5: Motors 1+2 (M1=75, M2=75, M3=0, M4=0) - THIS IS WHAT ROS DOES"
i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 0x33 && i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 75 && i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 75 && i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 0 && i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 0
echo "Listening for 3 seconds..."
sleep 3

# Test M3+M4
echo "Test 6: Motors 3+4 (M1=0, M2=0, M3=75, M4=75)"
i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 0x33 && i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 0 && i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 0 && i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 75 && i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 75
echo "Listening for 3 seconds..."
sleep 3

# Stop all motors
echo ""
echo "Stopping all motors..."
i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 0x33 && i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 0 && i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 0 && i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 0 && i2cset -y ${I2C_BUS} ${MOTOR_ADDR} 0

echo ""
echo "Test complete. Which test made the motors move?"
echo "Test 1: Motor 1 only"
echo "Test 2: Motor 2 only"  
echo "Test 3: Motor 3 only"
echo "Test 4: Motor 4 only"
echo "Test 5: Motors 1+2 (current ROS setting)"
echo "Test 6: Motors 3+4"
echo ""