#!/usr/bin/env python3
import smbus
import time
import struct

# Set the I2C bus number, usually 1
I2C_BUS = 9  # Changed to 9 for your system

# Set the I2C address of the 4-ch encoder motor driver
MOTOR_ADDR = 0x34 

# Register addresses
ADC_BAT_ADDR = 0x00
MOTOR_TYPE_ADDR = 0x14 # Set the encoder motor type
MOTOR_ENCODER_POLARITY_ADDR = 0x15 # Set the polarity of the encoder
MOTOR_FIXED_PWM_ADDR = 0x1F # Fixed PWM control, open-loop control, range: (-100~100)
MOTOR_FIXED_SPEED_ADDR = 0x33 # Fixed speed control, closed-loop control
MOTOR_ENCODER_TOTAL_ADDR = 0x3C # Total pulse value of each of the four encoder motors

# Motor type values
MOTOR_TYPE_WITHOUT_ENCODER = 0
MOTOR_TYPE_TT = 1
MOTOR_TYPE_N20 = 2
MOTOR_TYPE_JGB37_520_12V_110RPM = 3 # 44 pulses per revolution, reduction ratio: 90, default

# Motor type and encoder direction polarity
MotorType = MOTOR_TYPE_JGB37_520_12V_110RPM
MotorEncoderPolarity = 0

bus = smbus.SMBus(I2C_BUS)
speed1 = [50,50,0,0]  # Only M1 and M2 for your setup
speed2 = [-50,-50,0,0]
speed3 = [0,0,0,0]

pwm1 = [50,50,0,0]
pwm2 = [-100,-100,0,0]
pwm3 = [0,0,0,0]

def Motor_Init():
    """Initialize motor"""
    print("Initializing motors...")
    try:
        bus.write_byte_data(MOTOR_ADDR, MOTOR_TYPE_ADDR, MotorType)
        print(f"✅ Motor type set to {MotorType}")
        time.sleep(0.5)
        
        bus.write_byte_data(MOTOR_ADDR, MOTOR_ENCODER_POLARITY_ADDR, MotorEncoderPolarity)
        print(f"✅ Encoder polarity set to {MotorEncoderPolarity}")
        print("Motor initialization complete!")
    except Exception as e:
        print(f"❌ Motor initialization failed: {e}")

def test_motors():
    """Test motor movement"""
    print("\n=== Testing Motors ===")
    
    try:
        # Test battery voltage
        battery = bus.read_i2c_block_data(MOTOR_ADDR, ADC_BAT_ADDR, 2)
        voltage = battery[0]+(battery[1]<<8)
        print(f"Battery voltage: {voltage}mV")
        
        # Test encoders
        try:
            encode_data = bus.read_i2c_block_data(MOTOR_ADDR, MOTOR_ENCODER_TOTAL_ADDR, 16)
            Encode = struct.unpack('iiii', bytes(encode_data))
            print(f"Encoders - M1: {Encode[0]}, M2: {Encode[1]}, M3: {Encode[2]}, M4: {Encode[3]}")
        except Exception as e:
            print(f"Encoder read failed: {e}")
        
        # Test forward motion
        print("Testing forward motion...")
        bus.write_i2c_block_data(MOTOR_ADDR, MOTOR_FIXED_SPEED_ADDR, speed1)
        time.sleep(3)
        
        # Test reverse motion  
        print("Testing reverse motion...")
        bus.write_i2c_block_data(MOTOR_ADDR, MOTOR_FIXED_SPEED_ADDR, speed2)
        time.sleep(3)
        
        # Stop motors
        print("Stopping motors...")
        bus.write_i2c_block_data(MOTOR_ADDR, MOTOR_FIXED_SPEED_ADDR, speed3)
        
        print("✅ Motor test complete!")
        
    except Exception as e:
        print(f"❌ Motor test failed: {e}")
    finally:
        # Ensure motors are stopped
        try:
            bus.write_i2c_block_data(MOTOR_ADDR, MOTOR_FIXED_SPEED_ADDR, [0,0,0,0])
        except:
            pass

if __name__ == "__main__":
    Motor_Init()
    test_motors()
    bus.close()