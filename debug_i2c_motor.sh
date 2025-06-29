#!/bin/bash

# I2C Motor Controller Debug Script
# Tests Hiwonder motor controller communication directly

MOTOR_ADDR=0x34
I2C_BUS=5

# Register addresses from ESP32 code
ADC_BAT_ADDR=0x00
MOTOR_TYPE_ADDR=0x14
MOTOR_ENCODER_POLARITY_ADDR=0x15
MOTOR_FIXED_SPEED_ADDR=0x33
MOTOR_ENCODER_TOTAL_ADDR=0x3C

# Motor types
MOTOR_TYPE_JGB37=3

echo "=================================================="
echo "    I2C MOTOR CONTROLLER DEBUG TOOL"
echo "=================================================="
echo "Motor Address: ${MOTOR_ADDR} on I2C bus ${I2C_BUS}"
echo ""

# Function to test I2C communication
test_i2c_basic() {
    echo "=== BASIC I2C COMMUNICATION TEST ==="
    
    echo "1. Scanning I2C bus ${I2C_BUS}..."
    i2cdetect -y ${I2C_BUS}
    echo ""
    
    echo "2. Testing simple read from motor controller..."
    echo "   Trying to read from address ${MOTOR_ADDR} register 0x00..."
    
    # Test if device responds to basic communication
    if i2cget -y ${I2C_BUS} ${MOTOR_ADDR} 0x00 2>/dev/null; then
        echo "✅ Basic I2C communication working"
    else
        echo "❌ Direct register read failed, trying detection method..."
        
        # Try i2cdetect approach
        if i2cdetect -y ${I2C_BUS} | grep -q "34"; then
            echo "✅ Device detected by i2cdetect but register read fails"
            echo "   This might be normal - some devices need specific protocols"
            echo "   Continuing with initialization tests..."
        else
            echo "❌ Device not responding at all"
            echo "   Check connections and power"
            return 1
        fi
    fi
    echo ""
}

# Function to test battery voltage reading
test_battery_read() {
    echo "=== BATTERY VOLTAGE TEST ==="
    echo "Reading battery voltage from register 0x${ADC_BAT_ADDR}..."
    
    # Read 2 bytes for voltage
    byte1=$(i2cget -y ${I2C_BUS} ${MOTOR_ADDR} ${ADC_BAT_ADDR} 2>/dev/null || echo "FAIL")
    byte2=$(i2cget -y ${I2C_BUS} ${MOTOR_ADDR} $((ADC_BAT_ADDR + 1)) 2>/dev/null || echo "FAIL")
    
    if [ "$byte1" != "FAIL" ] && [ "$byte2" != "FAIL" ]; then
        echo "✅ Battery voltage read successful"
        echo "   Byte 1: $byte1"
        echo "   Byte 2: $byte2"
        # Convert to decimal for voltage calculation
        val1=$(printf "%d" $byte1)
        val2=$(printf "%d" $byte2)
        voltage=$((val1 + (val2 << 8)))
        echo "   Raw voltage: ${voltage}mV"
    else
        echo "❌ Battery voltage read failed"
    fi
    echo ""
}

# Function to test motor initialization
test_motor_init() {
    echo "=== MOTOR INITIALIZATION TEST (Manufacturer Protocol) ==="
    
    echo "1. Setting motor type to JGB37 (value ${MOTOR_TYPE_JGB37}) using manufacturer method..."
    # Use block write like manufacturer: WireWriteDataArray(MOTOR_TYPE_ADDR,&MotorType,1)
    if i2cset -y ${I2C_BUS} ${MOTOR_ADDR} ${MOTOR_TYPE_ADDR} ${MOTOR_TYPE_JGB37} i 2>/dev/null; then
        echo "✅ Motor type set successfully (block write)"
        sleep 0.005  # 5ms delay like manufacturer
        
        # Verify by reading back
        read_val=$(i2cget -y ${I2C_BUS} ${MOTOR_ADDR} ${MOTOR_TYPE_ADDR} 2>/dev/null || echo "FAIL")
        if [ "$read_val" != "FAIL" ]; then
            echo "   Readback value: $read_val"
            expected=$(printf "0x%02x" ${MOTOR_TYPE_JGB37})
            if [ "$read_val" = "$expected" ]; then
                echo "   ✅ Value matches expected"
            else
                echo "   ⚠️  Value mismatch: expected $expected, got $read_val"
            fi
        fi
    else
        echo "❌ Failed to set motor type with block write"
        echo "   Trying single byte write..."
        if i2cset -y ${I2C_BUS} ${MOTOR_ADDR} ${MOTOR_TYPE_ADDR} ${MOTOR_TYPE_JGB37} 2>/dev/null; then
            echo "   ✅ Motor type set with single byte write"
            sleep 0.005
        else
            echo "   ❌ Both methods failed"
        fi
    fi
    echo ""
    
    echo "2. Setting encoder polarity to 0 using manufacturer method..."
    if i2cset -y ${I2C_BUS} ${MOTOR_ADDR} ${MOTOR_ENCODER_POLARITY_ADDR} 0 i 2>/dev/null; then
        echo "✅ Encoder polarity set successfully (block write)"
        sleep 0.005  # 5ms delay like manufacturer
        
        # Verify by reading back
        read_val=$(i2cget -y ${I2C_BUS} ${MOTOR_ADDR} ${MOTOR_ENCODER_POLARITY_ADDR} 2>/dev/null || echo "FAIL")
        if [ "$read_val" != "FAIL" ]; then
            echo "   Readback value: $read_val"
            if [ "$read_val" = "0x00" ]; then
                echo "   ✅ Value matches expected (0x00)"
            else
                echo "   ⚠️  Value mismatch: expected 0x00, got $read_val"
            fi
        fi
    else
        echo "❌ Failed to set encoder polarity with block write"
        echo "   Trying single byte write..."
        if i2cset -y ${I2C_BUS} ${MOTOR_ADDR} ${MOTOR_ENCODER_POLARITY_ADDR} 0 2>/dev/null; then
            echo "   ✅ Encoder polarity set with single byte write"
            sleep 0.005
        else
            echo "   ❌ Both methods failed"
        fi
    fi
    echo ""
}

# Function to test motor speed commands
test_motor_speeds() {
    echo "=== MOTOR SPEED COMMAND TEST ==="
    
    echo "1. Testing forward motion (speed 25)..."
    # Write 4 bytes: M1=25, M2=25, M3=0, M4=0
    if i2cset -y ${I2C_BUS} ${MOTOR_ADDR} ${MOTOR_FIXED_SPEED_ADDR} 25 25 0 0 i 2>/dev/null; then
        echo "✅ Forward command sent successfully"
        sleep 2
    else
        echo "❌ Failed to send forward command"
    fi
    
    echo "2. Testing stop..."
    if i2cset -y ${I2C_BUS} ${MOTOR_ADDR} ${MOTOR_FIXED_SPEED_ADDR} 0 0 0 0 i 2>/dev/null; then
        echo "✅ Stop command sent successfully"
        sleep 1
    else
        echo "❌ Failed to send stop command"
    fi
    
    echo "3. Testing reverse motion (speed -15)..."
    # For negative values, we need to handle 2's complement
    # -15 in 8-bit 2's complement is 241 (0xF1)
    neg15=241
    if i2cset -y ${I2C_BUS} ${MOTOR_ADDR} ${MOTOR_FIXED_SPEED_ADDR} ${neg15} ${neg15} 0 0 i 2>/dev/null; then
        echo "✅ Reverse command sent successfully"
        sleep 2
    else
        echo "❌ Failed to send reverse command"
    fi
    
    echo "4. Final stop..."
    i2cset -y ${I2C_BUS} ${MOTOR_ADDR} ${MOTOR_FIXED_SPEED_ADDR} 0 0 0 0 i 2>/dev/null
    echo "✅ Motors stopped"
    echo ""
}

# Function to test encoder reading (manufacturer protocol)
test_encoder_read() {
    echo "=== ENCODER READING TEST (Manufacturer Protocol) ==="
    echo "Reading 16 bytes from encoder register 0x${MOTOR_ENCODER_TOTAL_ADDR}..."
    
    # Method 1: Try to read block directly (like manufacturer code)
    echo "Method 1: Direct block read attempt..."
    
    # First write register address
    echo "  Writing register address 0x${MOTOR_ENCODER_TOTAL_ADDR}..."
    if i2cset -y ${I2C_BUS} ${MOTOR_ADDR} ${MOTOR_ENCODER_TOTAL_ADDR} 2>/dev/null; then
        echo "  ✅ Register address written"
        
        # Try to read 16 bytes using different approaches
        echo "  Reading 16 bytes..."
        
        # Approach A: Sequential single-byte reads after register setup
        echo "  Method A: Sequential reads after register setup..."
        encoder_bytes=()
        success=true
        for i in {0..15}; do
            byte_val=$(i2cget -y ${I2C_BUS} ${MOTOR_ADDR} 2>/dev/null || echo "FAIL")
            if [ "$byte_val" = "FAIL" ]; then
                success=false
                break
            fi
            encoder_bytes+=($byte_val)
            printf "    Byte %2d: %s\n" $i "$byte_val"
        done
        
        if [ "$success" = true ]; then
            echo "  ✅ Sequential read successful"
            echo "  Attempting to decode as 4 x int32 (little-endian):"
            # Group bytes into 4-byte integers (little-endian)
            for motor in {0..3}; do
                base=$((motor * 4))
                if [ ${#encoder_bytes[@]} -gt $((base + 3)) ]; then
                    b0=$(printf "%d" ${encoder_bytes[$base]})
                    b1=$(printf "%d" ${encoder_bytes[$((base + 1))]})
                    b2=$(printf "%d" ${encoder_bytes[$((base + 2))]})
                    b3=$(printf "%d" ${encoder_bytes[$((base + 3))]})
                    
                    # Little-endian: LSB first
                    value=$((b0 + (b1 << 8) + (b2 << 16) + (b3 << 24)))
                    
                    # Handle signed 32-bit (2's complement)
                    if [ $value -gt 2147483647 ]; then
                        value=$((value - 4294967296))
                    fi
                    
                    printf "    Motor %d encoder: %d\n" $((motor + 1)) $value
                fi
            done
        else
            echo "  ❌ Sequential read failed"
        fi
        
    else
        echo "  ❌ Failed to write register address"
    fi
    
    echo ""
    echo "Method 2: Individual byte reads with register addressing..."
    for i in {0..15}; do
        addr=$((MOTOR_ENCODER_TOTAL_ADDR + i))
        byte_val=$(i2cget -y ${I2C_BUS} ${MOTOR_ADDR} $(printf "0x%02x" $addr) 2>/dev/null || echo "XX")
        printf "  Byte %2d (reg 0x%02x): %s\n" $i $addr "$byte_val"
    done
    echo ""
}

# Function for interactive testing
interactive_test() {
    echo "=== INTERACTIVE MODE ==="
    echo "Commands:"
    echo "  f <speed>  - Forward at speed (1-127)"
    echo "  r <speed>  - Reverse at speed (1-127)" 
    echo "  s          - Stop motors"
    echo "  b          - Read battery"
    echo "  e          - Read encoders"
    echo "  t          - Test sequence"
    echo "  q          - Quit"
    echo ""
    
    while true; do
        echo -n "motor> "
        read -r cmd args
        
        case $cmd in
            f)
                speed=${args:-25}
                echo "Forward speed $speed"
                i2cset -y ${I2C_BUS} ${MOTOR_ADDR} ${MOTOR_FIXED_SPEED_ADDR} $speed $speed 0 0 i
                ;;
            r)
                speed=${args:-15}
                # Convert to 2's complement for negative
                neg_speed=$((256 - speed))
                echo "Reverse speed $speed (sent as $neg_speed)"
                i2cset -y ${I2C_BUS} ${MOTOR_ADDR} ${MOTOR_FIXED_SPEED_ADDR} $neg_speed $neg_speed 0 0 i
                ;;
            s)
                echo "Stop"
                i2cset -y ${I2C_BUS} ${MOTOR_ADDR} ${MOTOR_FIXED_SPEED_ADDR} 0 0 0 0 i
                ;;
            b)
                test_battery_read
                ;;
            e)
                test_encoder_read
                ;;
            t)
                test_motor_speeds
                ;;
            q)
                echo "Stopping motors and exiting..."
                i2cset -y ${I2C_BUS} ${MOTOR_ADDR} ${MOTOR_FIXED_SPEED_ADDR} 0 0 0 0 i 2>/dev/null
                break
                ;;
            *)
                echo "Unknown command: $cmd"
                ;;
        esac
    done
}

# Main execution
case "${1:-auto}" in
    auto)
        test_i2c_basic && test_motor_init && test_motor_speeds && test_encoder_read
        ;;
    basic)
        test_i2c_basic
        ;;
    init)
        test_motor_init
        ;;
    speed)
        test_motor_speeds
        ;;
    encoder)
        test_encoder_read
        ;;
    battery)
        test_battery_read
        ;;
    interactive|i)
        interactive_test
        ;;
    *)
        echo "Usage: $0 [auto|basic|init|speed|encoder|battery|interactive]"
        echo "  auto        - Run all tests (default)"
        echo "  basic       - Basic I2C communication test"
        echo "  init        - Motor initialization test"
        echo "  speed       - Motor speed command test"
        echo "  encoder     - Encoder reading test"
        echo "  battery     - Battery voltage test"
        echo "  interactive - Interactive testing mode"
        ;;
esac

echo ""
echo "Debug complete. Motors should be stopped."
