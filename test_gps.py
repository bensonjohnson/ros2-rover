#!/usr/bin/env python3

import serial
import time

def test_gps_connection():
    """Test GPS connection on /dev/ttyS6 with different baud rates"""
    baud_rates = [9600, 38400, 57600, 115200]
    
    for baud in baud_rates:
        print(f"\nTesting /dev/ttyS6 at {baud} baud...")
        try:
            with serial.Serial('/dev/ttyS6', baud, timeout=2) as ser:
                print(f"Serial port opened successfully at {baud} baud")
                
                # Read for 5 seconds
                start_time = time.time()
                data_received = False
                
                while time.time() - start_time < 5:
                    if ser.in_waiting > 0:
                        data = ser.readline().decode('ascii', errors='ignore').strip()
                        if data:
                            print(f"Data received: {data}")
                            data_received = True
                            if data.startswith('$'):  # NMEA sentence
                                print("GPS NMEA data detected!")
                                return baud
                
                if not data_received:
                    print("No data received")
                    
        except Exception as e:
            print(f"Error at {baud} baud: {e}")
    
    print("\nNo GPS data found at any tested baud rate")
    return None

if __name__ == "__main__":
    working_baud = test_gps_connection()
    if working_baud:
        print(f"\nGPS is working at {working_baud} baud!")
    else:
        print("\nGPS connection test failed")
        print("Please check:")
        print("1. GPS wiring connections")
        print("2. GPS power supply (3.3V or 5V)")
        print("3. GPS module is functioning")
