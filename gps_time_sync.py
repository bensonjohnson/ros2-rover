#!/usr/bin/env python3
"""
GPS Time Synchronization Script
Reads NMEA data from GPS and synchronizes system time
"""

import serial
import pynmea2
import subprocess
import datetime
import time
import sys
import argparse
import logging

class GPSTimeSync:
    def __init__(self, gps_port='/dev/ttyS6', gps_baudrate=115200, verbose=False):
        self.gps_port = gps_port
        self.gps_baudrate = gps_baudrate
        self.verbose = verbose
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def get_gps_time(self, timeout=30):
        """Read GPS time from NMEA RMC or GGA messages"""
        try:
            with serial.Serial(self.gps_port, self.gps_baudrate, timeout=1) as gps_serial:
                self.logger.info(f"Connected to GPS on {self.gps_port} at {self.gps_baudrate} baud")
                
                start_time = time.time()
                while time.time() - start_time < timeout:
                    try:
                        line = gps_serial.readline().decode('ascii', errors='replace').strip()
                        
                        if line.startswith('$') and ('RMC' in line or 'GGA' in line):
                            msg = pynmea2.parse(line)
                            
                            # Get time and date from RMC message (preferred for date)
                            if isinstance(msg, pynmea2.RMC) and msg.timestamp and msg.datestamp:
                                gps_datetime = datetime.datetime.combine(msg.datestamp, msg.timestamp)
                                self.logger.info(f"GPS time from RMC: {gps_datetime} UTC")
                                return gps_datetime
                            
                            # Get time from GGA message (time only, use current date)
                            elif isinstance(msg, pynmea2.GGA) and msg.timestamp:
                                today = datetime.date.today()
                                gps_datetime = datetime.datetime.combine(today, msg.timestamp)
                                self.logger.info(f"GPS time from GGA: {gps_datetime} UTC")
                                return gps_datetime
                                
                    except pynmea2.ParseError:
                        continue
                    except Exception as e:
                        self.logger.debug(f"GPS parsing error: {e}")
                        continue
                        
                self.logger.error(f"No valid GPS time received within {timeout} seconds")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to connect to GPS: {e}")
            return None
    
    def set_system_time(self, gps_datetime):
        """Set system time using sudo date command"""
        try:
            # Format: YYYY-MM-DD HH:MM:SS
            time_string = gps_datetime.strftime("%Y-%m-%d %H:%M:%S")
            
            # Use sudo date to set system time
            cmd = ['sudo', 'date', '-s', time_string]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"System time set to: {time_string} UTC")
                return True
            else:
                self.logger.error(f"Failed to set system time: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error setting system time: {e}")
            return False
    
    def sync_time(self, timeout=30):
        """Main function to sync system time with GPS"""
        self.logger.info("Starting GPS time synchronization...")
        
        # Get current system time
        system_time = datetime.datetime.now(datetime.timezone.utc)
        self.logger.info(f"Current system time: {system_time} UTC")
        
        # Get GPS time
        gps_time = self.get_gps_time(timeout)
        if not gps_time:
            self.logger.error("Failed to get GPS time")
            return False
        
        # Ensure GPS time is timezone-aware (UTC)
        if gps_time.tzinfo is None:
            gps_time = gps_time.replace(tzinfo=datetime.timezone.utc)
        
        # Calculate time difference
        time_diff = abs((gps_time - system_time).total_seconds())
        self.logger.info(f"Time difference: {time_diff:.2f} seconds")
        
        # Only sync if difference is significant (> 1 second)
        if time_diff > 1.0:
            self.logger.info(f"Time difference ({time_diff:.2f}s) is significant, updating system time")
            success = self.set_system_time(gps_time)
            if success:
                self.logger.info("GPS time synchronization completed successfully")
                return True
            else:
                self.logger.error("Failed to set system time")
                return False
        else:
            self.logger.info(f"Time difference ({time_diff:.2f}s) is small, no sync needed")
            return True

def main():
    parser = argparse.ArgumentParser(description='Synchronize system time with GPS')
    parser.add_argument('--port', default='/dev/ttyS6', help='GPS serial port (default: /dev/ttyS6)')
    parser.add_argument('--baudrate', type=int, default=115200, help='GPS baudrate (default: 115200)')
    parser.add_argument('--timeout', type=int, default=30, help='GPS read timeout in seconds (default: 30)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--continuous', '-c', action='store_true', help='Continuous monitoring mode')
    parser.add_argument('--interval', type=int, default=3600, help='Sync interval in seconds for continuous mode (default: 3600)')
    
    args = parser.parse_args()
    
    gps_sync = GPSTimeSync(args.port, args.baudrate, args.verbose)
    
    if args.continuous:
        print(f"Starting continuous GPS time sync (interval: {args.interval}s)")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                gps_sync.sync_time(args.timeout)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nGPS time sync stopped")
            sys.exit(0)
    else:
        # Single sync
        success = gps_sync.sync_time(args.timeout)
        sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()