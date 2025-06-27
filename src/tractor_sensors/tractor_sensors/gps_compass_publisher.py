#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, MagneticField, Imu
from geometry_msgs.msg import QuaternionStamped
from geographic_msgs.msg import GeoPointStamped
import serial
import pynmea2
import math
import threading
import time


class GPSCompassPublisher(Node):
    def __init__(self):
        super().__init__('gps_compass_publisher')
        
        # Parameters
        self.declare_parameter('gps_port', '/dev/ttyUSB0')
        self.declare_parameter('gps_baudrate', 9600)
        self.declare_parameter('compass_port', '/dev/ttyUSB1')
        self.declare_parameter('compass_baudrate', 9600)
        self.declare_parameter('magnetic_declination', 0.0)  # degrees
        self.declare_parameter('gps_frame_id', 'gps_link')
        self.declare_parameter('compass_frame_id', 'compass_link')
        
        self.gps_port = self.get_parameter('gps_port').value
        self.gps_baudrate = self.get_parameter('gps_baudrate').value
        self.compass_port = self.get_parameter('compass_port').value
        self.compass_baudrate = self.get_parameter('compass_baudrate').value
        self.magnetic_declination = math.radians(self.get_parameter('magnetic_declination').value)
        self.gps_frame_id = self.get_parameter('gps_frame_id').value
        self.compass_frame_id = self.get_parameter('compass_frame_id').value
        
        # Initialize serial connections
        self.gps_serial = None
        self.compass_serial = None
        
        try:
            self.gps_serial = serial.Serial(self.gps_port, self.gps_baudrate, timeout=1)
            self.get_logger().info(f'GPS serial port {self.gps_port} opened successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to open GPS serial port: {e}')
        
        try:
            self.compass_serial = serial.Serial(self.compass_port, self.compass_baudrate, timeout=1)
            self.get_logger().info(f'Compass serial port {self.compass_port} opened successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to open compass serial port: {e}')
        
        # Publishers
        self.navsat_pub = self.create_publisher(NavSatFix, 'gps/fix', 10)
        self.geopoint_pub = self.create_publisher(GeoPointStamped, 'gps/filtered', 10)
        self.magnetic_pub = self.create_publisher(MagneticField, 'magnetic_field', 10)
        self.heading_pub = self.create_publisher(QuaternionStamped, 'compass/heading', 10)
        self.imu_pub = self.create_publisher(Imu, 'imu/data', 10)
        
        # GPS data
        self.gps_fix_type = 0
        self.latitude = 0.0
        self.longitude = 0.0
        self.altitude = 0.0
        self.gps_lock = threading.Lock()
        
        # Compass data
        self.magnetic_heading = 0.0
        self.compass_lock = threading.Lock()
        
        # Start serial reading threads
        if self.gps_serial:
            self.gps_thread = threading.Thread(target=self.read_gps_data, daemon=True)
            self.gps_thread.start()
        
        if self.compass_serial:
            self.compass_thread = threading.Thread(target=self.read_compass_data, daemon=True)
            self.compass_thread.start()
        
        # Publisher timer
        self.timer = self.create_timer(0.1, self.publish_data)  # 10 Hz
        
        self.get_logger().info('GPS and Compass Publisher initialized')
    
    def read_gps_data(self):
        """Read GPS data from serial port"""
        while rclpy.ok():
            try:
                if self.gps_serial and self.gps_serial.in_waiting:
                    line = self.gps_serial.readline().decode('ascii', errors='replace').strip()
                    
                    if line.startswith('$'):
                        try:
                            msg = pynmea2.parse(line)
                            
                            if isinstance(msg, pynmea2.GGA):
                                with self.gps_lock:
                                    if msg.latitude and msg.longitude:
                                        self.latitude = float(msg.latitude)
                                        self.longitude = float(msg.longitude)
                                        self.altitude = float(msg.altitude) if msg.altitude else 0.0
                                        self.gps_fix_type = int(msg.gps_qual) if msg.gps_qual else 0
                            
                        except pynmea2.ParseError as e:
                            self.get_logger().debug(f'GPS parse error: {e}')
                        except Exception as e:
                            self.get_logger().debug(f'GPS processing error: {e}')
                
                time.sleep(0.01)  # Small delay
                
            except Exception as e:
                self.get_logger().error(f'GPS reading error: {e}')
                time.sleep(1.0)
    
    def read_compass_data(self):
        """Read compass data from serial port"""
        while rclpy.ok():
            try:
                if self.compass_serial and self.compass_serial.in_waiting:
                    line = self.compass_serial.readline().decode('ascii', errors='replace').strip()
                    
                    # Parse compass data (assuming NMEA HDT format or custom format)
                    if line.startswith('$') and 'HDT' in line:
                        try:
                            parts = line.split(',')
                            if len(parts) > 1:
                                heading = float(parts[1])
                                with self.compass_lock:
                                    self.magnetic_heading = math.radians(heading)
                        except (ValueError, IndexError) as e:
                            self.get_logger().debug(f'Compass parse error: {e}')
                    
                    # Custom compass format (example: "HEADING:123.45")
                    elif line.startswith('HEADING:'):
                        try:
                            heading = float(line.split(':')[1])
                            with self.compass_lock:
                                self.magnetic_heading = math.radians(heading)
                        except (ValueError, IndexError) as e:
                            self.get_logger().debug(f'Compass parse error: {e}')
                
                time.sleep(0.01)  # Small delay
                
            except Exception as e:
                self.get_logger().error(f'Compass reading error: {e}')
                time.sleep(1.0)
    
    def publish_data(self):
        """Publish GPS and compass data"""
        current_time = self.get_clock().now()
        
        # Publish GPS data
        with self.gps_lock:
            if self.gps_fix_type > 0:
                # NavSatFix message
                navsat_msg = NavSatFix()
                navsat_msg.header.stamp = current_time.to_msg()
                navsat_msg.header.frame_id = self.gps_frame_id
                navsat_msg.latitude = self.latitude
                navsat_msg.longitude = self.longitude
                navsat_msg.altitude = self.altitude
                
                # Set status based on fix type
                if self.gps_fix_type >= 4:  # RTK
                    navsat_msg.status.status = NavSatFix.STATUS_GBAS_FIX
                elif self.gps_fix_type >= 2:  # DGPS
                    navsat_msg.status.status = NavSatFix.STATUS_SBAS_FIX
                else:  # GPS
                    navsat_msg.status.status = NavSatFix.STATUS_FIX
                
                navsat_msg.status.service = NavSatFix.SERVICE_GPS
                
                # Simple covariance (should be calibrated based on actual GPS performance)
                navsat_msg.position_covariance[0] = 1.0  # East
                navsat_msg.position_covariance[4] = 1.0  # North
                navsat_msg.position_covariance[8] = 4.0  # Up
                navsat_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
                
                self.navsat_pub.publish(navsat_msg)
                
                # GeoPointStamped message
                geopoint_msg = GeoPointStamped()
                geopoint_msg.header.stamp = current_time.to_msg()
                geopoint_msg.header.frame_id = self.gps_frame_id
                geopoint_msg.position.latitude = self.latitude
                geopoint_msg.position.longitude = self.longitude
                geopoint_msg.position.altitude = self.altitude
                
                self.geopoint_pub.publish(geopoint_msg)
        
        # Publish compass data
        with self.compass_lock:
            # Magnetic field message
            mag_msg = MagneticField()
            mag_msg.header.stamp = current_time.to_msg()
            mag_msg.header.frame_id = self.compass_frame_id
            
            # Convert heading to magnetic field vector (simplified)
            mag_msg.magnetic_field.x = math.cos(self.magnetic_heading)
            mag_msg.magnetic_field.y = math.sin(self.magnetic_heading)
            mag_msg.magnetic_field.z = 0.0
            
            # Covariance
            mag_msg.magnetic_field_covariance[0] = 0.01
            mag_msg.magnetic_field_covariance[4] = 0.01
            mag_msg.magnetic_field_covariance[8] = 0.01
            
            self.magnetic_pub.publish(mag_msg)
            
            # Heading as quaternion
            true_heading = self.magnetic_heading + self.magnetic_declination
            heading_msg = QuaternionStamped()
            heading_msg.header.stamp = current_time.to_msg()
            heading_msg.header.frame_id = self.compass_frame_id
            heading_msg.quaternion.x = 0.0
            heading_msg.quaternion.y = 0.0
            heading_msg.quaternion.z = math.sin(true_heading / 2.0)
            heading_msg.quaternion.w = math.cos(true_heading / 2.0)
            
            self.heading_pub.publish(heading_msg)
            
            # IMU message (compass only provides yaw)
            imu_msg = Imu()
            imu_msg.header.stamp = current_time.to_msg()
            imu_msg.header.frame_id = self.compass_frame_id
            imu_msg.orientation = heading_msg.quaternion
            imu_msg.orientation_covariance[8] = 0.01  # yaw covariance
            
            # Mark other fields as unavailable
            imu_msg.angular_velocity_covariance[0] = -1
            imu_msg.linear_acceleration_covariance[0] = -1
            
            self.imu_pub.publish(imu_msg)
    
    def destroy_node(self):
        """Clean shutdown"""
        try:
            if self.gps_serial:
                self.gps_serial.close()
            if self.compass_serial:
                self.compass_serial.close()
        except Exception as e:
            self.get_logger().error(f'Error during serial cleanup: {e}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    gps_compass_pub = GPSCompassPublisher()
    
    try:
        rclpy.spin(gps_compass_pub)
    except KeyboardInterrupt:
        pass
    finally:
        gps_compass_pub.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()