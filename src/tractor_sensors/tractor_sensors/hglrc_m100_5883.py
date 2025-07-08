#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, NavSatStatus, MagneticField, Imu
from geometry_msgs.msg import QuaternionStamped
from geographic_msgs.msg import GeoPointStamped
import serial
import pynmea2
import math
import threading
import time
import struct

try:
    import smbus

    I2C_AVAILABLE = True
except ImportError:
    I2C_AVAILABLE = False
    print("smbus not available. Install with: sudo apt install python3-smbus")


class HGLRCM1005883Publisher(Node):
    def __init__(self):
        super().__init__("hglrc_m100_5883_publisher")

        # Parameters
        self.declare_parameter("gps_port", "/dev/ttyS6")
        # HGLRC M100-5883 uses 115200
        self.declare_parameter("gps_baudrate", 115200)
        # I2C bus number (QMC5883 on I2C port 5)
        self.declare_parameter("i2c_bus", 5)
        self.declare_parameter("qmc5883_address", 0x0D)  # QMC5883 I2C address
        self.declare_parameter("magnetic_declination", 0.0)  # degrees
        self.declare_parameter("gps_frame_id", "gps_link")
        self.declare_parameter("compass_frame_id", "compass_link")
        self.declare_parameter("compass_update_rate", 100.0)  # Hz

        self.gps_port = self.get_parameter("gps_port").value
        self.gps_baudrate = self.get_parameter("gps_baudrate").value
        self.i2c_bus = self.get_parameter("i2c_bus").value
        self.qmc5883_address = self.get_parameter("qmc5883_address").value
        self.magnetic_declination = math.radians(
            self.get_parameter("magnetic_declination").value
        )
        self.gps_frame_id = self.get_parameter("gps_frame_id").value
        self.compass_frame_id = self.get_parameter("compass_frame_id").value
        self.compass_update_rate = self.get_parameter("compass_update_rate").value

        # Initialize connections
        self.gps_serial = None
        self.i2c_bus_obj = None

        # Initialize GPS serial connection
        try:
            self.gps_serial = serial.Serial(
                self.gps_port,
                self.gps_baudrate,
                timeout=1,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
            )
            self.get_logger().info(
                f"HGLRC M100-5883 GPS serial port {
                    self.gps_port} opened at {
                    self.gps_baudrate} baud"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to open GPS serial port: {e}")

        # Initialize I2C for QMC5883
        if I2C_AVAILABLE:
            try:
                self.i2c_bus_obj = smbus.SMBus(self.i2c_bus)
                self.init_qmc5883()
                self.get_logger().info(
                    f"QMC5883 compass initialized on I2C bus {
                        self.i2c_bus}"
                )
            except Exception as e:
                self.get_logger().error(f"Failed to initialize QMC5883 compass: {e}")
                self.i2c_bus_obj = None
        else:
            self.get_logger().error("I2C not available - install python3-smbus")

        # Publishers
        self.navsat_pub = self.create_publisher(NavSatFix, "hglrc_gps/fix", 10)
        self.geopoint_pub = self.create_publisher(
            GeoPointStamped, "hglrc_gps/filtered", 10
        )
        self.magnetic_pub = self.create_publisher(
            MagneticField, "hglrc_gps/magnetic_field", 10
        )
        self.heading_pub = self.create_publisher(
            QuaternionStamped, "hglrc_gps/heading", 10
        )
        self.imu_pub = self.create_publisher(Imu, "hglrc_gps/imu", 10)

        # GPS data
        self.gps_fix_type = 0
        self.latitude = 0.0
        self.longitude = 0.0
        self.altitude = 0.0
        self.gps_lock = threading.Lock()

        # Compass data
        self.magnetic_x = 0.0
        self.magnetic_y = 0.0
        self.magnetic_z = 0.0
        self.compass_lock = threading.Lock()

        # Start threads
        if self.gps_serial:
            self.gps_thread = threading.Thread(target=self.read_gps_data, daemon=True)
            self.gps_thread.start()

        if self.i2c_bus_obj:
            self.compass_thread = threading.Thread(
                target=self.read_compass_data, daemon=True
            )
            self.compass_thread.start()

        # Publisher timer
        # 10 Hz for GPS, compass reads faster
        self.timer = self.create_timer(0.1, self.publish_data)

        self.get_logger().info(
            "HGLRC M100-5883 GPS and QMC5883 Compass Publisher initialized"
        )

    def init_qmc5883(self):
        """Initialize QMC5883 magnetometer"""
        try:
            # Soft reset - write 0x80 to control register 2 (0x0A)
            self.i2c_bus_obj.write_byte_data(self.qmc5883_address, 0x0A, 0x80)
            time.sleep(0.1)

            # Set/Reset period register (0x0B) - recommended value 0x01
            self.i2c_bus_obj.write_byte_data(self.qmc5883_address, 0x0B, 0x01)

            # Control register 1 (0x09): 
            # MODE[1:0] = 01 (continuous measurement mode)
            # ODR[3:2] = 00 (10Hz) - start with slower rate for stability
            # RNG[5:4] = 00 (±2G range)
            # OSR[7:6] = 00 (512 oversampling)
            # Combined: 0x01 for continuous mode, 10Hz, ±2G, OSR=512
            self.i2c_bus_obj.write_byte_data(self.qmc5883_address, 0x09, 0x01)
            time.sleep(0.1)

            # Read status register to verify communication
            status = self.i2c_bus_obj.read_byte_data(self.qmc5883_address, 0x06)
            self.get_logger().info(f"QMC5883 status after init: 0x{status:02X}")

            # Note: QMC5883L doesn't have a reliable chip ID register like HMC5883L
            # Instead, verify by checking if we can read status register
            self.get_logger().info("QMC5883 initialization completed")

        except Exception as e:
            self.get_logger().error(f"QMC5883 initialization error: {e}")
            raise

    def read_qmc5883_raw(self):
        """Read raw magnetometer data from QMC5883"""
        try:
            # Check data ready bit (DRDY) in status register 0x06
            status = self.i2c_bus_obj.read_byte_data(self.qmc5883_address, 0x06)
            if not (status & 0x01):  # DRDY bit
                return None, None, None

            # QMC5883L data registers: 0x00-0x05
            # 0x00: X LSB, 0x01: X MSB
            # 0x02: Y LSB, 0x03: Y MSB  
            # 0x04: Z LSB, 0x05: Z MSB
            data = []
            for reg in range(0x00, 0x06):
                data.append(self.i2c_bus_obj.read_byte_data(self.qmc5883_address, reg))

            # Convert to signed 16-bit values (little-endian: LSB first, then MSB)
            x = struct.unpack("<h", bytes([data[0], data[1]]))[0]
            y = struct.unpack("<h", bytes([data[2], data[3]]))[0]
            z = struct.unpack("<h", bytes([data[4], data[5]]))[0]

            return x, y, z

        except Exception as e:
            self.get_logger().debug(f"QMC5883 read error: {e}")
            return None, None, None

    def read_gps_data(self):
        """Read GPS NMEA data from serial port"""
        while rclpy.ok():
            try:
                if self.gps_serial and self.gps_serial.in_waiting:
                    line = (
                        self.gps_serial.readline()
                        .decode("ascii", errors="replace")
                        .strip()
                    )

                    if line.startswith("$"):
                        try:
                            msg = pynmea2.parse(line)

                            if isinstance(msg, pynmea2.GGA):
                                with self.gps_lock:
                                    if msg.latitude and msg.longitude:
                                        self.latitude = float(msg.latitude)
                                        self.longitude = float(msg.longitude)
                                        self.altitude = (
                                            float(msg.altitude) if msg.altitude else 0.0
                                        )
                                        self.gps_fix_type = (
                                            int(msg.gps_qual) if msg.gps_qual else 0
                                        )

                        except pynmea2.ParseError as e:
                            self.get_logger().debug(f"GPS parse error: {e}")
                        except Exception as e:
                            self.get_logger().debug(f"GPS processing error: {e}")

                time.sleep(0.01)  # Small delay

            except Exception as e:
                self.get_logger().error(f"GPS reading error: {e}")
                time.sleep(1.0)

    def read_compass_data(self):
        """Read compass data from QMC5883 via I2C"""
        update_period = 1.0 / self.compass_update_rate

        while rclpy.ok():
            try:
                x_raw, y_raw, z_raw = self.read_qmc5883_raw()

                if x_raw is not None:
                    # Convert to microTesla (QMC5883L specifications)
                    # QMC5883L ±2G range with 16-bit resolution (we set RNG=00 for ±2G)
                    # 1 Gauss = 100 microTesla
                    # ±2G range = ±200 microTesla over 16-bit signed range (-32768 to +32767)
                    scale_factor = 200.0 / 32768.0  # microTesla per LSB

                    with self.compass_lock:
                        self.magnetic_x = x_raw * scale_factor
                        self.magnetic_y = y_raw * scale_factor
                        self.magnetic_z = z_raw * scale_factor

                time.sleep(update_period)

            except Exception as e:
                self.get_logger().error(f"Compass reading error: {e}")
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
                    navsat_msg.status.status = NavSatStatus.STATUS_GBAS_FIX
                elif self.gps_fix_type >= 2:  # DGPS
                    navsat_msg.status.status = NavSatStatus.STATUS_SBAS_FIX
                else:  # GPS
                    navsat_msg.status.status = NavSatStatus.STATUS_FIX

                # M10 supports multiple constellations
                navsat_msg.status.service = (
                    NavSatStatus.SERVICE_GPS
                    | NavSatStatus.SERVICE_GLONASS
                    | NavSatStatus.SERVICE_GALILEO
                )

                # Improved covariance for M10 chip (~2.5m CEP)
                navsat_msg.position_covariance[0] = 6.25  # East (2.5m)^2
                navsat_msg.position_covariance[4] = 6.25  # North
                navsat_msg.position_covariance[8] = 16.0  # Up (4m)^2
                navsat_msg.position_covariance_type = (
                    NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
                )

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

            mag_msg.magnetic_field.x = self.magnetic_x * 1e-6  # Convert µT to T
            mag_msg.magnetic_field.y = self.magnetic_y * 1e-6
            mag_msg.magnetic_field.z = self.magnetic_z * 1e-6

            # QMC5883 covariance
            mag_msg.magnetic_field_covariance[0] = 1e-12  # Good accuracy
            mag_msg.magnetic_field_covariance[4] = 1e-12
            mag_msg.magnetic_field_covariance[8] = 1e-12

            self.magnetic_pub.publish(mag_msg)

            # Calculate heading from magnetometer
            if self.magnetic_x != 0 or self.magnetic_y != 0:
                magnetic_heading = math.atan2(self.magnetic_y, self.magnetic_x)
                true_heading = magnetic_heading + self.magnetic_declination

                # Heading as quaternion
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
                imu_msg.orientation_covariance[8] = 0.005  # Good yaw accuracy

                # Mark other fields as unavailable
                imu_msg.angular_velocity_covariance[0] = -1
                imu_msg.linear_acceleration_covariance[0] = -1

                self.imu_pub.publish(imu_msg)

    def destroy_node(self):
        """Clean shutdown"""
        try:
            if self.gps_serial:
                self.gps_serial.close()
            if self.i2c_bus_obj:
                self.i2c_bus_obj.close()
        except Exception as e:
            self.get_logger().error(f"Error during cleanup: {e}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    hglrc_pub = HGLRCM1005883Publisher()

    try:
        rclpy.spin(hglrc_pub)
    except KeyboardInterrupt:
        pass
    finally:
        hglrc_pub.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
