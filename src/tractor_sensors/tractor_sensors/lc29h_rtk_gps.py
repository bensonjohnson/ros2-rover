#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, NavSatStatus
from geographic_msgs.msg import GeoPointStamped
from geometry_msgs.msg import TwistWithCovarianceStamped
import serial
import pynmea2
import math
import threading
import time
import socket
import requests
from typing import Optional


class LC29HRTKPublisher(Node):
    def __init__(self):
        super().__init__("lc29h_rtk_gps_publisher")

        # Parameters
        self.declare_parameter("gps_port", "/dev/ttyUSB0")
        self.declare_parameter("gps_baudrate", 460800)  # LC29H typical high speed
        self.declare_parameter("gps_frame_id", "gps_link")
        self.declare_parameter("rtk_mode", "rover")  # "rover", "base", "disabled"
        
        # RTK Parameters
        self.declare_parameter("ntrip_host", "")
        self.declare_parameter("ntrip_port", 2101)
        self.declare_parameter("ntrip_mountpoint", "")
        self.declare_parameter("ntrip_username", "")
        self.declare_parameter("ntrip_password", "")
        self.declare_parameter("rtcm_timeout", 10.0)  # seconds
        
        # Base station parameters
        self.declare_parameter("base_latitude", 0.0)
        self.declare_parameter("base_longitude", 0.0)
        self.declare_parameter("base_altitude", 0.0)
        self.declare_parameter("base_accuracy_limit", 3.0)  # meters
        self.declare_parameter("base_observation_time", 300)  # seconds

        self.gps_port = self.get_parameter("gps_port").value
        self.gps_baudrate = self.get_parameter("gps_baudrate").value
        self.gps_frame_id = self.get_parameter("gps_frame_id").value
        self.rtk_mode = self.get_parameter("rtk_mode").value
        
        self.ntrip_host = self.get_parameter("ntrip_host").value
        self.ntrip_port = self.get_parameter("ntrip_port").value
        self.ntrip_mountpoint = self.get_parameter("ntrip_mountpoint").value
        self.ntrip_username = self.get_parameter("ntrip_username").value
        self.ntrip_password = self.get_parameter("ntrip_password").value
        self.rtcm_timeout = self.get_parameter("rtcm_timeout").value
        
        self.base_latitude = self.get_parameter("base_latitude").value
        self.base_longitude = self.get_parameter("base_longitude").value
        self.base_altitude = self.get_parameter("base_altitude").value
        self.base_accuracy_limit = self.get_parameter("base_accuracy_limit").value
        self.base_observation_time = self.get_parameter("base_observation_time").value

        # Initialize GPS serial connection
        self.gps_serial = None
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
                f"LC29H RTK GPS serial port {self.gps_port} opened at {self.gps_baudrate} baud"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to open GPS serial port: {e}")
            return

        # Publishers
        self.navsat_pub = self.create_publisher(NavSatFix, "gps/fix", 10)
        self.geopoint_pub = self.create_publisher(GeoPointStamped, "gps/filtered", 10)
        self.velocity_pub = self.create_publisher(
            TwistWithCovarianceStamped, "gps/velocity", 10
        )

        # GPS data
        self.gps_fix_type = 0
        self.latitude = 0.0
        self.longitude = 0.0
        self.altitude = 0.0
        self.speed = 0.0  # m/s
        self.course = 0.0  # degrees
        self.hdop = 99.9
        self.vdop = 99.9
        self.satellites_used = 0
        self.rtk_status = 0  # 0=none, 1=float, 2=fixed
        self.gps_lock = threading.Lock()

        # RTK/NTRIP connection
        self.ntrip_socket = None
        self.ntrip_connected = False
        self.last_rtcm_time = time.time()

        # Initialize GPS configuration
        self.configure_gps()

        # Start GPS reading thread
        if self.gps_serial:
            self.gps_thread = threading.Thread(target=self.read_gps_data, daemon=True)
            self.gps_thread.start()

            # Start NTRIP client if in rover mode
            if self.rtk_mode == "rover" and self.ntrip_host:
                self.ntrip_thread = threading.Thread(
                    target=self.ntrip_client_loop, daemon=True
                )
                self.ntrip_thread.start()

        # Publisher timer - 10 Hz
        self.timer = self.create_timer(0.1, self.publish_data)

        self.get_logger().info(
            f"LC29H RTK GPS Publisher initialized in {self.rtk_mode} mode"
        )

    def configure_gps(self):
        """Configure LC29H GPS for RTK operation"""
        try:
            # Wait for GPS to boot up
            time.sleep(2)

            # Restore default parameters
            self.send_command("$PQTMRESTOREPAR*13")
            time.sleep(1)

            if self.rtk_mode == "rover":
                self.get_logger().info("Configuring LC29H as RTK rover")
                # Set rover mode
                self.send_command("$PQTMCFGRCVRMODE,W,1*2A")
                # Disable certain NMEA messages to reduce noise
                self.send_command("$PAIR062,2,0*3C")
                # Set position output to 5 Hz (200ms interval)
                self.send_command("$PAIR050,200*21")
                
            elif self.rtk_mode == "base":
                self.get_logger().info("Configuring LC29H as RTK base station")
                # Set base mode
                self.send_command("$PQTMCFGRCVRMODE,W,2*29")
                # Output RTCM3 MSM7 messages
                self.send_command("$PAIR432,1*22")
                # Output RTCM3 antenna position
                self.send_command("$PAIR434,1*24")
                
                # Configure base station position if provided
                if (self.base_latitude != 0.0 or 
                    self.base_longitude != 0.0 or 
                    self.base_altitude != 0.0):
                    # Set known base position
                    cmd = (f"$PQTMCFGSVIN,W,2,0,0,"
                           f"{self.base_latitude},{self.base_longitude},{self.base_altitude}*")
                    checksum = self.calculate_checksum(cmd[1:-1])
                    self.send_command(f"{cmd}{checksum:02X}")
                else:
                    # Auto-survey mode
                    cmd = (f"$PQTMCFGSVIN,W,1,{self.base_observation_time},"
                           f"{self.base_accuracy_limit},0,0,0*")
                    checksum = self.calculate_checksum(cmd[1:-1])
                    self.send_command(f"{cmd}{checksum:02X}")

            else:  # Standard GPS mode
                self.get_logger().info("Configuring LC29H for standard GPS operation")
                # Standard GNSS mode
                self.send_command("$PQTMCFGRCVRMODE,W,0*2B")

            # Save configuration
            self.send_command("$PQTMSAVEPAR*5A")
            time.sleep(1)

        except Exception as e:
            self.get_logger().error(f"GPS configuration error: {e}")

    def calculate_checksum(self, data: str) -> int:
        """Calculate NMEA checksum"""
        checksum = 0
        for char in data:
            checksum ^= ord(char)
        return checksum

    def send_command(self, command: str):
        """Send command to GPS"""
        if self.gps_serial and self.gps_serial.is_open:
            try:
                self.gps_serial.write((command + "\r\n").encode())
                self.gps_serial.flush()
                time.sleep(0.1)
                self.get_logger().debug(f"Sent GPS command: {command}")
            except Exception as e:
                self.get_logger().error(f"Failed to send GPS command {command}: {e}")

    def ntrip_client_loop(self):
        """NTRIP client main loop for RTK corrections"""
        while rclpy.ok() and self.rtk_mode == "rover":
            try:
                if not self.ntrip_connected:
                    self.connect_ntrip()
                
                if self.ntrip_connected:
                    self.receive_rtcm_corrections()
                else:
                    time.sleep(5)  # Wait before retry

            except Exception as e:
                self.get_logger().error(f"NTRIP client error: {e}")
                self.ntrip_connected = False
                time.sleep(10)

    def connect_ntrip(self) -> bool:
        """Connect to NTRIP caster"""
        try:
            # Close existing connection
            if self.ntrip_socket:
                self.ntrip_socket.close()

            # Create HTTP request for NTRIP
            ntrip_request = (
                f"GET /{self.ntrip_mountpoint} HTTP/1.0\r\n"
                f"User-Agent: LC29H-ROS2-Client\r\n"
                f"Accept: */*\r\n"
                f"Connection: close\r\n"
            )

            if self.ntrip_username and self.ntrip_password:
                import base64
                credentials = base64.b64encode(
                    f"{self.ntrip_username}:{self.ntrip_password}".encode()
                ).decode()
                ntrip_request += f"Authorization: Basic {credentials}\r\n"

            ntrip_request += "\r\n"

            # Connect to NTRIP caster
            self.ntrip_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.ntrip_socket.settimeout(10)
            self.ntrip_socket.connect((self.ntrip_host, self.ntrip_port))
            
            # Send HTTP request
            self.ntrip_socket.send(ntrip_request.encode())
            
            # Read HTTP response
            response = self.ntrip_socket.recv(1024).decode()
            if "200 OK" in response:
                self.ntrip_connected = True
                self.last_rtcm_time = time.time()
                self.get_logger().info(
                    f"Connected to NTRIP caster {self.ntrip_host}:{self.ntrip_port}/{self.ntrip_mountpoint}"
                )
                return True
            else:
                self.get_logger().error(f"NTRIP connection failed: {response}")
                return False

        except Exception as e:
            self.get_logger().error(f"NTRIP connection error: {e}")
            return False

    def receive_rtcm_corrections(self):
        """Receive RTCM corrections and forward to GPS"""
        try:
            if not self.ntrip_socket:
                return

            # Set socket timeout
            self.ntrip_socket.settimeout(1.0)
            
            # Receive RTCM data
            rtcm_data = self.ntrip_socket.recv(1024)
            
            if rtcm_data:
                # Forward RTCM data to GPS receiver
                if self.gps_serial and self.gps_serial.is_open:
                    self.gps_serial.write(rtcm_data)
                    self.last_rtcm_time = time.time()
                    
                self.get_logger().debug(f"Received {len(rtcm_data)} bytes of RTCM data")
            
            # Check for timeout
            if time.time() - self.last_rtcm_time > self.rtcm_timeout:
                self.get_logger().warn("RTCM data timeout - reconnecting")
                self.ntrip_connected = False

        except socket.timeout:
            # Normal timeout - continue loop
            pass
        except Exception as e:
            self.get_logger().error(f"RTCM receive error: {e}")
            self.ntrip_connected = False

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
                            
                            # Process GGA messages (position and quality)
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
                                        self.hdop = (
                                            float(msg.horizontal_dil) if msg.horizontal_dil else 99.9
                                        )
                                        self.satellites_used = (
                                            int(msg.num_sats) if msg.num_sats else 0
                                        )
                            
                            # Process RMC messages (speed and course)
                            elif isinstance(msg, pynmea2.RMC):
                                with self.gps_lock:
                                    if msg.spd_over_grnd is not None:
                                        # Convert knots to m/s
                                        self.speed = float(msg.spd_over_grnd) * 0.514444
                                    if msg.true_course is not None:
                                        self.course = float(msg.true_course)

                            # Process GSA messages (DOP and satellites)
                            elif isinstance(msg, pynmea2.GSA):
                                with self.gps_lock:
                                    if hasattr(msg, 'vdop') and msg.vdop:
                                        self.vdop = float(msg.vdop)

                        except pynmea2.ParseError:
                            pass  # Ignore parse errors
                        except Exception as e:
                            self.get_logger().debug(f"GPS message processing error: {e}")

                time.sleep(0.01)  # Small delay

            except Exception as e:
                self.get_logger().error(f"GPS reading error: {e}")
                time.sleep(1.0)

    def publish_data(self):
        """Publish GPS data"""
        current_time = self.get_clock().now()

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
                if self.gps_fix_type >= 5:  # RTK Fixed
                    navsat_msg.status.status = NavSatStatus.STATUS_GBAS_FIX
                    self.rtk_status = 2
                elif self.gps_fix_type >= 4:  # RTK Float
                    navsat_msg.status.status = NavSatStatus.STATUS_SBAS_FIX
                    self.rtk_status = 1
                elif self.gps_fix_type >= 2:  # DGPS
                    navsat_msg.status.status = NavSatStatus.STATUS_SBAS_FIX
                    self.rtk_status = 0
                else:  # Standard GPS
                    navsat_msg.status.status = NavSatStatus.STATUS_FIX
                    self.rtk_status = 0

                # LC29H supports multiple constellations
                navsat_msg.status.service = (
                    NavSatStatus.SERVICE_GPS
                    | NavSatStatus.SERVICE_GLONASS
                    | NavSatStatus.SERVICE_GALILEO
                    | NavSatStatus.SERVICE_COMPASS  # BeiDou
                )

                # Set covariance based on fix quality
                if self.rtk_status == 2:  # RTK Fixed - cm level accuracy
                    pos_var = 0.01  # 10cm standard deviation
                elif self.rtk_status == 1:  # RTK Float - dm level accuracy
                    pos_var = 0.25  # 50cm standard deviation
                else:  # Standard GPS - use HDOP
                    pos_var = max(1.0, self.hdop) ** 2

                navsat_msg.position_covariance[0] = pos_var  # East
                navsat_msg.position_covariance[4] = pos_var  # North
                navsat_msg.position_covariance[8] = pos_var * 2  # Up (typically worse)
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

                # Velocity message
                if self.speed > 0.1:  # Only publish if moving
                    velocity_msg = TwistWithCovarianceStamped()
                    velocity_msg.header.stamp = current_time.to_msg()
                    velocity_msg.header.frame_id = self.gps_frame_id
                    
                    # Convert course to velocity components
                    course_rad = math.radians(self.course)
                    velocity_msg.twist.twist.linear.x = self.speed * math.cos(course_rad)
                    velocity_msg.twist.twist.linear.y = self.speed * math.sin(course_rad)
                    velocity_msg.twist.twist.linear.z = 0.0

                    # Velocity covariance (higher uncertainty at low speeds)
                    vel_var = max(0.1, 0.05 * self.speed) ** 2
                    velocity_msg.twist.covariance[0] = vel_var  # vx
                    velocity_msg.twist.covariance[7] = vel_var  # vy
                    velocity_msg.twist.covariance[14] = vel_var * 10  # vz
                    
                    self.velocity_pub.publish(velocity_msg)

    def destroy_node(self):
        """Clean shutdown"""
        try:
            if self.ntrip_socket:
                self.ntrip_socket.close()
            if self.gps_serial:
                self.gps_serial.close()
        except Exception as e:
            self.get_logger().error(f"Error during cleanup: {e}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    lc29h_pub = LC29HRTKPublisher()

    try:
        rclpy.spin(lc29h_pub)
    except KeyboardInterrupt:
        pass
    finally:
        lc29h_pub.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()