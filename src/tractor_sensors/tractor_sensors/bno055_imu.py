#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, MagneticField, Temperature
from geometry_msgs.msg import QuaternionStamped, Vector3Stamped
from std_msgs.msg import Int8, String
import time
import math
import threading
import struct

try:
    import smbus
    I2C_AVAILABLE = True
except ImportError:
    I2C_AVAILABLE = False
    print("smbus not available. Install with: sudo apt install python3-smbus")


class BNO055Publisher(Node):
    # BNO055 I2C addresses
    BNO055_ADDRESS_A = 0x28
    BNO055_ADDRESS_B = 0x29
    
    # BNO055 Register addresses
    BNO055_CHIP_ID_ADDR = 0x00
    BNO055_PAGE_ID_ADDR = 0x07
    BNO055_ACCEL_REV_ID_ADDR = 0x01
    BNO055_MAG_REV_ID_ADDR = 0x02
    BNO055_GYRO_REV_ID_ADDR = 0x03
    BNO055_SW_REV_ID_LSB_ADDR = 0x04
    BNO055_SW_REV_ID_MSB_ADDR = 0x05
    BNO055_BL_REV_ID_ADDR = 0x06

    # Mode registers
    BNO055_OPR_MODE_ADDR = 0x3D
    BNO055_PWR_MODE_ADDR = 0x3E
    BNO055_SYS_TRIGGER_ADDR = 0x3F

    # Data registers
    BNO055_EULER_H_LSB_ADDR = 0x1A
    BNO055_QUATERNION_DATA_W_LSB_ADDR = 0x20
    BNO055_LINEAR_ACCEL_DATA_X_LSB_ADDR = 0x28
    BNO055_GRAVITY_DATA_X_LSB_ADDR = 0x2E
    BNO055_GYRO_DATA_X_LSB_ADDR = 0x14
    BNO055_ACCEL_DATA_X_LSB_ADDR = 0x08
    BNO055_MAG_DATA_X_LSB_ADDR = 0x0E
    BNO055_TEMP_ADDR = 0x34

    # Calibration registers
    BNO055_CALIB_STAT_ADDR = 0x35
    BNO055_SELFTEST_RESULT_ADDR = 0x36
    BNO055_SYS_CLK_STAT_ADDR = 0x38
    BNO055_SYS_STAT_ADDR = 0x39
    BNO055_SYS_ERR_ADDR = 0x3A

    # Power modes
    POWER_MODE_NORMAL = 0x00
    POWER_MODE_LOWPOWER = 0x01
    POWER_MODE_SUSPEND = 0x02

    # Operation modes
    OPERATION_MODE_CONFIG = 0x00
    OPERATION_MODE_ACCONLY = 0x01
    OPERATION_MODE_MAGONLY = 0x02
    OPERATION_MODE_GYRONLY = 0x03
    OPERATION_MODE_ACCMAG = 0x04
    OPERATION_MODE_ACCGYRO = 0x05
    OPERATION_MODE_MAGGYRO = 0x06
    OPERATION_MODE_AMG = 0x07
    OPERATION_MODE_IMUPLUS = 0x08
    OPERATION_MODE_COMPASS = 0x09
    OPERATION_MODE_M4G = 0x0A
    OPERATION_MODE_NDOF_FMC_OFF = 0x0B
    OPERATION_MODE_NDOF = 0x0C

    def __init__(self):
        super().__init__("bno055_imu_publisher")

        # Parameters
        self.declare_parameter("i2c_bus", 1)
        self.declare_parameter("i2c_address", self.BNO055_ADDRESS_A)
        self.declare_parameter("frame_id", "imu_link")
        self.declare_parameter("operation_mode", "NDOF")  # NDOF, IMUPLUS, etc.
        self.declare_parameter("update_rate", 50.0)  # Hz
        self.declare_parameter("auto_calibrate", True)
        self.declare_parameter("temperature_compensation", True)

        self.i2c_bus = self.get_parameter("i2c_bus").value
        self.i2c_address = self.get_parameter("i2c_address").value
        self.frame_id = self.get_parameter("frame_id").value
        self.operation_mode_str = self.get_parameter("operation_mode").value
        self.update_rate = self.get_parameter("update_rate").value
        self.auto_calibrate = self.get_parameter("auto_calibrate").value
        self.temperature_compensation = self.get_parameter("temperature_compensation").value

        # Convert operation mode string to value
        self.operation_mode = getattr(
            self, f"OPERATION_MODE_{self.operation_mode_str}", 
            self.OPERATION_MODE_NDOF
        )

        # Initialize I2C
        self.i2c = None
        if not I2C_AVAILABLE:
            self.get_logger().error("I2C not available - install python3-smbus")
            return

        try:
            self.i2c = smbus.SMBus(self.i2c_bus)
            if not self.init_bno055():
                self.get_logger().error("Failed to initialize BNO055")
                return
        except Exception as e:
            self.get_logger().error(f"Failed to initialize I2C bus {self.i2c_bus}: {e}")
            return

        # Publishers
        self.imu_pub = self.create_publisher(Imu, "imu/data", 10)
        self.mag_pub = self.create_publisher(MagneticField, "imu/mag", 10)
        self.temp_pub = self.create_publisher(Temperature, "imu/temperature", 10)
        self.euler_pub = self.create_publisher(Vector3Stamped, "imu/euler", 10)
        self.quaternion_pub = self.create_publisher(QuaternionStamped, "imu/quaternion", 10)
        self.calibration_pub = self.create_publisher(Int8, "imu/calibration_status", 10)
        self.status_pub = self.create_publisher(String, "imu/status", 10)

        # Data storage
        self.sensor_data = {
            'quaternion': [0.0, 0.0, 0.0, 1.0],  # x, y, z, w
            'euler': [0.0, 0.0, 0.0],  # roll, pitch, yaw
            'gyro': [0.0, 0.0, 0.0],  # x, y, z (rad/s)
            'accel': [0.0, 0.0, 0.0],  # x, y, z (m/s²)
            'linear_accel': [0.0, 0.0, 0.0],  # x, y, z (m/s²)
            'gravity': [0.0, 0.0, 0.0],  # x, y, z (m/s²)
            'mag': [0.0, 0.0, 0.0],  # x, y, z (µT)
            'temperature': 0.0,  # °C
            'calibration': 0,
            'system_status': 0,
            'system_error': 0
        }
        self.data_lock = threading.Lock()

        # Start sensor reading thread
        self.sensor_thread = threading.Thread(target=self.read_sensor_data, daemon=True)
        self.sensor_thread.start()

        # Publisher timer
        update_period = 1.0 / self.update_rate
        self.timer = self.create_timer(update_period, self.publish_data)

        self.get_logger().info(
            f"BNO055 IMU Publisher initialized on I2C bus {self.i2c_bus}, "
            f"address 0x{self.i2c_address:02X}, mode: {self.operation_mode_str}"
        )

    def init_bno055(self) -> bool:
        """Initialize BNO055 sensor"""
        try:
            # Check chip ID
            chip_id = self.read_byte(self.BNO055_CHIP_ID_ADDR)
            if chip_id != 0xA0:
                self.get_logger().error(f"Invalid chip ID: 0x{chip_id:02X}, expected 0xA0")
                return False

            self.get_logger().info("BNO055 chip detected")

            # Read revision IDs
            accel_rev = self.read_byte(self.BNO055_ACCEL_REV_ID_ADDR)
            mag_rev = self.read_byte(self.BNO055_MAG_REV_ID_ADDR)
            gyro_rev = self.read_byte(self.BNO055_GYRO_REV_ID_ADDR)
            sw_rev_lsb = self.read_byte(self.BNO055_SW_REV_ID_LSB_ADDR)
            sw_rev_msb = self.read_byte(self.BNO055_SW_REV_ID_MSB_ADDR)
            bl_rev = self.read_byte(self.BNO055_BL_REV_ID_ADDR)

            sw_rev = (sw_rev_msb << 8) | sw_rev_lsb

            self.get_logger().info(
                f"BNO055 Firmware versions - Accel: {accel_rev}, "
                f"Mag: {mag_rev}, Gyro: {gyro_rev}, SW: {sw_rev}, BL: {bl_rev}"
            )

            # Set to config mode
            self.write_byte(self.BNO055_OPR_MODE_ADDR, self.OPERATION_MODE_CONFIG)
            time.sleep(0.025)

            # Reset
            self.write_byte(self.BNO055_SYS_TRIGGER_ADDR, 0x20)
            time.sleep(0.65)  # Wait for reset

            # Check chip ID again after reset
            chip_id = self.read_byte(self.BNO055_CHIP_ID_ADDR)
            if chip_id != 0xA0:
                self.get_logger().error(f"Chip ID lost after reset: 0x{chip_id:02X}")
                return False

            # Set power mode to normal
            self.write_byte(self.BNO055_PWR_MODE_ADDR, self.POWER_MODE_NORMAL)
            time.sleep(0.01)

            # Set page ID to 0
            self.write_byte(self.BNO055_PAGE_ID_ADDR, 0)

            # Use external crystal
            self.write_byte(self.BNO055_SYS_TRIGGER_ADDR, 0x80)
            time.sleep(0.01)

            # Set operation mode
            self.write_byte(self.BNO055_OPR_MODE_ADDR, self.operation_mode)
            time.sleep(0.02)

            self.get_logger().info(f"BNO055 initialized in mode: {self.operation_mode_str}")
            return True

        except Exception as e:
            self.get_logger().error(f"BNO055 initialization error: {e}")
            return False

    def read_byte(self, register: int) -> int:
        """Read single byte from register"""
        return self.i2c.read_byte_data(self.i2c_address, register)

    def write_byte(self, register: int, value: int):
        """Write single byte to register"""
        self.i2c.write_byte_data(self.i2c_address, register, value)

    def read_bytes(self, register: int, length: int) -> list:
        """Read multiple bytes from register"""
        return self.i2c.read_i2c_block_data(self.i2c_address, register, length)

    def read_vector(self, register: int, scale: float = 1.0) -> list:
        """Read 3-axis vector data (6 bytes, little endian)"""
        try:
            data = self.read_bytes(register, 6)
            
            # Convert to signed 16-bit integers (little endian)
            x = struct.unpack('<h', bytes([data[0], data[1]]))[0] * scale
            y = struct.unpack('<h', bytes([data[2], data[3]]))[0] * scale
            z = struct.unpack('<h', bytes([data[4], data[5]]))[0] * scale
            
            return [x, y, z]
        except:
            return [0.0, 0.0, 0.0]

    def read_quaternion(self) -> list:
        """Read quaternion data (8 bytes)"""
        try:
            data = self.read_bytes(self.BNO055_QUATERNION_DATA_W_LSB_ADDR, 8)
            
            # Convert to signed 16-bit integers and scale
            w = struct.unpack('<h', bytes([data[0], data[1]]))[0] / 16384.0
            x = struct.unpack('<h', bytes([data[2], data[3]]))[0] / 16384.0
            y = struct.unpack('<h', bytes([data[4], data[5]]))[0] / 16384.0
            z = struct.unpack('<h', bytes([data[6], data[7]]))[0] / 16384.0
            
            return [x, y, z, w]
        except:
            return [0.0, 0.0, 0.0, 1.0]

    def read_euler(self) -> list:
        """Read Euler angles in degrees"""
        try:
            data = self.read_bytes(self.BNO055_EULER_H_LSB_ADDR, 6)
            
            # Convert to signed 16-bit integers and scale to degrees
            heading = struct.unpack('<h', bytes([data[0], data[1]]))[0] / 16.0
            roll = struct.unpack('<h', bytes([data[2], data[3]]))[0] / 16.0
            pitch = struct.unpack('<h', bytes([data[4], data[5]]))[0] / 16.0
            
            return [roll, pitch, heading]
        except:
            return [0.0, 0.0, 0.0]

    def read_calibration_status(self) -> int:
        """Read calibration status"""
        try:
            cal_stat = self.read_byte(self.BNO055_CALIB_STAT_ADDR)
            # Extract system calibration (bits 7:6)
            sys_cal = (cal_stat >> 6) & 0x03
            return sys_cal
        except:
            return 0

    def read_sensor_data(self):
        """Main sensor reading loop"""
        while rclpy.ok():
            try:
                # Read all sensor data
                quaternion = self.read_quaternion()
                euler = self.read_euler()
                
                # Read raw sensor data
                gyro = self.read_vector(self.BNO055_GYRO_DATA_X_LSB_ADDR, 1/900.0)  # dps to rad/s
                accel = self.read_vector(self.BNO055_ACCEL_DATA_X_LSB_ADDR, 0.01)  # mg to m/s²
                linear_accel = self.read_vector(self.BNO055_LINEAR_ACCEL_DATA_X_LSB_ADDR, 0.01)
                gravity = self.read_vector(self.BNO055_GRAVITY_DATA_X_LSB_ADDR, 0.01)
                mag = self.read_vector(self.BNO055_MAG_DATA_X_LSB_ADDR, 0.16)  # µT
                
                # Convert gyro from degrees/s to rad/s
                gyro = [g * math.pi / 180.0 for g in gyro]

                # Read temperature
                try:
                    temp_raw = self.read_byte(self.BNO055_TEMP_ADDR)
                    temperature = temp_raw  # Temperature in Celsius
                except:
                    temperature = 0.0

                # Read status
                calibration = self.read_calibration_status()
                
                try:
                    system_status = self.read_byte(self.BNO055_SYS_STAT_ADDR)
                    system_error = self.read_byte(self.BNO055_SYS_ERR_ADDR)
                except:
                    system_status = 0
                    system_error = 0

                # Store data
                with self.data_lock:
                    self.sensor_data.update({
                        'quaternion': quaternion,
                        'euler': euler,
                        'gyro': gyro,
                        'accel': accel,
                        'linear_accel': linear_accel,
                        'gravity': gravity,
                        'mag': mag,
                        'temperature': temperature,
                        'calibration': calibration,
                        'system_status': system_status,
                        'system_error': system_error
                    })

                # Update rate control
                time.sleep(0.01)  # 100Hz max read rate

            except Exception as e:
                self.get_logger().debug(f"Sensor reading error: {e}")
                time.sleep(0.1)

    def publish_data(self):
        """Publish sensor data"""
        current_time = self.get_clock().now()

        with self.data_lock:
            data = self.sensor_data.copy()

        # IMU message (main output)
        imu_msg = Imu()
        imu_msg.header.stamp = current_time.to_msg()
        imu_msg.header.frame_id = self.frame_id

        # Orientation (quaternion)
        imu_msg.orientation.x = data['quaternion'][0]
        imu_msg.orientation.y = data['quaternion'][1]
        imu_msg.orientation.z = data['quaternion'][2]
        imu_msg.orientation.w = data['quaternion'][3]

        # Angular velocity
        imu_msg.angular_velocity.x = data['gyro'][0]
        imu_msg.angular_velocity.y = data['gyro'][1]
        imu_msg.angular_velocity.z = data['gyro'][2]

        # Linear acceleration (gravity-compensated)
        imu_msg.linear_acceleration.x = data['linear_accel'][0]
        imu_msg.linear_acceleration.y = data['linear_accel'][1]
        imu_msg.linear_acceleration.z = data['linear_accel'][2]

        # Set covariance based on calibration status
        cal_factor = max(1.0, 4.0 - data['calibration'])  # Lower is better
        
        # Orientation covariance
        ori_var = (0.01 * cal_factor) ** 2  # Better when calibrated
        imu_msg.orientation_covariance[0] = ori_var  # roll
        imu_msg.orientation_covariance[4] = ori_var  # pitch
        imu_msg.orientation_covariance[8] = ori_var  # yaw

        # Angular velocity covariance (gyro noise)
        gyro_var = (0.001) ** 2
        imu_msg.angular_velocity_covariance[0] = gyro_var
        imu_msg.angular_velocity_covariance[4] = gyro_var
        imu_msg.angular_velocity_covariance[8] = gyro_var

        # Linear acceleration covariance
        accel_var = (0.01) ** 2
        imu_msg.linear_acceleration_covariance[0] = accel_var
        imu_msg.linear_acceleration_covariance[4] = accel_var
        imu_msg.linear_acceleration_covariance[8] = accel_var

        self.imu_pub.publish(imu_msg)

        # Magnetometer message
        mag_msg = MagneticField()
        mag_msg.header.stamp = current_time.to_msg()
        mag_msg.header.frame_id = self.frame_id
        mag_msg.magnetic_field.x = data['mag'][0] * 1e-6  # µT to T
        mag_msg.magnetic_field.y = data['mag'][1] * 1e-6
        mag_msg.magnetic_field.z = data['mag'][2] * 1e-6

        # Magnetometer covariance
        mag_var = (0.1e-6) ** 2  # BNO055 has good mag accuracy
        mag_msg.magnetic_field_covariance[0] = mag_var
        mag_msg.magnetic_field_covariance[4] = mag_var
        mag_msg.magnetic_field_covariance[8] = mag_var

        self.mag_pub.publish(mag_msg)

        # Temperature message
        temp_msg = Temperature()
        temp_msg.header.stamp = current_time.to_msg()
        temp_msg.header.frame_id = self.frame_id
        temp_msg.temperature = float(data['temperature'])
        temp_msg.variance = 1.0  # ±1°C accuracy

        self.temp_pub.publish(temp_msg)

        # Euler angles message
        euler_msg = Vector3Stamped()
        euler_msg.header.stamp = current_time.to_msg()
        euler_msg.header.frame_id = self.frame_id
        euler_msg.vector.x = math.radians(data['euler'][0])  # roll
        euler_msg.vector.y = math.radians(data['euler'][1])  # pitch
        euler_msg.vector.z = math.radians(data['euler'][2])  # yaw

        self.euler_pub.publish(euler_msg)

        # Quaternion message (separate from IMU)
        quat_msg = QuaternionStamped()
        quat_msg.header.stamp = current_time.to_msg()
        quat_msg.header.frame_id = self.frame_id
        quat_msg.quaternion.x = data['quaternion'][0]
        quat_msg.quaternion.y = data['quaternion'][1]
        quat_msg.quaternion.z = data['quaternion'][2]
        quat_msg.quaternion.w = data['quaternion'][3]

        self.quaternion_pub.publish(quat_msg)

        # Calibration status message
        cal_msg = Int8()
        cal_msg.data = data['calibration']
        self.calibration_pub.publish(cal_msg)

        # Status message
        if data['system_error'] != 0:
            status_str = f"ERROR: System error {data['system_error']}"
        elif data['calibration'] < 3:
            status_str = f"CALIBRATING: System cal level {data['calibration']}/3"
        else:
            status_str = "OK: Fully calibrated"

        status_msg = String()
        status_msg.data = status_str
        self.status_pub.publish(status_msg)

    def destroy_node(self):
        """Clean shutdown"""
        try:
            if self.i2c:
                self.i2c.close()
        except Exception as e:
            self.get_logger().error(f"Error during cleanup: {e}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    bno055_pub = BNO055Publisher()

    try:
        rclpy.spin(bno055_pub)
    except KeyboardInterrupt:
        pass
    finally:
        bno055_pub.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()