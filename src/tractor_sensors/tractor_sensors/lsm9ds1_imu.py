#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, MagneticField, Temperature
from geometry_msgs.msg import Vector3Stamped
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


class LSM9DS1Publisher(Node):
    # LSM9DS1 I2C addresses
    LSM9DS1_ACCEL_GYRO_ADDR = 0x6B  # Accelerometer and gyroscope
    LSM9DS1_MAG_ADDR = 0x1E         # Magnetometer
    
    # Accelerometer/Gyroscope registers
    WHO_AM_I_XG = 0x0F
    CTRL_REG1_G = 0x10  # Gyroscope control
    CTRL_REG2_G = 0x11
    CTRL_REG3_G = 0x12
    ORIENT_CFG_G = 0x13
    INT_GEN_SRC_G = 0x14
    OUT_TEMP_L = 0x15
    OUT_TEMP_H = 0x16
    STATUS_REG = 0x17
    OUT_X_L_G = 0x18    # Gyroscope data
    OUT_X_H_G = 0x19
    OUT_Y_L_G = 0x1A
    OUT_Y_H_G = 0x1B
    OUT_Z_L_G = 0x1C
    OUT_Z_H_G = 0x1D
    CTRL_REG4 = 0x1E
    CTRL_REG5_XL = 0x1F
    CTRL_REG6_XL = 0x20  # Accelerometer control
    CTRL_REG7_XL = 0x21
    CTRL_REG8 = 0x22
    CTRL_REG9 = 0x23
    CTRL_REG10 = 0x24
    INT_GEN_SRC_XL = 0x26
    OUT_X_L_XL = 0x28   # Accelerometer data
    OUT_X_H_XL = 0x29
    OUT_Y_L_XL = 0x2A
    OUT_Y_H_XL = 0x2B
    OUT_Z_L_XL = 0x2C
    OUT_Z_H_XL = 0x2D
    
    # Magnetometer registers
    WHO_AM_I_M = 0x0F
    CTRL_REG1_M = 0x20  # Magnetometer control
    CTRL_REG2_M = 0x21
    CTRL_REG3_M = 0x22
    CTRL_REG4_M = 0x23
    CTRL_REG5_M = 0x24
    STATUS_REG_M = 0x27
    OUT_X_L_M = 0x28    # Magnetometer data
    OUT_X_H_M = 0x29
    OUT_Y_L_M = 0x2A
    OUT_Y_H_M = 0x2B
    OUT_Z_L_M = 0x2C
    OUT_Z_H_M = 0x2D
    INT_CFG_M = 0x30
    INT_SRC_M = 0x31
    INT_THS_L_M = 0x32
    INT_THS_H_M = 0x33

    # Scale factors (will be set based on configuration)
    ACCEL_SCALE = 0.000061  # Default ±2g scale: 2g/32768 = 0.000061 g/LSB
    GYRO_SCALE = 0.00875    # Default ±245dps scale: 245/32768 = 0.00875 dps/LSB  
    MAG_SCALE = 0.00014     # Default ±4gauss scale: 4/32768 = 0.00014 gauss/LSB

    def __init__(self):
        super().__init__("lsm9ds1_imu_publisher")

        # Parameters
        self.declare_parameter("i2c_bus", 5)
        self.declare_parameter("accel_gyro_addr", self.LSM9DS1_ACCEL_GYRO_ADDR)
        self.declare_parameter("mag_addr", self.LSM9DS1_MAG_ADDR)
        self.declare_parameter("frame_id", "imu_link")
        self.declare_parameter("update_rate", 50.0)  # Hz
        self.declare_parameter("accel_range", 2)     # ±2g
        self.declare_parameter("gyro_range", 245)    # ±245 dps
        self.declare_parameter("mag_range", 4)       # ±4 gauss
        self.declare_parameter("invert_x", True)     # X-axis inversion (upside down)
        self.declare_parameter("invert_y", False)    # Y-axis no inversion
        self.declare_parameter("invert_z", True)     # Z-axis inversion (upside down)

        self.i2c_bus = self.get_parameter("i2c_bus").value
        self.accel_gyro_addr = self.get_parameter("accel_gyro_addr").value
        self.mag_addr = self.get_parameter("mag_addr").value
        self.frame_id = self.get_parameter("frame_id").value
        self.update_rate = self.get_parameter("update_rate").value
        self.accel_range = self.get_parameter("accel_range").value
        self.gyro_range = self.get_parameter("gyro_range").value
        self.mag_range = self.get_parameter("mag_range").value
        self.invert_x = self.get_parameter("invert_x").value
        self.invert_y = self.get_parameter("invert_y").value
        self.invert_z = self.get_parameter("invert_z").value

        # Initialize I2C
        self.i2c = None
        if not I2C_AVAILABLE:
            self.get_logger().error("I2C not available - install python3-smbus")
            return

        try:
            self.i2c = smbus.SMBus(self.i2c_bus)
            if not self.init_lsm9ds1():
                self.get_logger().error("Failed to initialize LSM9DS1")
                return
        except Exception as e:
            self.get_logger().error(f"Failed to initialize I2C bus {self.i2c_bus}: {e}")
            return

        # Publishers
        self.imu_pub = self.create_publisher(Imu, "imu/data", 10)
        self.mag_pub = self.create_publisher(MagneticField, "imu/mag", 10)
        self.temp_pub = self.create_publisher(Temperature, "imu/temperature", 10)
        self.raw_accel_pub = self.create_publisher(Vector3Stamped, "imu/accel_raw", 10)
        self.raw_gyro_pub = self.create_publisher(Vector3Stamped, "imu/gyro_raw", 10)

        # Data storage
        self.sensor_data = {
            'accel': [0.0, 0.0, 0.0],      # m/s²
            'gyro': [0.0, 0.0, 0.0],       # rad/s
            'mag': [0.0, 0.0, 0.0],        # gauss
            'temperature': 0.0              # °C
        }
        self.data_lock = threading.Lock()

        # Start sensor reading thread
        self.sensor_thread = threading.Thread(target=self.read_sensor_data, daemon=True)
        self.sensor_thread.start()

        # Publisher timer
        update_period = 1.0 / self.update_rate
        self.timer = self.create_timer(update_period, self.publish_data)

        self.get_logger().info(
            f"LSM9DS1 IMU Publisher initialized on I2C bus {self.i2c_bus}, "
            f"addresses: accel/gyro=0x{self.accel_gyro_addr:02X}, mag=0x{self.mag_addr:02X}, "
            f"axis inversions: X={self.invert_x}, Y={self.invert_y}, Z={self.invert_z}"
        )

    def init_lsm9ds1(self) -> bool:
        """Initialize LSM9DS1 sensor"""
        try:
            # Check WHO_AM_I registers
            who_am_i_xg = self.read_byte(self.accel_gyro_addr, self.WHO_AM_I_XG)
            who_am_i_m = self.read_byte(self.mag_addr, self.WHO_AM_I_M)
            
            if who_am_i_xg != 0x68:
                self.get_logger().error(f"Invalid accel/gyro ID: 0x{who_am_i_xg:02X}, expected 0x68")
                return False
            
            if who_am_i_m != 0x3D:
                self.get_logger().error(f"Invalid magnetometer ID: 0x{who_am_i_m:02X}, expected 0x3D")
                return False

            self.get_logger().info("LSM9DS1 detected successfully")

            # Configure accelerometer
            # CTRL_REG6_XL: ±2g, 119Hz ODR, BW determined by ODR
            self.write_byte(self.accel_gyro_addr, self.CTRL_REG6_XL, 0x60)
            
            # Configure gyroscope  
            # CTRL_REG1_G: 119Hz ODR, 245dps scale
            self.write_byte(self.accel_gyro_addr, self.CTRL_REG1_G, 0x60)
            
            # Configure magnetometer
            # CTRL_REG1_M: Temp comp, high performance XY, 80Hz ODR
            self.write_byte(self.mag_addr, self.CTRL_REG1_M, 0xFC)
            
            # CTRL_REG2_M: ±4 gauss scale
            self.write_byte(self.mag_addr, self.CTRL_REG2_M, 0x00)
            
            # CTRL_REG3_M: Continuous conversion mode
            self.write_byte(self.mag_addr, self.CTRL_REG3_M, 0x00)
            
            # CTRL_REG4_M: High performance Z axis
            self.write_byte(self.mag_addr, self.CTRL_REG4_M, 0x0C)

            # Update scale factors based on configuration
            self.update_scale_factors()

            time.sleep(0.1)  # Allow sensors to stabilize
            
            self.get_logger().info("LSM9DS1 initialized successfully")
            return True

        except Exception as e:
            self.get_logger().error(f"LSM9DS1 initialization error: {e}")
            return False

    def update_scale_factors(self):
        """Update scale factors based on configured ranges"""
        # Accelerometer scale factors (g/LSB)
        accel_scales = {2: 0.000061, 4: 0.000122, 8: 0.000244, 16: 0.000732}
        self.ACCEL_SCALE = accel_scales.get(self.accel_range, 0.000061)
        
        # Gyroscope scale factors (dps/LSB)  
        gyro_scales = {245: 0.00875, 500: 0.01750, 2000: 0.07000}
        self.GYRO_SCALE = gyro_scales.get(self.gyro_range, 0.00875)
        
        # Magnetometer scale factors (gauss/LSB)
        mag_scales = {4: 0.00014, 8: 0.00029, 12: 0.00043, 16: 0.00058}
        self.MAG_SCALE = mag_scales.get(self.mag_range, 0.00014)

    def apply_axis_corrections(self, data: list) -> list:
        """Apply axis inversions for sensor mounting orientation"""
        corrected = data.copy()
        if self.invert_x:
            corrected[0] = -corrected[0]
        if self.invert_y:
            corrected[1] = -corrected[1] 
        if self.invert_z:
            corrected[2] = -corrected[2]
        return corrected

    def compute_orientation(self, accel: list, mag: list) -> list:
        """Compute orientation quaternion from accelerometer and magnetometer
        Returns [x, y, z, w] quaternion"""
        try:
            # Normalize accelerometer vector
            ax, ay, az = accel
            accel_norm = math.sqrt(ax*ax + ay*ay + az*az)
            if accel_norm < 0.1:  # Avoid division by zero
                return [0.0, 0.0, 0.0, 1.0]
            
            ax /= accel_norm
            ay /= accel_norm  
            az /= accel_norm
            
            # Normalize magnetometer vector
            mx, my, mz = mag
            mag_norm = math.sqrt(mx*mx + my*my + mz*mz)
            if mag_norm < 0.1:  # Avoid division by zero
                return [0.0, 0.0, 0.0, 1.0]
            
            mx /= mag_norm
            my /= mag_norm
            mz /= mag_norm
            
            # Tilt compensation: remove gravity component from magnetometer
            # Dot product of magnetometer and gravity (down is -accel direction)
            h_x = mx - ax * (mx * ax + my * ay + mz * az)  
            h_y = my - ay * (mx * ax + my * ay + mz * az)
            h_z = mz - az * (mx * ax + my * ay + mz * az)
            
            # Normalize horizontal magnetic field
            h_norm = math.sqrt(h_x*h_x + h_y*h_y + h_z*h_z)
            if h_norm < 0.1:
                return [0.0, 0.0, 0.0, 1.0]
                
            h_x /= h_norm
            h_y /= h_norm
            h_z /= h_norm
            
            # Compute roll and pitch from accelerometer
            roll = math.atan2(ay, az)
            pitch = math.atan2(-ax, math.sqrt(ay*ay + az*az))
            
            # Compute tilt-compensated yaw from horizontal magnetic field
            yaw = math.atan2(h_y, h_x)
            
            # Convert to quaternion (ZYX Euler order)
            cr = math.cos(roll * 0.5)
            sr = math.sin(roll * 0.5)
            cp = math.cos(pitch * 0.5)
            sp = math.sin(pitch * 0.5)  
            cy = math.cos(yaw * 0.5)
            sy = math.sin(yaw * 0.5)
            
            qw = cr * cp * cy + sr * sp * sy
            qx = sr * cp * cy - cr * sp * sy
            qy = cr * sp * cy + sr * cp * sy
            qz = cr * cp * sy - sr * sp * cy
            
            return [qx, qy, qz, qw]
            
        except:
            # Return identity quaternion on error
            return [0.0, 0.0, 0.0, 1.0]

    def read_byte(self, addr: int, register: int) -> int:
        """Read single byte from register"""
        return self.i2c.read_byte_data(addr, register)

    def write_byte(self, addr: int, register: int, value: int):
        """Write single byte to register"""
        self.i2c.write_byte_data(addr, register, value)

    def read_word_2c(self, addr: int, register: int) -> int:
        """Read 16-bit signed word (little endian)"""
        low = self.read_byte(addr, register)
        high = self.read_byte(addr, register + 1)
        val = (high << 8) + low
        if val >= 0x8000:
            return val - 0x10000
        return val

    def read_accel_data(self) -> list:
        """Read accelerometer data"""
        try:
            x = self.read_word_2c(self.accel_gyro_addr, self.OUT_X_L_XL)
            y = self.read_word_2c(self.accel_gyro_addr, self.OUT_Y_L_XL)  
            z = self.read_word_2c(self.accel_gyro_addr, self.OUT_Z_L_XL)
            
            # Convert to m/s² (1g = 9.80665 m/s²)
            x_ms2 = x * self.ACCEL_SCALE * 9.80665
            y_ms2 = y * self.ACCEL_SCALE * 9.80665
            z_ms2 = z * self.ACCEL_SCALE * 9.80665
            
            # Apply axis corrections for mounting orientation
            return self.apply_axis_corrections([x_ms2, y_ms2, z_ms2])
        except:
            return [0.0, 0.0, 0.0]

    def read_gyro_data(self) -> list:
        """Read gyroscope data"""
        try:
            x = self.read_word_2c(self.accel_gyro_addr, self.OUT_X_L_G)
            y = self.read_word_2c(self.accel_gyro_addr, self.OUT_Y_L_G)
            z = self.read_word_2c(self.accel_gyro_addr, self.OUT_Z_L_G)
            
            # Convert to rad/s
            x_rad = math.radians(x * self.GYRO_SCALE)
            y_rad = math.radians(y * self.GYRO_SCALE)  
            z_rad = math.radians(z * self.GYRO_SCALE)
            
            # Apply axis corrections for mounting orientation
            return self.apply_axis_corrections([x_rad, y_rad, z_rad])
        except:
            return [0.0, 0.0, 0.0]

    def read_mag_data(self) -> list:
        """Read magnetometer data"""
        try:
            x = self.read_word_2c(self.mag_addr, self.OUT_X_L_M)
            y = self.read_word_2c(self.mag_addr, self.OUT_Y_L_M)
            z = self.read_word_2c(self.mag_addr, self.OUT_Z_L_M)
            
            # Convert to gauss
            x_gauss = x * self.MAG_SCALE
            y_gauss = y * self.MAG_SCALE
            z_gauss = z * self.MAG_SCALE
            
            # Apply axis corrections for mounting orientation
            return self.apply_axis_corrections([x_gauss, y_gauss, z_gauss])
        except:
            return [0.0, 0.0, 0.0]

    def read_temperature(self) -> float:
        """Read temperature data"""
        try:
            temp_l = self.read_byte(self.accel_gyro_addr, self.OUT_TEMP_L)
            temp_h = self.read_byte(self.accel_gyro_addr, self.OUT_TEMP_H)
            temp_raw = (temp_h << 8) | temp_l
            
            # Convert to signed
            if temp_raw >= 0x8000:
                temp_raw -= 0x10000
                
            # Convert to Celsius (datasheet formula)
            temp_c = (temp_raw / 256.0) + 25.0
            return temp_c
        except:
            return 0.0

    def read_sensor_data(self):
        """Main sensor reading loop"""
        while rclpy.ok():
            try:
                # Read all sensor data
                accel = self.read_accel_data()
                gyro = self.read_gyro_data()
                mag = self.read_mag_data()
                temperature = self.read_temperature()

                # Store data
                with self.data_lock:
                    self.sensor_data.update({
                        'accel': accel,
                        'gyro': gyro,
                        'mag': mag,
                        'temperature': temperature
                    })

                # Update rate control
                time.sleep(0.02)  # 50Hz max read rate

            except Exception as e:
                self.get_logger().debug(f"Sensor reading error: {e}")
                time.sleep(0.1)

    def publish_data(self):
        """Publish sensor data"""
        current_time = self.get_clock().now()

        with self.data_lock:
            data = self.sensor_data.copy()

        # IMU message with computed orientation from magnetometer + accelerometer
        imu_msg = Imu()
        imu_msg.header.stamp = current_time.to_msg()
        imu_msg.header.frame_id = self.frame_id

        # Compute tilt-compensated orientation from accel + magnetometer
        orientation_quat = self.compute_orientation(data['accel'], data['mag'])
        imu_msg.orientation.x = orientation_quat[0]
        imu_msg.orientation.y = orientation_quat[1] 
        imu_msg.orientation.z = orientation_quat[2]
        imu_msg.orientation.w = orientation_quat[3]

        # Angular velocity
        imu_msg.angular_velocity.x = data['gyro'][0]
        imu_msg.angular_velocity.y = data['gyro'][1]
        imu_msg.angular_velocity.z = data['gyro'][2]

        # Linear acceleration
        imu_msg.linear_acceleration.x = data['accel'][0]
        imu_msg.linear_acceleration.y = data['accel'][1]
        imu_msg.linear_acceleration.z = data['accel'][2]

        # Set covariances (LSM9DS1 typical noise characteristics)
        # Orientation covariance (tilt-compensated compass)
        ori_var = (0.1) ** 2  # ~6° uncertainty in orientation
        imu_msg.orientation_covariance[0] = ori_var  # roll
        imu_msg.orientation_covariance[4] = ori_var  # pitch  
        imu_msg.orientation_covariance[8] = ori_var * 4  # yaw (worse due to magnetometer)

        # Angular velocity covariance
        gyro_var = (0.01) ** 2  # 0.01 rad/s noise
        imu_msg.angular_velocity_covariance[0] = gyro_var
        imu_msg.angular_velocity_covariance[4] = gyro_var
        imu_msg.angular_velocity_covariance[8] = gyro_var

        # Linear acceleration covariance
        accel_var = (0.02) ** 2  # 0.02 m/s² noise
        imu_msg.linear_acceleration_covariance[0] = accel_var
        imu_msg.linear_acceleration_covariance[4] = accel_var
        imu_msg.linear_acceleration_covariance[8] = accel_var

        self.imu_pub.publish(imu_msg)

        # Magnetometer message
        mag_msg = MagneticField()
        mag_msg.header.stamp = current_time.to_msg()
        mag_msg.header.frame_id = self.frame_id
        mag_msg.magnetic_field.x = data['mag'][0] * 1e-4  # Convert gauss to Tesla
        mag_msg.magnetic_field.y = data['mag'][1] * 1e-4
        mag_msg.magnetic_field.z = data['mag'][2] * 1e-4

        # Magnetometer covariance
        mag_var = (0.01e-4) ** 2  # 0.01 gauss noise in Tesla
        mag_msg.magnetic_field_covariance[0] = mag_var
        mag_msg.magnetic_field_covariance[4] = mag_var
        mag_msg.magnetic_field_covariance[8] = mag_var

        self.mag_pub.publish(mag_msg)

        # Temperature message
        temp_msg = Temperature()
        temp_msg.header.stamp = current_time.to_msg()
        temp_msg.header.frame_id = self.frame_id
        temp_msg.temperature = float(data['temperature'])
        temp_msg.variance = 2.0  # ±2°C accuracy

        self.temp_pub.publish(temp_msg)

        # Raw sensor data messages (useful for debugging)
        accel_raw_msg = Vector3Stamped()
        accel_raw_msg.header.stamp = current_time.to_msg()
        accel_raw_msg.header.frame_id = self.frame_id
        accel_raw_msg.vector.x = data['accel'][0]
        accel_raw_msg.vector.y = data['accel'][1]
        accel_raw_msg.vector.z = data['accel'][2]
        self.raw_accel_pub.publish(accel_raw_msg)

        gyro_raw_msg = Vector3Stamped()
        gyro_raw_msg.header.stamp = current_time.to_msg()
        gyro_raw_msg.header.frame_id = self.frame_id
        gyro_raw_msg.vector.x = data['gyro'][0]
        gyro_raw_msg.vector.y = data['gyro'][1]
        gyro_raw_msg.vector.z = data['gyro'][2]
        self.raw_gyro_pub.publish(gyro_raw_msg)

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
    lsm9ds1_pub = LSM9DS1Publisher()

    try:
        rclpy.spin(lsm9ds1_pub)
    except KeyboardInterrupt:
        pass
    finally:
        lsm9ds1_pub.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()