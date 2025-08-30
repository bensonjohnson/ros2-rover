from setuptools import find_packages, setup
import os
from glob import glob

package_name = "tractor_sensors"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*launch.[pxy][yma]*")),
        ),
        (
            os.path.join("share", package_name, "config"),
            glob(os.path.join("config", "*.yaml")),
        ),
    ],
    install_requires=["setuptools", "pyserial", "pynmea2", "smbus2"],
    zip_safe=True,
    maintainer="benson",
    maintainer_email="benson@todo.todo",
    description="Sensor interfaces for tractor robot including rotary encoders, GPS, and compass",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "hglrc_m100_5883 = tractor_sensors.hglrc_m100_5883:main",
            "lc29h_rtk_gps = tractor_sensors.lc29h_rtk_gps:main",
            "bno055_imu = tractor_sensors.bno055_imu:main",
            "lsm9ds1_imu = tractor_sensors.lsm9ds1_imu:main",
        ],
    },
)
