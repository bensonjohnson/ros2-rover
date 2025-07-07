from setuptools import find_packages, setup

package_name = "tractor_control"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "pygame", "smbus2"],
    zip_safe=True,
    maintainer="benson",
    maintainer_email="benson@todo.todo",
    description="Tank steering control package for tractor robot with I2C motor control",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "hiwonder_motor_driver = tractor_control.hiwonder_motor_driver:main",
            "xbox_controller_teleop = tractor_control.xbox_controller_teleop:main",
            "velocity_feedback_controller = tractor_control.velocity_feedback_controller:main",
        ],
    },
)
