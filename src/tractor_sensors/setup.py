from setuptools import find_packages, setup

package_name = 'tractor_sensors'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='benson',
    maintainer_email='benson@todo.todo',
    description='Sensor interfaces for tractor robot including rotary encoders, GPS, and compass',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'encoder_publisher = tractor_sensors.encoder_publisher:main',
            'gps_compass_publisher = tractor_sensors.gps_compass_publisher:main',
        ],
    },
)
