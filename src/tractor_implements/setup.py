from setuptools import find_packages, setup

package_name = 'tractor_implements'

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
    description='Implement control package for tractor attachments like lawn mower and sprayer',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mower_controller = tractor_implements.mower_controller:main',
            'sprayer_controller = tractor_implements.sprayer_controller:main',
        ],
    },
)
