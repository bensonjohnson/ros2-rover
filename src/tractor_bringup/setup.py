from setuptools import find_packages, setup

package_name = "tractor_bringup"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/control.launch.py"]),
        ("share/" + package_name + "/launch", ["launch/implements.launch.py"]),
        ("share/" + package_name + "/launch", ["launch/navigation.launch.py"]),
        ("share/" + package_name + "/launch", ["launch/robot_description.launch.py"]),
        ("share/" + package_name + "/launch", ["launch/sensors.launch.py"]),
        ("share/" + package_name + "/launch", ["launch/tractor_bringup.launch.py"]),
        ("share/" + package_name + "/launch", ["launch/tractor_gazebo.launch.py"]),
        ("share/" + package_name + "/launch", ["launch/tractor_sim.launch.py"]),
        ("share/" + package_name + "/launch", ["launch/vision.launch.py"]),
        ("share/" + package_name + "/launch", ["launch/hiwonder_control.launch.py"]),
        ("share/" + package_name + "/launch", ["launch/robot_localization.launch.py"]),
        ("share/" + package_name + "/launch", ["launch/slam_mapping.launch.py"]),
        ("share/" + package_name + "/launch", ["launch/control_with_feedback.launch.py"]), # Added this line
        ("share/" + package_name + "/launch", ["launch/xbox_teleop_with_feedback.launch.py"]), # Added this line
        ("share/" + package_name + "/urdf", ["urdf/tractor.urdf.xacro"]),
        ("share/" + package_name + "/config", ["config/nav2_params.yaml"]),
        ("share/" + package_name + "/config", ["config/robot_localization.yaml"]),
        ("share/" + package_name + "/config", ["config/slam_toolbox_params.yaml"]),
        ("share/" + package_name + "/maps", ["maps/yard_map.yaml"]),
        ("share/" + package_name + "/maps", ["maps/yard_map.pgm"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="benson",
    maintainer_email="benson@todo.todo",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [],
    },
)
