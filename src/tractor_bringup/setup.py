from setuptools import find_packages, setup
import os
from glob import glob

package_name = "tractor_bringup"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        ("share/" + package_name + "/urdf", ["urdf/tractor.urdf.xacro"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "maps"), glob("maps/*.yaml") + glob("maps/*.pgm")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="benson",
    maintainer_email="benson@todo.todo",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "autonomous_mapping.py = tractor_bringup.autonomous_mapping:main",
            "safety_monitor.py = tractor_bringup.safety_monitor:main",
            "frontier_explorer.py = tractor_bringup.frontier_explorer:main",
        ],
    },
)
