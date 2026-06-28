from setuptools import find_packages, setup
import os
from glob import glob

package_name = "tractor_explorer"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="rover",
    maintainer_email="rover@local",
    description="Deep Exploration Network for autonomous home mapping with RKNN NPU inference",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "explorer_runner = tractor_explorer.explorer_runner:main",
            "explore_manager = tractor_explorer.explore_manager:main",
            "map_integrator = tractor_explorer.map_integrator:main",
            "data_collector = tractor_explorer.data_collector:main",
            "convert_to_rknn = tractor_explorer.convert_to_rknn:main",
        ],
    },
)
