from setuptools import find_packages, setup

package_name = "tractor_vision"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="benson",
    maintainer_email="benson@todo.todo",
    description="Vision processing package for tractor robot with Intel RealSense 435i integration",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "obstacle_detector = tractor_vision.obstacle_detector:main",
        ],
    },
)
