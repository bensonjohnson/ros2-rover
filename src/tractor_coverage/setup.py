from setuptools import find_packages, setup

package_name = "tractor_coverage"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/coverage_system.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="benson",
    maintainer_email="benson@todo.todo",
    description="Coverage path planning for agricultural operations with implement-specific patterns",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "coverage_action_server = tractor_coverage.coverage_action_server:main",
            "coverage_client = tractor_coverage.coverage_client:main",
            "coverage_visualizer = tractor_coverage.coverage_visualizer:main",
        ],
    },
)
