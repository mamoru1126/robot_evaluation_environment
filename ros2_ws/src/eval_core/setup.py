from glob import glob

from setuptools import find_packages, setup

package_name = "eval_core"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    package_data={"eval_core.web": ["static/*"]},
    include_package_data=True,
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="mamoru1126",
    maintainer_email="mamoru1126@users.noreply.github.com",
    description="Robot evaluation environment core (controllers, scenarios, metrics, web UI)",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "local_planner = eval_core.ros.local_planner_node:main",
            "eval_server = eval_core.app:main",
            "scan_relay = eval_core.ros.scan_relay_node:main",
            "example_external_controller = eval_core.ros.example_external_controller:main",
        ],
    },
)
