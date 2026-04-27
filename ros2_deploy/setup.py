from setuptools import setup

package_name = "f1tenth_rl_deploy"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/rl_foxglove.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ESE615 Team 7",
    maintainer_email="team7@example.com",
    description="ROS2 launch helpers for F1TENTH RL policy deployment and Foxglove visualization.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "rl_inference = f1tenth_rl_deploy.rl_inference:main",
        ],
    },
)
