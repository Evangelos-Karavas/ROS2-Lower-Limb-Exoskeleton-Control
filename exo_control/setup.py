from setuptools import find_packages, setup
import os

package_name = "exo_control"

def package_files(src_dir, install_dir):
    """
    Collect all files under src_dir and map them to the corresponding install_dir.
    Returns a list of (dest, [files...]) tuples suitable for data_files.
    """
    data = {}
    for root, _, files in os.walk(src_dir):
        files = [f for f in files if not f.startswith(".")]  # ignore hidden
        if not files:
            continue
        rel = os.path.relpath(root, src_dir)
        dest = os.path.join(install_dir, rel) if rel != "." else install_dir
        data.setdefault(dest, [])
        for f in files:
            data[dest].append(os.path.join(root, f))
    return sorted(data.items())

data_files = [
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    ("share/" + package_name, ["package.xml"]),
]

# launch/config (non-recursive is fine, but recursive also works)
data_files += package_files("launch", os.path.join("share", package_name, "launch"))
data_files += package_files("config", os.path.join("share", package_name, "config"))

# everything under neural_network_parameters installed preserving structure
data_files += package_files(
    "neural_network_parameters",
    os.path.join("share", package_name, "neural_network_parameters"),
)

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=data_files,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Vaggelis Karavas",
    maintainer_email="vaggeliskaravas@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "joint_publisher = exo_control.joint_publisher_nn:main",
            "data_publisher = exo_control.data_publisher:main",
            "randomized_data_publisher = exo_control.randomized_data_publisher:main",
            "joint_publisher_kinetics = exo_control.joint_publisher_kinetics:main",
            "joint_publisher_kinematics_kinetics = exo_control.joint_publisher_kinematics_kinetics:main",
        ],
    },
)
