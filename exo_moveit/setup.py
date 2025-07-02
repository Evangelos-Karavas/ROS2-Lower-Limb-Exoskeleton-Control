from setuptools import find_packages, setup

package_name = 'exo_moveit'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    ('share/' + package_name + '/launch', ['launch/demo.launch.py']),
    ],

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vaggelis',
    maintainer_email='vaggeliskaravas@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'log_joint_states = exo_moveit.logjointstates:main',  
        ],
    },
)
