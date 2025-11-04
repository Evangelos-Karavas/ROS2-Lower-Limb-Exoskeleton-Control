from setuptools import find_packages, setup
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

package_name = 'exo_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    ('share/' + package_name + '/launch', ['launch/joint_publisher.launch.py']),
    ('share/' + package_name + '/launch', ['launch/joint_publisher_pv.launch.py']),
    ('share/' + package_name + '/launch', ['launch/data_publisher.launch.py']),
    ('share/' + package_name + '/launch', ['launch/randomized_data_publisher.launch.py']),
    ('share/' + package_name + '/config', ['config/ros2_controller.yaml']),

    ('share/exo_control/config', ['config/ros2_controller.yaml']),

    # Load Keras models
    ('share/exo_control/neural_network_parameters/models/', ['neural_network_parameters/models/PV_cnn_model.keras']),
    ('share/exo_control/neural_network_parameters/models/', ['neural_network_parameters/models/PV_lstm_model.keras']),
    ('share/exo_control/neural_network_parameters/models/', ['neural_network_parameters/models/Timestamp_cnn_model.keras']),
    ('share/exo_control/neural_network_parameters/models/', ['neural_network_parameters/models/Timestamp_lstm_model.keras']),

    # Load scalers
    ('share/exo_control/neural_network_parameters/scaler', ['neural_network_parameters/scaler/standard_scaler_typical_lstm.save']),
    ('share/exo_control/neural_network_parameters/scaler', ['neural_network_parameters/scaler/standard_scaler_cp_lstm.save']),
    ('share/exo_control/neural_network_parameters/scaler', ['neural_network_parameters/scaler/standard_scaler_typical_cnn.save']),
    ('share/exo_control/neural_network_parameters/scaler', ['neural_network_parameters/scaler/standard_scaler_cp_cnn.save']),
    # Load excel of data
    ('share/exo_control/neural_network_parameters/excel', ['neural_network_parameters/excel/timestamps_typical_cnn.xlsx']),
    ('share/exo_control/neural_network_parameters/excel', ['neural_network_parameters/excel/timestamps_typical_lstm.xlsx']),
    ('share/exo_control/neural_network_parameters/excel', ['neural_network_parameters/excel/timestamps_cp_cnn.xlsx']),
    ('share/exo_control/neural_network_parameters/excel', ['neural_network_parameters/excel/timestamps_cp_lstm.xlsx']),
    ('share/exo_control/neural_network_parameters/excel', ['neural_network_parameters/excel/PV_typical_cnn.xlsx']),
    ('share/exo_control/neural_network_parameters/excel', ['neural_network_parameters/excel/PV_typical_lstm.xlsx']),
    ('share/exo_control/neural_network_parameters/excel', ['neural_network_parameters/excel/PV_cp_cnn.xlsx']),
    ('share/exo_control/neural_network_parameters/excel', ['neural_network_parameters/excel/PV_cp_lstm.xlsx']),
    ('share/exo_control/neural_network_parameters/excel', ['neural_network_parameters/excel/data_healthy.xlsx']),
    ('share/exo_control/neural_network_parameters/excel', ['neural_network_parameters/excel/data_cp.xlsx']),
    ],

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Vaggelis Karavas',
    maintainer_email='vaggeliskaravas@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'joint_publisher = exo_control.joint_publisher_nn:main',
        'joint_publisher_pv = exo_control.joint_publisher_pv:main',
        'data_publisher = exo_control.data_publisher:main',
        'randomized_data_publisher = exo_control.randomized_data_publisher:main',
        ],
    },
)
