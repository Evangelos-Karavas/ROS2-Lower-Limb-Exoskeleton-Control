from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    exo_description_pkg = FindPackageShare('exo_description')
    exo_control_pkg = FindPackageShare('exo_control')
    urdf_path = PathJoinSubstitution([
        exo_description_pkg, 'urdf', 'lleap_exo.urdf.xacro'
    ])
    controllers_config = PathJoinSubstitution([
        exo_control_pkg, 'config', 'ros2_controller.yaml'
    ])
    # Load robot description
    robot_description = ParameterValue(
        Command(['xacro ', urdf_path]), value_type=str
    )

    return LaunchDescription([
        # Start joint state broadcaster and trajectory controller
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster'],
            output='screen'
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['trajectory_controller'],
            output='screen'
        ),
        # Start the joint publisher node
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='exo_control',
                    executable='data_publisher',
                    name='data_publisher_node',
                    output='screen'
                )
            ]
        )
    ])
