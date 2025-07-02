import os
from ament_index_python import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
import xacro

def generate_launch_description():
    # Packages
    exo_description_pkg = FindPackageShare('exo_description')
    exo_control_pkg = FindPackageShare('exo_control')

    # File paths
    xacro_file = os.path.join(get_package_share_directory('exo_description'), 'urdf', 'lleap_exo.urdf.xacro')
    controllers_config = PathJoinSubstitution([exo_control_pkg, 'config', 'ros2_controller.yaml'])

    # Robot description
    robot_description_raw = xacro.process_file(xacro_file).toxml()
    robot_description = {'robot_description': robot_description_raw}

    # Use simulation time
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Gazebo model path setup
    pkg_share_path = os.pathsep + os.path.join(get_package_prefix('exo_description'), 'share')
    if 'GAZEBO_MODEL_PATH' in os.environ:
        os.environ['GAZEBO_MODEL_PATH'] += pkg_share_path
    else:
        os.environ['GAZEBO_MODEL_PATH'] = pkg_share_path

    return LaunchDescription([
        DeclareLaunchArgument(
            name='use_sim_time',
            default_value='true',
            description='Use simulation clock (Gazebo)'
        ),

        LogInfo(msg='Starting unified simulation and control launch...'),
        SetEnvironmentVariable('GAZEBO_VERBOSITY', 'error'),

        # Start Gazebo
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
            ])
        ),

        # Start robot_state_publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[robot_description, {'use_sim_time': use_sim_time}]
        ),

        # Spawn robot into Gazebo
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-topic', 'robot_description', '-entity', 'lleap_exo'],
            output='screen'
        ),

        # Start ros2_control node (controller manager)
        Node(
            package='controller_manager',
            executable='ros2_control_node',
            parameters=[robot_description, controllers_config],
            output='screen'
        ),

        # Spawners for controllers
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster'],
            output='screen'
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['left_leg_controller'],
            output='screen'
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['right_leg_controller'],
            output='screen'
        ),

        # Delay joint publisher until controllers are up
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='exo_control',
                    executable='joint_publisher',
                    name='joint_trajectory_publisher_node',
                    output='screen'
                )
            ]
        )
    ])
