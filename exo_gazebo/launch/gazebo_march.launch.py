import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    declare_use_sim_time = DeclareLaunchArgument(
        name='use_sim_time',
        default_value='true',
        description='Use simulation time'
    )

    # Add exo models to Gazebo resource path
    exo_models_path = os.path.join(
        get_package_share_directory('exo_description'),
        'models'
    )
    existing_gz_path = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
    gz_resource_path_value = f"{existing_gz_path}:{exo_models_path}" if existing_gz_path else exo_models_path

    set_gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=gz_resource_path_value
    )

    # Robot description
    urdf_file = PathJoinSubstitution([
        FindPackageShare('exo_description'),
        'urdf',
        'march_6.urdf.xacro'
    ])
    robot_description = Command(['xacro ', urdf_file])

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': ParameterValue(robot_description, value_type=str),
            'use_sim_time': use_sim_time
        }]
    )

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            )
        ),
        launch_arguments={
            'gz_args': '-r -v 4 empty.sdf'
        }.items()
    )

    # Spawn robot
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'exosuit',
            '-topic', 'robot_description',
            '-x', '0', '-y', '0', '-z', '0.05'
        ],
        output='screen'
    )

    spawn_after_gz = TimerAction(
        period=2.0,
        actions=[spawn_entity]
    )

    # Bridge contact topics
    bridge_contacts = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        arguments=[
            '/world/empty/model/exosuit/link/left_ankle_plate/sensor/left_sole_contact/contact'
            '@ros_gz_interfaces/msg/Contacts[ignition.msgs.Contacts',
            '/world/empty/model/exosuit/link/right_ankle_plate/sensor/right_sole_contact/contact'
            '@ros_gz_interfaces/msg/Contacts[ignition.msgs.Contacts',
        ],
        remappings=[
            (
                '/world/empty/model/exosuit/link/left_ankle_plate/sensor/left_sole_contact/contact',
                '/left_sole/contacts'
            ),
            (
                '/world/empty/model/exosuit/link/right_ankle_plate/sensor/right_sole_contact/contact',
                '/right_sole/contacts'
            ),
        ],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    bridge_after_spawn = TimerAction(
        period=3.0,
        actions=[bridge_contacts]
    )

    foot_contact_bool = Node(
        package='exo_control',
        executable='foot_contact_bool',
        output='screen',
        parameters=[{
            'use_sim_time': False,   # <-- IMPORTANT
            'left_contacts_topic': '/left_sole/contacts',
            'right_contacts_topic': '/right_sole/contacts',
            'left_bool_topic': '/left_sole/in_contact',
            'right_bool_topic': '/right_sole/in_contact',
            'hold_time_sec': 0.02,
            'publish_rate_hz': 20.0,  # if your node supports it
        }]
    )

    contacts_bool_after_bridge = TimerAction(
        period=3.5,
        actions=[foot_contact_bool]
    )
    return LaunchDescription([
        declare_use_sim_time,
        set_gz_resource_path,
        robot_state_publisher,
        gz_sim,
        spawn_after_gz,
        bridge_after_spawn,
        contacts_bool_after_bridge
    ])
