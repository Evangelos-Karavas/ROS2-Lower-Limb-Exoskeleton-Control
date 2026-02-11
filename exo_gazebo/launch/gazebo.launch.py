import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory, get_package_prefix

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Set GZ_SIM_RESOURCE_PATH so it can find models and URDFs
    gz_model_path = PathJoinSubstitution([
        FindPackageShare('exo_description'), 'models'
    ])

    set_gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=gz_model_path
    )

    declare_use_sim_time = DeclareLaunchArgument(
        name='use_sim_time',
        default_value='true',
        description='Use simulation time'
    )

    # URDF
    urdf_file = PathJoinSubstitution([
        FindPackageShare('exo_description'), 'urdf', 'lleap_exo.urdf.xacro'
    ])
    robot_description = Command(['xacro ', urdf_file])

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': ParameterValue(robot_description, value_type=str),
            'use_sim_time': use_sim_time
        }]
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'exosuit',
            '-topic', 'robot_description',
            '-x', '0.0', '-y', '0.0', '-z', '1.4',
            '-R', '0.0', '-P', '0.0', '-Y', '0.0',
        ],
        output='screen'
    )

    # Launch Gazebo
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch', 'gz_sim.launch.py'
            )
        ]),
        launch_arguments={'gz_args': '-r empty.sdf'}.items()
    )
    # Bridge contact topics
    bridge_contacts = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        arguments=[
            '/world/empty/model/exosuit/link/left_foot/sensor/left_sole_contact/contact'
            '@ros_gz_interfaces/msg/Contacts[ignition.msgs.Contacts',
            '/world/empty/model/exosuit/link/right_foot/sensor/right_sole_contact/contact'
            '@ros_gz_interfaces/msg/Contacts[ignition.msgs.Contacts',
        ],
        remappings=[
            (
                '/world/empty/model/exosuit/link/left_foot/sensor/left_sole_contact/contact',
                '/left_sole/contacts'
            ),
            (
                '/world/empty/model/exosuit/link/right_foot/sensor/right_sole_contact/contact',
                '/right_sole/contacts'
            ),
        ],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    bridge_after_spawn = TimerAction(
        period=3.0,
        actions=[bridge_contacts]
    )
    bridge_imu = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        arguments=[
            '/world/empty/model/exosuit/link/slider_x/sensor/base_link_imu/imu'
            '@sensor_msgs/msg/Imu[gz.msgs.IMU',
        ],
        remappings=[
            (
                '/world/empty/model/exosuit/link/slider_x/sensor/base_link_imu/imu',  #/world/empty/model/exosuit/link/base_link/sensor/base_link_imu/imu
                '/imu/data'
            ),
        ],
        parameters=[{'use_sim_time': use_sim_time}],
    )
    bridge_imu_after_spawn = TimerAction(
        period=3.0,
        actions=[bridge_imu]
    )
    foot_contact_bool = Node(
        package='exo_control',
        executable='foot_contact_bool',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'left_contacts_topic': '/left_sole/contacts',
            'right_contacts_topic': '/right_sole/contacts',
            'left_bool_topic': '/left_sole/in_contact',
            'right_bool_topic': '/right_sole/in_contact',
            'hold_time_sec': 0.02,
            'publish_rate_hz': 20.0,
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
        spawn_entity,
        bridge_after_spawn,
        bridge_imu_after_spawn,
        contacts_bool_after_bridge
    ])
