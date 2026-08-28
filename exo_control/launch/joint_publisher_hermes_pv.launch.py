from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node


def generate_launch_description():
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen'
    )

    forward_position_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['forward_position_controller'],
        output='screen'
    )

    # hermes_description's Gazebo bridge publishes sole contacts as
    # /sole_L/contacts and /sole_R/contacts, but foot_contact_bool
    # (and the hermes control node downstream) expect /left_sole/contacts
    # and /right_sole/contacts, so remap inputs instead of the node.
    foot_contact_bool = Node(
        package='exo_control',
        executable='foot_contact_bool',
        output='screen',
        remappings=[
            ('/left_sole/contacts', '/sole_L/contacts'),
            ('/right_sole/contacts', '/sole_R/contacts'),
        ],
    )

    joint_publisher_hermes_pv = Node(
        package='exo_control',
        executable='joint_publisher_hermes_pv',
        name='joint_publisher_hermes_pv',
        output='screen'
    )

    return LaunchDescription([
        joint_state_broadcaster_spawner,
        foot_contact_bool,
        RegisterEventHandler(
            OnProcessExit(
                target_action=joint_state_broadcaster_spawner,
                on_exit=[forward_position_controller_spawner]
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=forward_position_controller_spawner,
                on_exit=[joint_publisher_hermes_pv]
            )
        ),
    ])
