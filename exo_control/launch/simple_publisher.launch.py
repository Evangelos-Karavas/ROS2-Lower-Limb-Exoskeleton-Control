from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    return LaunchDescription([

        # Start the simple_publisher node
        TimerAction(
            period=0.5,
            actions=[
                Node(
                    package='exo_control',
                    executable='simple_publisher',
                    name='simple_publisher_node',
                    output='screen'
                )
            ]
        )
    ])
