import os
import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
from ament_index_python.packages import get_package_share_directory

class JointPublisherFromExcel(Node):
    """This node simply takes data from the Excel files (look at neural_network_parameters/excel/--all_available_excel_files--)
        and publishes them directly to gazebo simulation. After all points have been sent, it restarts publishing from the start
        of the excel file."""
    def __init__(self):
        super().__init__('joint_publisher_from_excel')

        pub_rate = 0.1  # seconds between publishes
        self.joint_names = [
            'left_hip_revolute_joint',
            'right_hip_revolute_joint',
            'left_knee_revolute_joint',
            'right_knee_revolute_joint',
            'left_ankle_revolute_joint',
            'right_ankle_revolute_joint'
        ]

        self.trajectory_publisher_ = self.create_publisher(
            JointTrajectory, '/trajectory_controller/joint_trajectory', 10
        )

        # Load joint trajectory data directly from Excel
        pkg_dir = get_package_share_directory('exo_control')
        excel_path = os.path.join(pkg_dir, 'neural_network_parameters/excel', 'timestamps_typical_cnn.xlsx')
        df = pd.read_excel(excel_path)
        self.joint_data = df.values.astype(np.float32)
        self.get_logger().info(f"Loaded joint data from Excel with shape {self.joint_data.shape}")

        self.traj_index = 0
        self.timer = self.create_timer(pub_rate, self.timer_callback)

    def timer_callback(self):
        if self.traj_index >= len(self.joint_data):
            self.get_logger().info("All joint positions sent.")
            self.traj_index = 0  # restart if needed, or `rclpy.shutdown()` to stop
            return
        joint_positions = np.radians(self.joint_data[self.traj_index])

        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = joint_positions.tolist()
        point.time_from_start = Duration(sec=0, nanosec=500_000_000)  # 0.5s
        msg.points.append(point)

        self.trajectory_publisher_.publish(msg)
        self.get_logger().info(f"Published (timestep {self.traj_index}): {np.round(joint_positions, 2)}")
        self.traj_index += 1


def main(args=None):
    rclpy.init(args=args)
    node = JointPublisherFromExcel()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
