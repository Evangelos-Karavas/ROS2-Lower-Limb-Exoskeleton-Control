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
    """This node takes data from typically developed patients or cerebral palsy patients from an Excel file
    and publishes them directly to gazebo simulation. After all points have been sent, it restarts."""
    
    def __init__(self):
        super().__init__('joint_publisher_from_excel')

        self.trajectory_publisher_ = self.create_publisher(JointTrajectory, '/trajectory_controller/joint_trajectory', 10)

        self.joint_names = [
            'left_hip_revolute_joint',
            'right_hip_revolute_joint',
            'left_knee_revolute_joint',
            'right_knee_revolute_joint',
            'left_ankle_revolute_joint',
            'right_ankle_revolute_joint'
        ]
        pub_rate = 0.1
        pkg_dir = get_package_share_directory('exo_control')
        excel_path = os.path.join(pkg_dir, 'neural_network_parameters/excel', 'data_cp.xlsx') # data_cp, data_healthy
        columns = ['LHipAngles (1)', 'RHipAngles (1)', 'LKneeAngles (1)', 'RKneeAngles (1)', 'LAnkleAngles (1)', 'RAnkleAngles (1)']
        df = pd.read_excel(excel_path, sheet_name="Data", usecols=columns, skiprows=[1, 2])
        self.joint_data = df.values.astype(np.float32)
        self.get_logger().info(f"Loaded joint data from Excel (sheet='Data') with columns {list(df.columns)} and shape {self.joint_data.shape}")
        self.traj_index = 0
        self.timer = self.create_timer(pub_rate, self.timer_callback)

    def timer_callback(self):
        num_samples = len(self.joint_data)
        if self.traj_index >= num_samples:
            self.get_logger().info("All joint positions sent.")
            self.traj_index = 0
            return
        joint_positions = np.radians(self.joint_data[self.traj_index].copy())
        right_side_indices = [1, 3, 5]
        right_offset = 25
        for idx in right_side_indices:
            shifted_index = (self.traj_index + right_offset) % 50
            if shifted_index < num_samples:
                joint_positions[idx] = np.radians(self.joint_data[shifted_index, idx])
            else:
                joint_positions[idx] = np.radians(self.joint_data[shifted_index % num_samples, idx])
        for i, joint in enumerate(self.joint_names):
            if 'knee' in joint and joint_positions[i] >= 0.0:
                joint_positions[i] = -joint_positions[i]

        # publish trajectory message
        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = joint_positions.tolist()
        point.time_from_start = Duration(sec=0, nanosec=500_000_000)
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
