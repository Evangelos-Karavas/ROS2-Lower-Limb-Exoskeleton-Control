import os
import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from ament_index_python.packages import get_package_share_directory


class PhaseVariableJointPublisher(Node):
    """This node simply takes data from the Excel files (look at neural_network_parameters/excel/--all_available_excel_files--)
        computes the pahse variable to correct some values, and publishes them to gazebo simulation. After all points have been sent, 
        it restarts publishing from the start of the excel file."""
    def __init__(self):
        super().__init__('phase_variable_joint_publisher')

        pub_timer = 0.06
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
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # Load data
        pkg_dir = get_package_share_directory('exo_control')
        excel_path = os.path.join(pkg_dir, 'neural_network_parameters/excel', 'timestamps_typical_cnn.xlsx')
        df = pd.read_excel(excel_path)
        self.joint_data = df.values.astype(np.float32)
        self.get_logger().info(f"Loaded joint data with shape: {self.joint_data.shape}")

        # Phase variable parameters
        self.state = 'S1'
        self.q0h = None
        self.qh_min = None
        self.qhm = None
        self.sm = None
        self.phase = 0.0
        self.c = 0.53  # stance duration ratio
        self.prev_thigh_angle = None
        self.traj_index = 0
        self.predicted_traj = None
        self.goal_sent = False
        self.last_goal_position = None
        self.timer = self.create_timer(pub_timer, self.timer_callback)

    def compute_phase_variable(self, qh):
        if self.q0h is None:
            self.q0h = qh  # Initialize on first timestep
            self.qh_min = qh

        # FSM transitions
        if self.state == 'S1':
            if qh < self.qh_min:
                self.qh_min = qh
            if qh <= -0.15:  # replace with threshold like q_po
                self.state = 'S2'
        elif self.state == 'S2':
            if qh > self.qh_min:
                self.qhm = qh
                self.sm = self.phase
                self.state = 'S3'
        elif self.state == 'S3':
            if qh > self.q0h:
                self.q0h = qh
                self.qh_min = qh
                self.state = 'S1'

        # Phase computation
        if self.state in ['S1', 'S2']: # Stance Phase
            s = ((self.q0h - qh) / (self.q0h - self.qh_min + 1e-5)) * self.c
        else:   # Swing phase
            s = 1.0 + ((1 - self.sm) / (self.q0h - self.qhm + 1e-5)) * (qh - self.q0h)

        self.phase = max(0.0, min(1.0, s))
        return self.phase

    def timer_callback(self):
        if self.traj_index >= len(self.joint_data):
            self.get_logger().info("All joint positions sent.")
            self.traj_index = 0
            return

        joint_row = self.joint_data[self.traj_index]
        joint_positions = np.radians(joint_row)

        # Compute qh_dot = derivative of left hip angle
        current_thigh_angle = joint_positions[0]
        qh_dot = 0.0
        if self.prev_thigh_angle is not None:
            qh_dot = (current_thigh_angle - self.prev_thigh_angle) / 0.03

        # Control ankle based on qh_dot sign
        flex = 0.2
        if qh_dot >= 0:
            # Swing phase → dorsiflex (positive)
            joint_positions[4] = max(joint_positions[4], flex)  # left ankle
            joint_positions[5] = min(joint_positions[5], -flex - 0.1)  # right ankle
        elif qh_dot < 0:
            # Stance phase → plantarflex (negative)
            joint_positions[4] = min(joint_positions[4], -flex - 0.1)
            joint_positions[5] = max(joint_positions[5], flex)

        # Save last thing angle for next step q_dot
        self.prev_thigh_angle = current_thigh_angle

        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = joint_positions.tolist()
        point.time_from_start = Duration(sec=0, nanosec=300_000_000)
        msg.points.append(point)

        self.trajectory_publisher_.publish(msg)
        self.get_logger().info(f"Published positions: {np.round(joint_positions, 2)}")
        self.traj_index += 1


    def joint_state_callback(self, msg):
        if not self.goal_sent or self.last_goal_position is None:
            return
        if not all(name in msg.name for name in self.joint_names):
            return

        current_pos = np.array([msg.position[msg.name.index(j)] for j in self.joint_names])
        if self.positions_close(current_pos, self.last_goal_position, tolerance=0.5):
            self.get_logger().info("Reached goal, moving to next point.")
            self.goal_sent = False
            self.traj_index += 1

    @staticmethod
    def positions_close(actual, goal, tolerance=0.1):
        return all(abs(a - g) < tolerance for a, g in zip(actual, goal))

def main(args=None):
    rclpy.init(args=args)
    node = PhaseVariableJointPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
