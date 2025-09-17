import os
import rclpy
import signal
import time
from rclpy.node import Node
import numpy as np
import pandas as pd
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Float32 

class PhaseVariableJointPublisher(Node):
    """This node simply takes data from the Excel files (look at neural_network_parameters/excel/--all_available_excel_files--)
        computes the pahse variable to correct some values, and publishes them to gazebo simulation. After all points have been sent, 
        it restarts publishing from the start of the excel file."""
    def __init__(self):
        super().__init__('phase_variable_joint_publisher')

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
        self.phase_pub_left = self.create_publisher(Float32, '/phase_variable_left', 10)
        self.phase_pub_right = self.create_publisher(Float32, '/phase_variable_right', 10)
        # Load data
        pkg_dir = get_package_share_directory('exo_control')
        excel_path = os.path.join(pkg_dir, 'neural_network_parameters/excel', 'timestamps_typical_cnn.xlsx')
        df = pd.read_excel(excel_path)
        self.joint_data = df.values.astype(np.float32)
        self.get_logger().info(f"Loaded joint data with shape: {self.joint_data.shape}")

        self.pub_timer = 0.06

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
        self.current_joint_positions = 0.0 * np.ones(6)
        self.timer = self.create_timer(self.pub_timer , self.timer_callback)
        self.kp_hip = 20.0
        self.kd_hip = 0.5
        self.kp_knee = 20.0
        self.kd_knee = 0.5
        self.kp_ankle = 10.0
        self.kd_ankle = 0.2
        self.last_joint_pos = np.zeros(6)
        self.state_left = 'S1'
        self.q0h_left = None
        self.qh_min_left = None
        self.qhm_left = None
        self.sm_left = None
        self.phase_left = 0.0

        self.state_right = 'S1'
        self.q0h_right = None
        self.qh_min_right = None
        self.qhm_right = None
        self.sm_right = None
        self.phase_right = 0.0     

    def compute_phase_variable(self, thigh_angle, thigh_velocity, leg):
        """
        Computes phase variable as described in Quintero et al. (2017).
        Implements state-based formulation with transitions and piecewise PV.
        """
        c = 0.53  # stance duration ratio

        if leg == 'left':
            if self.state_left == 'S1':
                if self.q0h_left is None:
                    self.q0h_left = thigh_angle
                if self.qh_min_left is None or thigh_angle < self.qh_min_left:
                    self.qh_min_left = thigh_angle

                s = c * (self.q0h_left - thigh_angle) / (self.q0h_left - self.qh_min_left + 1e-5)

                if thigh_angle <= self.qh_min_left and thigh_velocity > 0:
                    self.state_left = 'S2'
                    self.qhm_left = thigh_angle  # qh min at transition
            elif self.state_left == 'S2':
                if self.qh_max_left is None or thigh_angle > self.qh_max_left:
                    self.qh_max_left = thigh_angle

                s = c + (1 - c) * (thigh_angle - self.qhm_left) / (self.qh_max_left - self.qhm_left + 1e-5)

                if thigh_angle >= self.qh_max_left and thigh_velocity < 0:
                    self.state_left = 'S1'
                    self.q0h_left = thigh_angle
                    self.qh_min_left = None
                    self.qh_max_left = None
            return np.clip(s, 0.0, 1.0)

        elif leg == 'right':
            if self.state_right == 'S1':
                if self.q0h_right is None:
                    self.q0h_right = thigh_angle
                if self.qh_min_right is None or thigh_angle < self.qh_min_right:
                    self.qh_min_right = thigh_angle

                s = c * (self.q0h_right - thigh_angle) / (self.q0h_right - self.qh_min_right + 1e-5)

                if thigh_angle <= self.qh_min_right and thigh_velocity > 0:
                    self.state_right = 'S2'
                    self.qhm_right = thigh_angle
            elif self.state_right == 'S2':
                if self.qh_max_right is None or thigh_angle > self.qh_max_right:
                    self.qh_max_right = thigh_angle

                s = c + (1 - c) * (thigh_angle - self.qhm_right) / (self.qh_max_right - self.qhm_right + 1e-5)

                if thigh_angle >= self.qh_max_right and thigh_velocity < 0:
                    self.state_right = 'S1'
                    self.q0h_right = thigh_angle
                    self.qh_min_right = None
                    self.qh_max_right = None
            return s



    def timer_callback(self):
        if self.traj_index >= len(self.joint_data):
            self.get_logger().info("All joint positions sent.")
            self.traj_index = 0
            return

        joint_row = self.joint_data[self.traj_index]
        joint_positions = np.radians(joint_row)

        # Compute qh_dot = derivative of left hip angle
        current_thigh_angle_left = joint_positions[0]
        current_thigh_angle_right = joint_positions[1]
        qh_dot_left = 0.0
        qh_dot_right = 0.0
        if self.prev_thigh_angle is not None:
            qh_dot_left = (current_thigh_angle_left - self.prev_thigh_angle_left) / self.pub_timer 
            qh_dot_right = (current_thigh_angle_right - self.prev_thigh_angle_right) / self.pub_timer 
        self.prev_thigh_angle_left = current_thigh_angle_left
        self.prev_thigh_angle_right = current_thigh_angle_right

        # --- Phase Variable ---
        phase_left = self.compute_phase_variable(current_thigh_angle_left, qh_dot_left, 'left')
        phase_right = self.compute_phase_variable(current_thigh_angle_right, qh_dot_right, 'right')
        self.phase_pub_left.publish(Float32(data=phase_left))
        self.phase_pub_right.publish(Float32(data=phase_right))
        # --- Desired values from phase variable ---
        desired_left_knee = -joint_positions[2] * phase_left
        desired_right_knee = -joint_positions[3] * phase_right
        desired_left_ankle = joint_positions[4] * phase_left
        desired_right_ankle = joint_positions[5] * phase_right

        if self.current_joint_positions is None:
            self.get_logger().info("Waiting for joint state data...")
            return
        actual_left_hip = self.current_joint_positions[0]
        actual_right_hip = self.current_joint_positions[1]
        actual_left_knee = self.current_joint_positions[2]
        actual_right_knee = self.current_joint_positions[3]
        actual_left_ankle = self.current_joint_positions[4]
        actual_right_ankle = self.current_joint_positions[5]


        # --- Joint velocity estimate (finite difference) ---
        dt = self.pub_timer
        joint_velocities = (joint_positions - self.last_joint_pos) / dt


        # --- PD control for hips ---
        cmd_left_hip = self.kp_hip * (current_thigh_angle_left - actual_left_hip) - self.kd_hip * joint_velocities[0]
        cmd_right_hip = self.kp_hip * (current_thigh_angle_right - actual_right_hip) - self.kd_hip * joint_velocities[1]

        # --- PD control for knees ---
        cmd_left_knee = self.kp_knee * (desired_left_knee - actual_left_knee) - self.kd_knee * joint_velocities[2]
        cmd_right_knee = self.kp_knee * (desired_right_knee - actual_right_knee) - self.kd_knee * joint_velocities[3]

        # --- PD control for ankles ---
        cmd_left_ankle = self.kp_ankle * (desired_left_ankle - actual_left_ankle) - self.kd_ankle * joint_velocities[4]
        cmd_right_ankle = self.kp_ankle * (desired_right_ankle - actual_right_ankle) - self.kd_ankle * joint_velocities[5]

        # --- Apply PD commands ---
        joint_positions[0] += cmd_left_hip * dt
        joint_positions[1] += cmd_right_hip * dt
        joint_positions[2] += cmd_left_knee * dt
        joint_positions[3] += cmd_right_knee * dt
        joint_positions[4] += cmd_left_ankle * dt
        joint_positions[5] += cmd_right_ankle * dt
        # Saturate joint limits for knees and ankles (degrees)
        for i, joint in enumerate(self.joint_names):
            if 'knee' in joint and joint_positions[i] >= -0.5:
                joint_positions[i] = -joint_positions[i]
            if 'ankle' in joint and joint_positions[i] <= -25.0:
                joint_positions[i] = -25.0
            if 'ankle' in joint and joint_positions[i] >= 16.5:
                joint_positions[i] = 16.0
        self.last_joint_pos = joint_positions.copy()
        self.last_goal_position = joint_positions.tolist()  # radians
        self.goal_sent = True
        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = joint_positions.tolist()
        point.time_from_start = Duration(sec=0, nanosec=300_000_000)
        msg.points.append(point)

        self.trajectory_publisher_.publish(msg)
        self.get_logger().info(f"Phase Left: {round(phase_left,2)}, Phase Right: {round(phase_right,2)} | Sent: {np.round(joint_positions, 2)}")
        self.traj_index += 1


    def joint_state_callback(self, msg: JointState):
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        try:
            self.current_joint_positions = np.array(
                [msg.position[name_to_idx[j]] for j in self.joint_names],
                dtype=np.float32
            )
        except KeyError:
            return

        # Only run the goal check if you actually set these when you publish a goal
        if not (self.goal_sent and self.last_goal_position is not None):
            return

        error_threshold_rad = 0.1
        # NOTE: last_goal_position should be in **radians** and in the same order as joint_names
        joint_errors = np.abs(self.current_joint_positions - np.array(self.last_goal_position, dtype=np.float32))
        if np.all(joint_errors < error_threshold_rad):
            self.get_logger().info("All joints reached goal. Proceeding to next point.")
            self.goal_sent = False


def main(args=None):
    rclpy.init(args=args)
    node = PhaseVariableJointPublisher()
    def shutdown_handler(signum, frame):
        node.get_logger().info("User exit detected. Sending exoskeleton to double stance and exiting...")
        node.send_joints_to_zero()
        time.sleep(2.0)  # Wait to ensure the message is sent
        node.destroy_node()
        rclpy.shutdown()
        exit(0)
    signal.signal(signal.SIGINT, shutdown_handler)

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
