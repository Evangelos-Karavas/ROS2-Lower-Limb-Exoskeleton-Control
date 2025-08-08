import os
import rclpy
import signal
import time
from rclpy.node import Node
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float32, Float32MultiArray
from builtin_interfaces.msg import Duration
from ament_index_python.packages import get_package_share_directory
from tensorflow.python.keras.models import load_model

import joblib

class JointPublisherFromModel(Node):
    def __init__(self):
        super().__init__('joint_publisher_from_model')

        # Publisher
        self.trajectory_publisher_ = self.create_publisher(JointTrajectory, '/trajectory_controller/joint_trajectory', 10)
        self.phase_var_pub_left = self.create_publisher(Float32MultiArray, '/phase_variable_left', 10)
        self.phase_var_pub_right = self.create_publisher(Float32MultiArray, '/phase_variable_right', 10)
        # Subscriber
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        # Neural Networks - Load model - excel - scaler
        pkg_dir = get_package_share_directory('exo_control')
        self.model_path = os.path.join(pkg_dir, 'neural_network_parameters/models', 'Timestamp_lstm_model.keras')
        self.excel_path = os.path.join(pkg_dir, 'neural_network_parameters/excel', 'timestamps_typical_lstm.xlsx')
        self.scaler_path = os.path.join(pkg_dir, 'neural_network_parameters/scaler', 'standard_scaler.save')
        self.model = load_model(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.get_logger().info("Loaded Keras model and scaler.")
        excel_input = pd.read_excel(self.excel_path)
        data = excel_input.values.astype(np.float32).reshape((-1, 51, 6))
        self.input_window = data[-1:]
        self.get_logger().info(f"Loaded Excel values with shape {self.input_window.shape}")
        # Joint names for the exoskeleton
        self.joint_names = [
            'left_hip_revolute_joint',
            'right_hip_revolute_joint',
            'left_knee_revolute_joint',
            'right_knee_revolute_joint',
            'left_ankle_revolute_joint',
            'right_ankle_revolute_joint'
        ]
        # All parameters for tuning the joint position publisher
        self.pub_timer = 0.1
        self.predicted_index = None
        self.goal_sent = False
        self.last_goal_position = None
        self.current_joint_positions = None
        self.prev_joint_pos = None
        self.traj_index = 0
        self.q_po = -8.3  # pushoff trigger thigh angle (deg)
        self.c = 0.53  # stance phase portion of stride (from paper)
        # Node startup
        self.timer = self.create_timer( self.pub_timer, self.timer_callback)


    def filter_data(self, joint_angles):
        lowcut = 10.0
        highcut = 300.0
        fs = 1000.0
        order = 3
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        filtered = np.zeros_like(joint_angles)
        for j in range(joint_angles.shape[1]):
            filtered[:, j] = filtfilt(b, a, joint_angles[:, j])
        return filtered
    def simulate_foot_contact(self, length, stance_percent):
        contact = np.zeros(length)
        stance_frames = int(length * (stance_percent / 100))
        contact[:stance_frames] = 1
        return contact
    def compute_phase_variable_fsm(self, thigh_angle, foot_contact):
        """
        Computation of phase variable for rythmic and non-rythmic walking
        based on Bobby Gregg's paper: Phase-Variable Control of a Powered
        Knee-Ankle Prosthesis over Continuously Varying Speeds and Inclines
        """
        N = len(thigh_angle)
        s = np.zeros(N)
        state = 'S1'
        theta_td = thigh_angle[0] + 0.05  # use first frame as touchdown
        theta_min = thigh_angle.min()
        s_m = 0
        theta_m = theta_min -0.05
        for i in range(N):
            qh = thigh_angle[i]
            qh_dot = (thigh_angle[i] - thigh_angle[i - 1]) / self.pub_timer if i > 0 else 0
            fc = foot_contact[i]
            # --- State transitions ---
            if state == 'S1' and qh <= self.q_po:
                state = 'S2'
            elif state == 'S2' and qh_dot >= 0:
                state = 'S3'
                s_m = s[i - 1]
                theta_m = qh
            elif state == 'S3' and fc == 0:
                state = 'S4'
            elif state == 'S4' and fc == 1:
                state = 'S1'
                theta_td = qh
                lookahead = thigh_angle[i:i+5]
                theta_min = lookahead.min() if len(lookahead) > 0 else theta_min
                s_m = 0
            # --- Compute s ---
            if state in ['S1', 'S2']:
                s[i] = (theta_td - qh) / (theta_td - theta_min) * self.c
            elif state in ['S3', 'S4']:
                s[i] = 1 + ((1 - s_m) / (theta_td - theta_m)) * (qh - theta_td)

            # Keep it in bounds
            s[i] = np.clip(s[i], 0, 1)

        return s

    def timer_callback(self):
        # Publish next point in trajectory using model output
        if self.predicted_index is None:
            prediction = self.model.predict(self.input_window, verbose=0)  # Prediction output is (1, 51, 6)
            self.last_prediction_step = prediction[0]  # Shape (51, 6)
            predicted_deg = self.scaler.inverse_transform(self.last_prediction_step)
            predicted_deg = self.filter_data(predicted_deg)
            fc_left = self.simulate_foot_contact(len(predicted_deg), 64.18)
            phase_left = self.compute_phase_variable_fsm(predicted_deg[:, 0], fc_left)
            fc_right = self.simulate_foot_contact(len(predicted_deg), 65.0)
            phase_right = self.compute_phase_variable_fsm(predicted_deg[:, 1], fc_right)
            self.phase_var = np.stack([phase_left, phase_right], axis=1)  # Shape (51, 2)
            # Publish phase_var arrays
            for i in range(len(predicted_deg)):
                msg_phase_left = Float32MultiArray()
                msg_phase_right = Float32MultiArray()
                msg_phase_left.data = phase_left.tolist()
                msg_phase_right.data = phase_right.tolist()
            self.phase_var_pub_left.publish(msg_phase_left)
            self.phase_var_pub_right.publish(msg_phase_right)
            self.predicted_index = predicted_deg

        # If trajectory is not finished, publish the next point of the step
        if self.traj_index < len(self.predicted_index): 
            next_point = self.predicted_index[self.traj_index]
            self.publish_joint_trajectory(next_point)
            self.last_goal_position = next_point
            self.goal_sent = True
            self.traj_index += 1
        # If trajectory is finished, reset the index and predicted trajectory to start the next step
        else: 
            self.predicted_index = None
            self.traj_index = 0
            self.input_window = self.last_prediction_step.reshape((1, 51, 6))  # Update input window for next prediction

    # Joint Publisher to Gazebo Simulation
    def publish_joint_trajectory(self, positions_from_model):
        # Saturate joint limits for ankle
        for i, joint in enumerate(self.joint_names):
            if 'knee' in joint and positions_from_model[i] > 0.0:
                positions_from_model[i] = -positions_from_model[i]

        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = np.radians(positions_from_model).tolist()
        # point.positions = positions_from_model.tolist()
        # self.get_logger().info(f"Published positions in rad: {np.round((point.positions), 2)}")
        point.time_from_start = Duration(sec=0, nanosec=300_000_000)
        msg.points.append(point)
        self.trajectory_publisher_.publish(msg)
        return msg

    # Joint State function to update predicted values if previous is reached
    def joint_state_callback(self, msg):
        if not self.goal_sent or self.last_goal_position is None:
            return
        if all(name in msg.name for name in self.joint_names):
            self.current_joint_positions = np.array(
                [msg.position[msg.name.index(j)] for j in self.joint_names]
            )
            error_threshold_deg = 1.0  # degrees
            error_threshold_rad = np.radians(error_threshold_deg)
            joint_errors = np.abs(self.current_joint_positions - np.radians(self.last_goal_position))
            joints_within_tolerance = joint_errors < error_threshold_rad
            if np.all(joints_within_tolerance):
                self.get_logger().info("✅ All joints reached goal. Proceeding to next point.")
                self.goal_sent = False
                self.traj_index += 1
            else:
                self.get_logger().info("Not all joints reached their goal. Holding position.")
                # self.publish_joint_trajectory(self.last_goal_position)

    def send_joints_to_zero(self):
        zero_positions = np.zeros(len(self.joint_names))
        self.publish_joint_trajectory(zero_positions)
        self.get_logger().info("Sent all joints to zero before shutdown.")

def main(args=None):
    rclpy.init(args=args)
    node = JointPublisherFromModel()

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

