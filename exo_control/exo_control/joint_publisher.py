import os
import rclpy
import signal
import time
from rclpy.node import Node
import numpy as np
import pandas as pd
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
from ament_index_python.packages import get_package_share_directory
from tensorflow.python.keras.models import load_model
import joblib

class JointPublisherFromModel(Node):
    def __init__(self):
        super().__init__('joint_publisher_from_model')


        # Publisher
        self.trajectory_publisher_ = self.create_publisher(JointTrajectory, '/trajectory_controller/joint_trajectory', 10)
        # Subscriber
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        # Joint names for the exoskeleton
        self.joint_names = [
            'left_hip_revolute_joint',
            'right_hip_revolute_joint',
            'left_knee_revolute_joint',
            'right_knee_revolute_joint',
            'left_ankle_revolute_joint',
            'right_ankle_revolute_joint'
        ]
        # NN - Load model excel and scaler
        pkg_dir = get_package_share_directory('exo_control')
        self.model_path = os.path.join(pkg_dir, 'neural_network_parameters/models', 'Timestamp_lstm_model.keras')
        self.excel_path = os.path.join(pkg_dir, 'neural_network_parameters/excel', 'timestamps_typical_lstm.xlsx')
        self.scaler_path = os.path.join(pkg_dir, 'neural_network_parameters/scaler', 'standard_scaler.save')

        self.model = load_model(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.get_logger().info("Loaded Keras model and scaler.")

        prediction = pd.read_excel(self.excel_path)
        data = prediction.values.astype(np.float32).reshape((-1, 51, 6))
        self.input_window = data[-1:]
        self.get_logger().info(f"Loaded Excel input with shape {self.input_window.shape}")

        # All parameters for tuning the joing position publisher

        self.pub_timer = 0.06
        self.traj_index = 0

        self.predicted_index = None
        self.goal_sent = False
        self.last_goal_position = None
        self.state_left = 'Stance'
        self.state_right = 'Stance'
        self.q0h_left = None
        self.qh_min_left = None
        self.qh_max_left = None
        self.qhm_left = None

        self.current_joint_positions = None
        self.prev_joint_pos = None

        self.kp_knee = 20.0
        self.kd_knee = 0.5
        self.kp_ankle = 10.0
        self.kd_ankle = 0.2

        # Node startup
        self.timer = self.create_timer( self.pub_timer, self.timer_callback)


    def compute_phase_variable(self, prediction):
        """Computes the phase variable from predicted joint angles."""
        theta_touchdown_left = prediction[0][0]  # Thigh angle at touchdown (left)
        theta_min_left = prediction[0].min()  # Minimum thigh angle (left)
        theta_touchdown_right = prediction[1][0]  # Thigh angle at touchdown (right)
        theta_min_right = prediction[1].min()  # Minimum thigh angle (right)
        c = 0.53  # Default scaling constant
        s_left = np.zeros(len(prediction))
        s_right = np.zeros(len(prediction))
        
        foot_contact_binary_left = np.zeros(len(prediction))  # Initialize stance phase indicator (left)
        foot_contact_binary_right = np.zeros(len(prediction))  # Initialize stance phase indicator (right)
        for i in range(0, len(prediction[0]), 51):
            foot_contact_percent_left = 66.19
            stance_rows_left = int(51 * (foot_contact_percent_left / 100))  
            foot_contact_binary_left[i:i + stance_rows_left] = 1  
            foot_contact_percent_right = 64.03
            stance_rows_right = int(51 * (foot_contact_percent_right / 100))  
            foot_contact_binary_right[i:i + stance_rows_right] = 1  
        for i in range(len(prediction[0])):
            if foot_contact_binary_left[i] == 1 and foot_contact_binary_right[i] == 0:  # Stance phase (left) and swing phase (right)
                s_left[i] = ((theta_touchdown_left - prediction[0][i]) / (theta_touchdown_left - theta_min_left)) * c
                theta_m_right = prediction[1].min()
                s_right[i] = 1 + ((1 - s_right[i-1]) / (theta_touchdown_right - theta_m_right)) * (prediction[1][i] - theta_touchdown_right)
            elif foot_contact_binary_right[i] == 1 and foot_contact_binary_left[i] == 0:  # Stance phase (right) and swing phase (left)
                s_right[i] = ((theta_touchdown_right - prediction[1][i]) / (theta_touchdown_right - theta_min_right)) * c
                theta_m_left = prediction[0].min()
                s_left[i] = 1 + ((1 - s_left[i-1]) / (theta_touchdown_left - theta_m_left)) * (prediction[0][i] - theta_touchdown_left)
            elif foot_contact_binary_left[i] == 1 and foot_contact_binary_right[i] == 1:  # Both legs in stance phase
                s_left[i] = ((theta_touchdown_left - prediction[0][i]) / (theta_touchdown_left - theta_min_left)) * c
                s_right[i] = ((theta_touchdown_right - prediction[1][i]) / (theta_touchdown_right - theta_min_right)) * c
            elif foot_contact_binary_left[i] == 0 and foot_contact_binary_right[i] == 0:  # Both legs in swing phase
                theta_m_left = prediction[0].min()
                s_left[i] = 1 + ((1 - s_left[i-1]) / (theta_touchdown_left - theta_m_left)) * (prediction[0][i] - theta_touchdown_left)
                theta_m_right = prediction[1].min()
                s_right[i] = 1 + ((1 - s_right[i-1]) / (theta_touchdown_right - theta_m_right)) * (prediction[1][i] - theta_touchdown_right)
        return s_left, s_right

    def timer_callback(self):

        # Publish next point in trajectory using model output
        if self.predicted_index is None:
            prediction = self.model.predict(self.input_window, verbose=0)  # Prediction output is (1, 51, 6)
            self.last_prediction_step = prediction[0]  # Shape (51, 6)
            predicted_deg = self.scaler.inverse_transform(self.last_prediction_step)
            phase_left, phase_right = self.compute_phase_variable(predicted_deg)
            self.phase_var = np.stack([phase_left, phase_right], axis=1)  # Shape (51, 2)
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
            self.get_logger().info(f"Full step completed. With total timesteps: {self.traj_index} ")
            self.predicted_index = None
            self.traj_index = 0
            self.input_window = self.last_prediction_step.reshape((1, 51, 6))  # Update input window for next prediction

    # Joint Publisher
    def publish_joint_trajectory(self, positions_from_model):
        # Saturate joint limits for ankle
        for i, joint in enumerate(self.joint_names):
            if 'knee' in joint and positions_from_model[i] > 0.0:
                positions_from_model[i] = -positions_from_model[i]

        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = np.radians(positions_from_model).tolist()
        # self.get_logger().info(f"Published positions in rad: {np.round((point.positions), 2)}")
        point.time_from_start = Duration(sec=0, nanosec=300_000_000)
        msg.points.append(point)
        self.trajectory_publisher_.publish(msg)


    def joint_state_callback(self, msg):
        if not self.goal_sent or self.last_goal_position is None:
            return
        if all(name in msg.name for name in self.joint_names):
            self.current_joint_positions = np.array([msg.position[msg.name.index(j)] for j in self.joint_names])
        current_pos = np.array([msg.position[msg.name.index(j)] for j in self.joint_names])
        if np.all(np.abs(current_pos - self.last_goal_position) < 0.1):
            self.get_logger().info("Reached goal, moving to next point.")
            self.goal_sent = False
            self.traj_index += 1
            

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

