import os
import rclpy
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

        # Timer for publishing next joint angles
        pub_timer = 0.1
        # Publisher
        self.trajectory_publisher_ = self.create_publisher(JointTrajectory, '/trajectory_controller/joint_trajectory', 10)
        # Subscriber
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        self.joint_names = [
            'left_hip_revolute_joint',
            'right_hip_revolute_joint',
            'left_knee_revolute_joint',
            'right_knee_revolute_joint',
            'left_ankle_revolute_joint',
            'right_ankle_revolute_joint'
        ]

        pkg_dir = get_package_share_directory('exo_control')
        self.model_path = os.path.join(pkg_dir, 'neural_network_parameters/models', 'Timestamp_lstm_model.keras')
        self.excel_path = os.path.join(pkg_dir, 'neural_network_parameters/excel', 'timestamps_typical_lstm.xlsx')
        self.scaler_path = os.path.join(pkg_dir, 'neural_network_parameters/scaler', 'standard_scaler.save')

        self.model = load_model(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.get_logger().info("Loaded Keras model and scaler.")

        df = pd.read_excel(self.excel_path)
        data = df.values.astype(np.float32).reshape((-1, 51, 6))
        self.input_window = data[-1:]
        self.get_logger().info(f"Loaded Excel input with shape {self.input_window.shape}")

        self.predicted_traj = None
        self.traj_index = 0
        self.goal_sent = False
        self.last_goal_position = None

        self.timer = self.create_timer( pub_timer, self.timer_callback)

    # Trajectory timer_callback
    def timer_callback(self):
        # Only for first input from excel
        if self.predicted_traj is None:
            prediction = self.model.predict(self.input_window, verbose=0)
            last_prediction_step = prediction[0]
            predicted_deg = self.scaler.inverse_transform(last_prediction_step)
            predicted_deg = predicted_deg
            self.predicted_traj = np.radians(predicted_deg)
            self.traj_index = 0
            self.get_logger().info("Predicted and transformed trajectory.")

        if self.traj_index < len(self.predicted_traj):
            next_point = self.predicted_traj[self.traj_index]
            self.publish_joint_trajectory(next_point)
            self.last_goal_position = next_point
            self.goal_sent = True
        else:
            self.get_logger().info("All trajectory points sent.")
            self.predicted_traj = None
            self.traj_index = 0

    # Joint Publisher
    def publish_joint_trajectory(self, positions_from_model):
        # Change the knee values from positive to negative from nn model output to simulation in Gazebo
        for i, joint in enumerate(self.joint_names):
            if 'knee' in joint and positions_from_model[i] > 0.0:
                positions_from_model[i] = -positions_from_model[i]
        for i, joint in enumerate(self.joint_names):
            if 'ankle' in joint and positions_from_model[i] <= -0.3:
                positions_from_model[i] = -0.4
            if 'ankle' in joint and positions_from_model[i] >= 0.36:
                positions_from_model[i] = 0.36

        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = positions_from_model.tolist()
        point.time_from_start = Duration(sec=0, nanosec=300)
        msg.points.append(point)

        self.trajectory_publisher_.publish(msg)
        self.get_logger().info(f"Published positions: {np.round(positions_from_model, 2)}")


    def joint_state_callback(self, msg):
        if not self.goal_sent or self.last_goal_position is None:
            return
        if not all(name in msg.name for name in self.joint_names):
            return

        current_pos = np.array([msg.position[msg.name.index(j)] for j in self.joint_names])
        if self.positions_close(current_pos, self.last_goal_position, tolerance=0.1):
            self.get_logger().info("Reached goal, moving to next point.")
            self.goal_sent = False
            self.traj_index += 1

    @staticmethod
    def positions_close(actual, goal, tolerance=0.1):
        return all(abs(a - g) < tolerance for a, g in zip(actual, goal))


def main(args=None):
    rclpy.init(args=args)
    node = JointPublisherFromModel()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

