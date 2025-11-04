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
from tensorflow.python.keras.models import load_model
import joblib

class JointPublisherFromModel(Node):
    def __init__(self):
        super().__init__('joint_publisher_from_model')

        # Publisher
        self.trajectory_publisher_ = self.create_publisher(JointTrajectory, '/trajectory_controller/joint_trajectory', 10)
        # Subscriber
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        # Load model, excel, scaler
        pkg_dir = get_package_share_directory('exo_control')
        self.model_path = os.path.join(pkg_dir, 'neural_network_parameters/models', 'Timestamp_lstm_model.keras') #Timestamp_cnn_model.keras
        self.excel_path = os.path.join(pkg_dir, 'neural_network_parameters/excel', 'timestamps_typical_lstm.xlsx') #timestamps_typical_cnn.xlsx
        self.scaler_path = os.path.join(pkg_dir, 'neural_network_parameters/scaler', 'standard_scaler_typical_lstm.save')
        self.model = load_model(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.get_logger().info("Loaded Keras model and scaler.")
        excel_input = pd.read_excel(self.excel_path)
        raw = excel_input.values.astype(np.float32).reshape((-1, 51, 6))

        # transform with the scaler used for training
        raw_2d = raw.reshape(-1, 6)
        raw_scaled_2d = self.scaler.transform(raw_2d)
        self.input_window = raw_scaled_2d.reshape(1, 51, 6)
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
        self.pub_timer = 0.02  # 20ms
        self.predicted_index = None
        self.goal_sent = False
        self.last_goal_position = None
        self.prev_joint_pos = None
        self.traj_index = 0
        self.last_send_time = 0.0
        self.resend_timeout = 1.0  # seconds
        self.error_threshold_rad = 0.05  # radians
        self.settle_cycles_required = 3  # number of consecutive OK callbacks
        self._consecutive_ok = 0
        self._last_js_time = None
        self._last_js_pos = None
        self.velocity_quasi_threshold = 0.02
        # Node startup
        self.timer = self.create_timer( self.pub_timer, self.timer_callback)

    def timer_callback(self):
        # Prepare a new predicted step if needed
        if self.predicted_index is None:
            prediction_scaled = self.model.predict(self.input_window, verbose=0)[0]  # (51,6) in scaled space
            self.last_prediction_step = prediction_scaled  # (51, 6)
            # Blend last 2 of previous with first 2 of current
            if hasattr(self, 'last_pred_scaled') and self.last_pred_scaled is not None:
                overlap = 5
                prev = self.last_pred_scaled[-overlap:].copy()   # last 2 of previous stride
                curr = prediction_scaled[:overlap+4].copy()         # first 2 of new stride

                # Crossfade smoothly (linear ramp)
                for i in range(overlap):
                    alpha = (i + 1) / (overlap + 1)
                    blended = (1 - alpha) * prev[i] + alpha * curr[i]
                    prediction_scaled[i] = blended

            # Store for next time (scaled) and prepare inputs (still scaled)
            self.last_pred_scaled = prediction_scaled
            self.input_window = prediction_scaled.reshape(1, 51, 6)

            # For publishing, convert to degrees once
            predicted_deg = self.scaler.inverse_transform(prediction_scaled)
            self.predicted_index = predicted_deg
            self.traj_index = 0
            self.goal_sent = False
            self.last_goal_position = None
            # self.get_logger().info(f"prediction {predicted_deg}")
            # predicted_deg = self.filter_data(predicted_deg)
            # self.get_logger().info(f"Filtered prediction: {predicted_deg}")
            # Store the sequence for this window
            self.predicted_index = predicted_deg
            self.traj_index = 0
            self.goal_sent = False
            self.last_goal_position = None

            self.predicted_index = predicted_deg
            self.traj_index = 0
            self.goal_sent = False
            self.last_goal_position = None

        # If we still have points to execute
        if self.traj_index < len(self.predicted_index):
            if not self.goal_sent:
                next_point = self.predicted_index[self.traj_index]

                # Now publish the joint point for tracking
                self.publish_joint_trajectory(next_point)
                self.last_goal_position = next_point
                self.goal_sent = True
                self.last_send_time = time.time()
            else:
                if time.time() - self.last_send_time > self.resend_timeout:
                    self.publish_joint_trajectory(self.last_goal_position)
                    self.last_send_time = time.time()
        else:
            # === Finished this predicted window; prepare next ===
            next_pred_scaled = self.model.predict(self.input_window, verbose=0)[0]
            end_of_prev = self.last_pred_scaled[-1].copy()  # last point of previous stride
            for i in range(10):  # adjust 10 for however many frames you want to adjust
                alpha = (i + 1) / 11.0  # ramp up smoothly
                next_pred_scaled[i] = (1 - alpha) * end_of_prev + alpha * next_pred_scaled[i]

            # --- Temporal overlap for model continuity ---
            overlap = 5  # last N frames of prev stride to seed next
            self.input_window = np.concatenate(
                [self.last_pred_scaled[-overlap:], next_pred_scaled[:-overlap]],
                axis=0
            ).reshape(1, 51, 6)

            # Store for next iteration
            self.last_pred_scaled = next_pred_scaled
            self.predicted_index = self.scaler.inverse_transform(next_pred_scaled)
            self.traj_index = 0
            self.goal_sent = False


    # Joint Publisher to Gazebo Simulation
    def publish_joint_trajectory(self, positions_from_model):
        for i, joint in enumerate(self.joint_names):
            if 'knee' in joint and positions_from_model[i] >= 0.0:
                positions_from_model[i] = -positions_from_model[i]
        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        # self.get_logger().info(f"Next goal positions (deg): {np.round(positions_from_model, 2)}")
        point.positions = np.radians(positions_from_model).tolist()
        self.get_logger().info(f"Next goal positions (rad): {np.round(point.positions, 2)}")
        # self.get_logger().info(f"Publishing new goal positions: {np.round(point.positions, 2)}")
        point.time_from_start = Duration(sec=0, nanosec=50_000_000)  # 50ms
        msg.points.append(point)
        self.trajectory_publisher_.publish(msg)
        return msg

    def joint_state_callback(self, msg):
        if not self.goal_sent or self.last_goal_position is None:
            return
        if not all(name in msg.name for name in self.joint_names):
            return

        # Extract in controller order
        current = np.array([msg.position[msg.name.index(j)] for j in self.joint_names])
        desired = np.radians(self.last_goal_position)
        pos_err = np.abs(current - desired)

        # Simple velocity proxy (finite difference per callback)
        now = self.get_clock().now().nanoseconds * 1e-9
        vel_ok = True
        if self._last_js_time is not None and self._last_js_pos is not None:
            dt = max(1e-3, now - self._last_js_time)
            est_vel = np.abs((current - self._last_js_pos) / dt)
            vel_ok = np.all(est_vel < self.velocity_quasi_threshold)

        self._last_js_time = now
        self._last_js_pos = current

        pos_ok = np.all(pos_err < self.error_threshold_rad)

        if pos_ok and vel_ok:
            self._consecutive_ok += 1
        else:
            self._consecutive_ok = 0

        if self._consecutive_ok >= self.settle_cycles_required:
            # self.get_logger().info("Goal reached & settled. Proceeding to next point.")
            self.goal_sent = False
            self.traj_index += 1
            self._consecutive_ok = 0
        else:
            self.goal_sent = True

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
        time.sleep(1.0)  # Wait to ensure the message is sent
        node.destroy_node()
        rclpy.shutdown()
        exit(0)
    signal.signal(signal.SIGINT, shutdown_handler)

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()