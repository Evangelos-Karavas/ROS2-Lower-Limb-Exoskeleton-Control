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
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import subprocess

class JointPublisherFromModel(Node):
    """
    Streams LSTM-predicted joint positions to ros2_control by publishing
    multi-point JointTrajectory segments (sliding horizon).
    """

    def __init__(self):
        super().__init__('joint_publisher_from_nn_model')

        if not self.has_parameter('use_sim_time'): self.declare_parameter('use_sim_time', True)
        # QoS for joint_states: shallow queue + best effort for low latency
        js_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1, durability=DurabilityPolicy.VOLATILE)

        # Publisher
        self.trajectory_publisher_ = self.create_publisher(JointTrajectory, '/trajectory_controller/joint_trajectory', 10)
        # Subscriber
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, js_qos)

        # --- Load model + scaler + initial window from Excel ---
        pkg_dir = get_package_share_directory('exo_control')
        self.model_path = os.path.join(pkg_dir, 'neural_network_parameters/models', 'Timestamp_lstm_model.keras')
        self.excel_path = os.path.join(pkg_dir, 'neural_network_parameters/excel', 'timestamps_cp_lstm.xlsx')
        self.scaler_path = os.path.join(pkg_dir, 'neural_network_parameters/scaler', 'standard_scaler_typical_lstm.save')

        self.model = load_model(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.get_logger().info("Loaded Keras model and scaler.")
        excel_input = pd.read_excel(self.excel_path)
        raw = excel_input.values.astype(np.float32).reshape((-1, 51, 6))
        self.get_logger().info("Loaded excel file of cerebral palsy data.")
        # Transform with the same scaler used from training
        raw_2d = raw.reshape(-1, 6)
        raw_scaled_2d = self.scaler.transform(raw_2d)
        self.input_window = raw_scaled_2d.reshape(1, 51, 6)
        self.get_logger().info(f"Loaded Excel values with shape {self.input_window.shape}")

        self.joint_names = [
            'left_hip_revolute_joint',
            'right_hip_revolute_joint',
            'left_knee_revolute_joint',
            'right_knee_revolute_joint',
            'left_ankle_revolute_joint',
            'right_ankle_revolute_joint'
        ]
        self.pub_timer = 0.3                # seconds between publishes (20 Hz)
        self.segment_len = 3                # number of points per message (sliding horizon)
        self.predicted_deg_window = None    # current window in degrees, shape (51, 6)
        self.last_pred_scaled = None        # last window (scaled), for crossfades
        self.traj_index = 0                 # index within current window
        self._last_js_time = None           # time of last joint state received
        self._last_js_pos = None            # last joint state positions received

        # Start publishing
        self.timer = self.create_timer(self.pub_timer, self.timer_callback)

    # ----------------------------
    # Helpers
    # ----------------------------
    def _flip_knee_signs_inplace(self, arr_2d_or_1d):
        knees_idx = [2, 3]
        if arr_2d_or_1d.ndim == 1:
            for ki in knees_idx:
                if arr_2d_or_1d[ki] >= 0.0:
                    arr_2d_or_1d[ki] = -arr_2d_or_1d[ki]
        else:
            for ki in knees_idx:
                mask = arr_2d_or_1d[:, ki] >= 0.0
                arr_2d_or_1d[mask, ki] = -arr_2d_or_1d[mask, ki]

    def _predict_next_window_scaled(self, seed_scaled_window):
        return self.model.predict(seed_scaled_window, verbose=0)[0]

    def _prepare_first_window_if_needed(self):
        if self.predicted_deg_window is not None:
            return
        prediction_scaled = self._predict_next_window_scaled(self.input_window)

        # If we had a previous scaled window, crossfade the first few frames
        if self.last_pred_scaled is not None:
            overlap = 5
            prev = self.last_pred_scaled[-overlap:].copy()
            curr = prediction_scaled[:overlap + 4].copy()
            for i in range(overlap):
                alpha = (i + 1) / (overlap + 1)
                prediction_scaled[i] = (1 - alpha) * prev[i] + alpha * curr[i]

        self.last_pred_scaled = prediction_scaled
        self.input_window = prediction_scaled.reshape(1, 51, 6)

        predicted_deg = self.scaler.inverse_transform(prediction_scaled)
        self.predicted_deg_window = predicted_deg
        self.traj_index = 0

    def _roll_to_next_window(self):
        """
        After finishing the current window, predict the next window and create
        a smooth transition from the previous end.
        """
        next_pred_scaled = self._predict_next_window_scaled(self.input_window)

        # Smooth ramp from end of previous to beginning of next
        end_of_prev = self.last_pred_scaled[-1].copy()
        for i in range(10):
            alpha = (i + 1) / 11.0
            next_pred_scaled[i] = (1 - alpha) * end_of_prev + alpha * next_pred_scaled[i]

        # Temporal overlap for model continuity
        overlap = 5
        self.input_window = np.concatenate([self.last_pred_scaled[-overlap:], next_pred_scaled[:-overlap]], axis=0).reshape(1, 51, 6)
        self.last_pred_scaled = next_pred_scaled
        self.predicted_deg_window = self.scaler.inverse_transform(next_pred_scaled)
        self.traj_index = 0

    def publish_trajectory(self, positions_seq_deg, dt):

        if len(positions_seq_deg) == 0:
            return
        seg = np.array(positions_seq_deg, dtype=np.float32).copy()

        # Knee conversion from positive to negative angles
        self._flip_knee_signs_inplace(seg)

        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        for i in range(seg.shape[0]):
            point = JointTrajectoryPoint()
            point.positions = np.radians(seg[i]).tolist()
            point.time_from_start = Duration(sec=0, nanosec=int((i + 1) * dt * 1e9))
            msg.points.append(point)

        # Optional: log first point briefly
        self.get_logger().debug(f"Publishing segment of {seg.shape[0]} points. First(rad): {np.round(np.radians(seg[0]), 3)}")

        self.trajectory_publisher_.publish(msg)

    # ----------------------------
    # Timer: streaming publisher
    # ----------------------------
    def timer_callback(self):
        self._prepare_first_window_if_needed()

        # If we still have points remaining in current window, send the next segment
        if self.traj_index < len(self.predicted_deg_window):
            end = min(self.traj_index + self.segment_len, len(self.predicted_deg_window))
            segment = self.predicted_deg_window[self.traj_index:end]
            self.publish_trajectory(segment, dt=self.pub_timer)
            self.traj_index = end
        else:
            # Finished current window → prepare next and immediately send first segment
            self._roll_to_next_window()
            end = min(self.segment_len, len(self.predicted_deg_window))
            segment = self.predicted_deg_window[:end]
            self.publish_trajectory(segment, dt=self.pub_timer)
            self.traj_index = end


    def joint_state_callback(self, msg: JointState):
        expected = set(self.joint_names)
        if not expected.issubset(set(msg.name)):
            return
        current = np.array([msg.position[msg.name.index(j)] for j in self.joint_names], dtype=np.float32)
        self._last_js_pos = current
        self._last_js_time = self.get_clock().now().nanoseconds * 1e-9


    def send_joints_to_zero(self, duration=2.0):
        zeros = np.zeros((1, len(self.joint_names)), dtype=np.float32)
        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = np.radians(zeros[0]).tolist()
        point.time_from_start = Duration(sec=int(duration))
        msg.points.append(point)
        self.trajectory_publisher_.publish(msg)
        self.get_logger().info("Sent all joints to zero.")


def main(args=None):
    rclpy.init(args=args)
    node = JointPublisherFromModel()
    # Handle CTRL+C for safe controller shutdown and return to zero position
    def shutdown_handler(signum, frame):
        node.get_logger().info("CTRL+C: Shutting Down...")
        node.send_joints_to_zero(duration=2.0)
        time.sleep(2.0)  # wait until motion completes
        controllers = ['trajectory_controller', 'joint_state_broadcaster']
        for ctrl in controllers:
            try:
                subprocess.run(['ros2', 'control', 'set_controller_state', ctrl, 'inactive'], check=False, timeout=2)
                subprocess.run(['ros2', 'control', 'unload_controller', ctrl], check=False, timeout=2)
            except subprocess.TimeoutExpired:
                node.get_logger().warn(f"Timeout unloading {ctrl}")
        node.get_logger().info("Controllers unloaded.")
        node.destroy_node()
        rclpy.shutdown()
        os._exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()