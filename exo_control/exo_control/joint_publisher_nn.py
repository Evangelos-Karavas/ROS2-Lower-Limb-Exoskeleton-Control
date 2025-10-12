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
        self.model_path = os.path.join(pkg_dir, 'neural_network_parameters/models', 'Timestamp_lstm_model.keras') #Timestamp_cnn_model.keras
        self.excel_path = os.path.join(pkg_dir, 'neural_network_parameters/excel', 'timestamps_typical_lstm.xlsx') #timestamps_typical_cnn.xlsx
        self.scaler_path = os.path.join(pkg_dir, 'neural_network_parameters/scaler', 'standard_scaler_typical_lstm.save')
        self.model = load_model(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.get_logger().info("Loaded Keras model and scaler.")
        excel_input = pd.read_excel(self.excel_path)
        raw = excel_input.values.astype(np.float32).reshape((-1, 51, 6))

        # IMPORTANT: transform with the SAME scaler used for training
        raw_2d = raw.reshape(-1, 6)                        # (51,6) -> (51*1,6)
        raw_scaled_2d = self.scaler.transform(raw_2d)      # scale
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
        self.pub_timer = 0.05
        self.predicted_index = None
        self.goal_sent = False
        self.last_goal_position = None
        self.current_joint_positions = None
        self.prev_joint_pos = None
        self.traj_index = 0
        self.qh_dot = 0.0  # thigh angular velocity (deg/s)
        self.q_po = -8.3  # pushoff trigger thigh angle (deg)
        self.c = 0.53  # stance phase portion of stride (from paper)
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


    def simulate_foot_contact(self, length, stance_percent): 
        contact = np.zeros(length) 
        stance_frames = int(length * (stance_percent / 100)) 
        contact[:stance_frames] = 1 
        return contact
    
    def _init_fsm(self, qh0):
        # Initialize per Bobby Gregg’s structure
        return {
            "state": "S1",
            "theta_td": qh0 + 0.05,  # touchdown ref
            "theta_min": qh0,        # will be updated with lookahead or running min
            "s_m": 0.0,
            "theta_m": qh0 - 0.05,
            "qh_prev": None,
        }

    def _step_fsm(self, fsm, qh, fc, dt):
        # Finite difference for qh_dot
        if fsm["qh_prev"] is None:
            qh_dot = 0.0
        else:
            qh_dot = (qh - fsm["qh_prev"]) / dt
        fsm["qh_prev"] = qh

        state = fsm["state"]
        theta_td = fsm["theta_td"]
        theta_min = fsm["theta_min"]
        s_m = fsm["s_m"]
        theta_m = fsm["theta_m"]

        # --- State transitions ---
        if state == "S1" and qh <= self.q_po:
            state = "S2"
        elif state == "S2" and qh_dot >= 0:
            state = "S3"
            s_m = getattr(self, "_last_s", 0.0)  # previous s
            theta_m = qh
        elif state == "S3" and fc == 0:
            state = "S4"
        elif state == "S4" and fc == 1:
            state = "S1"
            theta_td = qh
            # Update theta_min conservatively (no lookahead in step mode)
            theta_min = min(theta_min, qh)
            s_m = 0.0

        # --- Compute s ---
        if state in ("S1", "S2"):
            denom = max(1e-6, (theta_td - theta_min))
            s = (theta_td - qh) / denom * self.c
        else:  # S3, S4
            denom = max(1e-6, (theta_td - theta_m))
            s = 1.0 + ((1.0 - s_m) / denom) * (qh - theta_td)

        # Bound and write back
        s = float(np.clip(s, 0.0, 1.0))
        self._last_s = s  # remember last s for s_m capture
        fsm["state"] = state
        fsm["theta_td"] = theta_td
        fsm["theta_min"] = theta_min
        fsm["s_m"] = s_m
        fsm["theta_m"] = theta_m

        return s, qh_dot

    def filter_data(self, joint_angles):
        lowcut = 15.0
        highcut = 250.0
        fs = 501.0
        order = 3
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        filtered = np.zeros_like(joint_angles)
        for j in range(joint_angles.shape[1]):
            filtered[:, j] = filtfilt(b, a, joint_angles[:, j])
        return filtered

    def timer_callback(self):
        # Prepare a new predicted step if needed
        if self.predicted_index is None:
            prediction = self.model.predict(self.input_window, verbose=0)
            self.last_prediction_step = prediction[0]  # (51, 6)
            predicted_deg = self.scaler.inverse_transform(self.last_prediction_step)
            # self.get_logger().info(f"prediction {predicted_deg}")
            # predicted_deg = self.filter_data(predicted_deg)
            # self.get_logger().info(f"Filtered prediction: {predicted_deg}")
            # Store the sequence for this window
            self.predicted_index = predicted_deg
            self.traj_index = 0
            self.goal_sent = False
            self.last_goal_position = None

            # Simulate FC per window (or replace with your real detector)
            self.fc_left_seq = self.simulate_foot_contact(len(predicted_deg), 64.18)
            self.fc_right_seq = self.simulate_foot_contact(len(predicted_deg), 65.0)

            # Init FSMs at first sample’s thigh angle
            qhL0 = predicted_deg[0, 0]
            qhR0 = predicted_deg[0, 1]
            self.fsm_left = self._init_fsm(qhL0)
            self.fsm_right = self._init_fsm(qhR0)

            #msg_phase_left = Float32MultiArray(); msg_phase_left.data = phase_left.tolist()
            #msg_phase_right = Float32MultiArray(); msg_phase_right.data = phase_right.tolist()
            #self.phase_var_pub_left.publish(msg_phase_left)
            #self.phase_var_pub_right.publish(msg_phase_right)

            self.predicted_index = predicted_deg
            self.traj_index = 0
            self.goal_sent = False
            self.last_goal_position = None

        # If we still have points to execute
        if self.traj_index < len(self.predicted_index):
            if not self.goal_sent:
                next_point = self.predicted_index[self.traj_index]

                # --- per-timestep phase update ---
                dt = self.pub_timer
                qh_left = next_point[0]   # deg
                qh_right = next_point[1]  # deg
                fcL = int(self.fc_left_seq[self.traj_index])
                fcR = int(self.fc_right_seq[self.traj_index])

                s_left, qh_dot_left = self._step_fsm(self.fsm_left, qh_left, fcL, dt)
                s_right, qh_dot_right = self._step_fsm(self.fsm_right, qh_right, fcR, dt)
                self.qh_dot_left = qh_dot_left
                self.qh_dot_right = qh_dot_right

                # Publish current phase values (one-sample messages)
                msg_phase_left = Float32MultiArray();  msg_phase_left.data = [s_left]
                msg_phase_right = Float32MultiArray(); msg_phase_right.data = [s_right]
                self.phase_var_pub_left.publish(msg_phase_left)
                self.phase_var_pub_right.publish(msg_phase_right)
                # --- end per-timestep phase update ---

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
            # Finished this predicted window; prep next
            self.predicted_index = None
            self.input_window = self.last_prediction_step.reshape((1, 51, 6))


    # Joint Publisher to Gazebo Simulation
    def publish_joint_trajectory(self, positions_from_model):
        for i, joint in enumerate(self.joint_names):
            if 'knee' in joint and positions_from_model[i] >= 0.0:
                positions_from_model[i] = -positions_from_model[i]
            # if 'hip' in joint and positions_from_model[i] >= 50.0:
            #     positions_from_model[i] = 50.0
            # if 'hip' in joint and positions_from_model[i] <= -20.0:
            #     positions_from_model[i] = -20.0
            # if 'ankle' in joint:
            #     positions_from_model[i] = 0.0
            # self.get_logger().info(f"qh_dot: {self.qh_dot:.2f} deg/s")
            # if self.qh_dot_left < -10:
            #     if 'ankle' in joint and 'left' in joint:
            #         positions_from_model[i] = 15.0
            #     else:
            #         positions_from_model[i] = -10.0
            # if self.qh_dot_right < -10:
            #     if 'ankle' in joint and 'right' in joint:
            #         positions_from_model[i] = 15.0
            #     else:
            #         positions_from_model[i] = -10.0
        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        self.get_logger().info(f"Next goal positions (deg): {np.round(positions_from_model, 2)}")
        point.positions = np.radians(positions_from_model).tolist()
        self.get_logger().info(f"Next goal positions (rad): {np.round(point.positions, 2)}")
        # self.get_logger().info(f"Publishing new goal positions: {np.round(point.positions, 2)}")
        point.time_from_start = Duration(sec=0, nanosec=500_000_000)  # 0.5s
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
            # Still holding current goal
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

