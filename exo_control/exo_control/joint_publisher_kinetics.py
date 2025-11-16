import os
import rclpy
import signal
import time
import subprocess

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import numpy as np
import pandas as pd

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from ament_index_python.packages import get_package_share_directory

import tensorflow as tf
import keras
import joblib


class JointPublisherFromModel(Node):
    """
    Streams LSTM-predicted joint positions to ros2_control using JointTrajectory.

    Model:
      - Input:  (1, 51, 54)
            36 dynamics (moments + forces) + 18 joint angles, all scaled
      - Output: (1, 51, 18)
            scaled joint angles (same scaler as input angles)

    Runtime:
      - Publishes only 6 sagittal DOFs (L/R hip,knee,ankle) in degrees → radians
      - Uses sliding horizon segments for smooth streaming.
    """

    def __init__(self):
        super().__init__('joint_publisher_from_nn_model')

        # ----------------------------------------------------
        #   1) Parameters, QoS, pubs/subs
        # ----------------------------------------------------
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)

        # joint_states QoS: low latency, no deep queue
        js_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )

        # Publisher to your trajectory controller
        # Adjust topic if needed (e.g. /left_leg_controller/joint_trajectory, /right_leg_controller/joint_trajectory)
        self.trajectory_publisher_ = self.create_publisher(
            JointTrajectory,
            '/trajectory_controller/joint_trajectory',
            10
        )

        # Subscribe to /joint_states for monitoring / future feedback
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            js_qos
        )

        # ----------------------------------------------------
        #   2) Register custom loss (for loading model)
        # ----------------------------------------------------
        @keras.saving.register_keras_serializable()
        def weighted_mse(y_true, y_pred):
            weights = tf.constant(
                [3, 3, 1,  3, 3, 1,  3, 3, 1,
                 3, 3, 1,  3, 3, 1,  3, 3, 1],
                dtype=tf.float32
            )
            return tf.reduce_mean(weights * tf.square(y_true - y_pred))

        # ----------------------------------------------------
        #   3) Column definitions (36 dyn + 18 angles)
        # ----------------------------------------------------
        MOMENT_COLS = [
            'LHipMoment (1)','RHipMoment (1)','LHipMoment (2)','RHipMoment (2)','LHipMoment (3)','RHipMoment (3)',
            'LKneeMoment (1)','RKneeMoment (1)','LKneeMoment (2)','RKneeMoment (2)','LKneeMoment (3)','RKneeMoment (3)',
            'LAnkleMoment (1)','RAnkleMoment (1)','LAnkleMoment (2)','RAnkleMoment (2)','LAnkleMoment (3)','RAnkleMoment (3)'
        ]
        FORCE_COLS = [
            'LHipForce (1)','RHipForce (1)','LHipForce (2)','RHipForce (2)','LHipForce (3)','RHipForce (3)',
            'LKneeForce (1)','RKneeForce (1)','LKneeForce (2)','RKneeForce (2)','LKneeForce (3)','RKneeForce (3)',
            'LAnkleForce (1)','RAnkleForce (1)','LAnkleForce (2)','RAnkleForce (2)','LAnkleForce (3)','RAnkleForce (3)'
        ]
        ANGLE_COLS = [
            'LHipAngles (1)','RHipAngles (1)','LHipAngles (2)','RHipAngles (2)','LHipAngles (3)','RHipAngles (3)',
            'LKneeAngles (1)','RKneeAngles (1)','LKneeAngles (2)','RKneeAngles (2)','LKneeAngles (3)','RKneeAngles (3)',
            'LAnkleAngles (1)','RAnkleAngles (1)','LAnkleAngles (2)','RAnkleAngles (2)','LAnkleAngles (3)','RAnkleAngles (3)'
        ]
        self.ALL_COLS = MOMENT_COLS + FORCE_COLS + ANGLE_COLS
        self.NUM_DYN = len(MOMENT_COLS) + len(FORCE_COLS)  # 36
        self.NUM_ANG = len(ANGLE_COLS)                      # 18

        # ----------------------------------------------------
        #   4) Load model + scalers + initial Excel window
        # ----------------------------------------------------
        pkg_dir = get_package_share_directory('exo_control')

        model_path = os.path.join(
            pkg_dir,
            'neural_network_parameters/models',
            'dynamics_lstm.keras'
        )

        self.scaler_dyn = joblib.load(
            os.path.join(pkg_dir, 'neural_network_parameters/scaler', 'dyn_scaler.save')
        )
        self.scaler_ang = joblib.load(
            os.path.join(pkg_dir, 'neural_network_parameters/scaler', 'ang_scaler.save')
        )

        # model output uses the same scaler as angles
        self.scaler_out = self.scaler_ang

        excel_path = os.path.join(pkg_dir, 'neural_network_parameters/excel', 'data_healthy_dynamics.xlsx')

        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={"weighted_mse": weighted_mse}
        )
        self.get_logger().info("Loaded LSTM model + scalers.")

        # Load one 51×54 trajectory from Excel as seed
        df = pd.read_excel(
            excel_path,
            sheet_name="Data",
            usecols=self.ALL_COLS,
            skiprows=[1, 2]
        )
        arr = df.values.astype(np.float32)

        if arr.shape != (51, 54):
            raise ValueError(f"Excel must be shape (51, 54). Got {arr.shape}")

        # Split into dynamics and angles
        momforce = arr[:, :self.NUM_DYN]       # (51,36)
        angles   = arr[:, self.NUM_DYN:]       # (51,18)

        # Scale like in training
        self.momforce_scaled = self.scaler_dyn.transform(momforce)  # (51,36)
        angles_scaled        = self.scaler_ang.transform(angles)    # (51,18)

        window_scaled = np.concatenate([self.momforce_scaled, angles_scaled], axis=1)
        self.input_window = window_scaled.reshape(1, 51, 54)        # first model input

        # ----------------------------------------------------
        #   5) ROS params & state for streaming
        # ----------------------------------------------------
        self.joint_names = [
            'left_hip_revolute_joint',
            'right_hip_revolute_joint',
            'left_knee_revolute_joint',
            'right_knee_revolute_joint',
            'left_ankle_revolute_joint',
            'right_ankle_revolute_joint'
        ]

        self.pub_timer = 0.3       # seconds between publishes
        self.segment_len = 3       # points per JointTrajectory segment

        # streaming state
        self.predicted_deg_window = None   # (51, 6) in degrees
        self.last_pred_scaled = None       # (51,18) scaled (angles only) used for overlap
        self.traj_index = 0

        # joint_state monitoring
        self._last_js_time = None
        self._last_js_pos = None

        # Periodic streaming timer
        self.timer = self.create_timer(self.pub_timer, self.timer_callback)

    # --------------------------------------------------------
    #  Helpers: sign flip, prediction utilities
    # --------------------------------------------------------
    def _flip_knee_signs_inplace(self, arr):
        """
        Flip knee signs (indices 2 and 3) if they are positive.
        Handles both 1D and 2D arrays in-place.
        """
        knees_idx = [2, 3]
        arr = np.asarray(arr)

        if arr.ndim == 1:
            for ki in knees_idx:
                if arr[ki] >= 0.0:
                    arr[ki] = -arr[ki]
        elif arr.ndim == 2:
            for ki in knees_idx:
                mask = arr[:, ki] >= 0.0
                arr[mask, ki] = -arr[mask, ki]

    def _predict_scaled_angles(self):
        """
        Run the LSTM on current input_window (1,51,54)
        Returns:
            (51,18) scaled angles (same scaler as scalers for ANGLE_COLS)
        """
        scaled_pred = self.model.predict(self.input_window, verbose=0)[0]
        # Expect shape (51,18)
        return scaled_pred

    def _build_angle_window_with_overlap(self, new_pred_scaled, overlap=5):
        """
        Combines tail of last_pred_scaled and head of new_pred_scaled
        to build a smooth angle window in scaled space.
        """
        if self.last_pred_scaled is None:
            # First time: no overlap available
            return new_pred_scaled

        # Tail from previous window
        prev_tail = self.last_pred_scaled[-overlap:]            # (overlap,18)
        # Head from new window (drop first 'overlap' to keep total length 51)
        new_head = new_pred_scaled[:-overlap]                   # (51-overlap,18)

        angle_window_scaled = np.concatenate([prev_tail, new_head], axis=0)  # (51,18)
        return angle_window_scaled

    # --------------------------------------------------------
    #  Prepare first window if needed
    # --------------------------------------------------------
    def _prepare_first_window_if_needed(self):
        if self.predicted_deg_window is not None:
            return

        # Predict scaled angles
        new_pred_scaled = self._predict_scaled_angles()        # (51,18)

        # Build overlapped angle window (scaled)
        angle_window_scaled = self._build_angle_window_with_overlap(
            new_pred_scaled,
            overlap=5
        )

        # Update state
        self.last_pred_scaled = angle_window_scaled

        # Rebuild full 54-dim input window for next prediction
        window_scaled = np.concatenate(
            [self.momforce_scaled, angle_window_scaled],
            axis=1
        )  # (51,54)
        self.input_window = window_scaled.reshape(1, 51, 54)

        # Convert to degrees for publishing
        angles_deg = self.scaler_out.inverse_transform(angle_window_scaled)  # (51,18)
        # Extract only the 6 sagittal DOFs
        self.predicted_deg_window = self.extract_6_sagittal(angles_deg)     # (51,6)
        self.traj_index = 0

        self.get_logger().info("Prepared first prediction window.")

    # --------------------------------------------------------
    #  Roll to next window (continuous streaming)
    # --------------------------------------------------------
    def _roll_to_next_window(self):
        """
        After finishing current window, predict next window and create
        a smooth temporal and angular transition using overlap.
        """
        new_pred_scaled = self._predict_scaled_angles()  # (51,18)

        # Optional: soft ramp at the very beginning based on last frame
        # of previous window to avoid sudden jumps in the very first step.
        end_of_prev = self.last_pred_scaled[-1].copy()
        for i in range(10):
            alpha = (i + 1) / 11.0
            new_pred_scaled[i] = (1 - alpha) * end_of_prev + alpha * new_pred_scaled[i]

        # Overlap angles across windows
        angle_window_scaled = self._build_angle_window_with_overlap(
            new_pred_scaled,
            overlap=5
        )

        # Update state
        self.last_pred_scaled = angle_window_scaled

        window_scaled = np.concatenate(
            [self.momforce_scaled, angle_window_scaled],
            axis=1
        )  # (51,54)
        self.input_window = window_scaled.reshape(1, 51, 54)

        angles_deg = self.scaler_out.inverse_transform(angle_window_scaled)
        self.predicted_deg_window = self.extract_6_sagittal(angles_deg)
        self.traj_index = 0

        self.get_logger().info("Rolled to next prediction window.")

    # --------------------------------------------------------
    #  Extract 6 sagittal DOFs from 18-angle predictions
    # --------------------------------------------------------
    def extract_6_sagittal(self, angles18_deg):
        """
        angles18_deg: (51,18) [degrees]
        Returns: (51,6) [degrees] in order:
          [LHip1, RHip1, LKnee1, RKnee1, LAnkle1, RAnkle1]
        """
        idx = [0, 1, 6, 7, 12, 13]
        return angles18_deg[:, idx]

    # --------------------------------------------------------
    #  Publish a trajectory segment in degrees -> radians
    # --------------------------------------------------------
    def publish_trajectory(self, positions_seq_deg, dt):
        if len(positions_seq_deg) == 0:
            return

        seg = np.array(positions_seq_deg, dtype=np.float32).copy()
        # Convert knee sign convention
        self._flip_knee_signs_inplace(seg)

        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        for i in range(seg.shape[0]):
            pt = JointTrajectoryPoint()
            pt.positions = np.radians(seg[i]).tolist()
            pt.time_from_start = Duration(
                sec=0,
                nanosec=int((i + 1) * dt * 1e9)
            )
            msg.points.append(pt)

        # Optional debug
        # self.get_logger().debug(f"Publishing {seg.shape[0]} pts, first(rad) = {np.round(np.radians(seg[0]), 3)}")

        self.trajectory_publisher_.publish(msg)

    # --------------------------------------------------------
    #  Streaming timer callback
    # --------------------------------------------------------
    def timer_callback(self):
        # Ensure first prediction window exists
        self._prepare_first_window_if_needed()

        # If we have remaining points in current window, send next segment
        if self.traj_index < len(self.predicted_deg_window):
            end = min(self.traj_index + self.segment_len, len(self.predicted_deg_window))
            segment = self.predicted_deg_window[self.traj_index:end]
            self.publish_trajectory(segment, dt=self.pub_timer)
            self.traj_index = end
        else:
            # Current window done → roll to next and immediately send first segment
            self._roll_to_next_window()
            end = min(self.segment_len, len(self.predicted_deg_window))
            segment = self.predicted_deg_window[:end]
            self.publish_trajectory(segment, dt=self.pub_timer)
            self.traj_index = end

    # --------------------------------------------------------
    #  joint_states callback (for monitoring / future feedback)
    # --------------------------------------------------------
    def joint_state_callback(self, msg: JointState):
        expected = set(self.joint_names)
        if not expected.issubset(set(msg.name)):
            return

        current = np.array(
            [msg.position[msg.name.index(j)] for j in self.joint_names],
            dtype=np.float32
        )
        self._last_js_pos = current
        self._last_js_time = self.get_clock().now().nanoseconds * 1e-9
        # You can log occasionally if needed:
        # self.get_logger().debug(f"JointState: {np.round(current, 2)}")

    # --------------------------------------------------------
    #  Send all joints to zero safely
    # --------------------------------------------------------
    def send_joints_to_zero(self, duration=2.0):
        zeros = np.zeros((1, len(self.joint_names)), dtype=np.float32)
        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        pt = JointTrajectoryPoint()
        pt.positions = np.radians(zeros[0]).tolist()
        pt.time_from_start = Duration(sec=int(duration))
        msg.points.append(pt)

        self.trajectory_publisher_.publish(msg)
        self.get_logger().info("Sent all joints to zero.")


def main(args=None):
    rclpy.init(args=args)
    node = JointPublisherFromModel()

    def shutdown_handler(signum, frame):
        node.get_logger().info("CTRL+C: Shutting down...")
        # Send joints to zero
        node.send_joints_to_zero(duration=2.0)
        time.sleep(2.0)  # wait until motion completes (rough)

        # Try to cleanly stop & unload controllers (adjust names to your setup!)
        controllers = ['trajectory_controller', 'joint_state_broadcaster']
        for ctrl in controllers:
            try:
                subprocess.run(
                    ['ros2', 'control', 'set_controller_state', ctrl, 'inactive'],
                    check=False,
                    timeout=2
                )
                subprocess.run(
                    ['ros2', 'control', 'unload_controller', ctrl],
                    check=False,
                    timeout=2
                )
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
