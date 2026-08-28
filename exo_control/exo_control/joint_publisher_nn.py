#!/usr/bin/env python3
import os
import time
import signal
import subprocess

import numpy as np
import pandas as pd
import rclpy

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from ament_index_python.packages import get_package_share_directory

import cv2  # must be imported before tensorflow to avoid protobuf version conflict
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM as _LSTM
import joblib


class _CompatLSTM(_LSTM):
    """Drops legacy `time_major` kwarg that Keras 3 no longer accepts."""
    def __init__(self, *args, time_major=False, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        config.pop('time_major', None)
        return super().from_config(config)


class JointPublisherNNTimestamp(Node):
    def __init__(self):
        super().__init__('joint_publisher_nn')

        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)

        js_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.trajectory_publisher_ = self.create_publisher(
            JointTrajectory, '/trajectory_controller/joint_trajectory', 10
        )
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self._on_joint_state, js_qos
        )

        # ======== CONFIG ========
        self.W = 51
        self.pub_timer = 0.03  # seconds per tick — matches Timestamp model training rate (20 Hz)

        # How many ticks we feed measured angles into the window before free-run
        self.measured_angle_warmup_ticks = 50

        # Controller joint order (used when publishing)
        self.joint_names_ctrl = [
            'left_hip_revolute_joint',
            'right_hip_revolute_joint',
            'left_knee_revolute_joint',
            'right_knee_revolute_joint',
            'left_ankle_revolute_joint',
            'right_ankle_revolute_joint',
        ]

        # Training order: column order the model and scaler expect.
        # The Timestamp model was trained with the same order as the controller.
        self.angle_names_train = [
            'left_hip_revolute_joint',
            'right_hip_revolute_joint',
            'left_knee_revolute_joint',
            'right_knee_revolute_joint',
            'left_ankle_revolute_joint',
            'right_ankle_revolute_joint',
        ]

        # Per-tick delta clamp (degrees, controller order)
        self.max_delta_deg = 3.0
        # ========================

        pkg_dir = get_package_share_directory('exo_control')
        self.model_path  = os.path.join(pkg_dir, 'neural_network_parameters/models',  'Timestamp_lstm_model.h5')
        self.scaler_path = os.path.join(pkg_dir, 'neural_network_parameters/scaler',  'standard_scaler_typical_lstm.save')
        self.excel_path  = os.path.join(pkg_dir, 'neural_network_parameters/excel',   'timestamps_cp_lstm.xlsx')

        for p in [self.model_path, self.scaler_path, self.excel_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f'Missing file: {p}')

        self.model  = load_model(self.model_path, custom_objects={'LSTM': _CompatLSTM}, compile=False)
        self.scaler = joblib.load(self.scaler_path)
        self.get_logger().info('Loaded Timestamp LSTM model and scaler.')

        df  = pd.read_excel(self.excel_path)
        raw = df.values.astype(np.float32)
        if raw.shape[1] != 6:
            raise RuntimeError(f'Excel must have 6 columns [6 joint angles deg], got {raw.shape[1]}')
        if raw.shape[0] < self.W:
            raise RuntimeError(f'Excel must have at least {self.W} rows, got {raw.shape[0]}')

        # Rolling window in degrees, shape (W, 6) — seeded from Excel
        self.window_unscaled = raw[:self.W].copy()

        # Latest measured angles in training order (initialised from Excel seed)
        self.last_angles_train_deg          = self.window_unscaled[-1].copy()
        self.last_predicted_angles_train_deg = self.last_angles_train_deg.copy()

        # Prediction buffer: (W, 6) degrees in training order.
        # _traj_buf_idx >= W triggers a new prediction.
        self._traj_buf     = None
        self._traj_buf_idx = self.W  # force prediction on first tick

        self.tick_count = 0
        self.last_published_angles_ctrl_deg = None

        self.timer = self.create_timer(self.pub_timer, self._timer_cb)
        self.get_logger().info(
            f'READY. dt={self.pub_timer}s, W={self.W}, warmup_ticks={self.measured_angle_warmup_ticks}'
        )

    # ----------------------------
    # Scaling helpers
    # ----------------------------
    def _scale_window(self, win_unscaled: np.ndarray) -> np.ndarray:
        """(W, 6) degrees → (1, W, 6) scaled."""
        scaled = self.scaler.transform(win_unscaled.reshape(-1, 6))
        return scaled.reshape(1, self.W, 6)

    def _predict_window_deg(self) -> np.ndarray:
        """Run model on current window → (W, 6) predicted degrees (training order)."""
        X       = self._scale_window(self.window_unscaled)
        y_scaled = self.model.predict(X, verbose=0)[0]   # (W, 6)
        return self.scaler.inverse_transform(y_scaled).astype(np.float32)

    # ----------------------------
    # Order mapping: training → controller
    # For the Timestamp model training order == controller order (identity).
    # ----------------------------
    @staticmethod
    def _train_order_to_ctrl_order(a_train6: np.ndarray) -> np.ndarray:
        return a_train6.copy()

    # ----------------------------
    # JointState callback
    # ----------------------------
    def _on_joint_state(self, msg: JointState):
        name_to_i = {n: i for i, n in enumerate(msg.name)}
        try:
            self.last_angles_train_deg = np.array(
                [np.degrees(float(msg.position[name_to_i[j]])) for j in self.angle_names_train],
                dtype=np.float32,
            )
        except KeyError:
            return

    # ----------------------------
    # Timer: rolling window + next-tick publish
    # ----------------------------
    def _timer_cb(self):
        # 1) Choose angles to append to the rolling window this tick
        if self.tick_count < self.measured_angle_warmup_ticks:
            angles_for_window = self.last_angles_train_deg
        else:
            angles_for_window = self.last_predicted_angles_train_deg

        # 2) Shift window left by one and append
        self.window_unscaled[:-1] = self.window_unscaled[1:]
        self.window_unscaled[-1]  = angles_for_window.astype(np.float32)

        # 3) Refill prediction buffer when exhausted
        if self._traj_buf_idx >= self.W:
            self._traj_buf     = self._predict_window_deg()
            self._traj_buf_idx = 0

        # 4) Take the next predicted point from the buffer
        next_angles_train_deg = self._traj_buf[self._traj_buf_idx].copy()
        self.last_predicted_angles_train_deg = next_angles_train_deg.copy()
        self._traj_buf_idx += 1

        # 5) Map to controller order + knee inversion
        next_angles_ctrl_deg    = self._train_order_to_ctrl_order(next_angles_train_deg)
        next_angles_ctrl_deg[2] = -abs(next_angles_ctrl_deg[2])  # left knee
        next_angles_ctrl_deg[3] = -abs(next_angles_ctrl_deg[3])  # right knee

        # 6) Per-tick delta clamp for safety
        if self.last_published_angles_ctrl_deg is not None:
            delta = next_angles_ctrl_deg - self.last_published_angles_ctrl_deg
            next_angles_ctrl_deg = self.last_published_angles_ctrl_deg + np.clip(
                delta, -self.max_delta_deg, self.max_delta_deg
            )
        self.last_published_angles_ctrl_deg = next_angles_ctrl_deg.copy()

        # 7) Publish single next-tick point
        msg              = JointTrajectory()
        msg.joint_names  = self.joint_names_ctrl
        pt               = JointTrajectoryPoint()
        pt.positions     = np.radians(next_angles_ctrl_deg).tolist()
        pt.time_from_start = Duration(sec=0, nanosec=int(self.pub_timer * 1e9))
        msg.points.append(pt)
        self.trajectory_publisher_.publish(msg)

        self.tick_count += 1

    # ----------------------------
    # Safety: return to zero on shutdown
    # ----------------------------
    def send_joints_to_zero(self, duration=2.0):
        msg             = JointTrajectory()
        msg.joint_names = self.joint_names_ctrl
        pt              = JointTrajectoryPoint()
        pt.positions    = [0.0] * len(self.joint_names_ctrl)
        pt.time_from_start = Duration(sec=int(duration))
        msg.points.append(pt)
        self.trajectory_publisher_.publish(msg)
        self.get_logger().info('Sent all joints to zero.')


def main(args=None):
    rclpy.init(args=args)
    node = JointPublisherNNTimestamp()

    def shutdown_handler(signum, frame):
        node.get_logger().info('CTRL+C: Shutting Down...')
        try:
            node.send_joints_to_zero(duration=2.0)
            time.sleep(2.0)
        except Exception:
            pass
        controllers = ['trajectory_controller', 'joint_state_broadcaster']
        for ctrl in controllers:
            try:
                subprocess.run(['ros2', 'control', 'set_controller_state', ctrl, 'inactive'], check=False, timeout=2)
                subprocess.run(['ros2', 'control', 'unload_controller', ctrl], check=False, timeout=2)
            except subprocess.TimeoutExpired:
                node.get_logger().warning(f'Timeout unloading {ctrl}')
        node.get_logger().info('Controllers unloaded.')
        node.destroy_node()
        rclpy.shutdown()
        os._exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
