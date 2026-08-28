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
from std_msgs.msg import Float64MultiArray
from ament_index_python.packages import get_package_share_directory

import cv2  # must be imported before tensorflow to avoid protobuf version conflict
import joblib
from tensorflow.keras.models import load_model


# ============================================================
#  Hermes exoskeleton — Timestamp next-tick CNN publisher, ALL PLANES (18ch)
#
#  Trained by timestamps_CNN_all_planes.py. No phase variable / foot
#  contact FSM here — this is a plain rolling-window next-tick predictor
#  driven purely by the angle history. The CNN was trained to predict the
#  next HORIZON=10 ticks at once; we only ever use the immediate next-tick
#  slice (index 0) to advance the rollout, one tick at a time.
#
#  The model predicts all 18 angle channels (sagittal/frontal/transverse
#  x 6 joints), but the physical hermes rig only actuates and measures
#  the sagittal plane (1 revolute joint per hip/knee/ankle). So:
#    - sagittal channels are teacher-forced from /joint_states during
#      warmup, then free-run from the model's own predictions
#    - frontal/transverse channels have no hardware measurement at all,
#      so they ALWAYS free-run from the model's own predictions (from
#      the very first tick, seeded by the excel window)
#    - only the 6 sagittal predictions are ever sent to the controller
#
#  Controller order (8 joints, forward_position_controller):
#    [joint_back_R=0, joint_hip_R, joint_knee_R, joint_ankle_R,
#     joint_back_L=0, joint_hip_L, joint_knee_L, joint_ankle_L]
#
#  Training order (18 angles fed to/from the CNN):
#    [LHip(1,2,3), LKnee(1,2,3), LAnkle(1,2,3),
#     RHip(1,2,3), RKnee(1,2,3), RAnkle(1,2,3)]
#  Sagittal (plane 1) sits at offsets [0, 3, 6, 9, 12, 15].
#
#  Input window (W×18): <18 angles in training order>
# ============================================================
class JointPublisherHermesTSCNN18ch(Node):
    def __init__(self):
        super().__init__("joint_publisher_hermes_ts_cnn_18ch")

        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", True)

        js_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Publisher
        self.cmd_publisher_ = self.create_publisher(
            Float64MultiArray, "/forward_position_controller/commands", 10
        )

        # Subscriber
        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self._on_joint_state, js_qos
        )

        # ======== CONFIG (edit here) ========
        self.W = 51
        self.pub_timer = 0.03  # seconds
        self.N_ANGLES = 18

        self.measured_angle_warmup_ticks = 50

        # The only 6 channels hermes hardware can actually measure, in the
        # same L-then-R / hip-knee-ankle grouping as the 18ch training order.
        self.angle_names_meas = [
            "joint_hip_L",
            "joint_knee_L",
            "joint_ankle_L",
            "joint_hip_R",
            "joint_knee_R",
            "joint_ankle_R",
        ]
        # Where those 6 sagittal channels land inside the 18-channel training vector.
        self.sagittal_idx = np.array([0, 3, 6, 9, 12, 15], dtype=np.int64)

        # Files produced by timestamps_CNN_all_planes.py
        self.HORIZON = 10
        model_file = "Timestamp_cnn_next_tick_model_18.keras"
        scaler_file = "standard_scaler_typical_cnn_next_tick_18.save"
        # 18 cols: <18 angles in training order>
        excel_file = "timestamps_cp_cnn_18ch.xlsx"
        # ================================

        pkg_dir = get_package_share_directory("exo_control")
        self.model_path = os.path.join(pkg_dir, "neural_network_parameters/models", model_file)
        self.scaler_path = os.path.join(pkg_dir, "neural_network_parameters/scaler", scaler_file)
        self.excel_path = os.path.join(pkg_dir, "neural_network_parameters/excel", excel_file)

        for p in [self.model_path, self.scaler_path, self.excel_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing file: {p}")

        self.model = load_model(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        self.get_logger().info("Loaded Timestamp (18ch) next-tick CNN model + scaler.")

        df = pd.read_excel(self.excel_path)
        raw = df.values.astype(np.float32)
        if raw.shape[1] != self.N_ANGLES:
            raise RuntimeError(f"Excel must have {self.N_ANGLES} columns [angles_deg], got {raw.shape}")
        if raw.shape[0] < self.W:
            raise RuntimeError(f"Excel must have at least {self.W} rows, got {raw.shape[0]}")

        self.window_unscaled = raw[: self.W].copy()

        # Full 18-channel state. Sagittal entries get overwritten by live
        # measurement during warmup; frontal/transverse always come from here.
        self.last_predicted_angles_train_deg = self.window_unscaled[-1].astype(np.float32).copy()
        self.last_measured_sagittal_deg = self.last_predicted_angles_train_deg[self.sagittal_idx].copy()

        self.tick_count = 0

        # Per-tick delta clamp (8 values, controller order, degrees)
        self.max_delta_deg = 3.0
        self.last_published_angles_ctrl_deg = None

        self.timer = self.create_timer(self.pub_timer, self._timer_cb)
        self.get_logger().info(
            f"READY. dt={self.pub_timer}s, W={self.W}, warmup_ticks={self.measured_angle_warmup_ticks}"
        )

    # ----------------------------
    # Scaling helpers
    # ----------------------------
    def _scale_window(self, win_unscaled_Wx18: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(win_unscaled_Wx18.reshape(-1, self.N_ANGLES))
        return scaled.reshape(1, self.W, self.N_ANGLES)

    def _predict_next_angles_deg_train_order(self) -> np.ndarray:
        X = self._scale_window(self.window_unscaled)
        y_scaled = np.array(self.model.predict(X, verbose=0))

        if y_scaled.ndim == 3 and y_scaled.shape == (1, self.HORIZON, self.N_ANGLES):
            # Model predicts HORIZON future ticks at once; only use the
            # immediate next-tick slice to advance the rollout one step.
            y_scaled = y_scaled[0, 0, :]
        else:
            raise RuntimeError(f"Unexpected model output shape: {y_scaled.shape}")

        y_deg = self.scaler.inverse_transform(y_scaled.reshape(1, -1))[0].astype(np.float32)
        return y_deg

    # ----------------------------
    # Order mapping: 6 sagittal (training order) -> controller (8 joints)
    #   training: [Lhip, Lknee, Lankle, Rhip, Rknee, Rankle]
    #   ctrl:     [back_R=0, hip_R, knee_R, ankle_R, back_L=0, hip_L, knee_L, ankle_L]
    # ----------------------------
    @staticmethod
    def _sagittal_train_order_deg_to_ctrl_order_deg(a_train6: np.ndarray) -> np.ndarray:
        Lhip, Lknee, Lankle, Rhip, Rknee, Rankle = a_train6.tolist()
        return np.array(
            [0.0, Rhip, Rknee, Rankle, 0.0, Lhip, Lknee, Lankle],
            dtype=np.float32,
        )

    # ----------------------------
    # JointState: measured sagittal angles
    # ----------------------------
    def _on_joint_state(self, msg: JointState):
        name_to_i = {n: i for i, n in enumerate(msg.name)}
        try:
            a_meas = np.array(
                [np.degrees(float(msg.position[name_to_i[j]])) for j in self.angle_names_meas],
                dtype=np.float32,
            )
        except KeyError:
            return

        self.last_measured_sagittal_deg = a_meas

    # ----------------------------
    # Timer: teacher forcing (sagittal only) -> free-run + publish
    # ----------------------------
    def _timer_cb(self):
        self.window_unscaled[:-1] = self.window_unscaled[1:]

        # Frontal/transverse always come from the model's own last prediction
        # (no hardware measurement exists for those channels). Sagittal is
        # teacher-forced from /joint_states during warmup only.
        angles_for_last_row = self.last_predicted_angles_train_deg.copy()
        if self.tick_count < self.measured_angle_warmup_ticks:
            angles_for_last_row[self.sagittal_idx] = self.last_measured_sagittal_deg

        self.window_unscaled[-1] = angles_for_last_row.astype(np.float32)

        next_angles_train_deg = self._predict_next_angles_deg_train_order()
        self.last_predicted_angles_train_deg = next_angles_train_deg.copy()

        # Only the 6 sagittal predictions are ever sent to hardware.
        sagittal6_deg = next_angles_train_deg[self.sagittal_idx]
        next_angles_ctrl_deg = self._sagittal_train_order_deg_to_ctrl_order_deg(sagittal6_deg)
        next_angles_ctrl_deg[2] = -abs(next_angles_ctrl_deg[2])  # knee_R always negative
        next_angles_ctrl_deg[6] = -abs(next_angles_ctrl_deg[6])  # knee_L always negative

        if self.last_published_angles_ctrl_deg is not None:
            delta = next_angles_ctrl_deg - self.last_published_angles_ctrl_deg
            next_angles_ctrl_deg = self.last_published_angles_ctrl_deg + np.clip(
                delta, -self.max_delta_deg, self.max_delta_deg
            )

        self.last_published_angles_ctrl_deg = next_angles_ctrl_deg.copy()

        cmd_msg = Float64MultiArray()
        cmd_msg.data = np.radians(next_angles_ctrl_deg).tolist()
        self.cmd_publisher_.publish(cmd_msg)

        self.tick_count += 1

    # ----------------------------
    # Safety
    # ----------------------------
    def send_joints_to_zero(self):
        msg = Float64MultiArray()
        msg.data = [0.0] * 8
        self.cmd_publisher_.publish(msg)
        self.get_logger().info("Sent all joints to zero.")


def main(args=None):
    rclpy.init(args=args)
    node = JointPublisherHermesTSCNN18ch()

    def shutdown_handler(signum, frame):
        node.get_logger().info("CTRL+C: Shutting Down...")
        try:
            node.send_joints_to_zero()
            time.sleep(2.0)
        except Exception:
            pass

        controllers = ["forward_position_controller", "joint_state_broadcaster"]
        for ctrl in controllers:
            try:
                subprocess.run(["ros2", "control", "set_controller_state", ctrl, "inactive"], check=False, timeout=2)
                subprocess.run(["ros2", "control", "unload_controller", ctrl], check=False, timeout=2)
            except subprocess.TimeoutExpired:
                node.get_logger().warning(f"Timeout unloading {ctrl}")

        node.destroy_node()
        rclpy.shutdown()
        os._exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
