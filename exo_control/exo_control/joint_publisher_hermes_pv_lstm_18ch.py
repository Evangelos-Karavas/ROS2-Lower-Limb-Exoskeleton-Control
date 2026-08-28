#!/usr/bin/env python3
import os
import time
import signal
import subprocess
from enum import Enum

import numpy as np
import pandas as pd
import rclpy

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, Float64MultiArray, Bool
from ament_index_python.packages import get_package_share_directory

import cv2  # must be imported before tensorflow to avoid protobuf version conflict
import joblib
from tensorflow.keras.models import load_model


# ============================================================
#  Rezazadeh-style PV FSM (qh + foot contact)
# ============================================================
class PVState(Enum):
    S1_STANCE = 1
    S2_PUSH_OFF = 2
    S3_PRESWING = 3
    S4_SWING = 4


class PhaseVariableFSM:
    def __init__(self, q0_deg=20.0, qmin_deg=-11.0, c=0.53, qpo_deg=-8.4, vel_eps_deg_s=1.0):
        self.q0 = float(q0_deg)
        self.qmin = float(qmin_deg)
        self.c = float(c)
        self.qpo = float(qpo_deg)
        self.vel_eps = float(vel_eps_deg_s)

        self.state = PVState.S1_STANCE
        self._qh_prev = None
        self._t_prev = None
        self.sm = None
        self.qhm = None
        self.s_prev = 0.0

    def _qh_dot(self, qh, t):
        if self._qh_prev is None:
            self._qh_prev = qh
            self._t_prev = t
            return 0.0
        dt = max(1e-6, t - self._t_prev)
        qdot = (qh - self._qh_prev) / dt
        self._qh_prev = qh
        self._t_prev = t
        return float(qdot)

    def _s_desc(self, qh):
        denom = (self.q0 - self.qmin)
        if abs(denom) < 1e-6:
            return 0.0
        s = ((self.q0 - qh) / denom) * self.c
        return float(np.clip(s, 0.0, 1.0))

    def _s_asc(self, qh):
        if self.sm is None or self.qhm is None:
            return float(np.clip(self.s_prev, 0.0, 1.0))
        denom = (self.q0 - self.qhm)
        if abs(denom) < 1e-6:
            return float(np.clip(self.s_prev, 0.0, 1.0))
        s = 1.0 + ((1.0 - self.sm) / denom) * (qh - self.q0)
        return float(np.clip(s, 0.0, 1.0))

    def reset_stride(self):
        self.state = PVState.S1_STANCE
        self.sm = None
        self.qhm = None
        self.s_prev = 0.0

    def update(self, qh_deg, fc, t_sec):
        qh = float(qh_deg)
        fc = bool(fc)
        t = float(t_sec)

        qdot = self._qh_dot(qh, t)

        if self.state != PVState.S4_SWING and (not fc):
            self.state = PVState.S4_SWING
        elif self.state == PVState.S4_SWING and fc:
            self.reset_stride()
        elif self.state == PVState.S1_STANCE and fc and (qh <= self.qpo):
            self.state = PVState.S2_PUSH_OFF
        elif self.state == PVState.S2_PUSH_OFF and fc and (qdot > self.vel_eps):
            self.sm = self._s_desc(qh)
            self.qhm = qh
            self.state = PVState.S3_PRESWING

        if self.state in (PVState.S1_STANCE, PVState.S2_PUSH_OFF):
            s = self._s_desc(qh)
        else:
            s = self._s_asc(qh)
            if self.state == PVState.S3_PRESWING:
                s = max(s, self.s_prev)

        self.s_prev = float(np.clip(s, 0.0, 1.0 - 1e-6))
        return self.s_prev


# ============================================================
#  Hermes exoskeleton — PV next-tick LSTM publisher, ALL PLANES (18ch)
#
#  Trained by phase_variable_lstm_all_planes.py. The model predicts all
#  18 angle channels (sagittal/frontal/transverse x 6 joints), but the
#  physical hermes rig only actuates and measures the sagittal plane
#  (1 revolute joint per hip/knee/ankle). So:
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
#  Training order (18 angles fed to/from the LSTM):
#    [LHip(1,2,3), LKnee(1,2,3), LAnkle(1,2,3),
#     RHip(1,2,3), RKnee(1,2,3), RAnkle(1,2,3)]
#  Sagittal (plane 1) sits at offsets [0, 3, 6, 9, 12, 15].
#
#  PV input window (W×20):
#    [pvL, pvR, <18 angles in training order>]
# ============================================================
class JointPublisherHermesPVLSTM18ch(Node):
    def __init__(self):
        super().__init__("joint_publisher_hermes_pv_lstm_18ch")

        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", True)

        js_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Publishers
        self.cmd_publisher_ = self.create_publisher(
            Float64MultiArray, "/forward_position_controller/commands", 10
        )
        self.pv_publisher_ = self.create_publisher(Float32MultiArray, "/phase_variable", 10)

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self._on_joint_state, js_qos
        )

        self.fc_topic_left = "/left_sole/in_contact"
        self.fc_topic_right = "/right_sole/in_contact"
        self.fcL = False
        self.fcR = False
        self.sub_fcL = self.create_subscription(Bool, self.fc_topic_left, self._on_fcL, 10)
        self.sub_fcR = self.create_subscription(Bool, self.fc_topic_right, self._on_fcR, 10)

        # ======== CONFIG (edit here) ========
        self.W = 51
        self.pub_timer = 0.03  # seconds
        self.N_ANGLES = 18

        self.measured_angle_warmup_ticks = 50

        # Joints used to drive the PV FSM (sagittal hip only — same as training)
        self.pv_joint_left = "joint_hip_L"
        self.pv_joint_right = "joint_hip_R"
        self.pv_flip_hip_sign = False

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

        # PV FSM params (deg) — carried over from the 6ch hermes node;
        # re-tune against the new 18ch CP dataset if hip kinematics differ.
        q0 = 34.0
        qmin = 2.9
        c = 0.53
        qpo = 10.0
        vel_eps = 5.0

        # Files produced by phase_variable_lstm_all_planes.py
        model_file = "PV_lstm_best_rollout_dtw_18ch.keras"
        scaler_pv_file = "scaler_pv_lstm_18ch.save"
        scaler_ang_file = "scaler_angles_lstm_18ch.save"
        # 20 cols: [PhaseVariable_Left, PhaseVariable_Right, <18 angles in training order>]
        excel_file = "PV_cp_lstm_18ch.xlsx"
        # ================================

        pkg_dir = get_package_share_directory("exo_control")
        self.model_path = os.path.join(pkg_dir, "neural_network_parameters/models", model_file)
        self.scaler_pv_path = os.path.join(pkg_dir, "neural_network_parameters/scaler", scaler_pv_file)
        self.scaler_ang_path = os.path.join(pkg_dir, "neural_network_parameters/scaler", scaler_ang_file)
        self.excel_path = os.path.join(pkg_dir, "neural_network_parameters/excel", excel_file)

        for p in [self.model_path, self.scaler_pv_path, self.scaler_ang_path, self.excel_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing file: {p}")

        self.model = load_model(self.model_path)
        self.scaler_pv = joblib.load(self.scaler_pv_path)
        self.scaler_ang = joblib.load(self.scaler_ang_path)
        self.get_logger().info("Loaded PV (18ch) next-tick LSTM model + scalers.")

        df = pd.read_excel(self.excel_path)
        raw = df.values.astype(np.float32)
        if raw.shape[1] != 2 + self.N_ANGLES:
            raise RuntimeError(
                f"Excel must have {2 + self.N_ANGLES} columns [pvL,pvR,{self.N_ANGLES} angles_deg], got {raw.shape}"
            )
        if raw.shape[0] < self.W:
            raise RuntimeError(f"Excel must have at least {self.W} rows, got {raw.shape[0]}")

        self.window_unscaled = raw[: self.W].copy()

        self.pv_fsm_L = PhaseVariableFSM(q0_deg=q0, qmin_deg=qmin, c=c, qpo_deg=qpo, vel_eps_deg_s=vel_eps)
        self.pv_fsm_R = PhaseVariableFSM(q0_deg=q0, qmin_deg=qmin, c=c, qpo_deg=qpo, vel_eps_deg_s=vel_eps)

        self.last_pv = (float(self.window_unscaled[-1, 0]), float(self.window_unscaled[-1, 1]))
        # Full 18-channel state. Sagittal entries get overwritten by live
        # measurement during warmup; frontal/transverse always come from here.
        self.last_predicted_angles_train_deg = self.window_unscaled[-1, 2:].astype(np.float32).copy()
        self.last_measured_sagittal_deg = self.last_predicted_angles_train_deg[self.sagittal_idx].copy()

        self.tick_count = 0

        # Per-tick delta clamp (8 values, controller order, degrees)
        self.max_delta_deg = 3.0
        self.last_published_angles_ctrl_deg = None

        self.strides_to_skip = 3
        self.strides_completed = 0
        self.max_blocked_ticks = 500

        self.timer = self.create_timer(self.pub_timer, self._timer_cb)
        self.get_logger().info(
            f"READY. dt={self.pub_timer}s, W={self.W}, warmup_ticks={self.measured_angle_warmup_ticks}"
        )

    # ----------------------------
    # Foot contact callbacks
    # ----------------------------
    def _on_fcL(self, msg: Bool):
        self.fcL = bool(msg.data)

    def _on_fcR(self, msg: Bool):
        self.fcR = bool(msg.data)

    # ----------------------------
    # Scaling helpers
    # ----------------------------
    def _scale_window(self, win_unscaled_Wx20: np.ndarray) -> np.ndarray:
        pv_un = win_unscaled_Wx20[:, :2].astype(np.float32)
        ang_un = win_unscaled_Wx20[:, 2:].astype(np.float32)
        pv_sc = self.scaler_pv.transform(pv_un)
        ang_sc = self.scaler_ang.transform(ang_un)
        X = np.concatenate([pv_sc, ang_sc], axis=1).astype(np.float32)
        return X.reshape(1, self.W, 2 + self.N_ANGLES)

    def _predict_next_angles_deg_train_order(self) -> np.ndarray:
        X = self._scale_window(self.window_unscaled)
        y_scaled = np.array(self.model.predict(X, verbose=0))

        if y_scaled.ndim == 2 and y_scaled.shape == (1, self.N_ANGLES):
            y_scaled = y_scaled[0]
        elif y_scaled.ndim == 1 and y_scaled.shape == (self.N_ANGLES,):
            pass
        else:
            raise RuntimeError(f"Unexpected model output shape: {y_scaled.shape}")

        y_deg = self.scaler_ang.inverse_transform(y_scaled.reshape(1, -1))[0].astype(np.float32)
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
    # JointState: compute PV + measured sagittal angles
    # ----------------------------
    def _on_joint_state(self, msg: JointState):
        name_to_i = {n: i for i, n in enumerate(msg.name)}

        if self.pv_joint_left not in name_to_i or self.pv_joint_right not in name_to_i:
            return

        t_now = self.get_clock().now().nanoseconds * 1e-9

        qL_deg = float(np.degrees(float(msg.position[name_to_i[self.pv_joint_left]])))
        qR_deg = float(np.degrees(float(msg.position[name_to_i[self.pv_joint_right]])))

        if self.pv_flip_hip_sign:
            qL_deg = -qL_deg
            qR_deg = -qR_deg

        prev_state_L = self.pv_fsm_L.state
        pvL = self.pv_fsm_L.update(qL_deg, self.fcL, t_now)
        pvR = self.pv_fsm_R.update(qR_deg, self.fcR, t_now)
        self.last_pv = (pvL, pvR)

        if prev_state_L == PVState.S4_SWING and self.pv_fsm_L.state == PVState.S1_STANCE:
            self.strides_completed += 1

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
        pvL, pvR = self.last_pv

        pv_msg = Float32MultiArray()
        pv_msg.data = [float(pvL), float(pvR)]
        self.pv_publisher_.publish(pv_msg)

        self.window_unscaled[:-1] = self.window_unscaled[1:]

        # Frontal/transverse always come from the model's own last prediction
        # (no hardware measurement exists for those channels). Sagittal is
        # teacher-forced from /joint_states during warmup only.
        angles_for_last_row = self.last_predicted_angles_train_deg.copy()
        if self.tick_count < self.measured_angle_warmup_ticks:
            angles_for_last_row[self.sagittal_idx] = self.last_measured_sagittal_deg

        self.window_unscaled[-1, 0] = float(pvL)
        self.window_unscaled[-1, 1] = float(pvR)
        self.window_unscaled[-1, 2:] = angles_for_last_row.astype(np.float32)

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

        if self.strides_completed < self.strides_to_skip:
            if self.tick_count < self.max_blocked_ticks:
                self.tick_count += 1
                return

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
    node = JointPublisherHermesPVLSTM18ch()

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
