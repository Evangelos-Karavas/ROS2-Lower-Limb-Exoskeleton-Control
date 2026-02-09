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
from std_msgs.msg import Float32MultiArray, Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from ament_index_python.packages import get_package_share_directory

import joblib
from tensorflow.keras.models import load_model


# ============================================================
#  Rezazadeh-style PV FSM (thigh angle + foot contact)
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

        # transitions
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

        # compute s
        if self.state in (PVState.S1_STANCE, PVState.S2_PUSH_OFF):
            s = self._s_desc(qh)
        else:
            s = self._s_asc(qh)
            if self.state == PVState.S3_PRESWING:
                s = max(s, self.s_prev)

        self.s_prev = float(np.clip(s, 0.0, 1.0 - 1e-6))
        return self.s_prev


# ============================================================
#  MAIN NODE
# ============================================================
class PVExcelRecursivePublisher(Node):
    def __init__(self):
        super().__init__("joint_publisher_pv_excel_recursive")

        # ---------------- parameters ----------------
        self.declare_parameter("window", 51)
        self.declare_parameter("pub_timer", 0.3)   # 0.3s = 3 Hz publishing
        self.declare_parameter("segment_len", 3)

        self.declare_parameter("invert_knee_if_positive", False)

        self.declare_parameter("pv_joint_left", "left_hip_revolute_joint")
        self.declare_parameter("pv_joint_right", "right_hip_revolute_joint")
        self.declare_parameter("pv_flip_hip_sign", True)

        self.declare_parameter("fc_topic_left", "/left_sole/in_contact")
        self.declare_parameter("fc_topic_right", "/right_sole/in_contact")

        # PV params (deg)
        self.declare_parameter("pv_q0_deg", 34.0)
        self.declare_parameter("pv_qmin_deg", 2.9)
        self.declare_parameter("pv_c", 0.53)
        self.declare_parameter("pv_qpo_deg", 10.0)
        self.declare_parameter("pv_vel_eps_deg_s", 5.0)

        # PV rollout progression for preview points
        self.declare_parameter("assumed_stride_period", 1.2)  # seconds per stride

        # Files
        self.declare_parameter("model_file", "PV_rolling_next_tick_lstm.keras")
        self.declare_parameter("scaler_pv_file", "scaler_pv_lstm.save")
        self.declare_parameter("scaler_ang_file", "scaler_angles_lstm.save")
        self.declare_parameter("excel_file", "rolling_gt_next_tick_with_pv_cnn.xlsx")

        self.declare_parameter("angle_joint_order", [
            "left_hip_revolute_joint",
            "left_knee_revolute_joint",
            "left_ankle_revolute_joint",
            "right_hip_revolute_joint",
            "right_knee_revolute_joint",
            "right_ankle_revolute_joint",
        ])

        self.declare_parameter("traj_joints", [
            "left_hip_revolute_joint",
            "right_hip_revolute_joint",
            "left_knee_revolute_joint",
            "right_knee_revolute_joint",
            "left_ankle_revolute_joint",
            "right_ankle_revolute_joint",
        ])

        # ---------------- read params ----------------
        self.W = int(self.get_parameter("window").value)
        self.pub_timer = float(self.get_parameter("pub_timer").value)
        self.segment_len = int(self.get_parameter("segment_len").value)

        self.invert_knee_if_positive = bool(self.get_parameter("invert_knee_if_positive").value)

        self.pv_joint_left = str(self.get_parameter("pv_joint_left").value)
        self.pv_joint_right = str(self.get_parameter("pv_joint_right").value)
        self.pv_flip_hip_sign = bool(self.get_parameter("pv_flip_hip_sign").value)

        self.fc_topic_left = str(self.get_parameter("fc_topic_left").value)
        self.fc_topic_right = str(self.get_parameter("fc_topic_right").value)

        q0 = float(self.get_parameter("pv_q0_deg").value)
        qmin = float(self.get_parameter("pv_qmin_deg").value)
        c = float(self.get_parameter("pv_c").value)
        qpo = float(self.get_parameter("pv_qpo_deg").value)
        vel_eps = float(self.get_parameter("pv_vel_eps_deg_s").value)

        self.assumed_stride_period = float(self.get_parameter("assumed_stride_period").value)

        model_file = str(self.get_parameter("model_file").value)
        scaler_pv_file = str(self.get_parameter("scaler_pv_file").value)
        scaler_ang_file = str(self.get_parameter("scaler_ang_file").value)
        excel_file = str(self.get_parameter("excel_file").value)

        self.angle_joint_order = list(self.get_parameter("angle_joint_order").value)
        self.traj_joints = list(self.get_parameter("traj_joints").value)

        # ---------------- load model/scalers/excel ----------------
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
        self.get_logger().info("Loaded PV model + scalers.")

        df = pd.read_excel(self.excel_path)
        raw = df.values.astype(np.float32)
        if raw.shape[1] != 8:
            raise RuntimeError(f"Excel must have 8 columns [pvL,pvR,6 angles_deg], got {raw.shape}")
        if raw.shape[0] < self.W:
            raise RuntimeError(f"Excel must have at least {self.W} rows, got {raw.shape[0]}")

        self.rolling_unscaled = raw[:self.W].copy()
        self.last_pv = (float(self.rolling_unscaled[-1, 0]), float(self.rolling_unscaled[-1, 1]))

        # ---------------- PV engines + FC ----------------
        self.pv_fsm_L = PhaseVariableFSM(q0_deg=q0, qmin_deg=qmin, c=c, qpo_deg=qpo, vel_eps_deg_s=vel_eps)
        self.pv_fsm_R = PhaseVariableFSM(q0_deg=q0, qmin_deg=qmin, c=c, qpo_deg=qpo, vel_eps_deg_s=vel_eps)

        self.fcL = False
        self.fcR = False

        # ---------------- ROS IO ----------------
        js_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.sub_js = self.create_subscription(JointState, "/joint_states", self._on_joint_state, js_qos)
        self.sub_fcL = self.create_subscription(Bool, self.fc_topic_left, self._on_fcL, 10)
        self.sub_fcR = self.create_subscription(Bool, self.fc_topic_right, self._on_fcR, 10)

        self.pub_traj = self.create_publisher(JointTrajectory, "/trajectory_controller/joint_trajectory", 10)
        self.pub_pv = self.create_publisher(Float32MultiArray, "/phase_variable", 10)

        # Publish every pub_timer (0.05 sec)
        self.timer = self.create_timer(self.pub_timer, self._timer_cb)

        self.get_logger().info(
            f"READY. dt={self.pub_timer}s, segment_len={self.segment_len}, assumed_stride_period={self.assumed_stride_period}s"
        )

    # ---------------- scaling helpers ----------------
    def _scale_features(self, unscaled_Wx8: np.ndarray) -> np.ndarray:
        pv_un = unscaled_Wx8[:, :2].astype(np.float32)
        ang_un = unscaled_Wx8[:, 2:].astype(np.float32)
        pv_sc = self.scaler_pv.transform(pv_un)
        ang_sc = self.scaler_ang.transform(ang_un)
        X = np.concatenate([pv_sc, ang_sc], axis=1).astype(np.float32)
        return X.reshape(1, self.W, 8)

    def _unscale_angles(self, y_scaled_6: np.ndarray) -> np.ndarray:
        y_scaled_6 = y_scaled_6.reshape(1, -1).astype(np.float32)
        return self.scaler_ang.inverse_transform(y_scaled_6)[0].astype(np.float32)

    # ---------------- conventions ----------------
    def _flip_knee_signs_inplace_deg(self, a: np.ndarray):
        if not self.invert_knee_if_positive:
            return
        for ki in [1, 4]:
            if a[ki] >= 0.0:
                a[ki] = -a[ki]

    # ---------------- model next step ----------------
    def _predict_next_step_scaled(self, X_1xWx8: np.ndarray) -> np.ndarray:
        y = self.model.predict(X_1xWx8, verbose=0)
        y = np.array(y)
        if y.ndim == 2 and y.shape == (1, 6):
            return y[0].astype(np.float32)
        if y.ndim == 1 and y.shape == (6,):
            return y.astype(np.float32)
        raise RuntimeError(f"Unexpected PV model output shape: {y.shape}")

    # ---------------- publishing ----------------
    def _publish_segment_deg(self, seg_deg: np.ndarray):
        msg = JointTrajectory()
        msg.joint_names = self.traj_joints

        for i in range(seg_deg.shape[0]):
            a = seg_deg[i].copy()
            self._flip_knee_signs_inplace_deg(a)

            m = {j: float(np.radians(a[k])) for k, j in enumerate(self.angle_joint_order)}

            pt = JointTrajectoryPoint()
            pt.positions = [m[j] for j in self.traj_joints]
            pt.time_from_start = Duration(sec=0, nanosec=int((i + 1) * self.pub_timer * 1e9))
            msg.points.append(pt)

        self.pub_traj.publish(msg)

    # ---------------- FC callbacks ----------------
    def _on_fcL(self, msg: Bool):
        self.fcL = bool(msg.data)

    def _on_fcR(self, msg: Bool):
        self.fcR = bool(msg.data)

    # ---------------- JointState callback: LIVE PV ----------------
    def _on_joint_state(self, msg: JointState):
        name_to_i = {n: i for i, n in enumerate(msg.name)}
        if self.pv_joint_left not in name_to_i or self.pv_joint_right not in name_to_i:
            return

        t_now = self.get_clock().now().nanoseconds * 1e-9

        qL_rad = float(msg.position[name_to_i[self.pv_joint_left]])
        qR_rad = float(msg.position[name_to_i[self.pv_joint_right]])

        qL_deg = float(np.degrees(qL_rad))
        qR_deg = float(np.degrees(qR_rad))

        if self.pv_flip_hip_sign:
            qL_deg = -qL_deg
            qR_deg = -qR_deg

        pvL = self.pv_fsm_L.update(qL_deg, self.fcL, t_now)
        pvR = self.pv_fsm_R.update(qR_deg, self.fcR, t_now)

        self.last_pv = (pvL, pvR)

    # ---------------- Timer: publish PV + rollout ----------------
    def _timer_cb(self):
        # publish current PV every tick
        pvL_now, pvR_now = self.last_pv
        pv_msg = Float32MultiArray()
        pv_msg.data = [float(pvL_now), float(pvR_now)]
        self.pub_pv.publish(pv_msg)

        # rollout segment with PV progressing forward
        seg_deg = np.zeros((self.segment_len, 6), dtype=np.float32)

        # PV increment per step (preview progression)
        dp = self.pub_timer / max(1e-3, self.assumed_stride_period)

        pvL = pvL_now
        pvR = pvR_now

        for k in range(self.segment_len):
            # Write PV into the "current" last row so X uses up-to-date PV
            self.rolling_unscaled[-1, 0] = float(pvL)
            self.rolling_unscaled[-1, 1] = float(pvR)

            Xk = self._scale_features(self.rolling_unscaled)
            y_scaled = self._predict_next_step_scaled(Xk)
            y_deg = self._unscale_angles(y_scaled)

            self._flip_knee_signs_inplace_deg(y_deg)
            seg_deg[k] = y_deg

            # advance PV for next predicted point (monotonic, wrap at 1)
            pvL = (pvL + dp) % 1.0
            pvR = (pvR + dp) % 1.0

            # shift rolling window and append new row
            new_row = np.concatenate([[pvL, pvR], y_deg]).astype(np.float32)
            self.rolling_unscaled[:-1] = self.rolling_unscaled[1:]
            self.rolling_unscaled[-1] = new_row

        self._publish_segment_deg(seg_deg)

    # ---------------- safety: send joints to zero ----------------
    def send_joints_to_zero(self, duration=2.0):
        msg = JointTrajectory()
        msg.joint_names = self.traj_joints
        pt = JointTrajectoryPoint()
        pt.positions = [0.0] * len(self.traj_joints)
        pt.time_from_start = Duration(sec=int(duration))
        msg.points.append(pt)
        self.pub_traj.publish(msg)
        self.get_logger().info("Sent joints to zero.")


def main(args=None):
    rclpy.init(args=args)
    node = PVExcelRecursivePublisher()

    def shutdown_handler(signum, frame):
        node.get_logger().info("CTRL+C: Shutting Down...")
        try:
            node.send_joints_to_zero(duration=2.0)
            time.sleep(2.0)
        except Exception:
            pass

        controllers = ["trajectory_controller", "joint_state_broadcaster"]
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
