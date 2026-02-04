#!/usr/bin/env python3
"""
PV Excel-seeded recursive JointTrajectory publisher (PV-conditioned model).

This node:
- Seeds an initial rolling window (W,8) from an Excel file:
    [pvL, pvR, 6 joint angles in degrees]
- Each timer tick, recursively predicts `segment_len` future steps using a
  single-step PV model (next-tick angle prediction).
- Publishes a multi-point JointTrajectory to /trajectory_controller/joint_trajectory
- Publishes PV to /phase_variable

Two PV modes (IMPORTANT: choose ONE):
1) Timer PV (debug / simplest, robust):
   - use_live_pv := False
   - PV is advanced internally with dp = pub_timer / assumed_stride_period,
     so PV goes 0->1 once per stride.

2) Live PV (from joint_states hip peaks):
   - use_live_pv := True
   - PV is computed from hip angle peaks (StrideTimedPV).
   - Timer does NOT manually advance PV (prevents double-driving PV).

Notes:
- Use self.get_clock().now() for timestamps (more reliable in sim than msg stamps).
- Keep pub_timer close to the rate used when generating your CSV / training data
  (e.g., 0.02–0.03s). Do NOT set pub_timer to huge values like 0.75s.
"""

import os
import time
import signal
import numpy as np
import pandas as pd
import rclpy
import subprocess

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from ament_index_python.packages import get_package_share_directory

import joblib
from tensorflow.keras.models import load_model


# ============================================================
#  Stride-timed PV from hip angle peaks
# ============================================================
class WindowedStridePV:
    """
    Online PV that mimics training PV computation:
    - Collects last stride_len hip samples
    - Computes PV curve for that window using compute_pv_stride(q, c)
    - Returns the current PV = last element of that curve

    This avoids peak detection completely.
    """

    def __init__(self, stride_len=51, c=0.63, enforce_monotonic=True):
        self.stride_len = int(stride_len)
        self.c = float(c)
        self.enforce_monotonic = bool(enforce_monotonic)
        self.buf = []

    @staticmethod
    def compute_pv_stride(q: np.ndarray, c: float, enforce_monotonic: bool = True) -> np.ndarray:
        q = q.astype(np.float64)
        N = q.shape[0]
        c = float(np.clip(c, 0.05, 0.95))

        q0 = float(q[0])
        idx_min = int(np.argmin(q))
        qmin = float(q[idx_min])

        denom = (q0 - qmin)
        if abs(denom) < 1e-9:
            s = np.linspace(0.0, 1.0, N, endpoint=False, dtype=np.float32)
            return np.clip(s, 0.0, 1.0 - 1e-6)

        s = np.zeros(N, dtype=np.float64)
        s[:idx_min + 1] = ((q0 - q[:idx_min + 1]) / denom) * c
        s[idx_min:] = 1.0 + ((1.0 - c) / denom) * (q[idx_min:] - q0)

        s = np.clip(s, 0.0, 1.0)
        if enforce_monotonic:
            s = np.maximum.accumulate(s)

        return np.clip(s.astype(np.float32), 0.0, 1.0 - 1e-6)

    def update(self, q_sample: float) -> float:
        self.buf.append(float(q_sample))
        if len(self.buf) > self.stride_len:
            self.buf.pop(0)

        # Until buffer full, just ramp by time index (gentle start-up)
        if len(self.buf) < self.stride_len:
            return float(np.clip(len(self.buf) / self.stride_len, 0.0, 1.0 - 1e-6))

        q = np.asarray(self.buf, dtype=np.float32)
        pv_curve = self.compute_pv_stride(q, c=self.c, enforce_monotonic=self.enforce_monotonic)
        return float(pv_curve[-1])


# ============================================================
#  MAIN NODE
# ============================================================
class PVExcelRecursivePublisher(Node):
    def __init__(self):
        super().__init__("joint_publisher_pv_excel_recursive")

        # ---- sim time ----
        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", True)

        # ---- timing / segment config ----
        self.declare_parameter("window", 51)
        self.declare_parameter("pub_timer", 0.06)
        self.declare_parameter("segment_len", 3)

        # ---- sign convention ----
        self.declare_parameter("invert_knee_if_positive", True)

        # ---- PV mode ----
        self.declare_parameter("use_live_pv", True)
        self.declare_parameter("pv_joint_left", "left_hip_revolute_joint")
        self.declare_parameter("pv_joint_right", "right_hip_revolute_joint")

        # Timer PV progression (used ONLY when use_live_pv=False)
        self.declare_parameter("assumed_stride_period", 1.2)  # seconds per stride

        # Live PV tuning
        self.declare_parameter("pv_stride_len", 51)
        self.declare_parameter("pv_contact_fraction", 0.63)

        S = int(self.get_parameter("pv_stride_len").value)
        c = float(self.get_parameter("pv_contact_fraction").value)

        # ---- package files ----
        self.declare_parameter("model_file", "PV_rolling_next_tick_lstm.keras")
        self.declare_parameter("scaler_pv_file", "scaler_pv_lstm.save")
        self.declare_parameter("scaler_ang_file", "scaler_angles_lstm.save")
        self.declare_parameter("excel_file", "rolling_gt_next_tick_with_pv_cnn.xlsx")

        # Training order
        self.declare_parameter("angle_joint_order", [
            "left_hip_revolute_joint",
            "left_knee_revolute_joint",
            "left_ankle_revolute_joint",
            "right_hip_revolute_joint",
            "right_knee_revolute_joint",
            "right_ankle_revolute_joint",
        ])

        # Controller order
        self.declare_parameter("traj_joints", [
            "left_hip_revolute_joint",
            "right_hip_revolute_joint",
            "left_knee_revolute_joint",
            "right_knee_revolute_joint",
            "left_ankle_revolute_joint",
            "right_ankle_revolute_joint",
        ])

        # ---- read params ----
        self.W = int(self.get_parameter("window").value)
        self.pub_timer = float(self.get_parameter("pub_timer").value)
        self.segment_len = int(self.get_parameter("segment_len").value)
        self.invert_knee_if_positive = bool(self.get_parameter("invert_knee_if_positive").value)

        self.use_live_pv = bool(self.get_parameter("use_live_pv").value)
        self.pv_joint_left = str(self.get_parameter("pv_joint_left").value)
        self.pv_joint_right = str(self.get_parameter("pv_joint_right").value)
        self.assumed_stride_period = float(self.get_parameter("assumed_stride_period").value)

        model_file = str(self.get_parameter("model_file").value)
        scaler_pv_file = str(self.get_parameter("scaler_pv_file").value)
        scaler_ang_file = str(self.get_parameter("scaler_ang_file").value)
        excel_file = str(self.get_parameter("excel_file").value)

        self.angle_joint_order = list(self.get_parameter("angle_joint_order").value)
        self.traj_joints = list(self.get_parameter("traj_joints").value)

        # ---- load files ----
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

        # ---- PV engines ----
        self.pvL_engine = WindowedStridePV(stride_len=S, c=c, enforce_monotonic=True)
        self.pvR_engine = WindowedStridePV(stride_len=S, c=c, enforce_monotonic=True)

        # ---- seed rolling window from Excel ----
        df = pd.read_excel(self.excel_path)
        raw = df.values.astype(np.float32)

        if raw.shape[1] != 8:
            raise RuntimeError(f"Excel must have 8 columns [pvL,pvR,6 angles_deg], got {raw.shape}")
        if raw.shape[0] < self.W:
            raise RuntimeError(f"Excel must have at least {self.W} rows, got {raw.shape[0]}")

        self.rolling_unscaled = raw[:self.W].copy()  # (W,8) unscaled

        # last PV start (from excel)
        self.last_pv = (float(self.rolling_unscaled[-1, 0]), float(self.rolling_unscaled[-1, 1]))
        self.have_live_pv = False

        # ---- ROS IO ----
        js_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1, durability=DurabilityPolicy.VOLATILE)
        self.sub_js = self.create_subscription(JointState, "/joint_states", self._on_joint_state, js_qos)

        self.pub_traj = self.create_publisher(JointTrajectory, "/trajectory_controller/joint_trajectory", 10)
        self.pub_pv = self.create_publisher(Float32MultiArray, "/phase_variable", 10)

        self.timer = self.create_timer(self.pub_timer, self._timer_cb)

        self.get_logger().info(
            f"PV publisher ready. W={self.W}, dt={self.pub_timer}, segment_len={self.segment_len}, "
            f"use_live_pv={self.use_live_pv}, assumed_stride_period={self.assumed_stride_period}"
        )

    # ----------------------------
    # Scaling helpers
    # ----------------------------
    def _scale_features(self, unscaled_Wx8: np.ndarray) -> np.ndarray:
        pv_un = unscaled_Wx8[:, :2].astype(np.float32)
        ang_un = unscaled_Wx8[:, 2:].astype(np.float32)

        pv_sc = self.scaler_pv.transform(pv_un)
        ang_sc = self.scaler_ang.transform(ang_un)

        X = np.concatenate([pv_sc, ang_sc], axis=1).astype(np.float32)  # (W,8)
        return X.reshape(1, self.W, 8)

    def _unscale_angles(self, y_scaled_6: np.ndarray) -> np.ndarray:
        y_scaled_6 = y_scaled_6.reshape(1, -1).astype(np.float32)
        return self.scaler_ang.inverse_transform(y_scaled_6)[0].astype(np.float32)

    # ----------------------------
    # Conventions
    # ----------------------------
    def _flip_knee_signs_inplace_deg(self, a: np.ndarray):
        if not self.invert_knee_if_positive:
            return
        for ki in [1, 4]:  # L knee, R knee indices in training order
            if a[ki] >= 0.0:
                a[ki] = -a[ki]

    # ----------------------------
    # Model next step
    # ----------------------------
    def _predict_next_step_scaled(self, X_1xWx8: np.ndarray) -> np.ndarray:
        y = self.model.predict(X_1xWx8, verbose=0)
        y = np.array(y)
        if y.ndim == 2 and y.shape == (1, 6):
            return y[0].astype(np.float32)
        if y.ndim == 1 and y.shape == (6,):
            return y.astype(np.float32)
        raise RuntimeError(f"Unexpected PV model output shape: {y.shape}. Expected (1,6) or (6,)")

    # ----------------------------
    # Publishing
    # ----------------------------
    def _publish_segment_deg(self, seg_deg: np.ndarray):
        msg = JointTrajectory()
        msg.joint_names = self.traj_joints

        for i in range(seg_deg.shape[0]):
            a = seg_deg[i].copy()
            self._flip_knee_signs_inplace_deg(a)

            # training order -> controller order (convert deg->rad)
            m = {j: float(np.radians(a[k])) for k, j in enumerate(self.angle_joint_order)}

            pt = JointTrajectoryPoint()
            pt.positions = [m[j] for j in self.traj_joints]
            pt.time_from_start = Duration(sec=0, nanosec=int((i + 1) * self.pub_timer * 1e9))
            msg.points.append(pt)

        self.pub_traj.publish(msg)

    # ----------------------------
    # JointState callback: LIVE PV (stride timed)
    # ----------------------------
    def _on_joint_state(self, msg: JointState):
        if not self.use_live_pv:
            return

        name_to_i = {n: i for i, n in enumerate(msg.name)}
        if self.pv_joint_left not in name_to_i or self.pv_joint_right not in name_to_i:
            return

        t_now = self.get_clock().now().nanoseconds * 1e-9
        qL = float(msg.position[name_to_i[self.pv_joint_left]])
        qR = float(msg.position[name_to_i[self.pv_joint_right]])

        pvL = self.pvL_engine.update(qL)
        pvR = self.pvR_engine.update(qR)

        self.last_pv = (pvL, pvR)
        self.have_live_pv = True

        # update only newest PV in rolling buffer
        self.rolling_unscaled[-1, 0] = float(pvL)
        self.rolling_unscaled[-1, 1] = float(pvR)

        pv_msg = Float32MultiArray()
        pv_msg.data = [float(pvL), float(pvR)]
        self.pub_pv.publish(pv_msg)

    # ----------------------------
    # Timer: recursive rollout segment_len points
    # ----------------------------
    def _timer_cb(self):
        seg_deg = np.zeros((self.segment_len, 6), dtype=np.float32)

        pvL, pvR = self.last_pv

        # If NOT using live PV, advance PV internally so it goes 0->1 per stride
        if not self.use_live_pv:
            dp = self.pub_timer / max(1e-3, self.assumed_stride_period)
        else:
            dp = 0.0  # do not double-drive PV

        for k in range(self.segment_len):
            Xk = self._scale_features(self.rolling_unscaled)

            y_scaled = self._predict_next_step_scaled(Xk)
            y_deg = self._unscale_angles(y_scaled)
            self._flip_knee_signs_inplace_deg(y_deg)

            seg_deg[k] = y_deg

            # Update PV for the next row
            if dp != 0.0:
                pvL = (pvL + dp) % 1.0
                pvR = (pvR + dp) % 1.0
                self.last_pv = (pvL, pvR)

            new_row = np.concatenate([[pvL, pvR], y_deg]).astype(np.float32)
            self.rolling_unscaled[:-1] = self.rolling_unscaled[1:]
            self.rolling_unscaled[-1] = new_row

        self._publish_segment_deg(seg_deg)

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
                node.get_logger().warn(f"Timeout unloading {ctrl}")

        node.destroy_node()
        rclpy.shutdown()
        os._exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
