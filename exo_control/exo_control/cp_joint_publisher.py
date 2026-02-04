#!/usr/bin/env python3
import os
import math
import numpy as np
import pandas as pd
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import JointState

class CPExcelJointStatePublisher(Node):
    """
    Publishes JointState messages from an Excel file containing CP joint angles.
    Publishes to /joint_states_playback by default for nn model input.
    Typical use:
      - Node A (this): publishes playback JointState from Excel
      - Node B (nn model + controller): subscribes to /joint_states_playback and subscribes to nn controller output and sends commands to exoskeleton
    """

    def __init__(self):
        super().__init__("cp_excel_jointstate_publisher")

        # -------------------------
        # Parameters
        # -------------------------
        self.declare_parameter("excel_path", "/home/vaggelis/ros2_ws/src/ROS2-Lower-Limb-Exoskeleton-Control/exo_control/neural_network_parameters/excel/timestamps_cp_cnn.xlsx")
        self.declare_parameter("sheet_name", "")
        self.declare_parameter("publish_topic", "/joint_states_playback")
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("loop", True)
        self.declare_parameter("time_col", "")
        self.declare_parameter("angles_in_degrees", True)
        self.declare_parameter("angle_cols", [
            "LHipAngles (1)", "LKneeAngles (1)", "LAnkleAngles (1)",
            "RHipAngles (1)", "RKneeAngles (1)", "RAnkleAngles (1)"
        ])
        # JointState joint names to publish (must match consumer expectations)
        self.declare_parameter("joint_names", [
            "left_hip_revolute_joint",
            "left_knee_revolute_joint",
            "left_ankle_revolute_joint",
            "right_hip_revolute_joint",
            "right_knee_revolute_joint",
            "right_ankle_revolute_joint",
        ])

        # Optional: publish velocity/effort as zeros
        self.declare_parameter("publish_velocity", False)
        self.declare_parameter("publish_effort", False)

        # -------------------------
        # Read params
        # -------------------------
        self.excel_path = str(self.get_parameter("excel_path").value)
        self.sheet_name = str(self.get_parameter("sheet_name").value).strip()
        self.publish_topic = str(self.get_parameter("publish_topic").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.loop = bool(self.get_parameter("loop").value)

        self.time_col = str(self.get_parameter("time_col").value).strip()
        self.angles_in_degrees = bool(self.get_parameter("angles_in_degrees").value)

        self.angle_cols = list(self.get_parameter("angle_cols").value)
        self.joint_names = list(self.get_parameter("joint_names").value)

        self.publish_velocity = bool(self.get_parameter("publish_velocity").value)
        self.publish_effort = bool(self.get_parameter("publish_effort").value)

        if len(self.angle_cols) != len(self.joint_names):
            raise ValueError(
                f"angle_cols ({len(self.angle_cols)}) must match joint_names ({len(self.joint_names)})"
            )

        if not self.excel_path:
            raise ValueError("Parameter 'excel_path' must be set to an .xlsx file path.")
        if not os.path.exists(self.excel_path):
            raise FileNotFoundError(f"Excel file not found: {self.excel_path}")

        # -------------------------
        # Load Excel into memory
        # -------------------------
        self.get_logger().info(f"Loading Excel: {self.excel_path}")
        if self.sheet_name:
            df = pd.read_excel(self.excel_path, sheet_name=self.sheet_name)
        else:
            df = pd.read_excel(self.excel_path)  # first sheet

        missing = [c for c in self.angle_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Excel missing angle columns: {missing}")

        self.have_time = bool(self.time_col) and (self.time_col in df.columns)
        if self.time_col and (self.time_col not in df.columns):
            self.get_logger().warn(
                f"time_col='{self.time_col}' not found. Will publish at fixed rate {self.publish_rate_hz} Hz."
            )
            self.have_time = False

        # Extract angles (N,6)
        ang = df[self.angle_cols].to_numpy(dtype=np.float32)
        ang = np.nan_to_num(ang)

        # Convert to radians for ROS JointState
        if self.angles_in_degrees:
            ang = np.deg2rad(ang)

        self.angles_rad = ang
        self.N = self.angles_rad.shape[0]

        # If time column exists, build dt schedule
        if self.have_time:
            t = df[self.time_col].to_numpy(dtype=np.float64)
            t = np.nan_to_num(t)
            # Ensure nondecreasing and start at 0-ish
            t = t - t[0]
            dt = np.diff(t, prepend=t[0])
            # Fix nonpositive dt
            dt[dt <= 0.0] = 1.0 / max(1e-6, self.publish_rate_hz)
            self.time_s = t
            self.dt_s = dt.astype(np.float64)
            self.get_logger().info(f"Time-based playback enabled using column '{self.time_col}'.")
        else:
            self.dt_fixed = 1.0 / max(1e-6, self.publish_rate_hz)

        self.get_logger().info(
            f"Loaded {self.N} samples. angles_in_degrees={self.angles_in_degrees}. "
            f"Publishing to {self.publish_topic}."
        )

        # -------------------------
        # Publisher
        # -------------------------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pub = self.create_publisher(JointState, self.publish_topic, qos)

        # Playback state
        self.i = 0

        # Timer: if time-based, we reschedule by recreating timer each tick (simple + robust)
        # If fixed-rate, normal periodic timer.
        if self.have_time:
            self.timer = self.create_timer(self.dt_s[0], self._tick_time_based)
        else:
            self.timer = self.create_timer(self.dt_fixed, self._tick_fixed_rate)

    def _make_msg(self, angles_rad_row: np.ndarray) -> JointState:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.joint_names)
        msg.position = [float(x) for x in angles_rad_row]

        if self.publish_velocity:
            msg.velocity = [0.0] * len(self.joint_names)
        if self.publish_effort:
            msg.effort = [0.0] * len(self.joint_names)
        return msg

    def _advance_index(self):
        self.i += 1
        if self.i >= self.N:
            if self.loop:
                self.i = 0
            else:
                self.get_logger().info("Playback finished (loop=False). Shutting down.")
                rclpy.shutdown()

    def _tick_fixed_rate(self):
        msg = self._make_msg(self.angles_rad[self.i])
        self.pub.publish(msg)
        self._advance_index()

    def _tick_time_based(self):
        # publish current
        msg = self._make_msg(self.angles_rad[self.i])
        self.pub.publish(msg)

        # advance index
        self._advance_index()

        # Reschedule timer to next dt
        # NOTE: rclpy timers can't change period; easiest approach is cancel+create new timer.
        try:
            self.timer.cancel()
        except Exception:
            pass

        next_dt = float(self.dt_s[self.i]) if self.have_time else 0.02
        # clamp to avoid crazy timer periods
        next_dt = float(np.clip(next_dt, 0.001, 0.2))
        self.timer = self.create_timer(next_dt, self._tick_time_based)


def main(args=None):
    rclpy.init(args=args)
    node = CPExcelJointStatePublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
