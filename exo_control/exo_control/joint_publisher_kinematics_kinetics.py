import os
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

import tensorflow as tf
import joblib
from ament_index_python.packages import get_package_share_directory


STRIDE = 51
PUB_DT = 0.05  # 20 Hz

# Controller joint order (must match ros2_control YAML)
CTRL_JOINTS = [
    'left_hip_roll_joint', 'left_hip_pitch_joint', 'left_hip_yaw_joint',
    'left_knee_roll_joint','left_knee_pitch_joint','left_knee_yaw_joint',
    'left_ankle_roll_joint','left_ankle_pitch_joint','left_ankle_yaw_joint',
    'right_hip_roll_joint','right_hip_pitch_joint','right_hip_yaw_joint',
    'right_knee_roll_joint','right_knee_pitch_joint','right_knee_yaw_joint',
    'right_ankle_roll_joint','right_ankle_pitch_joint','right_ankle_yaw_joint',
]

# Your model angle order (example):
# [LHip(1), RHip(1), LHip(2), RHip(2), LHip(3), RHip(3), LKnee(1), RKnee(1), ...]
# You said sagittal=(1) corresponds to yaw. With your controller order (roll,pitch,yaw),
# this mapping is:
#   (1)=yaw, (2)=roll, (3)=pitch
#
# CTRL order groups L then R; MODEL order interleaves L/R per axis.
# So we build the mapping from MODEL index -> CTRL index:
CTRL_TO_MODEL = [  # model_index -> ctrl_index
    2, 11,   # Hip (1)=yaw: left_hip_yaw, right_hip_yaw
    0, 9,    # Hip (2)=roll: left_hip_roll, right_hip_roll
    1, 10,   # Hip (3)=pitch: left_hip_pitch, right_hip_pitch

    5, 14,   # Knee (1)=yaw
    3, 12,   # Knee (2)=roll
    4, 13,   # Knee (3)=pitch

    8, 17,   # Ankle (1)=yaw
    6, 15,   # Ankle (2)=roll
    7, 16,   # Ankle (3)=pitch
]

# Inverse mapping: CTRL index -> MODEL index
MODEL_TO_CTRL = [0] * 18
for model_i, ctrl_i in enumerate(CTRL_TO_MODEL):
    MODEL_TO_CTRL[ctrl_i] = model_i


def to_model_order(v18_ctrl):
    """CTRL order -> MODEL order"""
    return v18_ctrl[np.array(CTRL_TO_MODEL, dtype=int)]


def to_ctrl_order(v18_model):
    """MODEL order -> CTRL order"""
    return v18_model[np.array(MODEL_TO_CTRL, dtype=int)]


class NextStrideFromLive(Node):
    """
    Collect 51 frames of live data from /joint_states,
    predict next stride (51x18 angles),
    publish as 51 streamed JointTrajectory points.
    """

    def __init__(self):
        super().__init__("next_stride_from_live")

        self.pub = self.create_publisher(
            JointTrajectory,
            "/trajectory_controller/joint_trajectory",
            10
        )

        self.sub = self.create_subscription(
            JointState,
            "/joint_states",
            self.on_js,
            50
        )

        # ---- Load model + scalers ----
        pkg = get_package_share_directory("exo_control")
        model_path = os.path.join(pkg, "neural_network_parameters/models", "td_nextstride_36to18.keras")
        sx_path    = os.path.join(pkg, "neural_network_parameters/scaler", "td_x36_scaler.save")
        sy_path    = os.path.join(pkg, "neural_network_parameters/scaler", "td_y18_scaler.save")

        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.sc_x = joblib.load(sx_path)
        self.sc_y = joblib.load(sy_path)

        # stride buffer: list of (36,) = [effort(18), angles_deg(18)] in MODEL order
        self.stride_buf = []

        # publishing state
        self.predicted_stride_deg_model = None  # (51,18) in MODEL order (deg)
        self.pub_idx = 0
        self.timer = self.create_timer(PUB_DT, self.publish_step)

        self.get_logger().info("Collecting 51 frames from /joint_states...")

    def on_js(self, msg: JointState):
        if self.predicted_stride_deg_model is not None:
            # we're currently publishing; ignore new data to keep logic simple
            return

        name_to_i = {n: i for i, n in enumerate(msg.name)}

        # ensure all controller joints exist in the message
        if not all(j in name_to_i for j in CTRL_JOINTS):
            return

        # angles in CTRL order (radians)
        ang_rad_ctrl = np.array([msg.position[name_to_i[j]] for j in CTRL_JOINTS], dtype=np.float32)
        ang_deg_ctrl = np.degrees(ang_rad_ctrl).astype(np.float32)

        # efforts/torques in CTRL order (if available)
        if msg.effort is not None and len(msg.effort) == len(msg.name):
            eff_ctrl = np.array([msg.effort[name_to_i[j]] for j in CTRL_JOINTS], dtype=np.float32)
        else:
            eff_ctrl = np.zeros(18, dtype=np.float32)

        # convert to MODEL order
        ang_deg_model = to_model_order(ang_deg_ctrl)
        eff_model     = to_model_order(eff_ctrl)

        feat36 = np.concatenate([eff_model, ang_deg_model], axis=0).astype(np.float32)

        self.stride_buf.append(feat36)

        if len(self.stride_buf) == STRIDE:
            self.run_prediction_and_schedule_publish()
            self.stride_buf = []

    def run_prediction_and_schedule_publish(self):
        X = np.stack(self.stride_buf, axis=0)[None, :, :]  # (1,51,36)

        xin = X.shape[-1]
        Xs = self.sc_x.transform(X.reshape(-1, xin)).reshape(X.shape)

        # predict (1,51,18) scaled
        Yhat_s = self.model.predict(Xs, verbose=0)

        # inverse-scale to degrees
        Yhat_deg = self.sc_y.inverse_transform(Yhat_s.reshape(-1, 18)).reshape(1, STRIDE, 18)[0]

        self.predicted_stride_deg_model = Yhat_deg.astype(np.float32)
        self.pub_idx = 0

        self.get_logger().info("Predicted next stride. Publishing 51 points...")

    def publish_step(self):
        if self.predicted_stride_deg_model is None:
            return

        if self.pub_idx >= STRIDE:
            self.get_logger().info("Finished publishing predicted stride. Collecting next stride...")
            self.predicted_stride_deg_model = None
            self.pub_idx = 0
            return

        frame_deg_model = self.predicted_stride_deg_model[self.pub_idx]  # (18,)
        frame_deg_ctrl  = to_ctrl_order(frame_deg_model)

        frame_rad_ctrl = np.radians(frame_deg_ctrl).astype(np.float32)

        msg = JointTrajectory()
        msg.joint_names = CTRL_JOINTS

        pt = JointTrajectoryPoint()
        pt.positions = frame_rad_ctrl.tolist()
        # For streaming single-point commands, keep this close to PUB_DT
        pt.time_from_start = Duration(sec=0, nanosec=int(PUB_DT * 1e9))

        msg.points = [pt]
        self.pub.publish(msg)

        self.pub_idx += 1


def main(args=None):
    rclpy.init(args=args)
    node = NextStrideFromLive()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
