import os
import numpy as np
import rclpy
from rclpy.node import Node
from collections import deque

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

import tensorflow as tf
import joblib
from ament_index_python.packages import get_package_share_directory


STRIDE = 51
PUB_DT = 0.05  # 20 Hz -> 51 points == 2.55s

CTRL_JOINTS = [
    'left_hip_roll_joint', 'left_hip_pitch_joint', 'left_hip_yaw_joint',
    'left_knee_roll_joint','left_knee_pitch_joint','left_knee_yaw_joint',
    'left_ankle_roll_joint','left_ankle_pitch_joint','left_ankle_yaw_joint',
    'right_hip_roll_joint','right_hip_pitch_joint','right_hip_yaw_joint',
    'right_knee_roll_joint','right_knee_pitch_joint','right_knee_yaw_joint',
    'right_ankle_roll_joint','right_ankle_pitch_joint','right_ankle_yaw_joint',
]

# MODEL index -> CTRL index mapping you already derived
CTRL_TO_MODEL = [
    2, 11,   # Hip (1)=yaw: left_hip_yaw, right_hip_yaw
    0, 9,    # Hip (2)=roll
    1, 10,   # Hip (3)=pitch
    5, 14,   # Knee (1)=yaw
    3, 12,   # Knee (2)=roll
    4, 13,   # Knee (3)=pitch
    8, 17,   # Ankle (1)=yaw
    6, 15,   # Ankle (2)=roll
    7, 16,   # Ankle (3)=pitch
]

MODEL_TO_CTRL = [0] * 18
for model_i, ctrl_i in enumerate(CTRL_TO_MODEL):
    MODEL_TO_CTRL[ctrl_i] = model_i


def to_model_order(v18_ctrl: np.ndarray) -> np.ndarray:
    return v18_ctrl[np.array(CTRL_TO_MODEL, dtype=int)]


def to_ctrl_order(v18_model: np.ndarray) -> np.ndarray:
    return v18_model[np.array(MODEL_TO_CTRL, dtype=int)]


class NextStridePublisher(Node):
    """
    - Keeps a rolling 51-sample buffer from /joint_states (never stops listening)
    - When buffer is full: runs LSTM once -> predicted (51,18)
    - Publishes ONE JointTrajectory message containing all 51 points
    """

    def __init__(self):
        super().__init__("next_stride_publisher")

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

        # Rolling buffer of last 51 frames, each (36,)
        self.buf = deque(maxlen=STRIDE)

        # Optional: prevent spamming predictions faster than you can execute
        self.busy = False

        self.get_logger().info("Listening to /joint_states and filling a rolling 51-sample window...")

    def on_js(self, msg: JointState):
        name_to_i = {n: i for i, n in enumerate(msg.name)}
        if not all(j in name_to_i for j in CTRL_JOINTS):
            return

        # angles in CTRL order (rad->deg)
        ang_rad_ctrl = np.array([msg.position[name_to_i[j]] for j in CTRL_JOINTS], dtype=np.float32)
        ang_deg_ctrl = np.degrees(ang_rad_ctrl).astype(np.float32)

        # IMPORTANT:
        # Your network expects 36 = 18 forces + 18 angles.
        # If you don't want to use msg.effort, you must provide a substitute.
        # For now: zeros (works structurally, but may reduce quality vs training).
        eff_ctrl = np.zeros(18, dtype=np.float32)

        ang_deg_model = to_model_order(ang_deg_ctrl)
        eff_model     = to_model_order(eff_ctrl)

        feat36 = np.concatenate([eff_model, ang_deg_model], axis=0).astype(np.float32)
        self.buf.append(feat36)

        if len(self.buf) < STRIDE:
            return

        # Trigger prediction if not busy
        if not self.busy:
            self.busy = True
            try:
                self.predict_and_publish()
            finally:
                self.busy = False

    def predict_and_publish(self):
        X = np.stack(self.buf, axis=0)[None, :, :]  # (1,51,36)

        Xs = self.sc_x.transform(X.reshape(-1, 36)).reshape(X.shape)
        Yhat_s = self.model.predict(Xs, verbose=0)             # (1,51,18)
        Yhat_deg_model = self.sc_y.inverse_transform(
            Yhat_s.reshape(-1, 18)
        ).reshape(STRIDE, 18).astype(np.float32)               # (51,18) deg

        traj = JointTrajectory()
        traj.joint_names = CTRL_JOINTS

        points = []
        for k in range(STRIDE):
            frame_deg_ctrl = to_ctrl_order(Yhat_deg_model[k])
            frame_rad_ctrl = np.radians(frame_deg_ctrl).astype(np.float32)

            pt = JointTrajectoryPoint()
            pt.positions = frame_rad_ctrl.tolist()

            # KEY FIX: time_from_start must be increasing across points
            t = (k + 1) * PUB_DT
            sec = int(t)
            nsec = int((t - sec) * 1e9)
            pt.time_from_start = Duration(sec=sec, nanosec=nsec)

            points.append(pt)

        traj.points = points
        self.pub.publish(traj)

        self.get_logger().info("Published full 51-point next-stride trajectory (single message).")


def main(args=None):
    rclpy.init(args=args)
    node = NextStridePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
