import rclpy
import numpy as np
import pandas as pd
import time
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
from ament_index_python.packages import get_package_share_directory
import tensorflow as tf
import keras
import joblib
import os


class NextStepPublisher(Node):

    def __init__(self):
        super().__init__('next_step_publisher')

        # -------- Joint names --------
        self.joint_names = [
            'left_hip_revolute_joint',
            'right_hip_revolute_joint',
            'left_knee_revolute_joint',
            'right_knee_revolute_joint',
            'left_ankle_revolute_joint',
            'right_ankle_revolute_joint'
        ]
        # -------- Column definitions --------
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
        # -------- Publishers + Subscribers --------
        self.pub = self.create_publisher(
            JointTrajectory,
            '/trajectory_controller/joint_trajectory',
            10
        )

        self.js_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.js_callback,
            10
        )

        self.last_js = None

        # -------- Load model + scalers --------
        pkg = get_package_share_directory('exo_control')
        model_path = f"{pkg}/neural_network_parameters/models/dynamics_lstm.keras"
        dyn = f"{pkg}/neural_network_parameters/scaler/dyn_scaler.save"
        ang = f"{pkg}/neural_network_parameters/scaler/ang_scaler.save"
        excel = f"{pkg}/neural_network_parameters/excel/data_healthy_dynamics.xlsx"

        @keras.saving.register_keras_serializable()
        def weighted_mse(y_true, y_pred):
            w = tf.constant([3,3,1]*6, dtype=tf.float32)
            return tf.reduce_mean(w * tf.square(y_true - y_pred))

        self.model = tf.keras.models.load_model(model_path,
                        custom_objects={'weighted_mse': weighted_mse})

        self.scaler_dyn = joblib.load(dyn)
        self.scaler_ang = joblib.load(ang)

        # -------- Load initial stride (51×54) --------
        df = pd.read_excel(excel, sheet_name="Data", skiprows=[1,2])
        df = df[self.ALL_COLS]   # force exact subset        
        arr = df.values.astype(np.float32)

        mom = arr[:, :36]
        ang = arr[:, 36:]

        mom_s = self.scaler_dyn.transform(mom)
        ang_s = self.scaler_ang.transform(ang)

        X = np.concatenate([mom_s, ang_s], axis=1).reshape(1,51,54)

        # -------- Predict next stride --------
        self.get_logger().info("Predicting next stride...")
        pred_scaled = self.model.predict(X)[0]
        angles_deg  = self.scaler_ang.inverse_transform(pred_scaled)

        # extract 6 sagittal
        idx = [0,1,6,7,12,13]
        self.predicted = angles_deg[:, idx].copy()

        # Knee sign fix continuous:
        self.predicted[:,2] = -np.abs(self.predicted[:,2])
        self.predicted[:,3] = -np.abs(self.predicted[:,3])

        self.current_idx = 0
        self.target = None

        self.get_logger().info("Prediction ready. Beginning step-by-step execution.")

        # Start the tight loop
        self.timer = self.create_timer(0.01, self.control_loop)



    # ---------------------------------------------------------
    def js_callback(self, msg):
        """Store latest joint state positions."""
        positions = []
        for j in self.joint_names:
            try:
                positions.append(msg.position[msg.name.index(j)])
            except:
                return
        self.last_js = np.array(positions, dtype=np.float32)


    # ---------------------------------------------------------
    def control_loop(self):
        """Send next target only when previous is reached, loop stride forever."""

        if self.last_js is None:
            return

        # If we have finished all frames → restart from the beginning
        if self.current_idx >= len(self.predicted):
            self.get_logger().info("Completed one predicted stride, restarting.")
            self.current_idx = 0
            self.target = None

        # If no current target: send one
        if self.target is None:
            self.target = self.predicted[self.current_idx]
            self.send_frame(self.target)
            return

        # Check if reached (joint_state close enough to target)
        if np.allclose(self.last_js, np.radians(self.target), atol=0.05):
            self.current_idx += 1
            self.target = None

    # ---------------------------------------------------------
    def send_frame(self, deg6):
        """Send a single joint configuration (no time constraints)."""

        rad = np.radians(deg6)

        traj = JointTrajectory()
        traj.joint_names = self.joint_names

        pt = JointTrajectoryPoint()
        pt.positions = rad.tolist()
        pt.time_from_start = Duration(sec=1)  # irrelevant because single point

        traj.points.append(pt)
        self.pub.publish(traj)
        self.get_logger().info(f"Sent frame {self.current_idx+1}/51")



def main(args=None):
    rclpy.init(args=args)
    node = NextStepPublisher()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
