import rclpy
import numpy as np
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Duration
from ament_index_python.packages import get_package_share_directory

class NextStepPublisher(Node):

    def __init__(self):
        super().__init__('next_step_publisher')

        # Exoskeleton joint names
        self.joint_names = [
            'left_hip_revolute_joint',
            'right_hip_revolute_joint',
            'left_knee_revolute_joint',
            'right_knee_revolute_joint',
            'left_ankle_revolute_joint',
            'right_ankle_revolute_joint'
        ]

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

        # --------------------------------------------
        # Load RL-corrected CP gait
        # --------------------------------------------
        pkg = get_package_share_directory('exo_control')
        gait_path = f"{pkg}/neural_network_parameters/excel/corrected_cp_gait.npy"

        self.get_logger().info(f"Loading RL corrected gait from {gait_path}")
        self.corrected_gait = np.load(gait_path)   # (51, 6) in DEGREES

        self.N = len(self.corrected_gait)          # stride length
        self.current_idx = 0
        self.target = None

        self.get_logger().info("Ready. Beginning execution...")

        # Timer loop (100 Hz)
        self.timer = self.create_timer(0.01, self.control_loop)


    # --------------------------------------------
    def js_callback(self, msg):
        """Store latest joint state positions."""
        try:
            self.last_js = np.array(
                [msg.position[msg.name.index(j)] for j in self.joint_names]
            )
        except:
            pass


    # --------------------------------------------
    def control_loop(self):

        if self.last_js is None:
            return

        # Finished?
        if self.current_idx >= self.N:
            self.get_logger().info("Finished executing corrected gait.")
            return

        # If no target yet → send it
        if self.target is None:
            self.send_frame(self.corrected_gait[self.current_idx])
            self.target = self.corrected_gait[self.current_idx]
            return

        # Check if joints reached target
        if np.allclose(self.last_js, np.radians(self.target), atol=0.05):
            self.current_idx += 1
            self.target = None


    # --------------------------------------------
    def send_frame(self, deg6):
        rad = np.radians(deg6)

        traj = JointTrajectory()
        traj.joint_names = self.joint_names

        pt = JointTrajectoryPoint()
        pt.positions = rad.tolist()
        pt.time_from_start = Duration(sec=1)

        traj.points.append(pt)
        self.pub.publish(traj)
        self.get_logger().info(f"Sent frame {self.current_idx+1}/{self.N}")


def main(args=None):
    rclpy.init(args=args)
    node = NextStepPublisher()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
