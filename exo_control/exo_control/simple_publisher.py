import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class TrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('trajectory_publisher')
        self.pub = self.create_publisher(JointTrajectory, '/trajectory_controller/joint_trajectory', 10)

        # Wait a few seconds to ensure controllers are active
        self.timer = self.create_timer(5.0, self.publish_once)
        self.has_published = False

    def publish_once(self):
        if self.has_published:
            return

        msg = JointTrajectory()
        msg.joint_names = [
            'left_hip_revolute_joint',
            'left_knee_revolute_joint',
            'left_ankle_revolute_joint',
            'right_hip_revolute_joint',
            'right_knee_revolute_joint',
            'right_ankle_revolute_joint'
        ]

        point = JointTrajectoryPoint()
        point.positions = [0.766, 0.2, 0.8, 0.6, -0.5, 0.9]
        point.time_from_start = Duration(sec=2)

        msg.points.append(point)
        self.pub.publish(msg)
        self.get_logger().info("Trajectory published.")
        self.has_published = True


def main():
    rclpy.init()
    node = TrajectoryPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
