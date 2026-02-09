#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from ros_gz_interfaces.msg import Contacts
from std_msgs.msg import Bool


class FootContactBool(Node):
    def __init__(self):
        super().__init__('foot_contact_bool')

        self.declare_parameter('period_sec', 0.05)
        self.period = float(self.get_parameter('period_sec').value)

        # How long since last contact msg before we force False
        self.declare_parameter('contact_timeout_sec', 0.15)
        self.timeout = float(self.get_parameter('contact_timeout_sec').value)

        self.left_pub = self.create_publisher(Bool, '/left_sole/in_contact', 10)
        self.right_pub = self.create_publisher(Bool, '/right_sole/in_contact', 10)

        self.left_in_contact = False
        self.right_in_contact = False

        # Track last time we received a contact message
        now = self.get_clock().now()
        self.left_last_msg_time = now
        self.right_last_msg_time = now

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(Contacts, '/left_sole/contacts', self.left_cb, sensor_qos)
        self.create_subscription(Contacts, '/right_sole/contacts', self.right_cb, sensor_qos)

        self.timer = self.create_timer(self.period, self.timer_cb)

        self.left_pub.publish(Bool(data=False))
        self.right_pub.publish(Bool(data=False))

        self.get_logger().info(
            f'Publishing contact bools every {self.period:.3f}s, timeout={self.timeout:.3f}s'
        )

    def left_cb(self, msg: Contacts):
        # If a msg arrives, it means contact exists (in your setup)
        self.left_in_contact = True
        self.left_last_msg_time = self.get_clock().now()

    def right_cb(self, msg: Contacts):
        self.right_in_contact = True
        self.right_last_msg_time = self.get_clock().now()

    def timer_cb(self):
        now = self.get_clock().now()

        # If we haven't heard from the sensor recently, assume contact ended
        if (now - self.left_last_msg_time).nanoseconds * 1e-9 > self.timeout:
            self.left_in_contact = False
        if (now - self.right_last_msg_time).nanoseconds * 1e-9 > self.timeout:
            self.right_in_contact = False

        self.left_pub.publish(Bool(data=self.left_in_contact))
        self.right_pub.publish(Bool(data=self.right_in_contact))


def main():
    rclpy.init()
    node = FootContactBool()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
