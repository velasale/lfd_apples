import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from builtin_interfaces.msg import Time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

class TestServo(Node):
    def __init__(self):
        super().__init__('test_servo')

        qos = QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        depth=10
        )

        # Publisher to the MoveIt Servo Cartesian command topic
        self.pub = self.create_publisher(TwistStamped, '/servo_node_lfd/delta_twist_cmds', qos)
        self.timer = self.create_timer(0.001, self.send_cmd)  # 100 Hz

    def send_cmd(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        # Cartesian velocity command (unitless)
        # [vx, vy, vz, wx, wy, wz] -- small steps
        msg.twist.linear.x = -0.2  # forward
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0

        self.pub.publish(msg)
        self.get_logger().info(f"Published Twist command: "
                               f"linear=({msg.twist.linear.x}, {msg.twist.linear.y}, {msg.twist.linear.z}) "
                               f"angular=({msg.twist.angular.x}, {msg.twist.angular.y}, {msg.twist.angular.z})")

def main(args=None):
    rclpy.init(args=args)
    node = TestServo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
