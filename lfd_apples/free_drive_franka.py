import rclpy
from rclpy.node import Node
from franka_msgs.srv import SetJointStiffness, SetCartesianStiffness
import time

class FreeDriveToggle(Node):
    def __init__(self):
        super().__init__('free_drive_toggle')

        # Franka services
        self.joint_cli = self.create_client(SetJointStiffness, '/service_server/set_joint_stiffness')
        self.cart_cli = self.create_client(SetCartesianStiffness, '/service_server/set_cartesian_stiffness')

        for cli, name in [(self.joint_cli, "joint stiffness"),
                          (self.cart_cli, "cartesian stiffness")]:
            while not cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f"Waiting for {name} service...")

    def free_drive(self):
        # Soften joint stiffness
        jreq = SetJointStiffness.Request()
        jreq.joint_stiffness = [0.0] * 7
        future = self.joint_cli.call_async(jreq)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        # Optional: soften Cartesian stiffness (safe values)
        val = 10.0
        creq = SetCartesianStiffness.Request()
        creq.cartesian_stiffness = [val, val, val, 1.0, 1.0, 1.0]
        future = self.cart_cli.call_async(creq)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        self.get_logger().info("✅ Robot in Free Drive mode. You can guide it by hand.")

    def moveit_mode(self):
        # Restore joint stiffness
        jreq = SetJointStiffness.Request()
        jreq.joint_stiffness = [200.0] * 7
        future = self.joint_cli.call_async(jreq)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        # Restore Cartesian stiffness
        creq = SetCartesianStiffness.Request()
        creq.cartesian_stiffness = [3000.0, 3000.0, 3000.0, 300.0, 300.0, 300.0]
        future = self.cart_cli.call_async(creq)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        self.get_logger().info("✅ Robot back in MoveIt control mode.")


def main():
    rclpy.init()
    node = FreeDriveToggle()

    input("Press Enter to enable Free Drive...")
    node.free_drive()

    input("Press Enter to restore and use MoveIt...")
    node.moveit_mode()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
