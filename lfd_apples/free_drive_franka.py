import rclpy
from rclpy.node import Node
from franka_msgs.srv import SetJointStiffness, SetCartesianStiffness
from controller_manager_msgs.srv import SwitchController, ListControllers


class FreeDriveToggle(Node):
    def __init__(self):
        super().__init__('free_drive_toggle')

        # Franka stiffness services
        self.joint_cli = self.create_client(SetJointStiffness, '/service_server/set_joint_stiffness')
        self.cart_cli = self.create_client(SetCartesianStiffness, '/service_server/set_cartesian_stiffness')

        # Controller manager services
        self.switch_cli = self.create_client(SwitchController, '/controller_manager/switch_controller')
        self.list_cli = self.create_client(ListControllers, '/controller_manager/list_controllers')

        for cli, name in [(self.joint_cli, "joint stiffness"),
                          (self.cart_cli, "cartesian stiffness"),
                          (self.switch_cli, "switch controller"),
                          (self.list_cli, "list controllers")]:
            while not cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f"Waiting for {name} service...")

        # Default controller names (adjust if your setup differs)
        self.moveit_controller = "fr3_arm_controller"
        self.freedrive_controller = "franka_cartesian_impedance_controller"

    def switch_to(self, start_list, stop_list):
        """Switch controllers via controller_manager."""
        req = SwitchController.Request()
        req.activate_controllers = start_list
        req.deactivate_controllers = stop_list
        req.strictness = req.STRICT
        req.activate_asap = True
        future = self.switch_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() is None:
            self.get_logger().error("❌ Controller switch failed")
        else:
            self.get_logger().info(f"✅ Switched controllers: start={start_list}, stop={stop_list}")

    def free_drive(self):
        # Switch controllers
        self.switch_to([self.freedrive_controller], [self.moveit_controller])

        # Soften joint stiffness
        jreq = SetJointStiffness.Request()
        jreq.joint_stiffness = [0.0] * 7
        future = self.joint_cli.call_async(jreq)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        # Soften Cartesian stiffness
        creq = SetCartesianStiffness.Request()
        creq.cartesian_stiffness = [10.0, 10.0, 10.0, 1.0, 1.0, 1.0]
        future = self.cart_cli.call_async(creq)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

        self.get_logger().info("🤲 Robot in Free Drive mode. You can guide it by hand.")

    def moveit_mode(self):
        # Switch controllers
        self.switch_to([self.moveit_controller], [self.freedrive_controller])

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

        self.get_logger().info("🤖 Robot back in MoveIt control mode.")


def main():
    rclpy.init()
    node = FreeDriveToggle()

    input("Press Enter to enable Free Drive...")
    node.free_drive()

    input("Press Enter to restore MoveIt control...")
    node.moveit_mode()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
