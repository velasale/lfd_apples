#!/usr/bin/env python3
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from controller_manager_msgs.srv import SwitchController, LoadController
import subprocess

from lfd_apples.listen_franka import main as listen_main

class MoveToHomeAndFreedrive(Node):
    def __init__(self):
        super().__init__('move_to_home_and_freedrive')

        # Action client for joint trajectory
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/fr3_arm_controller/follow_joint_trajectory'
        )

        # Controller names
        self.arm_controller = 'fr3_arm_controller'
        self.gravity_controller = 'gravity_compensation_example_controller'

        # Switch controller client
        self.switch_client = self.create_client(SwitchController, '/controller_manager/switch_controller')
        self.switch_client.wait_for_service()

        # Load controller client
        self.load_client = self.create_client(LoadController, '/controller_manager/load_controller')
        self.load_client.wait_for_service()

    def move_to_home(self):
        self.get_logger().info('Waiting for action server...')
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server not available')
            return False

        joint_names = [
            'fr3_joint1', 'fr3_joint2', 'fr3_joint3',
            'fr3_joint4', 'fr3_joint5', 'fr3_joint6', 'fr3_joint7'
        ]
              
        home_positions = [ 1.0,
                          -1.4,
                           0.66,
                          -2.2,
                           0.3,
                           2.23,
                           1.17]
    

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = joint_names
        point = JointTrajectoryPoint()
        point.positions = home_positions
        point.time_from_start.sec = 5
        goal_msg.trajectory.points.append(point)

        self.get_logger().info('Sending trajectory goal...')
        send_goal_future = self.action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return False

        self.get_logger().info('Goal accepted, waiting for result...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        self.get_logger().info(f'Motion complete with error code: {result.error_code}')
        return True

    def ensure_controller_loaded(self, controller_name):
        """Load a controller if not already loaded."""
        
        req = LoadController.Request()
        req.name = controller_name
        future = self.load_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()
        if resp.ok:
            self.get_logger().info(f'Controller {controller_name} loaded successfully.')
        else:
            self.get_logger().warn(f'Controller {controller_name} may already be loaded or failed to load.')

    def configure_controller(self, controller_name):
        """Configure controller to inactive using CLI (required for Humble)."""
        try:
            subprocess.run([
                'ros2', 'control', 'set_controller_state', controller_name, 'inactive'
            ], check=True)
            self.get_logger().info(f'Controller {controller_name} configured to inactive.')
        except subprocess.CalledProcessError:
            self.get_logger().error(f'Failed to configure {controller_name} to inactive.')

    def enable_freedrive(self):
        self.get_logger().info('Preparing freedrive mode...')

        # 1️⃣ Deactivate the arm controller
        switch_req = SwitchController.Request()
        switch_req.deactivate_controllers = [self.arm_controller]
        switch_req.activate_controllers = []
        switch_req.strictness = 2  # BEST_EFFORT
        future = self.switch_client.call_async(switch_req)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()
        if resp.ok:
            self.get_logger().info(f'{self.arm_controller} deactivated successfully.')
        else:
            self.get_logger().error(f'Failed to deactivate {self.arm_controller}.')
            return

        # 2️⃣ Load gravity compensation controller
        self.ensure_controller_loaded(self.gravity_controller)

        # 3️⃣ Configure gravity controller to inactive
        self.configure_controller(self.gravity_controller)

        # 4️⃣ Activate gravity compensation controller
        activate_req = SwitchController.Request()
        activate_req.deactivate_controllers = []
        activate_req.activate_controllers = [self.gravity_controller]
        activate_req.strictness = 2  # BEST_EFFORT
        future = self.switch_client.call_async(activate_req)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()
        if resp.ok:
            self.get_logger().info('Freedrive mode enabled! You can move the arm by hand.')
        else:
            self.get_logger().error('Failed to activate gravity compensation controller.')


def main():
    rclpy.init()

    node = MoveToHomeAndFreedrive()

    for demo in range(10):

        input(f"Press Enter to start demonstration {demo+1}/10: ")

        # Step 1: Move to home position and enable freedrive    
        node.get_logger().info("Retrying move to home...")
        while not node.move_to_home():
            pass
            
        # Step 2: Enable freedrive mode        
        node.enable_freedrive()
        listen_main()
        node.get_logger().info("Free-drive mode enabled, you can start the demo.")    

        # Step 3: Wait for user to finish demonstration
        print(f"Waiting to start demonstration {demo+1}/10. Press Enter to continue...")
        input()    
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
