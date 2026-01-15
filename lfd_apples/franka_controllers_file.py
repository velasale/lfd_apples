import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import JointState
import numpy as np
import time
import subprocess
from controller_manager_msgs.srv import SwitchController, LoadController, ListControllers

class CartesianVelocityZ(Node):
    def __init__(self):
        super().__init__('cartesian_velocity_z')

        # Publisher to Cartesian velocity controller
        self.pub = self.create_publisher(TwistStamped,
                                         '/cartesian_velocity_controller/command', 10)

        # Subscribe to EE pose
        self.sub_pose = self.create_subscription(
            PoseStamped,
            '/franka_robot_state_broadcaster/current_pose',
            self.pose_callback, 10
        )

        # Subscribe to joint states
        self.sub_joints = self.create_subscription(
            JointState,
            '/franka/joint_states',
            self.joint_callback, 10
        )

        # Controller names
        self.arm_controller = 'fr3_arm_controller'
        self.eef_velocity_controller = 'cartesian_velocity_controller'

        # Controller clients
        self.switch_client = self.create_client(SwitchController, '/controller_manager/switch_controller')
        self.switch_client.wait_for_service()
        self.load_client = self.create_client(LoadController, '/controller_manager/load_controller')
        self.load_client.wait_for_service()

        # Motion parameters
        self.amplitude = 0.02
        self.frequency = 0.2
        self.run_time = 10.0
        self.dt = 0.01

        # Limits
        self.v_max = 0.02
        self.a_max = 0.02
        self.j_max = 0.05

        # Internal state
        self.start_z = None
        self.current_z = None
        self.vz_cmd = 0.0
        self.az_cmd = 0.0
        self.motion_enabled = False
        self.start_time = time.time()
        self.joint_positions = None

        # Timer
        self.timer = self.create_timer(self.dt, self.update)
        self.create_timer(0.5, self.enable_motion)

        self.get_logger().info("Cartesian Z node initialized. Waiting to start motion...")

    # ---------------- Pose & joint callbacks ----------------
    def pose_callback(self, msg: PoseStamped):
        self.current_z = msg.pose.position.z
        if self.start_z is None:
            self.start_z = self.current_z

    def joint_callback(self, msg: JointState):
        self.joint_positions = np.array(msg.position)

    def enable_motion(self):
        self.motion_enabled = True
        self.get_logger().info("Motion enabled")

    def stop(self):
        self.pub.publish(TwistStamped())
        self.get_logger().info("Motion finished")
        rclpy.shutdown()

    # ---------------- Jacobian ----------------
    def compute_jacobian(self, q):
        """Placeholder: returns 7x6 Jacobian for Franka (simplified). Replace with real computation."""
        J = np.zeros((6, 7))
        J[2, :] = 0.2  # z-direction influence
        return J

    # ---------------- Update loop ----------------
    def update(self):
        t = time.time() - self.start_time
        if t > self.run_time:
            self.stop()
            return

        if not self.motion_enabled or self.current_z is None or self.start_z is None:
            self.pub.publish(TwistStamped())
            return

        # --- Quintic s-curve trajectory ---
        T = 1.0 / self.frequency
        s = (t % T) / T
        s5 = s**5; s4 = s**4; s3 = s**3
        factor = 10*s3 - 15*s4 + 6*s5
        factor_dot = (30*s**2 - 60*s**3 + 30*s**4)/T
        factor_ddot = (60*s - 180*s**2 + 120*s**3)/(T**2)

        z_target = self.start_z + self.amplitude * np.sin(2*np.pi*factor)
        vz_des = 2*np.pi*self.amplitude*np.cos(2*np.pi*factor) * factor_dot
        az_des = - (2*np.pi)**2 * self.amplitude * np.sin(2*np.pi*factor) * factor_dot**2 + \
                 2*np.pi*self.amplitude*np.cos(2*np.pi*factor) * factor_ddot

        # --- Apply jerk limit ---
        j_des = (az_des - self.az_cmd) / self.dt
        j_des = np.clip(j_des, -self.j_max, self.j_max)
        self.az_cmd += j_des * self.dt

        # --- Apply acceleration limit ---
        self.az_cmd = np.clip(self.az_cmd, -self.a_max, self.a_max)

        # --- Integrate to velocity ---
        self.vz_cmd += self.az_cmd * self.dt
        self.vz_cmd = np.clip(self.vz_cmd, -self.v_max, self.v_max)

        # --- Publish Twist ---
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.z = self.vz_cmd
        self.pub.publish(msg)

        # --- Compute joint velocities ---
        if self.joint_positions is not None:
            J = self.compute_jacobian(self.joint_positions)
            ee_twist = np.array([0, 0, self.vz_cmd, 0, 0, 0])
            qdot = np.linalg.pinv(J) @ ee_twist
            qdot_str = " ".join([f"{v:.4f}" for v in qdot])
            self.get_logger().info(f"EE vz: {self.vz_cmd:.5f} m/s | joint_velocities: {qdot_str}")

    # ---------------- Controller switching ----------------
    def swap_controller(self, stop_controller: str, start_controller: str, settle_time: float = 1.0):
        """Switch safely from one controller to another."""
        if stop_controller:
            self.get_logger().info(f"Deactivating controller: {stop_controller}")
            req = SwitchController.Request()
            req.deactivate_controllers = [stop_controller]
            req.activate_controllers = []
            req.strictness = 2
            future = self.switch_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            resp = future.result()
            if not resp or not resp.ok:
                self.get_logger().warn(f"Failed to deactivate {stop_controller}, continuing anyway.")
            else:
                self.get_logger().info(f"{stop_controller} deactivated successfully.")

        self.get_logger().info(f"Waiting {settle_time:.1f}s before activating {start_controller}...")
        time.sleep(settle_time)

        self.ensure_controller_loaded(start_controller)
        self.configure_controller(start_controller)

        self.get_logger().info(f"Activating controller: {start_controller}")
        req = SwitchController.Request()
        req.deactivate_controllers = []
        req.activate_controllers = [start_controller]
        req.strictness = 2
        future = self.switch_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()
        if not resp or not resp.ok:
            self.get_logger().error(f"Failed to activate {start_controller}.")
            return False
        self.get_logger().info(f"{start_controller} activated successfully.")

        # Publish zero velocity after switch
        self.publish_zero_velocity(0.5)
        self.vz_cmd = 0.0
        self.start_time = time.time()
        return True

    def ensure_controller_loaded(self, controller_name):
        list_client = self.create_client(ListControllers, '/controller_manager/list_controllers')
        list_client.wait_for_service()
        req = ListControllers.Request()
        future = list_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()
        if any(c.name == controller_name for c in resp.controller):
            self.get_logger().info(f"Controller '{controller_name}' already loaded.")
            return
        load_req = LoadController.Request()
        load_req.name = controller_name
        future = self.load_client.call_async(load_req)
        rclpy.spin_until_future_complete(self, future)
        load_resp = future.result()
        if load_resp.ok:
            self.get_logger().info(f"Controller '{controller_name}' loaded.")
        else:
            self.get_logger().error(f"Failed to load '{controller_name}'.")

    def configure_controller(self, controller_name):
        list_client = self.create_client(ListControllers, '/controller_manager/list_controllers')
        list_client.wait_for_service()
        req = ListControllers.Request()
        future = list_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()
        ctrl_state = None
        for c in resp.controller:
            if c.name == controller_name:
                ctrl_state = c.state
                break
        if ctrl_state == 'inactive':
            self.get_logger().info(f"Controller {controller_name} already inactive.")
            return
        try:
            subprocess.run(['ros2', 'control', 'set_controller_state', controller_name, 'inactive'], check=True)
            self.get_logger().info(f"{controller_name} configured to inactive.")
        except subprocess.CalledProcessError:
            self.get_logger().error(f"Failed to set {controller_name} inactive.")

    def publish_zero_velocity(self, duration=0.3):
        msg = TwistStamped()
        start = time.time()
        while time.time() - start < duration:
            self.pub.publish(msg)
            time.sleep(self.dt)


def main():
    rclpy.init()
    node = CartesianVelocityZ()
    # Swap to Cartesian velocity controller
    node.swap_controller(node.arm_controller, node.eef_velocity_controller)
    rclpy.spin(node)

if __name__ == "__main__":
    main()
