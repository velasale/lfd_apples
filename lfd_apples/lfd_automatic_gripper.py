import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.task import Future

from std_msgs.msg import Int16MultiArray
from std_srvs.srv import SetBool
from geometry_msgs.msg import PoseStamped


class GripperController(Node):
    def __init__(self):
        super().__init__('gripper_controller')

        # Parameters
        self.declare_parameter('distance_threshold', 50)   # mm
        self.declare_parameter('pressure_threshold', 600)  # hPa
        self.declare_parameter('release_timer', 10)        # sec

        self.distance_threshold = self.get_parameter('distance_threshold').value
        self.pressure_threshold = self.get_parameter('pressure_threshold').value
        self.timer_value = self.get_parameter('release_timer').value

        # Subscribers
        self.distance_sub = self.create_subscription(
            Int16MultiArray, 'microROS/sensor_data', self.gripper_sensors_callback, 10)
        self.eef_pose_sub = self.create_subscription(
            PoseStamped, '/franka_robot_state_broadcaster/current_pose', self.eef_pose_callback, 10)

        # Service Clients
        self.valve_client = self.create_client(SetBool, 'microROS/toggle_valve')
        self.fingers_client = self.create_client(SetBool, 'microROS/move_stepper')

        while not self.valve_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for valve service...')
        while not self.fingers_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for fingers service...')

        self.get_logger().info("GripperController node started.")

        # Flags
        self.flag_distance = False
        self.flag_engagement = False
        self.flag_disposal = False
        self.flag_init = False
        self.cooldown = False  # NEW: prevents immediate rearming
        self.auto_off_timer = None

        # Parameters
        self.apple_disposal_coord = [-0.42, 0.56, 0.19]       
        self.disposal_range = 0.05

    # ----------------------- Helper to safely destroy timers -----------------------
    def destroy_timer_safe(self, attr_name: str):
        t = getattr(self, attr_name, None)
        if t is not None:
            try:
                t.cancel()
                self.destroy_timer(t)
            except Exception as e:
                self.get_logger().warn(f"Error destroying timer {attr_name}: {e}")
            setattr(self, attr_name, None)

    # ----------------------- Main Sensor Callback -----------------------
    def gripper_sensors_callback(self, msg: Int16MultiArray):
        if (not self.cooldown) and (not self.flag_init):
            self.get_logger().info("--- State 0 ---: Initialization: Valve OFF, Fingers IN")
            self.fingers_and_valve_reset()
            self.flag_init = True

        # Target close
        if (not self.cooldown) and (not self.flag_distance) and (0 < msg.data[3] < self.distance_threshold):
            self.get_logger().info(f"--- State 1 ---: Target close ({msg.data[3]} < {self.distance_threshold}), valve ON")
            req = SetBool.Request()
            req.data = True
            self.valve_client.call_async(req)
            self.flag_distance = True

        # Target moved away
        elif self.flag_distance and (msg.data[3] > self.distance_threshold):
            self.get_logger().info(f"--- State 0 ---: Target moved away ({msg.data[3]} > {self.distance_threshold}), resetting gripper")
            req = SetBool.Request()
            req.data = False
            self.valve_client.call_async(req)
            self.flag_distance = False
            self.destroy_timer_safe("auto_off_timer")

        # Check pressure → engage fingers & start timer
        if (not self.cooldown) and (not self.flag_engagement) and max(msg.data[:3]) < self.pressure_threshold:
            self.get_logger().info(f"--- State 2 ---: Pressures {msg.data[:3]} < {self.pressure_threshold}, cups engaged, deploying fingers")
            req = SetBool.Request()
            req.data = True
            self.fingers_client.call_async(req)
            self.flag_engagement = True

            # Start one-shot auto-off timer
            if self.auto_off_timer is None:
                self.get_logger().info(f"Starting auto-off timer ({self.timer_value} s)...")
                self.auto_off_timer = self.create_timer(self.timer_value, self.auto_off_callback)

    # ----------------------- Auto-off Timer -----------------------
    def auto_off_callback(self):
        self.get_logger().warn("--- State 3 ---: Auto-off triggered: Turning OFF valve and retracting fingers")

        # Destroy the timer so it only runs once
        self.destroy_timer_safe("auto_off_timer")

        # Reset system
        self.fingers_and_valve_reset()

    # ----------------------- EEF Pose Callback -----------------------
    def eef_pose_callback(self, msg: PoseStamped):
        eef_x, eef_y, eef_z = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z

        if (not self.cooldown) and (not self.flag_disposal) and (
            abs(eef_x - self.apple_disposal_coord[0]) < self.disposal_range and
            abs(eef_y - self.apple_disposal_coord[1]) < self.disposal_range and
            abs(eef_z - self.apple_disposal_coord[2]) < self.disposal_range
        ):
            self.get_logger().info("--- State 3 ---: End-effector in disposal zone.")
            self.flag_disposal = True
            self.destroy_timer_safe("auto_off_timer")
            self.fingers_and_valve_reset()

    # ----------------------- Reset Sequence -----------------------
    def fingers_and_valve_reset(self):
        self.get_logger().info("Resetting fingers and valve to initial state.")
        req = SetBool.Request()
        req.data = False
        future_fingers = self.fingers_client.call_async(req)
        future_fingers.add_done_callback(self._after_fingers_call_reset)

    def _after_fingers_call_reset(self, future_fingers: Future):
        try:
            response = future_fingers.result()
            if response.success:
                self.get_logger().info("Successful Fingers In.")
            else:
                self.get_logger().warn("Failed Fingers In.")
        except Exception as e:
            self.get_logger().error(f"Fingers In service call failed: {e}")
        self.delay_timer = self.create_timer(0.5, self._call_valve_after_reset)

    def _call_valve_after_reset(self):
        self.destroy_timer_safe("delay_timer")
        req_valve = SetBool.Request()
        req_valve.data = False
        future_valve = self.valve_client.call_async(req_valve)
        future_valve.add_done_callback(self._after_valve_call_reset)

    def _after_valve_call_reset(self, future_valve: Future):
        try:
            response = future_valve.result()
            if response.success:
                self.get_logger().info("Successful Valve Closed.")
            else:
                self.get_logger().warn("Failed Valve Closed.")
        except Exception as e:
            self.get_logger().error(f"Valve service call failed: {e}")
        self.delay_reset_timer = self.create_timer(0.5, self._after_reset_complete)

    def _after_reset_complete(self):
        self.destroy_timer_safe("delay_reset_timer")
        self.flag_distance = False
        self.flag_engagement = False
        self.flag_disposal = False
        self.destroy_timer_safe("auto_off_timer")

        # NEW: short cooldown to prevent immediate re-triggering
        self.cooldown = True
        self.get_logger().info("Cooldown active (2 s)...")
        self.cooldown_timer = self.create_timer(2.0, self._clear_cooldown)

    def _clear_cooldown(self):
        self.destroy_timer_safe("cooldown_timer")
        self.cooldown = False
        self.get_logger().info("Cooldown cleared, ready for next cycle.")


def main(args=None):
    rclpy.init(args=args)
    node = GripperController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
