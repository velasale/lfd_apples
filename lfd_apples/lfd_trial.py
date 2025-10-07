import rclpy
from rclpy.node import Node

from std_msgs.msg import Int16MultiArray
from std_srvs.srv import SetBool   # Change to your actual service type if different


class GripperController(Node):
    def __init__(self):
        super().__init__('gripper_controller')

        # Parameters
        self.declare_parameter('distance_threshold', 50)   # mm
        self.declare_parameter('pressure_threshold', 600)  # hPa
        self.declare_parameter('release_timer', 10)  # sec

        self.distance_threshold = self.get_parameter('distance_threshold').value
        self.pressure_threshold = self.get_parameter('pressure_threshold').value
        self.timer_value = self.get_parameter('release_timer').value

        # Subscribers
        self.distance_sub = self.create_subscription(Int16MultiArray, 'microROS/sensor_data', self.gripper_sensors_callback, 10)
                
        # Service Clients
        self.valve_client = self.create_client(SetBool, 'microROS/toggle_valve')
        self.fingers_client = self.create_client(SetBool, 'microROS/move_stepper')

        # Wait for services
        while not self.valve_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for turn_valve_on service...')
        while not self.fingers_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for fingers_out service...')

        self.get_logger().info("GripperController node started.")

        # Flags
        self.flag_distance = False
        self.flag_engage = False
        self.flag_init = True
        self.auto_off_timer = None

    def gripper_sensors_callback(self, msg: Int16MultiArray):

        if self.flag_init:
            req = SetBool.Request()
            req.data = False
            self.valve_client.call_async(req)
            self.fingers_client.call_async(req)
            self.flag_init = False
            self.get_logger().info("Initialization: Valve OFF, Fingers IN")
            return
        
        if not self.flag_distance and msg.data[3] < self.distance_threshold:
            self.get_logger().info(f"Distance {msg.data[3]} < {self.distance_threshold}, calling valve ON")
            req = SetBool.Request()
            req.data = True
            self.valve_client.call_async(req)
            self.flag_distance = True

        # Check Pressure
        if not self.flag_engage and max(msg.data[:3]) < self.pressure_threshold:
            self.get_logger().info(f"Pressures {msg.data[0], msg.data[1], msg.data[2]} < {self.pressure_threshold}, cups engaged, deploying fingers")
            req = SetBool.Request()
            req.data = True
            self.fingers_client.call_async(req)
            self.flag_engage = True

            # Start auto-off timer once fingers are engaged
            if self.auto_off_timer is None:
                self.get_logger().info(f"Starting auto-off timer ({self.timer_value} s)...")
                self.auto_off_timer = self.create_timer(self.timer_value, self.auto_off_callback)

    def auto_off_callback(self):
        """Called by timer to switch off valve and retract fingers."""
        self.get_logger().warn("Auto-off triggered: Turning OFF valve and retracting fingers")

        req = SetBool.Request()
        req.data = False
        self.valve_client.call_async(req)
        self.fingers_client.call_async(req)

        # TODO: FIX this
        # Reset state
        # self.flag_distance = False
        # self.flag_engage = False

        # Destroy timer so it only runs once
        if self.auto_off_timer is not None:
            self.auto_off_timer.cancel()
            self.auto_off_timer = None

        
   


def main(args=None):
    rclpy.init(args=args)
    node = GripperController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
