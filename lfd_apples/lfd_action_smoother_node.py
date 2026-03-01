#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
import numpy as np

class LfDActionSmoother(Node):
    """
    Smoothly ramps incoming TwistStamped commands to prevent abrupt jumps
    that trigger Franka reflexes.
    """

    def __init__(self):
        super().__init__('lfd_action_smoother')

        # --- Parameters ---
        self.declare_parameter('linear_acc_limit', 0.5)   # m/s^2
        self.declare_parameter('angular_acc_limit', 0.5)  # rad/s^2        
       
        # --- Limits ---
        self.MAX_LINEAR_ACC = self.get_parameter('linear_acc_limit').value
        self.MAX_ANGULAR_ACC = self.get_parameter('angular_acc_limit').value

        # --- State ---
        self.current_cmd = TwistStamped()
        self.target_cmd = TwistStamped()
        self.running_lfd = False
        self.start_time = None
        self.last_cmd_time = None

        # --- Publishers and Subscribers ---
        self.servo_pub = self.create_publisher(
            TwistStamped,
            '/smoother/delta_twist_command',
            10
        )        
        self.tgt_twist_sub = self.create_subscription(
            TwistStamped,
            '/lfd/delta_twist_target',
            self.tgt_twist_callback,
            10
        )

        # --- Timer ---       
        self.create_timer(0.002, self.publish_smoothed_velocity)

        self.get_logger().info("LfDActionSmoother initialized. Waiting for first target twist...")

    def tgt_twist_callback(self, msg: TwistStamped):
        """
        Receive target twist and enable ramping.
        """
        self.target_cmd = msg
        if not self.running_lfd:
            self.running_lfd = True
            self.start_time = self.get_clock().now()
            self.last_cmd_time = self.start_time
            # Initialize current_cmd to first received twist to avoid discontinuity
            self.current_cmd = msg
            self.get_logger().info("First target received. Ramping enabled.")

    def publish_smoothed_velocity(self):
        """
        Ramps current_cmd towards target_cmd while respecting acceleration limits.
        Prevents sending commands until first target is received.
        """
        if not self.running_lfd:
            return

        now = self.get_clock().now()
        dt = (now - self.last_cmd_time).nanoseconds * 1e-9
      

        if dt <= 0.001:
            return  # avoid division by zero
        self.last_cmd_time = now

        def s_ramp(current, target, max_acc, dt):
            # Maximum allowed change
            max_delta = max_acc * dt
            # Monotonic update: cannot go beyond target
            if target > current:
                current = min(current + max_delta, target)
            else:
                current = max(current - max_delta, target)
            return current

        # --- Linear ramping ---
        for axis in ['x', 'y', 'z']:
            current = getattr(self.current_cmd.twist.linear, axis)
            target = getattr(self.target_cmd.twist.linear, axis)
            setattr(self.current_cmd.twist.linear, axis,
                    s_ramp(current, target, self.MAX_LINEAR_ACC, dt))

        # --- Angular ramping ---
        for axis in ['x', 'y', 'z']:
            current = getattr(self.current_cmd.twist.angular, axis)
            target = getattr(self.target_cmd.twist.angular, axis)
            setattr(self.current_cmd.twist.angular, axis,
                    s_ramp(current, target, self.MAX_ANGULAR_ACC, dt))

        # Header
        self.current_cmd.header.stamp = now.to_msg()
        self.current_cmd.header.frame_id = "fr3_hand_tcp"

        # Publish
        self.servo_pub.publish(self.current_cmd)

def main(args=None):
    rclpy.init(args=args)
    node = LfDActionSmoother()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()