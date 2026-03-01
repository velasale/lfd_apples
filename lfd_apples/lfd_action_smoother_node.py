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
       
        self.MAX_LINEAR_ACC  = 0.5  #m/s²
        self.MAX_LINEAR_JERK = 1.0  #m/s³

        self.MAX_ANGULAR_ACC  = 0.5 #rad/s²
        self.MAX_ANGULAR_JERK = 1.0 #rad/s³


        # --- State ---
        self.current_cmd = TwistStamped()
        self.target_cmd = TwistStamped()
        self.running_lfd = False
        self.start_time = None
        self.last_cmd_time = None

        self.current_acc = TwistStamped().twist

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
        self.create_timer(0.034, self.publish_smoothed_velocity_linear_ramp)

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



    def publish_smoothed_velocity_linear_ramp(self):
        """
        Ramps current_cmd towards target_cmd while respecting acceleration limits.
        Prevents sending commands until first target is received.
        """
        if not self.running_lfd:
            return

        now = self.get_clock().now()
        dt = (now - self.last_cmd_time).nanoseconds * 1e-9
      

        if dt <= 0.004:
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



    # def publish_smoothed_velocity_s_ramp(self):
    #     """
    #     Ramps current_cmd towards target_cmd while respecting acceleration limits.
    #     Prevents sending commands until first target is received.
    #     """
    #     if not self.running_lfd:
    #         return

    #     now = self.get_clock().now()
    #     dt = (now - self.last_cmd_time).nanoseconds * 1e-9
      

    #     if dt <= 0.001:
    #         return  # avoid division by zero
    #     self.last_cmd_time = now

    #     def s_ramp(current_v, target_v, current_a, max_acc, max_jerk, dt):
    #         """
    #         Jerk-limited S-curve ramp.
    #         Returns (new_velocity, new_acceleration)
    #         """

    #         # Velocity error
    #         error = target_v - current_v

    #         # Desired acceleration direction
    #         desired_a = max_acc if error > 0 else -max_acc

    #         # Jerk limit: move acceleration toward desired_a
    #         max_delta_a = max_jerk * dt

    #         if desired_a > current_a:
    #             current_a = min(current_a + max_delta_a, desired_a)
    #         else:
    #             current_a = max(current_a - max_delta_a, desired_a)

    #         # Integrate acceleration into velocity
    #         new_v = current_v + current_a * dt

    #         # Do not overshoot target
    #         if (error > 0 and new_v > target_v) or (error < 0 and new_v < target_v):
    #             new_v = target_v
    #             current_a = 0.0  # stop accelerating at target

    #         return new_v, current_a

    #     # --- Linear ---
    #     for axis in ['x', 'y', 'z']:
    #         current_v = getattr(self.current_cmd.twist.linear, axis)
    #         target_v  = getattr(self.target_cmd.twist.linear, axis)
    #         current_a = getattr(self.current_acc.linear, axis)

    #         new_v, new_a = s_ramp(
    #             current_v,
    #             target_v,
    #             current_a,
    #             self.MAX_LINEAR_ACC,
    #             self.MAX_LINEAR_JERK,
    #             dt
    #         )

    #         setattr(self.current_cmd.twist.linear, axis, new_v)
    #         setattr(self.current_acc.linear, axis, new_a)

    #     # --- Angular ---
    #     for axis in ['x', 'y', 'z']:
    #         current_v = getattr(self.current_cmd.twist.angular, axis)
    #         target_v  = getattr(self.target_cmd.twist.angular, axis)
    #         current_a = getattr(self.current_acc.angular, axis)

    #         new_v, new_a = s_ramp(
    #             current_v,
    #             target_v,
    #             current_a,
    #             self.MAX_ANGULAR_ACC,
    #             self.MAX_ANGULAR_JERK,
    #             dt
    #         )

    #         setattr(self.current_cmd.twist.angular, axis, new_v)
    #         setattr(self.current_acc.angular, axis, new_a)

    #     # Header
    #     self.current_cmd.header.stamp = now.to_msg()
    #     self.current_cmd.header.frame_id = "fr3_hand_tcp"

    #     # Publish
    #     self.servo_pub.publish(self.current_cmd)





def main(args=None):
    rclpy.init(args=args)
    node = LfDActionSmoother()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()