#!/usr/bin/env python3
import subprocess
import time
import signal
import os
# ROS 2 imports
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Int16MultiArray


# Global variables
BAG_DIR = os.path.expanduser("~/franka_bags")
os.makedirs(BAG_DIR, exist_ok=True)
BAG_NAME_MAIN = os.path.join(BAG_DIR, "lfd_bag_main")
BAG_NAME_CAMERA = os.path.join(BAG_DIR, "lfd_bag_ihcamera")


# === Desired joint configuration ===
JOINT_NAMES = [
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7"
]

JOINT_POSITIONS = [
    1.4646141805298691,
    -0.9702388178201579,
    0.19573473944695943,
    -2.6398837719728743,
    0.19467791672497642,
    1.6778097848819793,
    0.7138629799943128,
]

class MoveToStart(Node):
    def __init__(self):
        super().__init__('move_to_start')
        self.publisher = self.create_publisher(JointTrajectory, '/fr3_arm_controller/joint_trajectory', 10)

    def go_to_position(self):
        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = JOINT_POSITIONS
        point.time_from_start = rclpy.duration.Duration(seconds=5.0).to_msg()  # 5s motion

        traj.points.append(point)
        self.publisher.publish(traj)
        self.get_logger().info("Sent robot to start position!")


def move_robot_to_start():
    rclpy.init()
    node = MoveToStart()

    # Give publisher time to connect
    time.sleep(1.0)

    node.go_to_position()

    # Allow time for execution
    time.sleep(6.0)

    node.destroy_node()
    rclpy.shutdown()


class SuctionMonitor(Node):
    def __init__(self, bag_proc_camera):
        super().__init__('suction_monitor')
        self.bag_proc_camera = bag_proc_camera
        self.sub = self.create_subscription(Int16MultiArray, 'microROS/sensor_data', self.sensor_callback, 10)
        self.engaged = False

    def sensor_callback(self, msg):
        pressures = list(msg.data)
        # Define your threshold for engagement (adjust as needed)
        ENGAGE_THRESHOLD = 600  # hPa
        all_engaged = all(p < ENGAGE_THRESHOLD for p in pressures)

        if all_engaged and not self.engaged:
            self.engaged = True
            self.get_logger().info("All suction cups engaged! Stopping camera bag recording.")
            try:
                self.bag_proc_camera.terminate()
                self.bag_proc_camera.wait()
                self.get_logger().info("Camera recording stopped.")
            except Exception as e:
                self.get_logger().error(f"Error stopping camera bag: {e}")



def main():
    rclpy.init()

    # === Step 1: Move robot to start ===
    # print("Moving robot to start configuration...")
    # move_robot_to_start()

    # === Step 2: Start Free Drive mode ===
    # Start Free Drive in background
    input("Hit Enter to start Free Drive...")    

    input("Hit Enter to start recording ROS 2 bag while in Free Drive...")    

    print("Recording started! Press Ctrl+C to stop.")


    # Start ROS 2 bag recording
    bag_proc_main = subprocess.Popen([
        "ros2", "bag", "record",
        "-o", BAG_NAME_MAIN,
        "/joint_states",
        "/franka_robot_state_broadcaster/current_pose",
        "/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame",
        "/franka_robot_state_broadcaster/robot_state",
        "microROS/sensor_data",            
    ])

    bag_proc_camera = subprocess.Popen([
        "ros2", "bag", "record",
        "-o", BAG_NAME_CAMERA,
        "gripper/rgb_palm_camera/image_raw",                  
    ])

    node = SuctionMonitor(bag_proc_camera)

        
    try:

        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.5)
    except KeyboardInterrupt:
        print("\nStopping recording..." )

    finally:
        node.destroy_node()
        rclpy.shutdown()


        # Terminate bag recording if still running
        try:
            bag_proc_main.terminate()
            bag_proc_main.wait()
        except:
            pass

        # Stop Free Drive
        print("Stopping Free Drive mode...")        
        print(f"Free Drive stopped. Bags saved in:\n  - {BAG_NAME_MAIN}\n  - {BAG_NAME_CAMERA}")

if __name__ == "__main__":
    main()
