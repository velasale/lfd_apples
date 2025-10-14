#!/usr/bin/env python3
import subprocess
import time
import signal
import os

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# Replace with your robot's IP and path to the compiled C++ Free Drive executable
ROBOT_IP = "192.168.1.11"

# Folder where bag will be saved
BAG_DIR = os.path.expanduser("~/franka_bags")
os.makedirs(BAG_DIR, exist_ok=True)
BAG_NAME = os.path.join(BAG_DIR, "franka_joint_bag")

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




def main():

    # === Step 1: Move robot to start ===
    # print("Moving robot to start configuration...")
    # move_robot_to_start()

    # === Step 2: Start Free Drive mode ===
    # Start Free Drive in background
    input("Hit Enter to start Free Drive...")

    

    try:
        input("Hit Enter to start recording ROS 2 bag while in Free Drive...")

        print("Recording started! Press Ctrl+C to stop.")
        # Start ROS 2 bag recording
        bag_proc = subprocess.Popen([
            "ros2", "bag", "record",
            "-o", BAG_NAME,
            "/joint_states",
            "/franka_robot_state_broadcaster/current_pose",
            "/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame",
            "/franka_robot_state_broadcaster/robot_state",
            "microROS/sensor_data",
        ])

        # Wait for the bag recording to finish (user presses Ctrl+C)
        bag_proc.wait()

    except KeyboardInterrupt:
        print("\nStopping recording...")

    finally:
        # Terminate bag recording if still running
        try:
            bag_proc.terminate()
            bag_proc.wait()
        except:
            pass

        # Stop Free Drive
        print("Stopping Free Drive mode...")        
        print(f"Free Drive stopped. Bag saved in: {BAG_NAME}")

if __name__ == "__main__":
    main()
