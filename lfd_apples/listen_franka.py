#!/usr/bin/env python3
import subprocess
import time
import signal
import os

# Replace with your robot's IP and path to the compiled C++ Free Drive executable
ROBOT_IP = "192.168.1.11"

# Folder where bag will be saved
BAG_DIR = os.path.expanduser("~/franka_bags")
os.makedirs(BAG_DIR, exist_ok=True)
BAG_NAME = os.path.join(BAG_DIR, "franka_joint_bag")


def main():
    # Start Free Drive in background
    input("Hit Enter to start Free Drive...")

    

    try:
        input("Hit Enter to start recording ROS 2 bag while in Free Drive...")

        print("Recording started! Press Ctrl+C to stop.")
        # Start ROS 2 bag recording
        bag_proc = subprocess.Popen([
            "ros2", "bag", "record",
            "-o", BAG_NAME,
            "/NS_1/joint_states",
            "/NS_1/franka_robot_state_broadcaster/current_pose",
            "/NS_1/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame",
            "/NS_1/franka_robot_state_broadcaster/robot_state",
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
