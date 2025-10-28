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
from lfd_apples.ros2bag2csv import extract_data_and_plot



def find_next_trial_number(base_dir, prefix="trial_"):
    existing_trials = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith(prefix)]

    if not existing_trials:
        return "trial_1"
    existing_numbers = [int(d.replace(prefix, '')) for d in existing_trials if d.replace(prefix, '').isdigit()]

    # print(existing_numbers)
    
    return f"trial_{max(existing_numbers) + 1}" if existing_numbers else "trial_1"


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
        
    
    # --- STEP 1: Define trial filename ---
    # Global variables
    BAG_DIR = os.path.expanduser("~/lfd_bags/experiment_1")
    os.makedirs(BAG_DIR, exist_ok=True)
    # Search directory for existing trials and create next trial number
    TRIAL = find_next_trial_number(BAG_DIR, prefix="trial_")

    BAG_NAME_MAIN = os.path.join(BAG_DIR, TRIAL, "lfd_bag_main")
    BAG_NAME_PALM_CAMERA = os.path.join(BAG_DIR, TRIAL, "lfd_bag_palm_camera")
    BAG_NAME_FIXED_CAMERA = os.path.join(BAG_DIR, TRIAL, "lfd_bag_fixed_camera")


    # --- STEP 2: Record ros2 bagfiles ---
    input("Hit Enter to start recording ROS 2 bag while in Free Drive...")    
    bag_proc_main = subprocess.Popen([
        "ros2", "bag", "record",
        "-o", BAG_NAME_MAIN,
        "/joint_states",
        "/franka_robot_state_broadcaster/current_pose",
        "/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame",
        # "/franka_robot_state_broadcaster/robot_state",
        "microROS/sensor_data",            
        "/franka/joint_states"
    ], preexec_fn=os.setsid)

    bag_proc_palm_camera = subprocess.Popen([
        "ros2", "bag", "record",
        "-o", BAG_NAME_PALM_CAMERA,
        "gripper/rgb_palm_camera/image_raw",     
    ], preexec_fn=os.setsid)

    bag_proc_fixed_camera = subprocess.Popen([
        "ros2", "bag", "record",
        "-o", BAG_NAME_FIXED_CAMERA,
        "fixed/rgb_camera/image_raw",     
    ], preexec_fn=os.setsid)

    print("Recording started! Press Ctrl+C to stop.")


    # --- STEP 3: Monitor engagement to finish In-Hand Camera file
    # node = SuctionMonitor(bag_proc_palm_camera)   

        
    try:

        # Uncommment to use ROS2 node for monitoring suction engagement
        # while rclpy.ok():
            # rclpy.spin_once(node, timeout_sec=0.5)

        while True:
            time.sleep(1.0)  


    except KeyboardInterrupt:
        print("\nStopping recording..." )

    finally:
        print("Stopping bag recordings...")

        for proc in [bag_proc_main, bag_proc_palm_camera, bag_proc_fixed_camera]:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=5)  # ✅ wait up to 5s, then continue
            except subprocess.TimeoutExpired:
                print(f"Process {proc.pid} taking too long, killing forcefully.")
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception as e:
                print(f"Error stopping process {proc.pid}: {e}")

        print("✅ Recordings stopped.")
        print(f"Bags saved in:\n  - {BAG_NAME_MAIN}\n  - {BAG_NAME_PALM_CAMERA}\n  - {BAG_NAME_FIXED_CAMERA}")

        print("Extracting data and generating plots...")
        try:
            extract_data_and_plot(os.path.join(BAG_DIR, TRIAL), "")
            print("✅ Data extraction complete.")
        except Exception as e:
            print(f"❌ Error during data extraction: {e}")

        rclpy.shutdown()

        

if __name__ == "__main__":
    main()

    
