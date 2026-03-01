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
import json
import datetime
import matplotlib.pyplot as plt

from ament_index_python.packages import get_package_share_directory


def find_next_trial_number(base_dir, prefix="trial_"):
    existing_trials = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith(prefix)]

    if not existing_trials:
        return "trial_1"
    existing_numbers = []
    for d in existing_trials:
        num_part = d.replace(prefix, '').split('_')[0]  # remove prefix and anything after _
        if num_part.isdigit():
            existing_numbers.append(int(num_part))    
    
    return f"trial_{max(existing_numbers) + 1}" if existing_numbers else "trial_1"



def get_template_path():
    pkg_share = get_package_share_directory('lfd_apples')
    template_path = os.path.join(pkg_share, 'data', 'metadata_template.json')
    return template_path


def save_metadata(filename):
    """
    Create json file and save it with the same name as the bag file
    @param filename:
    @return:
    """
    
    # --- Open default metadata template for the experiment
    template_path = get_template_path()

    with open(template_path, 'r') as template_file:
        experiment_info = json.load(template_file)
    
    # Update some data
    experiment_info['general']['date'] = str(datetime.datetime.now())

    apple_id = input("Type the apple id: ")  # Wait for user to press Enter
    experiment_info['proxy']['apple']['id'] = apple_id

    spur_id = input("Type the spur id: ")  # Wait for user to press Enter
    experiment_info['proxy']['spur']['id'] = spur_id


    # --- Save metadata in file    
    with open(filename + '.json', "w") as outfile:
        json.dump(experiment_info, outfile, indent=4)


    # # --- Organize metatada
    # experiment_info = {
    #     "general": {
    #         "date": str(datetime.datetime.now()),
    #         "demonstrator": 'alejo',
    #         "experiment type": 'first_trials',                                    
    #         "pick pattern": 'pull_bend',            # twist, pull_straight, pull_bend
    #     },

    #     "robot": {
    #         "robot": 'Franka Arm',
    #         "gripper":{
    #             "type": 'Alejo tandem actuation gripper',
    #             "pressure @ valve": '65 PSI',
    #             "weight": '1300 g',
    #         },              
    #     },

    #     "proxy": {
    #         "branch":{
    #             "material": 'wood',
    #             "diameter": '25 mm',
    #             "length": '100 mm',
    #             "pose":{
    #                 "position": [0,0,0],
    #                 "orientation": [0,0,0],
    #             },
    #             "spring stiffness": 'high',        # soft, medium, high
    #         },
              
    #         "spur":{
    #             "material": 'TPU',
    #             "diameter": '10 mm',
    #             "length": '25 mm',
    #             "orientation": [0,0,0],
    #         },               

    #         "stem":{
    #             "material": 'steel cable',
    #             "diameter": '3 mm',
    #             "length": '10 mm',
    #             "magnet": "medium",             # low, medium, hard
    #         },
            
    #         "apple":{
    #             "mass": '173 g',
    #             "diameter": '80 mm',
    #             "height": '70 mm',
    #             "shape": 'round',                # round, oblong
    #             "pose": {
    #                 "position": [1,2,3],
    #                 "orientation": [0,0,0],
    #             },
    #         },            
    #     },

    #     "fixed camera": {
    #         "reference": 'Logitech, Inc. HD Webcam C615',
    #         "frame_id": 'robot base link',
    #         "position": [0.5, 0.0, 1.0],
    #         "orientation": [0, 0, 0],            
    #     },

    #     "results": {
    #         "success_approach": True,
    #         "success_grasp": True,
    #         "success_pick": True,
    #         "success_disposal": True,
    #         "comments": 'N/A',
    #     },      
        
    # }

    # # --- Save metadata in file
    # filename += ".json"
    # with open(filename, "w") as outfile:
    #     json.dump(experiment_info, outfile, indent=4)


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


def start_recording_bagfile(BAG_FILEPATH, arm_bag=True, inhand_camera_bag=True, fixed_camera_bag=True):

    bag_list = []

    BAG_NAME_MAIN = os.path.join(BAG_FILEPATH, "lfd_bag_main")
    BAG_NAME_PALM_CAMERA = os.path.join(BAG_FILEPATH, "lfd_bag_palm_camera")
    BAG_NAME_FIXED_CAMERA = os.path.join(BAG_FILEPATH, "lfd_bag_fixed_camera")

    timestamp = int(time.time() * 1000) % 100000  # unique suffix

    # --- STEP 2: Record ros2 bagfiles ---
    if arm_bag:        
        bag_proc_main = subprocess.Popen([
            "ros2", "bag", "record",
            "-o", BAG_NAME_MAIN,
            "/joint_states",
            "/franka_robot_state_broadcaster/current_pose",
            "/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame",
            "/franka_robot_state_broadcaster/external_joint_torques",
            "/franka_robot_state_broadcaster/desired_end_effector_twist",
            "/franka_robot_state_broadcaster/desired_joint_states",
            "/franka_robot_state_broadcaster/measured_joint_states",
            # "/franka_robot_state_broadcaster/robot_state",
            '/lfd/delta_twist_target',
            '/smoother/delta_twist_command',
            '/moveit2_servo/joint_vel_target',
            "microROS/sensor_data",              
        ],start_new_session=True, stdin=subprocess.DEVNULL)  # Detach from parent's stdin
        bag_list.append(bag_proc_main)

    if inhand_camera_bag:        
        bag_proc_palm_camera = subprocess.Popen([   
            "ros2", "bag", "record",
            "-b", "10000000000",
            "-o", BAG_NAME_PALM_CAMERA,
            'gripper/rgb_palm_camera/image_raw_with_artifacts', 
            # "gripper/rgb_palm_camera/image_raw",                 
        ],start_new_session=True, stdin=subprocess.DEVNULL)  # Detach from parent's stdin
        bag_list.append(bag_proc_palm_camera)    

    if fixed_camera_bag:        
        bag_proc_fixed_camera = subprocess.Popen([            
            "ros2", "bag", "record",
            "-b", "10000000000",
            "-o", BAG_NAME_FIXED_CAMERA,
            "fixed/rgb_camera/image_raw",                 
        ],start_new_session=True, stdin=subprocess.DEVNULL)  # Detach from parent's stdin
        bag_list.append(bag_proc_fixed_camera)

    time.sleep(1.0)   
    return bag_list


def stop_recording_bagfile(rosbag_list, timeout=5.0):
    """
    Stop all rosbag2 processes safely and wait for termination.
    """
    print("🛑 Stopping bag recordings...")

    # Graceful stop for known processes
    for proc in rosbag_list:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            proc.wait(timeout=timeout)
            print(f"✅ Recorder process {proc.pid} stopped.")
        except subprocess.TimeoutExpired:
            print(f"⚠️ Recorder {proc.pid} did not terminate in {timeout}s, killing.")
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            print(f"ℹ️ Recorder {proc.pid} already stopped.")

    # Fallback: ensure no rogue ros2 bag recorders remain
    time.sleep(0.5)
    os.system("pkill -f 'ros2 bag record' >/dev/null 2>&1")
    print("✅ All rosbag recordings stopped.")




def main():            

    input("\n\033[1;32m1 - Place apple on the proxy. Press ENTER when done.\033[0m\n")


    # STEP 1: Define trial filename
    BAG_DIR = os.path.expanduser("/media/alejo/Pruning25/03_IL_bagfiles/experiment_1_(robot)")
    # BAG_DIR = os.path.expanduser("/home/alejo/lfd_bags/experiment_1")

    os.makedirs(BAG_DIR, exist_ok=True)
    # Search directory for existing trials and create next trial number
    TRIAL = find_next_trial_number(BAG_DIR, prefix="trial_")


    # STEP 2: Start recording rosbags
    rosbag_list = start_recording_bagfile(BAG_DIR, TRIAL) 


    # STEP 3: Stop recording
    input("\n\033[1;32m2 - Now drive the arm to perform an apple-pick demonstration. Press ENTER when done.\033[0m\n")
    stop_recording_bagfile(rosbag_list)
    

    # --- STEP 4: Save metadata ---
    # print("Saving metadata...")
    try:
        save_metadata(os.path.join(BAG_DIR, TRIAL, "metadata_" + TRIAL))
        # print("✅ Metadata saved.")
    except Exception as e:
        print(f"❌ Error saving metadata: {e}")

    print("Extracting data and generating plots...")
    try:
        plt.ion()  # <-- interactive mode ON
        # extract_data_and_plot(os.path.join(BAG_DIR, TRIAL), "")
        extract_data_and_plot(BAG_DIR, TRIAL)
        print("✅ Data extraction complete.")

        # Prompt user to hit Enter to close figures
        input("\n\033[1;32m3 - Plots generated. Check how things look and press ENTER to close all figures.\033[0m\n")
        
        # Close all matplotlib figures
        plt.close('all')

        # Prompt user to take notes in YAML file if needed
        input("\n\033[1;32m3 - Annotate the yaml file if needed, and press ENTER to continue with next demo.\033[0m\n")


    except Exception as e:
        print(f"❌ Error during data extraction: {e}")
    

        

if __name__ == "__main__":
    main()

    
