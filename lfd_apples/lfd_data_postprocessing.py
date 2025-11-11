import os
import pandas as pd
import ast
import re

def parse_array_string(s):
    """
    Safely extract the list inside array('h', [ ... ]) and return it as a Python list.
    """
    if not isinstance(s, str):
        return s
    match = re.search(r"\[(.*?)\]", s)
    if match:
        inner = "[" + match.group(1) + "]"
        return ast.literal_eval(inner)
    return None


def downsample_pressure_and_tof_data(raw_data_path, destination_path):
    
    df_raw = pd.read_csv(raw_data_path)    
    df_raw["_data_as_list"] = df_raw["_data"].apply(parse_array_string)

    # Split that list into multiple independent columns
    data_expanded = pd.DataFrame(df_raw["_data_as_list"].tolist(), columns=["scA", "scB", "scC", "tof"])

    # Combine with the rest of the dataframe
    df_final = pd.concat([df_raw, data_expanded], axis=1)

    pass

def downsample_inhand_camera_raw_images(raw_data_path, destination_path):
    # Implement downsampling logic here
    pass    

def downsample_eef_wrench_data(raw_data_path, destination_path):
    # Implement downsampling logic here

    # filter data first

    # then downsample
    pass    

def downsample_robot_joint_states_data(raw_data_path, destination_path):
    # Implement downsampling logic here
    pass

def downsample_robot_ee_pose_data(raw_data_path, destination_path):
    # Implement downsampling logic here
    pass

def estimate_robot_ee_pose():
    pass




def main():

    # ---------- Step 1: Load raw data ----------
    MAIN_DIR = os.path.join("D:")                                   # windows OS
    # MAIN_DIR = os.path.join('media', 'alejo', 'Pruning25')        # ubuntu OS
    SOURCE_DIR = os.path.join(MAIN_DIR, "01_IL_bagfiles")
    DESTINATION_DIR = os.path.join(MAIN_DIR, "02_IL_postprocessed")

    EXPERIMENT = "experiment_4"

    SOURCE_PATH = os.path.join(SOURCE_DIR, EXPERIMENT)
    DESTINATION_PATH = os.path.join(DESTINATION_DIR, EXPERIMENT)
    FIXED_CAM_SUBDIR = os.path.join("robot", "lfd_bag_fixed_camera", "camera_frames", "fixed_rgb_camera_image_raw")
    INHAND_CAM_SUBDIR = os.path.join("robot", "lfd_bag_palm_camera", "camera_frames", "gripper_rgb_palm_camera_image_raw")
    ARM_SUBDIR = os.path.join("robot", "lfd_bag_main", "bag_csvs")
    GRIPPER_SUBDIR = os.path.join("robot", "lfd_bag_main", "bag_csvs")

    trials = [trial for trial in os.listdir(SOURCE_PATH) if os.path.isdir(os.path.join(SOURCE_PATH, trial))]

    # ---------- Step 2: Loop through all trials ----------
    for trial in trials:
        
        # Define paths to all raw data
        raw_pressure_and_tof_path = os.path.join(SOURCE_PATH, trial, GRIPPER_SUBDIR, "microROS_sensor_data.csv")
        raw_eef_wrench_path = os.path.join(SOURCE_PATH, trial, ARM_SUBDIR, "franka_robot_state_broadcaster_external_wrench_in_stiffness_frame.csv")
        raw_joint_states_path = os.path.join(SOURCE_PATH, trial, ARM_SUBDIR, "franka_robot_state_broadcaster_measured_joint_states.csv")
        raw_ee_pose_path = os.path.join(SOURCE_PATH, trial, ARM_SUBDIR, "franka_robot_state_broadcaster_current_pose.csv")
        
        # Define paths to all processed data
        post_pressure_and_tof_path = os.path.join(DESTINATION_PATH, trial, GRIPPER_SUBDIR, "microROS_sensor_data.csv")
        post_eef_wrench_path = os.path.join(DESTINATION_PATH, trial, ARM_SUBDIR, "franka_robot_state_broadcaster_external_wrench_in_stiffness_frame.csv")
        post_joint_states_path = os.path.join(DESTINATION_PATH, trial, ARM_SUBDIR, "franka_robot_state_broadcaster_measured_joint_states.csv")
        post_ee_pose_path = os.path.join(DESTINATION_PATH, trial, ARM_SUBDIR, "franka_robot_state_broadcaster_current_pose.csv")

        # Downsample data
        downsample_pressure_and_tof_data(raw_pressure_and_tof_path, post_pressure_and_tof_path)

    # Step 2: Downsample data and make all channels the same length

    # Step 3: Crop data for each phase


    # Plot data before and after downsampling to double check everything is fine
    # Unit test!!!

    print('done!')


if __name__ == '__main__':
    main()
