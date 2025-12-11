import os
import platform
import pandas as pd
import ast
import re
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from lfd_vision import extract_pooled_latent_vector
from ultralytics import YOLO
import cv2

from scipy.ndimage import gaussian_filter, median_filter, gaussian_filter1d

from ros2bag2csv import plot_pressure, plot_wrench
from pathlib import Path
import yaml
from pathlib import Path


# ====================== Handy functions =======================
def rename_folder(SOURCE_PATH, start_index=100):
    """ Rename trials"""

    # Sort by time of creation
    trials = sorted(
        [trial for trial in os.listdir(SOURCE_PATH) if os.path.isdir(os.path.join(SOURCE_PATH, trial))],
        key=lambda x: os.path.getctime(os.path.join(SOURCE_PATH, x))
    )

    for trial in trials:

        subfolder_old_name = trial
        subfolder_new_name = "trial_" + str(start_index)
        print(subfolder_old_name, subfolder_new_name)

        OLD_PATH = os.path.join(SOURCE_PATH, subfolder_old_name)
        NEW_PATH = os.path.join(SOURCE_PATH, subfolder_new_name)

        for item in os.listdir(os.path.join(SOURCE_PATH, trial)):
            if item.endswith("json"):
                metadata_old_name = item

        # Rename metadata json file
        metadata_new_name = "metadata_trial_" + str(start_index) + ".json"
        METADATA_OLD_PATH = os.path.join(OLD_PATH, metadata_old_name)
        METADATA_NEW_PATH = os.path.join(OLD_PATH, metadata_new_name)

        # Rename file and subfolder
        print('Be cautious about using this method')
        # os.rename(METADATA_OLD_PATH, METADATA_NEW_PATH)
        # os.rename(OLD_PATH, NEW_PATH)

        start_index += 1


def interpolate_to_reference_multi(df_values, df_ref, ts_col_values, ts_col_ref, method="linear"):
    """
    Interpolate multiple numeric columns in df_values to the timestamps in df_ref.
    This version avoids reindexing, so no NaN values appear.
    """

    # Extract timestamp arrays
    src_ts = df_values[ts_col_values].values
    ref_ts = df_ref[ts_col_ref].values

    # Identify numeric columns to interpolate
    numeric_cols = df_values.select_dtypes(include=[np.number]).columns.tolist()

    # Remove the timestamp column itself
    numeric_cols.remove(ts_col_values)

    # Create output DataFrame
    df_out = pd.DataFrame({ts_col_ref: ref_ts})

    # Interpolate every numeric column using numpy
    for col in numeric_cols:
        df_out[col] = np.interp(ref_ts, src_ts, df_values[col].values)

    return df_out


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


def get_phase_columns(phase_name):

    data_columns_path = config_path = Path(__file__).parent / "config" / "lfd_data_columns.yaml"

    with open(data_columns_path, "r") as f:
        cfg = yaml.safe_load(f)

    
    # In hand Camera Feature count
    out = {}
    for key, val in cfg.items():
        if isinstance(val, dict) and "prefix" in val:
            out[key] = [f"{val['prefix']}{i}" for i in range(1, val["count"] + 1)]
        else:
            out[key] = val
    

    groups = cfg["phases"][phase_name]
    return [col for group in groups for col in out[group]]


# ============ Topic-specific downsampling functions ===========
def downsample_pressure_and_tof_data(df, raw_data_path, compare_plots=True):
    
    df_raw = pd.read_csv(raw_data_path)    
    df_raw["_data_as_list"] = df_raw["_data"].apply(parse_array_string)

    # Split that list into multiple independent columns
    data_expanded = pd.DataFrame(df_raw["_data_as_list"].tolist(), columns=["scA", "scB", "scC", "tof"])

    # Combine with the rest of the dataframe
    df_final = pd.concat([df_raw, data_expanded], axis=1)
    df_final.drop(columns=["_data", "_data_as_list", "timestamp", "_layout._data_offset"], inplace=True)


    # Interpolate to reference timestamps
    df_downsampled = interpolate_to_reference_multi(df_final, df, ts_col_values="elapsed_time", ts_col_ref="timestamp_vector", method="linear")

    if compare_plots:
        # Compare plots before and after downsampling
        # Ensure numeric 1-D arrays
        plt.figure()    

        x = np.array(df_final['elapsed_time']).flatten()
        y = np.array(df_final['scA']).flatten()
        plt.plot(x, y, label='Original scA')

        x_ds = np.array(df_downsampled['timestamp_vector']).flatten()
        y_ds = np.array(df_downsampled['scA']).flatten()
        plt.plot(x_ds, y_ds, label='Downsampled scA', linestyle='--')
        
        plt.legend()
        plt.title('Pressure Sensor A Before and After Downsampling')        

    return df_downsampled


def reduce_size_inhand_camera_raw_images(raw_data_path, layer=12):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    pt_path = os.path.join(script_dir, "resources", "best_segmentation.pt")

    model = YOLO(pt_path)
    rows = []

    for fname in sorted(os.listdir(raw_data_path)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(raw_data_path, fname)
        img_cv = cv2.imread(image_path)

        if img_cv is None:
            print(f"Could not read {image_path}, skipping.")
            continue

        pooled_vector, feat_map = extract_pooled_latent_vector(
            img_cv,
            model,
            layer_index=layer
        )

        # build row: first filename, then 256 feature values
        row = [fname] + pooled_vector.tolist()
        rows.append(row)

        # print(fname, pooled_vector.shape)
    
    feature_dim = len(rows[0]) - 1  # typically 256
    columns = ["filename"] + [f"f{i}" for i in range(feature_dim)]
    df = pd.DataFrame(rows, columns=columns)

    df.drop(columns=["filename"], inplace=True)

    return df
      

def downsample_eef_wrench_data(df, raw_data_path, compare_plots=True):

    df_raw = pd.read_csv(raw_data_path)    
    # df_raw["_data_as_list"] = df_raw["_data"].apply(parse_array_string)

    # Split that list into multiple independent columns
    # data_expanded = pd.DataFrame(df_raw["_data_as_list"].tolist(), columns=["scA", "scB", "scC", "tof"])

    # Combine with the rest of the dataframe
    df_final = pd.concat([df_raw], axis=1)
    df_final.drop(columns=["_header._stamp._sec", "_header._stamp._nanosec", "_header._frame_id", "timestamp"], inplace=True)


    # Interpolate to reference timestamps
    df_downsampled = interpolate_to_reference_multi(df_final, df, ts_col_values="elapsed_time", ts_col_ref="timestamp_vector", method="linear")

    if compare_plots:
        # Compare plots before and after downsampling
        # Ensure numeric 1-D arrays
        plt.figure()    

        x = np.array(df_final['elapsed_time']).flatten()
        y = np.array(df_final['_wrench._force._z']).flatten()
        plt.plot(x, y, label='Original wrench Force Z', alpha=0.5)

        x_ds = np.array(df_downsampled['timestamp_vector']).flatten()
        y_ds = np.array(df_downsampled['_wrench._force._z']).flatten()
        plt.plot(x_ds, y_ds, label='Downsampled wrench Force z')
        
        plt.legend()
        plt.title('Wrench Force XBefore and After Downsampling')        

    return df_downsampled


def downsample_robot_joint_states_data(df, raw_data_path, compare_plots=True):

    df_raw = pd.read_csv(raw_data_path)        

    df_raw["_position_as_list"] = df_raw["_position"].apply(parse_array_string)
    df_raw["_velocity_as_list"] = df_raw["_velocity"].apply(parse_array_string)
    df_raw["_effort_as_list"] = df_raw["_effort"].apply(parse_array_string)

    # Split that list into multiple independent columns, and just take the first 7 joints
    pos_expanded = pd.DataFrame(df_raw["_position_as_list"].apply(lambda x: x[:7]).tolist(), columns=["pos_joint_1", "pos_joint_2", "pos_joint_3", "pos_joint_4", "pos_joint_5", "pos_joint_6", "pos_joint_7"])
    vel_expanded = pd.DataFrame(df_raw["_velocity_as_list"].apply(lambda x: x[:7]).tolist(), columns=["vel_joint_1", "vel_joint_2", "vel_joint_3", "vel_joint_4", "vel_joint_5", "vel_joint_6", "vel_joint_7"])
    eff_expanded = pd.DataFrame(df_raw["_effort_as_list"].apply(lambda x: x[:7]).tolist(), columns=["eff_joint_1", "eff_joint_2", "eff_joint_3", "eff_joint_4", "eff_joint_5", "eff_joint_6", "eff_joint_7"])  

    # Combine with the rest of the dataframe
    df_final = pd.concat([df_raw["elapsed_time"], pos_expanded, vel_expanded, eff_expanded], axis=1)
    # df_final.drop(columns=["_header._stamp._sec", "_header._stamp._nanosec", "_header._frame_id", "timestamp"], inplace=True)

    # Interpolate to reference timestamps
    df_downsampled = interpolate_to_reference_multi(df_final, df, ts_col_values="elapsed_time", ts_col_ref="timestamp_vector", method="linear")

    if compare_plots:
        # Compare plots before and after downsampling
        # Ensure numeric 1-D arrays
        plt.figure()    

        x = np.array(df_final['elapsed_time']).flatten()
        y = np.array(df_final['pos_joint_1']).flatten()
        plt.plot(x, y, label='Original pos joint 1')

        x_ds = np.array(df_downsampled['timestamp_vector']).flatten()
        y_ds = np.array(df_downsampled['pos_joint_1']).flatten()
        plt.plot(x_ds, y_ds, label='Downsampled pos joint 1', linestyle='--')
        
        plt.legend()
        plt.title('Pos joint 1 Before and After Downsampling')        

    return df_downsampled


def downsample_robot_ee_pose_data(df, raw_data_path, compare_plots=True):

    df_raw = pd.read_csv(raw_data_path)        
    
    # Combine with the rest of the dataframe
    df_final = pd.concat([df_raw], axis=1)
    df_final.drop(columns=["_header._stamp._sec", "_header._stamp._nanosec", "_header._frame_id", "timestamp"], inplace=True)

    # Interpolate to reference timestamps
    df_downsampled = interpolate_to_reference_multi(df_final, df, ts_col_values="elapsed_time", ts_col_ref="timestamp_vector", method="linear")

    if compare_plots:
        # Compare plots before and after downsampling
        # Ensure numeric 1-D arrays
        plt.figure()    

        x = np.array(df_final['elapsed_time']).flatten()
        y = np.array(df_final['_pose._position._x']).flatten()
        plt.plot(x, y, label='Original pose position x')

        x_ds = np.array(df_downsampled['timestamp_vector']).flatten()
        y_ds = np.array(df_downsampled['_pose._position._x']).flatten()
        plt.plot(x_ds, y_ds, label='Downsampled pose position x', linestyle='--')
        
        plt.legend()
        plt.title('Pose position x Before and After Downsampling')        

    return df_downsampled


def get_timestamp_vector_from_images(image_folder_path):
    """Get timestamp vector from images in a folder where filenames end in '_<timestamp>'."""

    image_filenames = sorted(os.listdir(image_folder_path))
    timestamps = []

    for filename in image_filenames:
        if filename.endswith('.png') or filename.endswith('.jpg'):
            name_without_ext = os.path.splitext(filename)[0]

            # Split by '_' and take the last part (the timestamp)
            parts = name_without_ext.split('_')
            timestamp_str = parts[-1]

            try:
                timestamp = float(timestamp_str)
                timestamps.append(timestamp)
            except ValueError:
                # Skip files that don't match expected pattern
                continue

    return timestamps


# =================== Action Space derivation ===================
def derive_actions_from_ee_pose(reference_df, raw_data_path, sigma=100, compare_plots=True):
    """ Applies a gaussian filter, Derive actions based on end-effector pose changes over time, and downsamples. 
    Actions are computed as differences in position and orientation over time.
    """

    df_raw = pd.read_csv(raw_data_path)        
    
    # Combine with the rest of the dataframe
    df_final = pd.concat([df_raw], axis=1)
    df_final.drop(columns=["_header._stamp._sec", "_header._stamp._nanosec", "_header._frame_id", "timestamp"], inplace=True)

    # Compute actions as differences in position and orientation
    positions = df_final[['_pose._position._x', '_pose._position._y', '_pose._position._z']].values
    orientations = df_final[['_pose._orientation._x', '_pose._orientation._y', '_pose._orientation._z', '_pose._orientation._w']].values

    # Filter signals if needed (e.g., smoothing) - optional step
    sigma = 100  # adjust for more/less smoothing
    filtered_positions = gaussian_filter1d(positions, sigma=sigma, axis=0)
    filtered_orientations = gaussian_filter1d(orientations, sigma=sigma, axis=0)  

    # Compute deltas
    delta_times = np.diff(df_final['elapsed_time'].values, prepend=df_final['elapsed_time'].values[0])
    # Avoid division by zero for first entry (set to 1 so speed = 0/1 = 0)
    delta_times[delta_times == 0] = 1e-9   # or 1.0 if timestamps are stable
    delta_positions = np.diff(filtered_positions, axis=0, prepend=filtered_positions[0:1, :])
    delta_orientations = np.diff(filtered_orientations, axis=0, prepend=filtered_orientations[0:1, :])

    # Compute speeds (m/s)
    linear_speeds = delta_positions / delta_times[:, None]
    orientation_speeds = delta_orientations / delta_times[:, None]
    # Create a new DataFrame for actions
    action_columns = ['delta_pos_x', 'delta_pos_y', 'delta_pos_z', 'delta_ori_x', 'delta_ori_y', 'delta_ori_z', 'delta_ori_w']
    actions_df = pd.DataFrame(np.hstack((linear_speeds, orientation_speeds)), columns=action_columns)
    # Copy the elapsed_time column
    actions_df['elapsed_time'] = df_final['elapsed_time'].values

    # Interpolate to reference timestamps
    df_downsampled = interpolate_to_reference_multi(actions_df, reference_df, ts_col_values="elapsed_time", ts_col_ref="timestamp_vector", method="linear")

    if compare_plots:
        # Compare plots before and after downsampling
        # Ensure numeric 1-D arrays
        plt.figure()    

        x_ds = np.array(df_downsampled['timestamp_vector']).flatten()
        y_ds = np.array(df_downsampled['delta_pos_x']).flatten()
        plt.plot(x_ds, y_ds, label='Downsampled Action delta pos x')

        x_ds = np.array(actions_df['elapsed_time']).flatten()
        y_ds = np.array(actions_df['delta_pos_x']).flatten()
        plt.plot(x_ds, y_ds, label='Original Action delta pos x', alpha=0.5)
        
        plt.legend()
        plt.title('Action delta pos x Over Time')        

    return df_downsampled


# =============== Crop functions for each Phase =================
def find_end_of_phase_1_approach(df, trial, tof_threshold=50):

    tof_values = df['tof'].values
    tof_filtered = gaussian_filter(tof_values, 3)
   

    # Create boolean mask: True if tof < threshold
    below_threshold = tof_filtered < tof_threshold

    # Detect transition: previous >= threshold, current < threshold
    transition_indices = np.where((~below_threshold[:-1]) & (below_threshold[1:]))[0] + 1

    if len(transition_indices) == 0:
        plot_pressure(df, time_vector='timestamp_vector')        
        print(f'No contact detected in {trial}, skipping cropping.')
        
        return None

    if len(transition_indices) > 1:

        # return "Multiple"       # TODO remove this line

        plot_pressure(df, time_vector='timestamp_vector')
        print(f'Multiple contact points detected in {trial}, needs attention.')        

        for index_phase_1_end in transition_indices:
            time_phase_1_end = df['timestamp_vector'].values[index_phase_1_end]     
            plt.axvline(x=time_phase_1_end, color='red', linestyle='--', label='Phase 1 End')        

        # Ask the user which contact point to pick
        plt.show()
        user_input = input(f"Enter index (0-{len(transition_indices)-1}) of correct phase 1 end: ")
        
        chosen_idx = int(user_input)
        idx_phase_1_end = transition_indices[chosen_idx]
    
                  
    if len(transition_indices) == 1:
        # Index of phase 1 end (first drop below threshold)
        idx_phase_1_end = transition_indices[0]

    return idx_phase_1_end


def find_end_of_phase_2_contact(df, trial, air_pressure_threshold=600, n_cups=2):

    scA_values = df['scA'].values
    scB_values = df['scB'].values
    scC_values = df['scC'].values

    scA_filtered = gaussian_filter(scA_values, 3)
    scB_filtered = gaussian_filter(scB_values, 3)
    scC_filtered = gaussian_filter(scC_values, 3)

    mask_A = scA_filtered < air_pressure_threshold
    mask_B = scB_filtered < air_pressure_threshold
    mask_C = scC_filtered < air_pressure_threshold

    # Count how many suction cups are below threshold at each time step
    num_below = mask_A.astype(int) + mask_B.astype(int) + mask_C.astype(int)

    # Indices where at least 2 cups are below threshold
    indices = np.where(num_below >= 2)[0]  

    if len(indices) == 0:
        plot_pressure(df, time_vector='timestamp_vector')        
        print(f'No engagement detected in {trial}, skipping cropping.')       

        plt.show()        
        return None

    idx_phase_2_end = indices[0]

    # plot_pressure(df, time_vector='timestamp_vector')
    # print(f'Index at which at least two suction cups engage in {trial}.')        
    # time_phase_2_end = df['timestamp_vector'].values[idx_phase_2_end]  
    # plt.axvline(x=time_phase_2_end, color='red', linestyle='--', label='Phase 2 End')        
    

    return idx_phase_2_end


def find_end_of_phase_3_contact(df, trial, total_force_threshold=20):
    
    fx = df['_wrench._force._x'].values
    fy = df['_wrench._force._y'].values
    fz = df['_wrench._force._z'].values
    tx = df['_wrench._torque._x'].values
    ty = df['_wrench._torque._y'].values
    tz = df['_wrench._torque._z'].values

    wrench = [fx, fy, fz, tx, ty, tz]
    t = np.array(df['timestamp_vector'].to_list(), dtype=float)

    # Apply median filter to smooth the signals
    wrench_filtered = [gaussian_filter(w, 3) for w in wrench]
    
    # Net Forces
    net_force = np.sqrt(wrench_filtered[0]**2 + wrench_filtered[1]**2 + wrench_filtered[2]**2)

    max_force_idx = np.argmax(net_force)
    time_phase_3_end = df['timestamp_vector'].values[max_force_idx]  
       
    # fig = plt.figure()
    # plt.plot(t, wrench_filtered[0], label='fx')
    # plt.plot(t, wrench_filtered[1], label='fy')
    # plt.plot(t, wrench_filtered[2], label='fz')
    # plt.axvline(x=time_phase_3_end, color='red', linestyle='--', label='Phase 3 End')        
    # plt.plot(t, net_force, label='net')
    # plt.legend()
    # plt.show()

    return max_force_idx



# ================ Main stages of data preprocessing ============
def stage_1_align_and_downsample():

    # ---------- Step 1: Load raw data ----------
    # MAIN_DIR = os.path.join("D:")                                   # windows OS
    MAIN_DIR = os.path.join('/media', 'alejo', 'IL_data')        # ubuntu OS
    SOURCE_DIR = os.path.join(MAIN_DIR, "01_IL_bagfiles")    
    # EXPERIMENT = "experiment_1_(pull)"
    EXPERIMENT = "only_human_demos/with_palm_cam"   
    SOURCE_PATH = os.path.join(SOURCE_DIR, EXPERIMENT)

    demonstrator = ""  # "human" or "robot"
    FIXED_CAM_SUBDIR = os.path.join(demonstrator, "lfd_bag_fixed_camera", "camera_frames", "fixed_rgb_camera_image_raw")
    INHAND_CAM_SUBDIR = os.path.join(demonstrator, "lfd_bag_palm_camera", "camera_frames", "gripper_rgb_palm_camera_image_raw")
    ARM_SUBDIR = os.path.join(demonstrator, "lfd_bag_main", "bag_csvs")
    GRIPPER_SUBDIR = os.path.join(demonstrator, "lfd_bag_main", "bag_csvs")

    # Destination path
    MAIN_DIR = os.path.join('/media', 'alejo', 'IL_data')  
    # DESTINATION_DIR = os.path.join(MAIN_DIR, "02_IL_preprocessed")    
    DESTINATION_DIR = os.path.join(MAIN_DIR, "99_checking_02_IL_preprocessed")   
    DESTINATION_PATH = os.path.join(DESTINATION_DIR, EXPERIMENT)
        
    
    trials = [trial for trial in os.listdir(SOURCE_PATH)
              if os.path.isdir(os.path.join(SOURCE_PATH, trial))]

    trials_sorted = sorted(
        trials, 
        key=lambda x: int(x.split("_")[-1])
        )
    
    # Type trial number in case you want to start from that one
    start_index = trials_sorted.index("trial_10045")
    # start_index = 0
    

    # ---------- Step 2: Loop through all trials ----------
    trials_without_subfolders = []
    trials_with_one_subfolder = []
    for trial in tqdm(trials_sorted[start_index:]):

        # Double check trial folders
        trial_subfolders = os.listdir(os.path.join(SOURCE_PATH, trial))
        if len(trial_subfolders) == 1:
            trials_without_subfolders.append(trial)
            continue
        elif len(trial_subfolders) == 2:
            trials_with_one_subfolder.append(trial)
            continue

        # Define paths to all raw data
        raw_palm_camera_images_path = os.path.join(SOURCE_PATH, trial, INHAND_CAM_SUBDIR)        
        raw_pressure_and_tof_path = os.path.join(SOURCE_PATH, trial, GRIPPER_SUBDIR, "microROS_sensor_data.csv")
        raw_eef_wrench_path = os.path.join(SOURCE_PATH, trial, ARM_SUBDIR, "franka_robot_state_broadcaster_external_wrench_in_stiffness_frame.csv")        
        raw_ee_pose_path = os.path.join(SOURCE_PATH, trial, ARM_SUBDIR, "franka_robot_state_broadcaster_current_pose.csv")
               
        # Downsample data and align datasets based on in-hand camera images timestamps
        compare_plots = False
        df = pd.DataFrame()
        df['timestamp_vector'] = get_timestamp_vector_from_images(raw_palm_camera_images_path)
        df_ds_1 = downsample_pressure_and_tof_data(df, raw_pressure_and_tof_path, compare_plots=compare_plots)
        df_ds_2 = downsample_eef_wrench_data(df, raw_eef_wrench_path, compare_plots=compare_plots)

        try:
            raw_joint_states_path = os.path.join(SOURCE_PATH, trial, ARM_SUBDIR, "franka_robot_state_broadcaster_measured_joint_states.csv")
            df_ds_3 = downsample_robot_joint_states_data(df, raw_joint_states_path, compare_plots=compare_plots)
        except FileNotFoundError:
            raw_joint_states_path = os.path.join(SOURCE_PATH, trial, ARM_SUBDIR, "joint_states.csv")
            df_ds_3 = downsample_robot_joint_states_data(df, raw_joint_states_path, compare_plots=compare_plots)
        
        df_ds_4 = downsample_robot_ee_pose_data(df, raw_ee_pose_path, compare_plots=compare_plots)
        df_ds_5 = reduce_size_inhand_camera_raw_images(raw_palm_camera_images_path, layer=12)

        # Compute ACTIONS based on ee pose
        df_ds_6 = derive_actions_from_ee_pose(df, raw_ee_pose_path, compare_plots)
        
        # Combine all downsampled data (STATES AND ACTIONS) into a single DataFrame
        df_ds_all = [df_ds_1, 
                     df_ds_2, 
                     df_ds_3,
                     df_ds_4,
                     df_ds_5,
                     df_ds_6]
        
        # dfs_trimmed = [df_ds_all[0]] + [df.iloc[:, 1:] for df in df_ds_all[1:]]  
        dfs_trimmed = [df_ds_1,
                       df_ds_2.iloc[:, 1:],     # drop timestamp column
                       df_ds_3.iloc[:, 1:],     # drop timestamp column
                       df_ds_4.iloc[:, 1:],     # drop timestamp column  
                       df_ds_5,                 # no timestamp column   
                       df_ds_6.iloc[:, 1:]           # drop timestamp column
                       ]
        
        combined_df = pd.concat(dfs_trimmed, axis=1)

        # Save combined downsampled data to CSV file        
        os.makedirs(DESTINATION_PATH, exist_ok=True)
        combined_csv_path = os.path.join(DESTINATION_PATH, trial + "_downsampled_aligned_data.csv")
        combined_df.to_csv(combined_csv_path, index=False)  

        if compare_plots:
            plt.show()
        

    print(f'Trials without subfolders: {trials_without_subfolders}\n')
    print(f'Trials with one subfolder: {trials_with_one_subfolder}\n')
    

    # Step 3: Crop data for each phase

    # Plot data before and after downsampling to double check everything is fine
    # Unit test!!!

    print('done!')


def stage_2_crop_data_to_task_phases():

    # --- Step 1: Define data columns for each phase ---
    phase_1_approach_cols = get_phase_columns("phase_1_approach")
    phase_2_contact_cols = get_phase_columns("phase_2_contact")
    phase_3_pick_cols = get_phase_columns("phase_3_pick")

    # --- Step 2: Define Data Source and Destination paths ----
    if platform.system() == "Windows":
        SOURCE_PATH = Path(r"D:\02_IL_preprocessed_(aligned_and_downsampled)\experiment_1_(pull)")
        DESTINATION_PATH = Path(r"D:\03_IL_preprocessed_(cropped_per_phase)\experiment_1_(pull)")
    else:
        SOURCE_PATH = Path("/media/alejo/IL_data/02_IL_preprocessed_(aligned_and_downsampled)/experiment_1_(pull)")
        DESTINATION_PATH = Path("/media/alejo/IL_data/03_IL_preprocessed_(cropped_per_phase)/experiment_1_(pull)")      

    trials = [f for f in os.listdir(SOURCE_PATH)
             if os.path.isfile(os.path.join(SOURCE_PATH, f)) and f.endswith(".csv")]    
    

    os.makedirs(DESTINATION_PATH, exist_ok=True)
    phases = ['phase_1_approach', 'phase_2_contact', 'phase_3_pick', 'phase_4_disposal']
    for phase in phases:
        os.makedirs(os.path.join(DESTINATION_PATH, phase), exist_ok=True)
    
    
    # --- Step 3: Loop through all trials ---
    trials_without_contact = []
    trials_with_multiple_contacts = []

    trials_without_engagement = []

    for trial in (trials):
        print(f'\nCropping {trial} into task phases...')

        df = pd.read_csv(os.path.join(SOURCE_PATH, trial))        
                
        # ------------------------ First: Define cropping indices --------------------------

        # End of phase 1: defined by tof < 5cm (contact)        
        idx_phase_1_end = find_end_of_phase_1_approach(df, trial, tof_threshold=50)
        if idx_phase_1_end is None:
            trials_without_contact.append(trial)
            continue  # Skip cropping for this trial
        elif idx_phase_1_end == "Multiple":
            trials_with_multiple_contacts.append(trial)
            continue
        
        idx_phase_2_start = idx_phase_1_end

        phase_1_time = 7.0  # in seconds
        idx_phase_1_start = idx_phase_1_end - int(phase_1_time * 30)  # assuming 30 Hz
        phase_1_extra_time_end = 2.0
        idx_phase_1_end += int(phase_1_extra_time_end * 30)

        # End of phase 2: defined by at least two suction cups engaged
        idx_phase_2_end = find_end_of_phase_2_contact(df, trial, air_pressure_threshold=600, n_cups=2)
        if idx_phase_2_end is None:
            trials_without_engagement.append(trial)
            continue  # Skip cropping for this trial

        idx_phase_3_start = idx_phase_2_end

        phase_2_extra_time_end = 2.0
        idx_phase_2_end += int(phase_2_extra_time_end * 30)

        # End of phase 3 defined by ...
        idx_phase_3_end = find_end_of_phase_3_contact(df, trial, total_force_threshold=20)
        phase_3_extra_time_end = 3.0
        idx_phase_3_end += int(phase_3_extra_time_end * 30)


        # ------------------------- Second: Crop data for each phase -----------------------
        df_phase_1 = df.iloc[idx_phase_1_start:idx_phase_1_end][['timestamp_vector'] + phase_1_approach_cols]
        # plt.plot(df_phase_1['timestamp_vector'],df_phase_1['tof'])        
        
        df_phase_2 = df.iloc[idx_phase_2_start:idx_phase_2_end][['timestamp_vector'] + phase_2_contact_cols]
        # fig = plt.figure()
        # plt.plot(df_phase_2['timestamp_vector'],df_phase_2['scA'])        
        # plt.plot(df_phase_2['timestamp_vector'],df_phase_2['scB'])  
        # plt.plot(df_phase_2['timestamp_vector'],df_phase_2['scC'])         

        df_phase_3 = df.iloc[idx_phase_3_start:idx_phase_3_end][['timestamp_vector'] + phase_3_pick_cols]
        # fig = plt.figure()
        # plt.plot(df_phase_3['timestamp_vector'],df_phase_3['_wrench._force._x'])        
        # plt.plot(df_phase_3['timestamp_vector'],df_phase_3['_wrench._force._y'])  
        # plt.plot(df_phase_3['timestamp_vector'],df_phase_3['_wrench._force._z'])  

        # plt.show()

        # df_phase_4 = df.iloc[idx_phase_3_end:][['timestamp_vector'] + phase_4_disposal_cols]

        # Save cropped data to CSV files
        base_filename = os.path.splitext(trial)[0]
        # df_phase_1.to_csv(os.path.join(DESTINATION_PATH, 'phase_1_approach', f"{base_filename}_(phase_1_approach).csv"), index=False)
        # df_phase_2.to_csv(os.path.join(DESTINATION_PATH, 'phase_2_contact', f"{base_filename}_(phase_2_contact).csv"), index=False)
        df_phase_3.to_csv(os.path.join(DESTINATION_PATH, 'phase_3_pick', f"{base_filename}_(phase_3_pick).csv"), index=False)
        # df_phase_4.to_csv(os.path.join(DESTINATION_PATH, 'phase_4_disposal', f"{base_filename}_(phase_4_disposal).csv"), index=False)


    # ========= ONLY HUMAN DEMOS: USEFUL FOR APPROACH PHASE ==========
    # Reason: Approach phase deosn't need the wrench topics

    if platform.system() == "Windows":
        SOURCE_PATH_ONLY_APPROACH = Path(r"D:\02_IL_preprocessed_(aligned_and_downsampled)\only_human_demos/with_palm_cam")
    else:
        SOURCE_PATH_ONLY_APPROACH = Path("/media/alejo/IL_data/02_IL_preprocessed_(aligned_and_downsampled)/only_human_demos/with_palm_cam")

    only_human_trials = [f for f in os.listdir(SOURCE_PATH_ONLY_APPROACH) 
                         if os.path.isfile(os.path.join(SOURCE_PATH_ONLY_APPROACH, f)) and f.endswith(".csv")]   
    
    for trial in only_human_trials:

        print(f'\nONLY HUMAN TRIALS - Cropping {trial} into approach phase...')

        df = pd.read_csv(os.path.join(SOURCE_PATH_ONLY_APPROACH, trial))        
                
        # ------------------------ First: Define cropping indices --------------------------
        # End of phase 1: defined by tof < 5cm (contact)        
        idx_phase_1_end = find_end_of_phase_1_approach(df, trial, tof_threshold=50)
        if idx_phase_1_end is None:
            trials_without_contact.append(trial)
            continue  # Skip cropping for this trial
        elif idx_phase_1_end == "Multiple":
            trials_with_multiple_contacts.append(trial)
            continue
               

        # ------------------------- Second: Crop data for each phase -----------------------
        df_phase_1 = df.iloc[idx_phase_1_start:idx_phase_1_end][['timestamp_vector'] + phase_1_approach_cols]

        # Save cropped data to CSV files
        base_filename = os.path.splitext(trial)[0]
        # df_phase_1.to_csv(os.path.join(DESTINATION_PATH, 'phase_1_approach', f"{base_filename}_(phase_1_approach).csv"), index=False)


    print('\n----Trials without contact:----')
    for trial in trials_without_contact:
        print(trial)

    print('\n----Trials with multiple contacts:----')
    for trial in trials_with_multiple_contacts:
        print(trial)

    print('\n----Trials without suction cups engagement:----')
    for trial in trials_without_engagement:
        print(trial)


def stage_3_fix_hw_issues():

    # Find trials whose pressure sensors had issues (e.g. pressure drops to -1)

    SOURCE_PATH = '/media/alejo/IL_data/02_IL_preprocessed/experiment_1_(pull)'
    trials = [f for f in os.listdir(SOURCE_PATH)
             if os.path.isfile(os.path.join(SOURCE_PATH, f)) and f.endswith(".csv")]
      
    
    # ---------- Step 2: Loop through all trials ----------   
    faulty_trials_scA = []  
    faulty_trials_scB = []
    faulty_trials_scC = []

    for trial in (trials):

        print(f'\nChecking {trial} for faulty pressure sensor data...')

        df = pd.read_csv(os.path.join(SOURCE_PATH, trial))

        scA = df['scA'].values 
        scB = df['scB'].values 
        scC = df['scC'].values 

        # Identify segments where pressure is -1
        faulty_indices_scA = np.where(scA == -1)[0]   
        faulty_indices_scB = np.where(scB == -1)[0]   
        faulty_indices_scC = np.where(scC == -1)[0]        


        if len(faulty_indices_scA) != 0:
            faulty_trials_scA.append(trial)
            # print(f'Fixing {trial}, found {len(faulty_indices_scA)} faulty indices in scA.')            
        elif len(faulty_indices_scB) != 0:
            faulty_trials_scB.append(trial)
            # print(f'Fixing {trial}, found {len(faulty_indices_scB)} faulty indices in scB.')           
        elif len(faulty_indices_scC) != 0:
            faulty_trials_scC.append(trial)
            # print(f'Fixing {trial}, found {len(faulty_indices_scC)} faulty indices in scC.')         
        else:       
            continue  # No issues detected  

        scA = np.copy(scA) / 10.0  # scale if needed
        scB = np.copy(scB) / 10.0
        scC = np.copy(scC) / 10.0

        # plot to visualize
        plt.figure()
        plt.plot(scA, label='Original scA', alpha=0.5, color='blue', linewidth=2)
        plt.plot(scB, label='Original scB', alpha=0.5, color='orange', linewidth=2)
        plt.plot(scC, label='Original scC', alpha=0.5, color='green', linewidth=2)
        plt.legend()
        plt.ylim([-10, 120])
        plt.grid()
        plt.show()

        
    
    print(f'\nTrials with faulty scA data: {faulty_trials_scA}\n'
          f'Trials with faulty scB data: {faulty_trials_scB}\n'
          f'Trials with faulty scC data: {faulty_trials_scC}\n')


def stage_4_short_time_memory(n_time_steps=3):
    """
    Generates a Dataframe with short-term memory given n_time_steps
    (e.g. t-2, t-1, t)
    """

    # Data Source and Destination folders
    if platform.system() == "Windows":
        SOURCE_PATH = Path(r"D:\03_IL_preprocessed_(cropped_per_phase)/experiment_1_(pull)/phase_3_pick")
        DESTINATION_PATH = Path(r"D:\04_IL_preprocessed_(memory)/experiment_1_(pull)/phase_3_pick")
    else:
        SOURCE_PATH = Path("/media/alejo/IL_data/03_IL_preprocessed_(cropped_per_phase)/experiment_1_(pull)/phase_3_pick")
        DESTINATION_PATH = Path("/media/alejo/IL_data/04_IL_preprocessed_(memory)/experiment_1_(pull)/phase_3_pick")         

    trials = [f for f in os.listdir(SOURCE_PATH)
             if os.path.isfile(os.path.join(SOURCE_PATH, f)) and f.endswith(".csv")]    
    
    DESTINATION_PATH = os.path.join(DESTINATION_PATH, f"{n_time_steps}_timesteps")
    os.makedirs(DESTINATION_PATH, exist_ok=True) 

    # Data Destination
    for trial in trials:

        print(f'\n Adjusting {trial} with time steps...')
        df = pd.read_csv(os.path.join(SOURCE_PATH, trial)) 
        total_rows = df.shape[0]
        
        df_combined = pd.DataFrame()

        for time_step in range(n_time_steps + 1):

            start_index = n_time_steps - time_step 
            end_index = total_rows - time_step
            df_time_step_ith = df.iloc[start_index: end_index]
            df_time_step_ith = df_time_step_ith.reset_index(drop=True)

            # Rename columns of ith timestep dataframe
            if time_step > 0:
                df_time_step_ith.columns = [col + f"_(t_{time_step})" for col in df_time_step_ith.columns]           

            # Combine dataframes            
            df_combined = pd.concat([df_time_step_ith, df_combined], axis=1)
            if time_step > 0:
                df_combined = df_combined.drop(f"timestamp_vector_(t_{time_step})", axis=1)
        
        df_combined = df_combined[["timestamp_vector"] + [c for c in df_combined.columns if c != "timestamp_vector"]]

        # Save cropped data to CSV files
        base_filename = os.path.splitext(trial)[0]
        df_combined.to_csv(os.path.join(DESTINATION_PATH, f"{base_filename}_({n_time_steps}_timesteps).csv"), index=False)

    
if __name__ == '__main__':

    stage_1_align_and_downsample()

    # stage_2_crop_data_to_task_phases()

    # stage_4_short_time_memory()
      
    # SOURCE_PATH = '/media/alejo/IL_data/01_IL_bagfiles/only_human_demos/with_palm_cam'
    # rename_folder(SOURCE_PATH, 10000)