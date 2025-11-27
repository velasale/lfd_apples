import os
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
        plt.plot(x, y, label='Original wrench Force Z')

        x_ds = np.array(df_downsampled['timestamp_vector']).flatten()
        y_ds = np.array(df_downsampled['_wrench._force._z']).flatten()
        plt.plot(x_ds, y_ds, label='Downsampled wrench Force z', linestyle='--')
        
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
        y_ds = np.array(df_downsampled['delta_pos_y']).flatten()
        plt.plot(x_ds, y_ds, label='Downsampled Action delta pos y')

        x_ds = np.array(actions_df['elapsed_time']).flatten()
        y_ds = np.array(actions_df['delta_pos_y']).flatten()
        plt.plot(x_ds, y_ds, label='Original Action delta pos y', alpha=0.5)
        
        plt.legend()
        plt.title('Action delta pos x Over Time')        

    return df_downsampled



def estimate_robot_ee_pose():
    pass


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


def main():

    # ---------- Step 1: Load raw data ----------
    # MAIN_DIR = os.path.join("D:")                                   # windows OS
    MAIN_DIR = os.path.join('/media', 'alejo', 'IL_data')        # ubuntu OS
    SOURCE_DIR = os.path.join(MAIN_DIR, "01_IL_bagfiles")
    # EXPERIMENT = "experiment_1_(pull)"
    EXPERIMENT = "only_human_demos/with_palm_cam"   

    DESTINATION_DIR = os.path.join(MAIN_DIR, "02_IL_preprocessed")    

    SOURCE_PATH = os.path.join(SOURCE_DIR, EXPERIMENT)
    DESTINATION_PATH = os.path.join(DESTINATION_DIR, EXPERIMENT)
    
    demonstrator = ""  # "human" or "robot"
    FIXED_CAM_SUBDIR = os.path.join(demonstrator, "lfd_bag_fixed_camera", "camera_frames", "fixed_rgb_camera_image_raw")
    INHAND_CAM_SUBDIR = os.path.join(demonstrator, "lfd_bag_palm_camera", "camera_frames", "gripper_rgb_palm_camera_image_raw")
    ARM_SUBDIR = os.path.join(demonstrator, "lfd_bag_main", "bag_csvs")
    GRIPPER_SUBDIR = os.path.join(demonstrator, "lfd_bag_main", "bag_csvs")
    
    trials = [trial for trial in os.listdir(SOURCE_PATH)
              if os.path.isdir(os.path.join(SOURCE_PATH, trial))]

    trials_sorted = sorted(
        trials, 
        key=lambda x: int(x.split("_")[-1])
        )
    
    start_index = trials_sorted.index("trial_64")
    

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
        df_dfs_6 = derive_actions_from_ee_pose(df, raw_ee_pose_path, compare_plots=True)
        
        # Combine all downsampled data into a single DataFrame
        df_ds_all = [df_ds_1, df_ds_2, df_ds_3, df_ds_4, df_ds_5, df_dfs_6]
        dfs_trimmed = [df_ds_all[0]] + [df.iloc[:, 1:] for df in df_ds_all[1:]]  
        combined_df = pd.concat(dfs_trimmed, axis=1)

        # Save combined downsampled data to CSV file        
        os.makedirs(DESTINATION_PATH, exist_ok=True)
        combined_csv_path = os.path.join(DESTINATION_PATH, trial + "_downsampled_aligned_data.csv")
        combined_df.to_csv(combined_csv_path, index=False)  

        if compare_plots:
            plt.show()
        plt.show()

    print(f'Trials without subfolders: {trials_without_subfolders}\n')
    print(f'Trials with one subfolder: {trials_with_one_subfolder}\n')
    

    # Step 3: Crop data for each phase

    # Plot data before and after downsampling to double check everything is fine
    # Unit test!!!

    print('done!')


if __name__ == '__main__':
    main()
   

    # # MAIN_DIR = os.path.join("D:")  # windows OS
    # MAIN_DIR = os.path.join('/media', 'guest', 'IL_data')        # ubuntu OS
    # SOURCE_DIR = os.path.join(MAIN_DIR, "01_IL_bagfiles")    
    # EXPERIMENT = "only_human_demos"    
    # SOURCE_PATH = os.path.join(SOURCE_DIR, EXPERIMENT)   

    # SOURCE_PATH = '/media/alejo/IL_data/01_IL_bagfiles/only_human_demos/with_palm_cam'
    # rename_folder(SOURCE_PATH, 1)