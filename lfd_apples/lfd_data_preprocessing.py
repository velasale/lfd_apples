import os
import platform
import pandas as pd
import ast
import re
import numpy as np


from tqdm import tqdm
from hampel import hampel

from lfd_vision import extract_pooled_latent_vector, bounding_box_centers
from ultralytics import YOLO
import cv2

from scipy.ndimage import gaussian_filter, median_filter, gaussian_filter1d
from scipy.signal import medfilt, butter, filtfilt, lfilter

from ros2bag2csv import plot_pressure, plot_wrench
from pathlib import Path
import yaml
from pathlib import Path
from tf_transformations import quaternion_multiply, quaternion_conjugate
from scipy.spatial.transform import Rotation as R

import matplotlib
matplotlib.use("TkAgg")  # Ensures interactive plotting
import matplotlib.pyplot as plt


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


def resolve_group(cfg, group):
    print(f"\nResolving group: {group}")

    parts = group.split(".")
    node = cfg

    for p in parts:
        print(f"  -> accessing key: {p}")
        print(f"     current node type: {type(node)}")

        if not isinstance(node, dict):
            raise TypeError(
                f"Node is not a dict while resolving '{group}'. "
                f"Stopped at '{p}', node={node}"
            )

        if p not in node:
            raise KeyError(
                f"Key '{p}' not found while resolving '{group}'. "
                f"Available keys: {list(node.keys())}"
            )

        node = node[p]

    print(f"  -> resolved node type: {type(node)}")
    print(f"  -> resolved node value: {node}")

    # Prefix-based features
    if isinstance(node, dict) and "prefix" in node:
        prefix = node["prefix"]
        count = node["count"]
        cols = [f"{prefix}{i}" for i in range(count)]
        print(f"  -> expanded prefix group: {cols[:3]} ...")
        return cols

    # Normal list
    if isinstance(node, list):
        return node

    raise TypeError(
        f"Group '{group}' resolved to unsupported type: {type(node)}"
    )


def get_phase_columns(phase_name):
    data_columns_path = Path(__file__).parent / "config" / "lfd_data_columns.yaml"

    with open(data_columns_path, "r") as f:
        cfg = yaml.safe_load(f)

    print("\nTop-level keys in YAML:")
    print(list(cfg.keys()))

    print(f"\nResolving phase: {phase_name}")
    groups = cfg["phases"][phase_name]
    print("Groups:", groups)

    cols = []
    for group in groups:
        cols.extend(resolve_group(cfg, group))

    print("\nFinal columns:")
    print(cols)

    return cols


def quat_to_angular_velocity(quaternions, delta_t):
    """
    quaternions: (N, 4) in xyzw, expressed in world frame
    delta_t: (N,) or scalar
    Returns: angular velocities (N, 3) in world frame, rad/s
    """

    quaternions = np.asarray(quaternions)    
    N = quaternions.shape[0]

    q_current = quaternions[1:]         # shape (N-1, 4)
    q_prev = quaternions[:-1]           # shape (N-1, 4)

    r_current = R.from_quat(q_current)
    r_prev = R.from_quat(q_prev)

    r_rel = r_current * r_prev.inv()    # shape (N-1,)        

    # Rotation vectors
    rotvec = r_rel.as_rotvec()          # shape (N-1, 3)   

    # Prepend a zero rotation vector for the first step
    rotvec_full = np.vstack([np.zeros((1, 3)), rotvec])  # shape (N, 3)

    # Angular velocity ω = rotvec / Δt
    omega = rotvec_full / delta_t[:, None]    

    return omega, rotvec_full


def plot_signals_before_and_after(df_before, df_after, signals, timestamp_vector=np.array([])):
    
    try:
        x = np.array(df_before['elapsed_time']).flatten()
        x_ds = np.array(df_after['timestamp_vector']).flatten()
    except KeyError:
        x = np.arange(len(df_before))
        x_ds = np.arange(len(df_after))

    n_signals = len(signals)
    # Decide layout
    if n_signals <= 3:
        n_rows = n_signals
        n_cols = 1
    else:
        n_cols = 2
        n_rows = 3

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        sharex=True,
        figsize=(12, 4 * n_rows)
        )
    
    # Force axes to always be (n_rows, n_cols)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]
    
    for i, sig in enumerate(signals):
        row = i % n_rows
        col = i // n_rows   

        ax = axes[row, col]

        y = np.array(df_before[sig]).flatten()
        y_ds = np.array(df_after[sig]).flatten()

        ax.plot(x, y, label=f'Original {sig}', color='orange', alpha=0.5, linewidth=1)
        ax.plot(x_ds, y_ds, label=f'Downsampled {sig}', color='black', linewidth=1)
        
        ax.set_title(f'Channel {sig} before and after Downsampling')
        ax.legend()
        ax.grid(True)

        # Hide x tick labels except bottom row
        if row != n_rows - 1:
            ax.tick_params(labelbottom=False)
    

    plt.tight_layout()

    plt.show()


def apple_pose_ground_truth(df, apple_location_index):
    ''' Get ground truth of apple location
    '''

    # @ base frame
    apple_pose_at_base_ground_truth = df.iloc[apple_location_index][['_pose._position._x', '_pose._position._y', '_pose._position._z']]    
    df['apple._x._base'] = apple_pose_at_base_ground_truth.values[0]
    df['apple._y._base'] = apple_pose_at_base_ground_truth.values[1]
    df['apple._z._base'] = apple_pose_at_base_ground_truth.values[2]

    # @ tcp frame    
    p_apple_base = np.tile(apple_pose_at_base_ground_truth.values, (len(df),1))
    p_ee_base = df[['_pose._position._x',
                    '_pose._position._y',
                    '_pose._position._z']].to_numpy()       # eef position 
    q_ee_base = df[['_pose._orientation._x',
                    '_pose._orientation._y',
                    '_pose._orientation._z',
                    '_pose._orientation._w']].to_numpy()    # eef orientation
    delta_base = p_apple_base - p_ee_base

    R_base_ee = R.from_quat(q_ee_base).as_matrix()

    p_apple_ee = np.einsum("nij,nj->ni", R_base_ee.transpose(0, 2, 1), delta_base)

    df[["apple._x._ee", "apple._y._ee", "apple._z._ee"]] = p_apple_ee


    return df


def fix_pressure_values(df):
    '''
    Some pressure signals had issues.
    In order to not toss away the entire trial dataset, it is better to fix it
    It may not be the cleanest, but it is reasonable

    - For air pressure values that suddenly raised to inf, simply average the other two along those instances
    - For air pressue channels that dropped to zero, average the other two along the entire trial
    
    :param df: Description
    '''


    # # --- Air Pressure Signals Check
    # Air Pressure Lower threshold
    pr_dn_thr = 150
    if (df['scA'] < pr_dn_thr).any() or (df['scB'] < pr_dn_thr).any() or (df['scC'] < pr_dn_thr).any():       

        if (df['scA'] < pr_dn_thr).any():           
            df['scA'] = df[['scB', 'scC']].mean(axis=1)           

        if (df['scB'] < pr_dn_thr).any():
            df['scB'] = df[['scA', 'scC']].mean(axis=1)


    # Air Pressure Upper Threshold
    pr_up_thr = 1100
    if (df['scA'] > pr_up_thr).any() or (df['scB'] > pr_up_thr).any() or (df['scC'] > pr_up_thr).any():
      
        if (df['scB'] > pr_up_thr).any():
            mask = df['scB'] > pr_up_thr
            df.loc[mask, 'scB'] = df.loc[mask, ['scA', 'scC']].mean(axis=1)
        
        if (df['scC'] > pr_up_thr).any():
            mask = df['scC'] > pr_up_thr
            df.loc[mask, 'scC'] = df.loc[mask, ['scA', 'scB']].mean(axis=1)
    
    return df


def check_singularity(df):
    '''
    Whenever there is a Singuality with Franka Arm, wrench drops to zero.
    Hence we need to remove this data
    
    :param df: Description
    '''

    # -- Force Singularity Check
    if (df['_wrench._force._x'].abs() < 1e-4).any(): 
        return True
    
    else:
        return False

# ============ Topic-specific downsampling functions ===========
def downsample_pressure_and_tof_data(df, raw_data_path, compare_plots=True):
    
    df_raw = pd.read_csv(raw_data_path)    
    df_raw["_data_as_list"] = df_raw["_data"].apply(parse_array_string)

    # Split raw data list into multiple independent columns
    data_expanded = pd.DataFrame(df_raw["_data_as_list"].tolist(), columns=["scA", "scB", "scC", "tof"])

    # Combine with the rest of the dataframe
    df_final = pd.concat([df_raw, data_expanded], axis=1)
    df_final.drop(columns=["_data", "_data_as_list", "timestamp", "_layout._data_offset"], inplace=True)

    # Filter signals
    df_filtered = df_final.copy()   
    signals = ['scA', 'scB', 'scC', 'tof']          
    for sign in signals:
        # df_filtered[sign] = hampel(
            # df_filtered[sign],
            # window_size=5,   # ~170 ms at 30 Hz
            # n_sigma=3.0).filtered_data
        df_filtered[sign] = gaussian_filter(df_filtered[sign],sigma=1.0)

    # Interpolate to reference timestamps
    df_downsampled = interpolate_to_reference_multi(df_filtered, df, ts_col_values="elapsed_time", ts_col_ref="timestamp_vector", method="linear")

    # Compare plots before and after downsampling
    if compare_plots:               
         
        plot_signals_before_and_after(df_before=df_final, df_after=df_downsampled, signals=signals)        

    return df_downsampled


def reduce_size_inhand_camera_raw_images(df_with_timestamps, raw_data_path, model, layer=15, compare_plots=True):
    
    timestamp_vector = df_with_timestamps['timestamp_vector'].values

    latent_vector_rows = []
    bbox_rows = []
    previous_bbox = [-1,1]
    for fname in sorted(os.listdir(raw_data_path)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(raw_data_path, fname)
        img_cv = cv2.imread(image_path)

        if img_cv is None:
            print(f"Could not read {image_path}, skipping.")
            continue

        # Pooled Latent Vector
        pooled_vector, feat_map = extract_pooled_latent_vector(
            img_cv,
            model,
            layer_index=layer
        )        
        # build row: first filename, then 64 feature values
        latent_row = [fname] + pooled_vector.tolist()       
        latent_vector_rows.append(latent_row)


        # Bounding Box Center
        bbox = bounding_box_centers(
            img_cv,
            model
        )
        
        if bbox:
            previous_bbox = bbox
        else:
            # In case that there is no bbox, and avoid sudden jumps, keep the previous one
            bbox = previous_bbox

        bbox_rows.append(bbox)       
        
    # Latent Vector DF
    feature_dim = len(latent_vector_rows[0]) - 1 # typically 64 , subtracting bbox and filename
    columns = ["filename"] + [f"f{i}" for i in range(feature_dim)] 
    df_latent = pd.DataFrame(latent_vector_rows, columns=columns)
    df_latent.drop(columns=["filename"], inplace=True)   

    df_latent_vector_filtered = pd.DataFrame(
        gaussian_filter1d(df_latent, sigma=2, axis=0),
        columns = [f"f{i}" for i in range(feature_dim)],
        index=df_latent.index
        )

    # BBOX DF
    bbox_columns = ["bbox._x._img_frame"] + ["bbox._y._img_frame"]
    df_bbox = pd.DataFrame(bbox_rows, columns=bbox_columns)
    df_bbox_filtered = df_bbox.copy()

    df_bbox_filtered["bbox._x._img_frame"] = hampel(df_bbox["bbox._x._img_frame"], window_size=5, n_sigma=1.0).filtered_data
    df_bbox_filtered["bbox._y._img_frame"] = hampel(df_bbox["bbox._y._img_frame"], window_size=5, n_sigma=1.0).filtered_data

    df_filtered = pd.concat([df_latent_vector_filtered, df_bbox_filtered], axis=1)    

    
    if compare_plots: 
        signals = ["f1", "f10", "f20"]
        plot_signals_before_and_after(df_latent, df_filtered, signals, timestamp_vector=timestamp_vector)
        signals = ["bbox._x._img_frame"]
        plot_signals_before_and_after(df_bbox, df_filtered, signals, timestamp_vector=timestamp_vector)
   
    return df_filtered
      

def downsample_eef_wrench_data(df, raw_data_path, compare_plots=True):

    df_raw = pd.read_csv(raw_data_path)        

    # Combine with the rest of the dataframe
    df_final = pd.concat([df_raw], axis=1)
    df_final.drop(columns=["_header._stamp._sec", "_header._stamp._nanosec", "_header._frame_id", "timestamp"], inplace=True)

    # Apply filter to smooth wrench signals
    df_filtered = df_final.copy()    
    signals = ['_wrench._force._x', '_wrench._force._y', '_wrench._force._z',
               '_wrench._torque._x', '_wrench._torque._y', '_wrench._torque._z']
    
    b, a = butter(N=4, Wn=4, fs=1000, btype='low')
    for sig in signals:      
        df_filtered[sig] = hampel(df_filtered[sig], window_size=21, n_sigma=3.0).filtered_data
        # Butterworth low-pass filter        
        df_filtered[sig] = filtfilt(b, a, df_filtered[sig])

    # Interpolate to reference timestamps
    df_downsampled = interpolate_to_reference_multi(df_filtered, df, ts_col_values="elapsed_time", ts_col_ref="timestamp_vector", method="linear")

    # Compare plots before and after downsampling
    if compare_plots:                
        signals = ['_wrench._force._x', '_wrench._force._y', '_wrench._force._z']        
        plot_signals_before_and_after(df_before=df_final, df_after=df_downsampled, signals=signals)                   
               

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

    # Compare plots before and after downsampling        
    if compare_plots:               
        signals = ['pos_joint_1', 'vel_joint_1', 'eff_joint_1']        
        plot_signals_before_and_after(df_before=df_final, df_after=df_downsampled, signals=signals)                       

    return df_downsampled


def downsample_robot_ee_pose_data(df, raw_data_path, compare_plots=True):

    df_raw = pd.read_csv(raw_data_path)        
    
    # Combine with the rest of the dataframe
    df_final = pd.concat([df_raw], axis=1)
    df_final.drop(columns=["_header._stamp._sec", "_header._stamp._nanosec", "_header._frame_id", "timestamp"], inplace=True)

    # Interpolate to reference timestamps
    df_downsampled = interpolate_to_reference_multi(df_final, df, ts_col_values="elapsed_time", ts_col_ref="timestamp_vector", method="linear")

    # Compare plots before and after downsampling    
    if compare_plots:
        signals = ['_pose._position._x', '_pose._position._y', '_pose._position._z']  
        plot_signals_before_and_after(df_before=df_final, df_after=df_downsampled, signals=signals)          

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
                print(f"Could not convert timestamp '{timestamp_str}' to float in filename '{filename}'.")
                # Skip files that don't match expected pattern
                continue

    return timestamps


# =================== Action Space derivation ===================
def derive_actions_from_ee_pose(reference_df, raw_data_path, sigma=100, compare_plots=True):
    """
    Derive Action Space from eef_pose. Action Space consists of linear and angular velocities.
     * Linear velocities are computed as differences in position over time.
     * Angular velocities are computed from differences in orientation (quaternions) over time. 
    
    :param reference_df: Description
    :param raw_data_path: Description
    :param sigma: Description
    :param compare_plots: Description
    """    

    # Step 1: Load raw data, and remove unnecessary columns
    df_raw = pd.read_csv(raw_data_path)        
    df_final = pd.concat([df_raw], axis=1)
    df_final.drop(columns=["_header._stamp._sec", "_header._stamp._nanosec", "_header._frame_id", "timestamp"], inplace=True)

    # Load actions yaml file for column names
    data_columns_path = config_path = Path(__file__).parent / "config" / "lfd_data_columns.yaml"
    with open(data_columns_path, "r") as f:
        cfg = yaml.safe_load(f)    
    action_columns = cfg["action_cols_at_base_frame"]

    # Step 2: Get eef positions and orientations
    positions = df_final[['_pose._position._x', '_pose._position._y', '_pose._position._z']].values
    orientations = df_final[['_pose._orientation._x', '_pose._orientation._y', '_pose._orientation._z', '_pose._orientation._w']].values

    # Step 3: Compute delta time Δt
    delta_times = np.diff(df_final['elapsed_time'].values, prepend=df_final['elapsed_time'].values[0])
    delta_times[delta_times <= 0.0001] = 0.001   # Avoid division by zero for first entry. Franka Arm sampling rate 1 kHz

    # Step 4: Compute linear velocities (m/s) and angular velocities (rad/s)
    delta_positions = np.diff(positions, axis=0, prepend=positions[0:1, :])
    linear_speeds = delta_positions / delta_times[:, None]
    orientation_speeds, delta_orientations = quat_to_angular_velocity(orientations, delta_times)  

    # Save before filtering for comparison
    delta_columns = ["delta_pos_x", "delta_pos_y", "delta_pos_z", "delta_ori_x", "delta_ori_y", "delta_ori_z"]
    all_columns = action_columns + delta_columns
    actions_df_before_filtering = pd.DataFrame(np.hstack((linear_speeds, orientation_speeds, delta_positions, delta_orientations)), columns=all_columns)    
    actions_df_before_filtering['elapsed_time'] = df_final['elapsed_time'].values


    b, a = butter(N=4, Wn=10, fs=1000, btype='low')
    # Step 6: Filter linear and angular speeds    
    for col in range(3):      
        linear_speeds[:,col] = hampel(linear_speeds[:,col], window_size=11, n_sigma=3.0).filtered_data
        # Butterworth low-pass filter        
        linear_speeds[:,col] = filtfilt(b, a, linear_speeds[:,col])

        orientation_speeds[:,col] = hampel(orientation_speeds[:,col], window_size=11, n_sigma=3.0).filtered_data
        # Butterworth low-pass filter
        orientation_speeds[:,col] = filtfilt(b, a, orientation_speeds[:,col])

        delta_positions[:,col] = hampel(delta_positions[:,col], window_size=11, n_sigma=3.0).filtered_data
        delta_positions[:,col] = filtfilt(b,a, delta_positions[:,col])

        delta_orientations[:,col] = hampel(delta_orientations[:,col], window_size=11, n_sigma=3.0).filtered_data
        delta_orientations[:,col] = filtfilt(b,a, delta_orientations[:,col])

    # window_size = 30
    # linear_speeds = median_filter(linear_speeds, size=(window_size,1))
    # # orientation_speeds = gaussian_filter1d(orientation_speeds, sigma=10, axis=0)
    # orientation_speeds = median_filter(orientation_speeds, size=(window_size,1))    

    actions_df = pd.DataFrame(np.hstack((linear_speeds, orientation_speeds, delta_positions, delta_orientations)), columns=all_columns)
    actions_df['elapsed_time'] = df_final['elapsed_time'].values

    # Interpolate to reference timestamps
    df_downsampled = interpolate_to_reference_multi(actions_df, reference_df, ts_col_values="elapsed_time", ts_col_ref="timestamp_vector", method="linear")

    # Compare plots before and after downsampling
    if compare_plots:
        signals = action_columns
        plot_signals_before_and_after(df_before=actions_df_before_filtering, df_after=df_downsampled, signals=signals)   
        signals = delta_columns
        plot_signals_before_and_after(df_before=actions_df_before_filtering, df_after=df_downsampled, signals=signals)   

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
        
        return None, None
    
    elif len(transition_indices) == 1:
        # Index of phase 1 end (first drop below threshold)
        idx_phase_1_end = transition_indices[0]
        idx_phase_2_start = idx_phase_1_end

        return idx_phase_1_end, idx_phase_2_start

    elif len(transition_indices) > 1:

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

        # Ask the user which contact point to pick
        # In this case, the index may differ, so it is safe to ask the user when to start phase 2
        plt.show()
        user_input = input(f"Enter index (0-{len(transition_indices)-1}) of correct phase 2 start: ")
        chosen_idx = int(user_input)
        idx_phase_2_start = transition_indices[chosen_idx]
    
        return idx_phase_1_end, idx_phase_2_start 
    

def find_end_of_phase_2_contact(df, trial, air_pressure_threshold=600, n_cups=2):

    scA_f = pd.Series(
    gaussian_filter(df['scA'].to_numpy(), 3),
    index=df.index
    )
    scB_f = pd.Series(
        gaussian_filter(df['scB'].to_numpy(), 3),
        index=df.index
    )
    scC_f = pd.Series(
        gaussian_filter(df['scC'].to_numpy(), 3),
        index=df.index
    )

    num_below = (
        (scA_f < air_pressure_threshold).astype(int) +
        (scB_f < air_pressure_threshold).astype(int) +
        (scC_f < air_pressure_threshold).astype(int)
    )

    idxs = num_below[num_below >= 2].index

    if idxs.empty:
        plot_pressure(df, time_vector='timestamp_vector')
        print(f'No engagement detected in {trial}, skipping cropping.')
        plt.show()
        return None

    idx_phase_2_end = idxs[0]

    # plot_pressure(df, time_vector='timestamp_vector')
    # print(f'Index at which at least two suction cups engage in {trial}.')     
    # time_phase_2_end = df.loc[idx_phase_2_end, 'timestamp_vector']       
    # plt.axvline(x=time_phase_2_end, color='red', linestyle='--', label='Phase 2 End')        
    # plt.show()
    

    return idx_phase_2_end


def find_end_of_phase_3_contact(df, trial, total_force_threshold=20):
    
    idx = df.index

    # --- Wrench signals (NumPy only at the boundary) ---
    fx_f = pd.Series(
        gaussian_filter(df['_wrench._force._x'].to_numpy(), 3),
        index=idx
    )
    fy_f = pd.Series(
        gaussian_filter(df['_wrench._force._y'].to_numpy(), 3),
        index=idx
    )
    fz_f = pd.Series(
        gaussian_filter(df['_wrench._force._z'].to_numpy(), 3),
        index=idx
    )

    # --- Net force (still indexed) ---
    net_force = np.sqrt(fx_f**2 + fy_f**2 + fz_f**2)

    # --- Phase boundary ---
    max_force_idx = net_force.idxmax()
    

    # fig = plt.figure()
    # t = df['timestamp_vector'].values    
    # time_max = df.loc[max_force_idx, 'timestamp_vector']
    # plt.plot(t, net_force, label='fx')    
    # plt.axvline(x=time_max, color='red', linestyle='--', label='Phase 3 End')        
    # plt.plot(t, net_force, label='net')
    # plt.legend()
    # plt.show()

    return max_force_idx



# ================ Main stages of data preprocessing ============
def stage_1_align_and_downsample():

    # ---------- Step 1: Load raw data ----------
    # MAIN_DIR = os.path.join("D:")                                     # windows OS
    MAIN_DIR = os.path.join('/media', 'alejo', 'IL_data')            # ubuntu OS
    SOURCE_DIR = os.path.join(MAIN_DIR, "01_IL_bagfiles")    
    EXPERIMENT = "experiment_1_(pull)"
    EXPERIMENT = "only_human_demos/with_palm_cam"   
    SOURCE_PATH = os.path.join(SOURCE_DIR, EXPERIMENT)

    demonstrator = ""  # "human" or "robot"

    FIXED_CAM_SUBDIR = os.path.join(demonstrator, "lfd_bag_fixed_camera", "camera_frames", "fixed_rgb_camera_image_raw")
    INHAND_CAM_SUBDIR = os.path.join(demonstrator, "lfd_bag_palm_camera", "camera_frames", "gripper_rgb_palm_camera_image_raw")
    ARM_SUBDIR = os.path.join(demonstrator, "lfd_bag_main", "bag_csvs")
    GRIPPER_SUBDIR = os.path.join(demonstrator, "lfd_bag_main", "bag_csvs")

    # Destination path
    # MAIN_DIR = os.path.join('/media', 'alejo', 'IL_data')  
    MAIN_DIR = os.path.join('/home/alejo/Documents/DATA')        # ubuntu OS
    DESTINATION_DIR = os.path.join(MAIN_DIR, "02_IL_preprocessed_(aligned_and_downsampled)")    
    DESTINATION_PATH = os.path.join(DESTINATION_DIR, EXPERIMENT)

    # Load YOLO model for in-hand camera feature extraction
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pt_path = os.path.join(script_dir, "resources", "best_segmentation.pt")
    cv_model = YOLO(pt_path)    
    
    trials = [trial for trial in os.listdir(SOURCE_PATH)
              if os.path.isdir(os.path.join(SOURCE_PATH, trial))]

    trials_sorted = sorted(
        trials, 
        key=lambda x: int(x.split("_")[-1])
        )
    
    # Type trial number in case you want to start from that one
    # start_index = trials_sorted.index("trial_237")
    start_index = 0
    

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

        # Define source paths to RAW data
        raw_palm_camera_images_path = os.path.join(SOURCE_PATH, trial, INHAND_CAM_SUBDIR)        
        raw_pressure_and_tof_path = os.path.join(SOURCE_PATH, trial, GRIPPER_SUBDIR, "microROS_sensor_data.csv")
        raw_eef_wrench_path = os.path.join(SOURCE_PATH, trial, ARM_SUBDIR, "franka_robot_state_broadcaster_external_wrench_in_stiffness_frame.csv")        
        raw_ee_pose_path = os.path.join(SOURCE_PATH, trial, ARM_SUBDIR, "franka_robot_state_broadcaster_current_pose.csv")
               
        # Downsample data and align datasets based on the timestamps of in-hand camera images
        compare_plots = True
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
        df_ds_5 = reduce_size_inhand_camera_raw_images(df, raw_palm_camera_images_path, model=cv_model, layer=15, compare_plots=compare_plots)

        # Compute ACTIONS based on ee pose
        df_ds_6 = derive_actions_from_ee_pose(df, raw_ee_pose_path, compare_plots=compare_plots)
        
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
                       df_ds_6.iloc[:, 1:]      # drop timestamp column
                       ]
        
        combined_df = pd.concat(dfs_trimmed, axis=1)

        # Save combined downsampled data to CSV file        
        os.makedirs(DESTINATION_PATH, exist_ok=True)
        combined_csv_path = os.path.join(DESTINATION_PATH, trial + "_downsampled_aligned_data.csv")
        combined_df.to_csv(combined_csv_path, index=False)  

        if compare_plots:
            plt.show()
        else:
            plt.close('all')        

    print(f'Trials without subfolders: {trials_without_subfolders}\n')
    print(f'Trials with one subfolder: {trials_with_one_subfolder}\n')   
    print('done!')


def stage_2_transform_data_to_eef_frame():

    # --- Step 2: Define Data Source and Destination paths ----
    if platform.system() == "Windows":
        SOURCE_PATH = Path(r"D:\02_IL_preprocessed_(aligned_and_downsampled)\experiment_1_(pull)")
        DESTINATION_PATH = Path(r"D:\03_IL_preprocessed_(cropped_per_phase)\experiment_1_(pull)")
    else:
        SOURCE_PATH = Path('/home/alejo/Documents/DATA/02_IL_preprocessed_(aligned_and_downsampled)/only_human_demos/with_palm_cam')
        DESTINATION_PATH = Path('/home/alejo/Documents/DATA/03_IL_preprocessed_(transformed_to_eef)/only_human_demos/with_palm_cam') #experiment_1_(pull)')

    trials = [f for f in os.listdir(SOURCE_PATH)
             if os.path.isfile(os.path.join(SOURCE_PATH, f)) and f.endswith(".csv")]    
    
    os.makedirs(DESTINATION_PATH, exist_ok=True)

    plt.close('all')

    for trial in (trials):
        
        print(f'\nTransforming {trial} to eef frame...')        
        df = pd.read_csv(os.path.join(SOURCE_PATH, trial))        

        # --- Transform actions ---
        # Actions are cartesian velocities in the base frame, need to transform to eef frame
        # This is done by applying the inverse of the current eef pose to the actions
        # Note: This is a placeholder for the actual transformation logic
        # In practice, you would need to implement the actual transformation using the eef pose data

        # Cartesian velocities in the base frame
        # cartesian_velocities_in_base_frame = df[['delta_pos_x', 'delta_pos_y', 'delta_pos_z', 'delta_angular_x', 'delta_angular_y', 'delta_angular_z']].values
        
        cartesian_velocities_in_base_frame = df[['v_eef_x', 'v_eef_y', 'v_eef_z', 'w_eef_x', 'w_eef_y', 'w_eef_z']].values
        # Deltas of eef in the base frame
        deltas_eef_in_base_frame = df[['delta_pos_x', 'delta_pos_y', 'delta_pos_z', 'delta_ori_x', 'delta_ori_y', 'delta_ori_z']].values

        # End-effector pose in the base frame
        eef_pose_in_base_frame = df[['_pose._position._x', '_pose._position._y', '_pose._position._z',
                       '_pose._orientation._x', '_pose._orientation._y', '_pose._orientation._z', '_pose._orientation._w']].values
        
        time_vector = df['timestamp_vector'].values

        v_eef_list = []
        w_eef_list = []

        delta_linear_eef_list = []
        delta_angular_eef_list = []

        for i,row in df.iterrows():

            v_base = cartesian_velocities_in_base_frame[i, :3]
            w_base = cartesian_velocities_in_base_frame[i, 3:]

            delta_linear_base = deltas_eef_in_base_frame[i, :3]
            delta_angular_base = deltas_eef_in_base_frame[i, 3:]

            q_base = eef_pose_in_base_frame[i, 3:]
            p_base = eef_pose_in_base_frame[i, :3]

            # Rotation matrix from quaternion   base <- eef
            R_base_eef = R.from_quat(q_base).as_matrix()            

            # Inverse rotation matrix : eef <- base
            R_eef_base = R_base_eef.T
            
            # Transform linear velocity
            v_eef = R_eef_base @ v_base
            delta_linear_eef = R_eef_base @ delta_linear_base
            
            # Transform angular velocity
            w_eef = R_eef_base @ w_base
            delta_angular_eef = R_eef_base @ delta_angular_base

            v_eef_list.append(v_eef)
            w_eef_list.append(w_eef)
            delta_linear_eef_list.append(delta_linear_eef)
            delta_angular_eef_list.append(delta_angular_eef)

        v_eef_array = np.vstack(v_eef_list)
        w_eef_array = np.vstack(w_eef_list)
        delta_linear_eef_array = np.vstack(delta_linear_eef_list)
        delta_angular_eef_array = np.vstack(delta_angular_eef_list)

        v_eef_array_filtered = np.zeros_like(v_eef_array)
        w_eef_array_filtered = np.zeros_like(w_eef_array)

        delta_linear_eef_array_filtered = np.zeros_like(delta_linear_eef_array)
        delta_angular_eef_array_filtered = np.zeros_like(delta_angular_eef_array)

        b, a = butter(N=2, Wn=2, fs=30, btype='low')
        # Step 6: Filter linear and angular speeds    
        for col in range(3):                              
            v_eef_array_filtered[:,col] = filtfilt(b, a, v_eef_array[:,col])
            w_eef_array_filtered[:,col] = filtfilt(b, a, w_eef_array[:,col])        
            delta_linear_eef_array_filtered[:,col] = filtfilt(b, a, delta_linear_eef_array[:,col])
            delta_angular_eef_array_filtered[:,col] = filtfilt(b, a, delta_angular_eef_array[:,col])

        plot_velocities = False    

        if plot_velocities:
            fig, ax = plt.subplots(4, 1, sharex=True, figsize=(8, 5))

            ax[0].plot(time_vector, v_eef_array[:,0], label='Unfiltered v_eef_x', alpha=0.5, linewidth=1, color='green')
            ax[0].plot(time_vector, v_eef_array_filtered[:,0], label='Filtered v_eef_x', linewidth=1, color='blue')
            ax[0].set_ylabel('Velocity [m/s]')
            ax[0].set_title(f'v_eef_x over time for {trial}')
            ax[0].grid(True)
            ax[0].legend()

            ax[1].plot(time_vector, w_eef_array[:,0], label='Unfiltered w_eef_x', alpha=0.5, linewidth=1, color='orange')
            ax[1].plot(time_vector, w_eef_array_filtered[:,0], label='Filtered w_eef_x', linewidth=1, color='red')           
            ax[1].set_ylabel('Angular velocity [rad/s]')
            ax[1].set_title(f'w_eef_x over time for {trial}')
            ax[1].grid(True)
            ax[1].legend()

            plt.show()

        # Switch back to DataFrame
        v_eef_df = pd.DataFrame(v_eef_array_filtered, columns=['v_eef._x._eef_frame',
                                                               'v_eef._y._eef_frame',
                                                               'v_eef._z._eef_frame'])
        w_eef_df = pd.DataFrame(w_eef_array_filtered, columns=['w_eef._x._eef_frame',
                                                               'w_eef._y._eef_frame',
                                                               'w_eef._z._eef_frame'])

        delta_linear_eef_df = pd.DataFrame(delta_linear_eef_array_filtered, columns=['Δ_lin_eef._x._eef_frame',
                                                                                     'Δ_lin_eef._y._eef_frame',
                                                                                     'Δ_lin_eef._z._eef_frame'])
        delta_angular_eef_df = pd.DataFrame(delta_angular_eef_array_filtered, columns=['Δ_ori_eef._x._eef_frame',
                                                                                       'Δ_ori_eef._y._eef_frame',
                                                                                       'Δ_ori_eef._z._eef_frame'])
        
        # ========= Safe Check of Air pressures =======
        df = fix_pressure_values(df)


        # Combine with original DataFrame
        df_eef_frame = pd.concat([df, v_eef_df, w_eef_df, delta_linear_eef_df, delta_angular_eef_df], axis=1)

        # Save transformed data to CSV files
        base_filename = os.path.splitext(trial)[0]
        df_eef_frame.to_csv(os.path.join(DESTINATION_PATH, f"{base_filename}_transformed.csv"), index=False)

       
        pass
        # 


def stage_3_crop_data_to_task_phases():

    # --- Step 1: Define data columns for each phase ---
    phase_1_approach_cols = get_phase_columns("phase_1_approach")
    phase_2_contact_cols = get_phase_columns("phase_2_contact")
    phase_3_pick_cols = get_phase_columns("phase_3_pick")

    # --- Step 2: Define Data Source and Destination paths ----
    if platform.system() == "Windows":
        SOURCE_PATH = Path(r"D:\02_IL_preprocessed_(aligned_and_downsampled)\experiment_1_(pull)")
        DESTINATION_PATH = Path(r"D:\03_IL_preprocessed_(cropped_per_phase)\experiment_1_(pull)")
    else:      
        SOURCE_PATH = Path('/home/alejo/Documents/DATA/03_IL_preprocessed_(transformed_to_eef)/experiment_1_(pull)')
        DESTINATION_PATH = Path('/home/alejo/Documents/DATA/04_IL_preprocessed_(cropped_per_phase)/experiment_1_(pull)')

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

    # Double check these:
    # trials = ['trial_309_downsampled_aligned_data_transformed.csv']

    for trial in (trials):
        print(f'\nCropping {trial} into task phases...')        

        df = pd.read_csv(os.path.join(SOURCE_PATH, trial))        
                
        # ------------------------ First: Define cropping indices --------------------------        

        # === PHASE 1: APPROACH PHASE ===
        PHASE_1_EXTRA_TIME_END = 0.5
        # End of phase 1: defined by tof < 5cm (contact)        
        idx_phase_1_end, idx_phase_2_start = find_end_of_phase_1_approach(df, trial, tof_threshold=60)
        if idx_phase_1_end is None:
            trials_without_contact.append(trial)
            continue  # Skip cropping for this trial
        elif idx_phase_1_end == "Multiple":
            trials_with_multiple_contacts.append(trial)
            continue
        
        # Get apple pose @ base and @tcp frame   
        df = apple_pose_ground_truth(df, idx_phase_1_end)

        fig = plt.figure()
        t = df['timestamp_vector'].values    
        time_ref = df.loc[idx_phase_1_end, 'timestamp_vector']
        plt.plot(t, df['tof'], label='tof')    
        plt.axvline(x=time_ref, color='red', linestyle='--', label='Phase 3 End')    
        plt.show()


        phase_1_time = 9.0  # in seconds
        idx_phase_1_start = max(0, (idx_phase_1_end - int(phase_1_time * 30)))  # assuming 30 Hz        
        idx_phase_1_end += int(PHASE_1_EXTRA_TIME_END * 30)

        # Crop data for phase 1
        df_phase_1 = df.iloc[idx_phase_1_start:idx_phase_1_end][['timestamp_vector'] + phase_1_approach_cols]        
        base_filename = os.path.splitext(trial)[0]
        df_phase_1.to_csv(os.path.join(DESTINATION_PATH, 'phase_1_approach', f"{base_filename}_(phase_1_approach).csv"), index=False)

        
        # === PHASE 2: CONTACT PHASE ===
        PHASE_2_EXTRA_TIME_END = 2.0
        # End of phase 2: defined by at least two suction cups engaged
        idx_phase_2_end = find_end_of_phase_2_contact(df[idx_phase_2_start:], trial, air_pressure_threshold=600, n_cups=2)
        
        if idx_phase_2_end is None:
            trials_without_engagement.append(trial)
            continue  # Skip cropping for this trial

        # Safety check
        if idx_phase_2_end < idx_phase_2_start:
            input(f"Issue with end and start of Contact phase: {trial} ")

        # Check ends at plot
        fig = plt.figure()
        t = df['timestamp_vector'].values    
        time_ref = df.loc[idx_phase_2_end, 'timestamp_vector']
        plt.plot(t, df['scA'], label='scA')    
        plt.plot(t, df['scB'], label='scB')   
        plt.plot(t, df['scC'], label='scC')   
        plt.axvline(x=time_ref, color='red', linestyle='--', label='Phase 3 End')    
        plt.show()


        idx_phase_3_start = idx_phase_2_end        
        idx_phase_2_end += int(PHASE_2_EXTRA_TIME_END * 30)

        # Crop data for phase 2
        df_phase_2 = df.iloc[idx_phase_2_start:idx_phase_2_end][['timestamp_vector'] + phase_2_contact_cols]
        df_phase_2.to_csv(os.path.join(DESTINATION_PATH, 'phase_2_contact', f"{base_filename}_(phase_2_contact).csv"), index=False)

        

        # === PHASE 3: PICK PHASE ===
        PHASE_3_EXTRA_TIME_END = 2.0
        # End of phase 3 defined by Max net Force
        idx_phase_3_end = find_end_of_phase_3_contact(df[idx_phase_3_start:], trial, total_force_threshold=20)        

        fig = plt.figure()
        t = df['timestamp_vector'].values    
        time_ref = df.loc[idx_phase_3_end, 'timestamp_vector']
        plt.plot(t, df['_wrench._force._x'], label='fx')    
        plt.plot(t, df['_wrench._force._y'], label='fy')    
        plt.plot(t, df['_wrench._force._z'], label='fz')    
        plt.axvline(x=time_ref, color='red', linestyle='--', label='Phase 3 End')    
        plt.show()

        idx_phase_3_end += int(PHASE_3_EXTRA_TIME_END * 30)

        # Safety check
        if idx_phase_3_end < idx_phase_3_start:
            input(f"Issue with end and start of Contact phase: {trial} ")

        if check_singularity(df):
            continue

        # Crop data for phase 3
        df_phase_3 = df.iloc[idx_phase_3_start:idx_phase_3_end][['timestamp_vector'] + phase_3_pick_cols]
        df_phase_3.to_csv(os.path.join(DESTINATION_PATH, 'phase_3_pick', f"{base_filename}_(phase_3_pick).csv"), index=False)

        
        
        # === PHASE 4: DISPOSAL PHASE ===              
        # df_phase_4.to_csv(os.path.join(DESTINATION_PATH, 'phase_4_disposal', f"{base_filename}_(phase_4_disposal).csv"), index=False)


    # ========= ONLY HUMAN DEMOS: USEFUL FOR APPROACH PHASE ==========
    # Reason: Approach phase doesn't need the wrench topics

    if platform.system() == "Windows":
        SOURCE_PATH_ONLY_APPROACH = Path(r"D:\02_IL_preprocessed_(aligned_and_downsampled)\only_human_demos/with_palm_cam")
    else:       
        SOURCE_PATH_ONLY_APPROACH = Path('/home/alejo/Documents/DATA/03_IL_preprocessed_(transformed_to_eef)/only_human_demos/with_palm_cam')

    only_human_trials = [f for f in os.listdir(SOURCE_PATH_ONLY_APPROACH) 
                         if os.path.isfile(os.path.join(SOURCE_PATH_ONLY_APPROACH, f)) and f.endswith(".csv")]   
    
    # only_human_trials = ['trial_10032_downsampled_aligned_data_transformed.csv',
    #                      'trial_10018_downsampled_aligned_data_transformed.csv',
    #                      'trial_10048_downsampled_aligned_data_transformed.csv',
    #                      'trial_10056_downsampled_aligned_data_transformed.csv',
    #                      'trial_10060_downsampled_aligned_data_transformed.csv',
    #                      'trial_10041_downsampled_aligned_data_transformed.csv',
    #                      ]
    for trial in only_human_trials:

        print(f'\nONLY HUMAN TRIALS - Cropping {trial} into approach phase...')

        df = pd.read_csv(os.path.join(SOURCE_PATH_ONLY_APPROACH, trial))        
                
        # ------------------------ First: Define cropping indices --------------------------
        # End of phase 1: defined by tof < 5cm (contact)        
        idx_phase_1_end,_ = find_end_of_phase_1_approach(df, trial, tof_threshold=50)
        if idx_phase_1_end is None:
            trials_without_contact.append(trial)
            continue  # Skip cropping for this trial
        elif idx_phase_1_end == "Multiple":
            trials_with_multiple_contacts.append(trial)
            continue
        
        # Get apple pose @ base and @tcp frame   
        df = apple_pose_ground_truth(df, idx_phase_1_end)

        phase_1_time = 7.0  # in seconds
        idx_phase_1_start = max(0, (idx_phase_1_end - int(phase_1_time * 30)))  # assuming 30 Hz        
        idx_phase_1_end += int(PHASE_1_EXTRA_TIME_END * 30)

        # Crop data for phase 1
        df_phase_1 = df.iloc[idx_phase_1_start:idx_phase_1_end][['timestamp_vector'] + phase_1_approach_cols]

        # Save cropped data to CSV files
        base_filename = os.path.splitext(trial)[0]
        df_phase_1.to_csv(os.path.join(DESTINATION_PATH, 'phase_1_approach', f"{base_filename}_(phase_1_approach).csv"), index=False)


    print('\n----Trials without contact:----')
    for trial in trials_without_contact:
        print(trial)

    print('\n----Trials with multiple contacts:----')
    for trial in trials_with_multiple_contacts:
        print(trial)

    print('\n----Trials without suction cups engagement:----')
    for trial in trials_without_engagement:
        print(trial)


def stage_4_short_time_memory(n_time_steps=0, phase='phase_1_contact', keep_actions_in_memory=False):
    """
    Generates a Dataframe with short-term memory given n_time_steps
    (e.g. t-2, t-1, t)
    """

    # Data Source and Destination folders
    if platform.system() == "Windows":
        SOURCE_PATH = Path(r"D:\03_IL_preprocessed_(cropped_per_phase)/experiment_1_(pull)/phase_3_pick")
        DESTINATION_PATH = Path(r"D:\04_IL_preprocessed_(memory)/experiment_1_(pull)/phase_3_pick")
    else:
        BASE_SOURCE_PATH = '/home/alejo/Documents/DATA'
        SOURCE_PATH = Path(BASE_SOURCE_PATH + '/04_IL_preprocessed_(cropped_per_phase)/experiment_1_(pull)/' + phase)
        DESTINATION_PATH = Path(BASE_SOURCE_PATH + '/05_IL_preprocessed_(memory)/experiment_1_(pull)/' + phase)         

    trials = [f for f in os.listdir(SOURCE_PATH)
             if os.path.isfile(os.path.join(SOURCE_PATH, f)) and f.endswith(".csv")]    
    
    DESTINATION_PATH = os.path.join(DESTINATION_PATH, f"{n_time_steps}_timesteps")
    os.makedirs(DESTINATION_PATH, exist_ok=True) 

    # Data Destination
    for trial in trials:

        print(f'\n Adjusting {trial} with time steps...')
        df = pd.read_csv(os.path.join(SOURCE_PATH, trial)) 

        # # ======= SPECIFIC IMAGE FEATURES ================
        # # Just leave the inputs and output cols       
        # phase_1_approach_cols = get_phase_columns("phase_1_approach")
        # phase_1_approach_cols.insert(0, 'timestamp_vector')
        # df = df[phase_1_approach_cols].copy()
        # # ================================================

        total_rows = df.shape[0]
        
        df_zero = df.iloc[[0]].copy()
        df_zero[:] = 0.0

        df_combined = pd.DataFrame()
        df_padding_combined = pd.DataFrame()

        for time_step in range(n_time_steps + 1):

            # # =============== Old Approach - No padding ============
            # start_index = n_time_steps - time_step 
            # end_index = total_rows - time_step
            # df_time_step_ith = df.iloc[start_index: end_index]
            # df_time_step_ith = df_time_step_ith.reset_index(drop=True)

            # # Rename columns of ith timestep dataframe
            # if time_step > 0:

            #     if not keep_actions_in_memory:
            #         df_time_step_ith = df_time_step_ith.iloc[:, :-6]
                
            #     df_time_step_ith.columns = [col + f"_(t_{time_step})" for col in df_time_step_ith.columns]           

            # # Combine dataframes            
            # df_combined = pd.concat([df_time_step_ith, df_combined], axis=1)
            # if time_step > 0:
            #     df_combined = df_combined.drop(f"timestamp_vector_(t_{time_step})", axis=1)


            # ============ New Approach - Padding ===============
            # Create Column for each previous time step
            #start_index = time_step
            end_index = total_rows - time_step
            df_padding_time_step_ith = df.iloc[0: end_index]

            # Add zeros
            for pad in range(time_step):                
                df_padding_time_step_ith = pd.concat([df_zero, df_padding_time_step_ith], ignore_index=True)

            # Rename columns
            if time_step > 0:

                if not keep_actions_in_memory:
                    df_padding_time_step_ith = df_padding_time_step_ith.iloc[:, :-6]                
                df_padding_time_step_ith.columns = [col + f"_(t_{time_step})" for col in df_padding_time_step_ith.columns]     
            
            # Combine Dataframes
            df_padding_combined = pd.concat([df_padding_time_step_ith, df_padding_combined], axis=1)
            if time_step > 0:
                df_padding_combined = df_padding_combined.drop(f"timestamp_vector_(t_{time_step})", axis=1)
            
        df_padding_combined = df_padding_combined[["timestamp_vector"] + [c for c in df_padding_combined.columns if c != "timestamp_vector"]]        
        # df_combined = df_combined[["timestamp_vector"] + [c for c in df_combined.columns if c != "timestamp_vector"]]

        # Save cropped data to CSV files
        base_filename = os.path.splitext(trial)[0]
        df_padding_combined.to_csv(os.path.join(DESTINATION_PATH, f"{base_filename}_({n_time_steps}_timesteps).csv"), index=False)


def stage_5_fix_hw_issues():

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
            
    
    print(f'\nTrials with faulty scA data: {faulty_trials_scA}\n'
          f'Trials with faulty scB data: {faulty_trials_scB}\n'
          f'Trials with faulty scC data: {faulty_trials_scC}\n')



if __name__ == '__main__':

    # stage_1_align_and_downsample()
    # stage_2_transform_data_to_eef_frame()
    stage_3_crop_data_to_task_phases()   
   
    # phases = ['phase_1_approach', 'phase_2_contact', 'phase_3_pick']    
    # # phases = ['phase_1_approach']    
    # for phase in phases:
    #     for step in [0, 5, 10, 15, 20]:
    #         stage_4_short_time_memory(n_time_steps=step, phase=phase, keep_actions_in_memory=False)  
      
    # SOURCE_PATH = '/media/alejo/IL_data/01_IL_bagfiles/only_human_demos/with_palm_cam'
    # rename_folder(SOURCE_PATH, 10000)