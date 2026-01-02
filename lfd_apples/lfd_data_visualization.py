
import cv2
import pandas as pd
import glob
import os
import numpy as np
from pathlib import Path
import platform
from scipy.spatial.transform import Rotation as R
import pickle
import matplotlib.pyplot as plt
import random
import joblib
import yaml

import torch

# Custom imports
from lfd_apples.lfd_learning import VelocityMLP  # make sure this class is imported
from lfd_apples.lfd_lstm import LSTMRegressor, create_sequences


def get_paths(trial_num="trial_1"):
    # Detect OS and set IL_data base directory
    if platform.system() == "Windows":
        # Windows: e.g. D:\01_IL_bagfiles\...
        BASE_IL = Path("D:/")
    else:
        # Linux: /media/alejo/IL_data/01_IL_bagfiles/...
        BASE_IL = Path("/media/alejo/IL_data")

    # Base folders
    bag_base = BASE_IL / "01_IL_bagfiles" / "experiment_1_(pull)"
    preproc_base = BASE_IL / "03_IL_preprocessed" / "experiment_1_(pull)"

    # Construct paths
    images_folder = (
        bag_base
        / trial_num
        / "robot"
        / "lfd_bag_palm_camera"
        / "camera_frames"
        / "gripper_rgb_palm_camera_image_raw"
    )

    csv_path = (
        preproc_base
        / "phase_1_approach"
        / f"{trial_num}_downsampled_aligned_data_(phase_1_approach).csv"
    )

    output_video_path = (
        preproc_base
        / "phase_1_approach"
        / f"{trial_num}.mp4"
    )

    return str(images_folder), str(csv_path), str(output_video_path)


def quat_to_omega(q, q_dot):
    """
    q: current quaternion [x,y,z,w]
    q_dot: quaternion derivative [dx,dy,dz,dw]
    Returns: omega = [wx,wy,wz] in rad/s
    """

    q_norm = np.linalg.norm(q)
    q = q / q_norm

    # quaternion inverse
    q_inv = np.array([-q[0], -q[1], -q[2], q[3]])

    # quaternion multiplication q_dot ⊗ q_inv
    x1, y1, z1, w1 = q_dot
    x2, y2, z2, w2 = q_inv
    prod = np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    ])

    # angular velocity vector part
    omega = 2 * prod[:3]
    return omega


def draw_vector_arrow(img, start, end, color, min_length=8, thickness=2, tipLength=0.3):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = np.hypot(dx, dy)

    if length < min_length:
        direction = np.array([dx, dy]) / (length + 1e-9)
        end = (int(start[0] + direction[0] * min_length), int(start[1] + direction[1] * min_length))

    # Draw black border for arrow
    cv2.arrowedLine(img, start, end, (0,0,0), thickness=thickness+2, tipLength=tipLength, line_type=cv2.LINE_AA)
    # Draw main arrow on top
    cv2.arrowedLine(img, start, end, color, thickness=thickness, tipLength=tipLength, line_type=cv2.LINE_AA)


def draw_omega_z_arrow(img, cx, cy, omega_z, radius=30, scale=10, color=(255, 0, 0), thickness=2, arrow_head_len=10):
    # Scale angular velocity
    omega_angle = omega_z * scale
    # Determine direction: positive CCW, negative CW
    direction = np.sign(omega_angle) if omega_angle != 0 else 1
    angle_deg = int(abs(omega_angle) * 180 / np.pi)

    start_angle = -90
    end_angle = start_angle + direction * angle_deg

    # Draw black border arc
    cv2.ellipse(img, (cx, cy), (radius, radius), 0, start_angle, end_angle, (0,0,0), thickness+2, lineType=cv2.LINE_AA)
    # Draw main arc
    cv2.ellipse(img, (cx, cy), (radius, radius), 0, start_angle, end_angle, color, thickness, lineType=cv2.LINE_AA)

    # Compute arrowhead
    theta = np.deg2rad(end_angle)
    x_end = int(cx + radius * np.cos(theta))
    y_end = int(cy + radius * np.sin(theta))

    # Tangent direction depends on omega sign
    tangent_angle = theta + direction * np.pi/2
    dx = np.cos(tangent_angle)
    dy = np.sin(tangent_angle)

    x1 = int(x_end - arrow_head_len * (dx + 0.3 * np.cos(theta)))
    y1 = int(y_end - arrow_head_len * (dy + 0.3 * np.sin(theta)))
    x2 = int(x_end - arrow_head_len * (dx - 0.3 * np.cos(theta)))
    y2 = int(y_end - arrow_head_len * (dy - 0.3 * np.sin(theta)))

    # Black border arrowhead
    cv2.line(img, (x_end, y_end), (x1, y1), (0,0,0), thickness+2, lineType=cv2.LINE_AA)
    cv2.line(img, (x_end, y_end), (x2, y2), (0,0,0), thickness+2, lineType=cv2.LINE_AA)
    # Main arrowhead
    cv2.line(img, (x_end, y_end), (x1, y1), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(img, (x_end, y_end), (x2, y2), color, thickness, lineType=cv2.LINE_AA)

    # Text label with black border
    # cv2.putText(img, "ωz", (cx + radius + 5, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, lineType=cv2.LINE_AA)
    # cv2.putText(img, "ωz", (cx + radius + 5, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, lineType=cv2.LINE_AA)


def draw_crosshair(img, cx, cy, color=(0, 255, 255), arm=15, corner=40, thickness=1):

    cv2.circle(img, (cx, cy), 3, color, -1)
    cv2.line(img, (cx, cy - arm), (cx, cy + arm), color, thickness)
    cv2.line(img, (cx - arm, cy), (cx + arm, cy), color, thickness)
    # Corner accents (L-shapes)
    cv2.line(img, (cx - corner, cy - corner), (cx - arm, cy - corner), color, thickness)
    cv2.line(img, (cx - corner, cy - corner), (cx - corner, cy - arm), color, thickness)
    cv2.line(img, (cx + corner, cy - corner), (cx + arm, cy - corner), color, thickness)
    cv2.line(img, (cx + corner, cy - corner), (cx + corner, cy - arm), color, thickness)
    cv2.line(img, (cx - corner, cy + corner), (cx - arm, cy + corner), color, thickness)
    cv2.line(img, (cx - corner, cy + corner), (cx - corner, cy + arm), color, thickness)
    cv2.line(img, (cx + corner, cy + corner), (cx + arm, cy + corner), color, thickness)
    cv2.line(img, (cx + corner, cy + corner), (cx + corner, cy + arm), color, thickness)


def combine_inhand_camera_and_actions(trial_name, images_folder, csv_path, output_video_path):

    # ==== LOAD CSV ====
    df = pd.read_csv(csv_path)

    # Convert timestamp number → string with *exact* 6 decimals
    df['timestamp_str'] = df['timestamp_vector'].apply(lambda x: f"{x:.6f}")

    # ==== LOAD IMAGES ====
    image_paths = sorted(glob.glob(os.path.join(images_folder, "*.jpg")))

    # Check size
    sample_img = cv2.imread(image_paths[0])
    height, width = sample_img.shape[:2]

    # ==== VIDEO WRITER ====
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # ==== MAIN LOOP ====
    for img_path in image_paths:
        filename = os.path.basename(img_path)

        # filename = frame_00000_0.082738.jpg → timestamp = "0.082738"
        timestamp = filename.split("_")[-1].replace(".jpg", "")
        timestamp = f"{float(timestamp):.6f}"  # normalize formatting

        # Find matching CSV row
        row = df[df["timestamp_str"] == timestamp]

        if row.empty:
            # print(f"No CSV match for timestamp {timestamp}")
            continue

        # eef linear velocity in base frame
        v_base = np.array([row["delta_pos_x"].values[0],
                           row["delta_pos_y"].values[0],
                           row["delta_pos_z"].values[0]])

        # eef quaternions in base frame
        q = np.array([
            row["_pose._orientation._x"].values[0],
            row["_pose._orientation._y"].values[0],
            row["_pose._orientation._z"].values[0],
            row["_pose._orientation._w"].values[0]
        ])

        # eef quaternion rates in base frame
        q_dot = np.array([
            row["delta_ori_x"].values[0],
            row["delta_ori_y"].values[0],
            row["delta_ori_z"].values[0],
            row["delta_ori_w"].values[0]
            ])

        # =================================================================
        #           Transform values from BASE to CAMERA frame
        # =================================================================

        # Compute angular velocity in EEF frame
        omega = quat_to_omega(q, q_dot)

        # Rotate linear velocity into EEF frame
        r_eef = R.from_quat(q)
        v_eef = r_eef.inv().apply(v_base)

        # Rotation-induced velocity at camera offset
        r_cam = np.array([0, 0, -0.06])  # camera offset in EEF frame
        v_rot = np.cross(omega, r_cam)

        # Total linear velocity at camera in EEF frame
        v_camera = v_eef + v_rot

        # Project to 2D for camera plane
        total_v_x = v_camera[0]
        total_v_y = v_camera[1]
        total_v_z = v_camera[2]

        # Rotation angle in degrees
        angle_deg = 90 #45      # 90
        angle_rad = np.radians(angle_deg)

        # Transform vector
        v_x_cam = total_v_x * np.cos(angle_rad) - total_v_y * np.sin(angle_rad)
        v_y_cam = total_v_x * np.sin(angle_rad) + total_v_y * np.cos(angle_rad)

        # ====================================================================
        #                  Draw crosshair and vectors in image
        # ====================================================================
        img = cv2.imread(img_path)
        cx, cy = width // 2, height // 2        # Center of image
        margin = 80
        # Scale vector to be visible
        scale = 600

        draw_crosshair(img, cx, cy)
        draw_vector_arrow(img, (cx, cy), (cx + int(v_x_cam * scale), cy), (0, 0, 255))  # Vx
        draw_vector_arrow(img, (cx, cy), (cx, cy - int(v_y_cam * scale)), (0, 255, 0))  # Vy
        draw_vector_arrow(img, (cx, cy), (cx + int(v_x_cam * scale), cy - int(v_y_cam * scale)), (0, 255, 255))  # Resultant
        # Center dot
        cv2.circle(img, (margin, margin), 3, (255, 255, 255), -1)
        draw_vector_arrow(img, (margin, margin), (margin + int(total_v_z * scale), margin), (255, 0, 0))  # Vz
        # draw_omega_z_arrow(img, cx, cy, omega[2], radius=30, scale=10)

        img = cv2.rotate(img, cv2.ROTATE_180)

        # ===============================
        # Draw timestamp after rotation
        # ===============================
        # Position: top-left corner with some margin
        ts_margin = 30
        cv2.putText(
            img,
            f"{trial_name}, t = {float(timestamp):.3f}s",  # three decimals
            (ts_margin, ts_margin),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),  # white text
            1
        )
        video.write(img)

    video.release()
    print("Video saved:", output_video_path)


def infer_actions(regressor='mlp_torch'):

    phase = 'phase_3_pick'
    timesteps = '10_timesteps'

    BASE_PATH = '/home/alejo/Documents/DATA'
    model_path = os.path.join(BASE_PATH, f'06_IL_learning/experiment_1_(pull)/{phase}/{timesteps}')

    # --- Load test trials ---
    test_trials_csv = os.path.join(model_path, 'test_trials.csv')
    df_trials = pd.read_csv(test_trials_csv)
    test_trials_list = df_trials['trial_id'].tolist()

    # --- Pick trial ---
    random_trial = False
    if random_trial:
        random_file = random.choice(test_trials_list)
    else:
        trial_number = 195
        main_folder = '/home/alejo/Documents/DATA/05_IL_preprocessed_(memory)/experiment_1_(pull)'
        random_file = os.path.join(main_folder, phase, timesteps)
        filename = 'trial_' + str(trial_number) + '_downsampled_aligned_data_transformed_(' + phase + ')_(' + timesteps + ').csv'
        random_file = os.path.join(random_file, filename)

    df = pd.read_csv(random_file)

    # --- Load action columns from config ---
    data_columns_path = Path(__file__).parent / "config" / "lfd_data_columns.yaml"
    with open(data_columns_path, "r") as f:
        cfg = yaml.safe_load(f)
    output_cols = cfg['action_cols']

    # --- Extract ground truth ---
    groundtruth = {col: df[col].values for col in output_cols}

    # --- Prepare input features ---
    df_inputs = df.drop(columns=['timestamp_vector'] + output_cols)
    X = df_inputs.to_numpy()

    # --- Load normalization stats ---
    X_mean = np.load(os.path.join(model_path, f"{regressor}_Xmean_experiment_1_(pull)_{phase}_{timesteps}.npy"))
    X_std  = np.load(os.path.join(model_path, f"{regressor}_Xstd_experiment_1_(pull)_{phase}_{timesteps}.npy"))
    X_norm = (X - X_mean) / X_std

    # --- Load target stats ---
    Y_mean = np.load(os.path.join(model_path, f"{regressor}_Ymean_experiment_1_(pull)_{phase}_{timesteps}.npy"))
    Y_std  = np.load(os.path.join(model_path, f"{regressor}_Ystd_experiment_1_(pull)_{phase}_{timesteps}.npy"))

    # --- Predict ---
    if regressor in ['rf', 'mlp']:
        model_name = f"{regressor}_experiment_1_(pull)_{phase}_{timesteps}.joblib"
        with open(os.path.join(model_path, model_name), "rb") as f:
            loaded_model = joblib.load(f)

        Y_pred = loaded_model.predict(X_norm)
        Y_pred_denorm = Y_pred * Y_std + Y_mean

    elif regressor == 'mlp_torch':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlp_model = VelocityMLP(input_dim=X_norm.shape[1], output_dim=len(output_cols)).to(device)
        mlp_model.load_state_dict(torch.load(os.path.join(model_path, "mlp_torch_model.pt"), map_location=device))
        mlp_model.eval()

        X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)
        with torch.no_grad():
            Y_pred = mlp_model(X_tensor).cpu().numpy()
        Y_pred_denorm = Y_pred * Y_std + Y_mean
    
    elif regressor == "lstm":
        SEQ_LEN = 6
        BATCH_SIZE = 256

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        lstm_model = LSTMRegressor(input_dim=65, hidden_dim=50, output_dim=6, num_layers=1)

        # Move model to device
        lstm_model.to(device)
        lstm_model.load_state_dict(torch.load(os.path.join(model_path,"lstm_model.pth")))

        # Set to evaluation mode
        lstm_model.eval()

        # Create sequence with 
        X_seq, idx_map = [], []  # keep track of which row each sequence corresponds to
        for i in range(len(X_norm) - SEQ_LEN + 1):
            X_seq.append(X_norm[i:i+SEQ_LEN])
            idx_map.append(i + SEQ_LEN - 1)  # last index of sequence

        X_seq = np.array(X_seq)

        # Convert to tensor
        
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)

        # Predict
        with torch.no_grad():
            Y_pred_seq = lstm_model(X_tensor).cpu().numpy()

        # Map predictions back to original row indices
        Y_pred_denorm = np.zeros((len(X), len(output_cols)))
        Y_pred_denorm[idx_map, :] = Y_pred_seq * Y_std + Y_mean



    # --- Assign predictions back to dataframe ---
    for i, col in enumerate(output_cols):
        df[col] = Y_pred_denorm[:, i]

    trial_description = (random_file.split('(pull)')[1]).split('steps/')[1]
    

    # --- Plot Linear Velocities ---
    fig, axs = plt.subplots(6, 1, figsize=(8, 12), sharex=True)    

    for i, col in enumerate(output_cols):
        axs[i].plot(df['timestamp_vector'], groundtruth[col], label='Ground Truth')
        axs[i].plot(df['timestamp_vector'], df[col], label='Predictions')
        axs[i].set_title(f'Action {col}')
        axs[i].legend()
        axs[i].grid(True)   
    plt.xlabel("Timestamp")
    plt.suptitle(f'Model: $\\bf{{{regressor}}}$ \n{trial_description}')
    plt.tight_layout()
    plt.show()

    # --- Optional: Create video overlay ---
    # trial_name = random_file.split('/')[-1].split('_downsampled')[0]
    # images_folder, _, output_video_path = get_paths(trial_name)
    # combine_inhand_camera_and_actions(trial_name, images_folder, random_file, output_video_path)

    print(f"Predictions done for trial: {random_file} using {regressor}")


def important_features(top=5):
    """Display the top features"""

    phases = ['phase_1_approach', 'phase_2_contact', 'phase_3_pick']
    for phase in phases:

        phase_df = pd.DataFrame()

        for n_timesteps in range(11):

            # Step 1: Define path
            folder = '/home/alejo/Documents/DATA/06_IL_learning/experiment_1_(pull)/' 
            steps = str(n_timesteps) + '_timesteps'
            filepath = os.path.join(folder, phase, steps, 'rf_feature_importances.csv')                       

            # Step 2: Open file
            df = pd.read_csv(filepath)
            sub_df = df.iloc[:top, 0]

            # Step 5: Append Column with column name = timesteps
            phase_df[steps] = sub_df

        print(f'\nFeature importance during \033[1m{phase}\033[0m$: \n\n {phase_df}')

                  


def main():

    # folder = r"D:\03_IL_preprocessed\experiment_1_(pull)\phase_1_approach"
    folder = '/media/alejo/IL_data/03_IL_preprocessed/experiment_1_(pull)/phase_1_approach'
    trials = [f for f in os.listdir(folder) if f.endswith(".csv")]

    for trial in trials:

        trial_name = trial.partition("_downsampled")[0]
        # ==== USER PATHS ====
        images_folder, csv_path, output_video_path = get_paths(trial_name)

        # Visualize Inhand Camera and Ground Truth Actions
        combine_inhand_camera_and_actions(trial_name, images_folder, csv_path, output_video_path)


if __name__ == '__main__':

    # main()
    infer_actions()

    # important_features(top=10)

