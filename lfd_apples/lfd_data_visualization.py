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
        angle_deg = 55
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
        draw_vector_arrow(img, (cx, cy), (cx, cy + int(v_y_cam * scale)), (0, 255, 0))  # Vy
        draw_vector_arrow(img, (cx, cy), (cx + int(v_x_cam * scale), cy + int(v_y_cam * scale)), (0, 255, 255))  # Resultant
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


def infer_actions():

    # Get the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'data', 'random_forest_model.pkl')

    # Load model
    with open(model_path, "rb") as f:
        rf_loaded = pickle.load(f)

    # Load inputs   62-113
    trial_number = '92' 

    dir = '/media/alejo/IL_data/04_IL_learning/experiment_1_(pull)/phase_1_validation/'
    trial = 'trial_' + trial_number + '_downsampled_aligned_data_(phase_1_approach).csv'
    filepath = os.path.join(dir, trial)
    df = pd.read_csv(filepath)

    groundtruth_delta_x = df['delta_pos_x'].values
    groundtruth_delta_y = df['delta_pos_y'].values
    groundtruth_delta_z = df['delta_pos_z'].values

    
    #  Names of the columns you dropped (ground truth)
    output_cols = ['delta_pos_x', 'delta_pos_y', 'delta_pos_z', 'delta_ori_x', 'delta_ori_y', 'delta_ori_z', 'delta_ori_w']
    # delete ground truth (last seven columns)
    df_just_inputs = df.drop(columns=['timestamp_vector'] + output_cols)
    arr = df_just_inputs.to_numpy()

    # --- 2. Load normalization stats (mean/std) if you saved them ---
    mean = np.load(os.path.join(script_dir, "data", "mean.npy"))
    std = np.load(os.path.join(script_dir, "data", "std.npy"))
    X_new_norm = (arr - mean) / std

    # Infere output
    Y_predictions = rf_loaded.predict(X_new_norm)

    # --- 4. Assign predictions back to original dataframe ---
    for i, col in enumerate(output_cols):
        df[col] = Y_predictions[:, i]
    
    # --- 5. Optionally, save to a new CSV ---
    DESTINATION_PATH = '/media/alejo/IL_data/04_IL_learning/experiment_1_(pull)/phase_1_predictions/'
    os.makedirs(DESTINATION_PATH, exist_ok=True)
    output_path = os.path.join(DESTINATION_PATH, 'trial_' + trial_number + '_predictions.csv')
    df.to_csv(output_path, index=False)


    # Create video to compare
    # Visualize Inhand Camera and Ground Truth Actions
    images_folder = '/media/alejo/IL_data/01_IL_bagfiles/experiment_1_(pull)/trial_' + trial_number + '/robot/lfd_bag_palm_camera/camera_frames/gripper_rgb_palm_camera_image_raw'
    csv_path = output_path
    output_video_path = os.path.join(DESTINATION_PATH, 'trial_' + trial_number + '_predictions.mp4')
    combine_inhand_camera_and_actions('trial_' + trial_number, images_folder, csv_path, output_video_path)

    fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

    # Row 1: X-axis
    axs[0].plot(df['timestamp_vector'], groundtruth_delta_x, label='Ground Truth')
    axs[0].plot(df['timestamp_vector'], df['delta_pos_x'], label='Predictions')
    axs[0].set_title('EEF linear velocity x-axis')
    axs[0].legend()
    axs[0].grid(True)

    # Row 2: Y-axis
    axs[1].plot(df['timestamp_vector'], groundtruth_delta_y, label='Ground Truth')
    axs[1].plot(df['timestamp_vector'], df['delta_pos_y'], label='Predictions')
    axs[1].set_title('EEF linear velocity y-axis')
    axs[1].legend()
    axs[1].grid(True)

    # Row 3: Z-axis
    axs[2].plot(df['timestamp_vector'], groundtruth_delta_z, label='Ground Truth')
    axs[2].plot(df['timestamp_vector'], df['delta_pos_z'], label='Predictions')
    axs[2].set_title('EEF linear velocity z-axis')
    axs[2].legend()
    axs[2].grid(True)

    plt.xlabel("Timestamp")
    plt.tight_layout()
    plt.show()

   


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