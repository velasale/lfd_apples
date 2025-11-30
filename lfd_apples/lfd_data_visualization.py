import cv2
import pandas as pd
import glob
import os
import numpy as np
from pathlib import Path
import platform
from scipy.spatial.transform import Rotation as R

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


def combine_inhand_camera_and_actions(images_folder, csv_path, output_video_path):

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

        # eef quaternions
        q = np.array([
            row["_pose._orientation._x"].values[0],
            row["_pose._orientation._y"].values[0],
            row["_pose._orientation._z"].values[0],
            row["_pose._orientation._w"].values[0]
        ])

        # eef quaternion rates
        q_dot = np.array([
            row["delta_ori_x"].values[0],
            row["delta_ori_y"].values[0],
            row["delta_ori_z"].values[0],
            row["delta_ori_w"].values[0]
            ])

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

        # Rotation angle in degrees
        angle_deg = 55
        angle_rad = np.radians(angle_deg)

        # Transform vector
        v_x_rot = total_v_x * np.cos(angle_rad) - total_v_y * np.sin(angle_rad)
        v_y_rot = total_v_x * np.sin(angle_rad) + total_v_y * np.cos(angle_rad)

        img = cv2.imread(img_path)

        # Center of image
        cx, cy = width // 2, height // 2

        # Scale vector to be visible
        scale = 400
        end_x = int(cx + v_x_rot * scale)
        end_y = int(cy - v_y_rot * scale)

        # ===============================
        #     DRAW FANCY CROSSHAIR
        # ===============================
        cross_color = (0, 255, 255)    # yellow BGR

        # Center dot
        cv2.circle(img, (cx, cy), 3, cross_color, -1)

        # Size of cross arms
        arm = 15
        thickness = 1

        # Vertical line
        cv2.line(img, (cx, cy - arm), (cx, cy + arm), cross_color, thickness)

        # Horizontal line
        cv2.line(img, (cx - arm, cy), (cx + arm, cy), cross_color, thickness)

        # Corner accents (L shapes)
        corner = 40
        cv2.line(img, (cx - corner, cy - corner), (cx - arm, cy - corner), cross_color, thickness)
        cv2.line(img, (cx - corner, cy - corner), (cx - corner, cy - arm), cross_color, thickness)

        cv2.line(img, (cx + corner, cy - corner), (cx + arm, cy - corner), cross_color, thickness)
        cv2.line(img, (cx + corner, cy - corner), (cx + corner, cy - arm), cross_color, thickness)

        cv2.line(img, (cx - corner, cy + corner), (cx - arm, cy + corner), cross_color, thickness)
        cv2.line(img, (cx - corner, cy + corner), (cx - corner, cy + arm), cross_color, thickness)

        cv2.line(img, (cx + corner, cy + corner), (cx + arm, cy + corner), cross_color, thickness)
        cv2.line(img, (cx + corner, cy + corner), (cx + corner, cy + arm), cross_color, thickness)

        # ===============================
        #   SETTINGS
        # ===============================
        scale = 500
        min_length = 8  # pixels – ensures you always see the arrow
        thick = 2  # main line thickness
        tip = 0.30  # arrowhead size

        def draw_arrow(img, start, end, color, thickness=2, tipLength=0.3):
            """
            Draw a clean arrow with outline + anti-aliasing.
            Ensures all coordinates are Python ints.
            """

            # Convert all inputs to pure ints
            start = (int(start[0]), int(start[1]))
            end = (int(end[0]), int(end[1]))

            # Outline (black)
            cv2.arrowedLine(
                img, start, end, (0, 0, 0),
                thickness=thickness + 2, tipLength=tipLength,
                line_type=cv2.LINE_AA
            )
            # Main arrow
            cv2.arrowedLine(
                img, start, end, color,
                thickness=thickness, tipLength=tipLength,
                line_type=cv2.LINE_AA
            )

        # ===============================
        #        DRAW Vx (RED)
        # ===============================
        end_x_vx = int(cx + v_x_rot * scale)
        end_y_vx = cy

        # enforce min length
        if abs(end_x_vx - cx) < min_length:
            end_x_vx = cx + min_length * np.sign(v_x_rot)

        draw_arrow(
            img,
            (cx, cy),
            (end_x_vx, end_y_vx),
            (0, 0, 255),  # red
            thickness=thick,
            tipLength=tip
        )

        # ===============================
        #        DRAW Vy (GREEN)
        # ===============================
        end_x_vy = cx
        end_y_vy = int(cy + v_y_rot * scale)

        if abs(end_y_vy - cy) < min_length:
            end_y_vy = cy + min_length * np.sign(v_y_rot)

        draw_arrow(
            img,
            (cx, cy),
            (end_x_vy, end_y_vy),
            (0, 255, 0),  # green
            thickness=thick,
            tipLength=tip
        )

        # ===============================
        #    DRAW RESULTANT (YELLOW)
        # ===============================
        end_x = int(cx + v_x_rot * scale)
        end_y = int(cy + v_y_rot * scale)

        # keep min length
        if np.hypot(end_x - cx, end_y - cy) < min_length:
            direction = np.array([v_x_rot, v_y_rot])
            direction = direction / (np.linalg.norm(direction) + 1e-9)
            end_x = int(cx + direction[0] * min_length)
            end_y = int(cy + direction[1] * min_length)

        draw_arrow(
            img,
            (cx, cy),
            (end_x, end_y),
            (0, 255, 255),  # yellow
            thickness=2,
            tipLength=0.35
        )

        # ===============================
        #           DRAW Vz (BLUE)
        # ===============================
        v_z = row["delta_pos_z"].values[0]

        margin = 80
        start_x_vz = margin
        start_y_vz = margin

        end_x_vz = int(start_x_vz + v_z * scale)

        if abs(end_x_vz - start_x_vz) < min_length:
            end_x_vz = start_x_vz + min_length * np.sign(v_z)

        draw_arrow(
            img,
            (start_x_vz, start_y_vz),
            (end_x_vz, start_y_vz),
            (255, 0, 0),  # blue
            thickness=2,
            tipLength=0.3
        )

        img = cv2.rotate(img, cv2.ROTATE_180)

        # ===============================
        # Draw timestamp after rotation
        # ===============================
        # Position: top-left corner with some margin
        ts_margin = 30
        cv2.putText(
            img,
            f"t = {float(timestamp):.3f}s",  # three decimals
            (ts_margin, ts_margin),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),  # white text
            1
        )

        video.write(img)

    video.release()
    print("Video saved:", output_video_path)

def main():

    folder = r"D:\03_IL_preprocessed\experiment_1_(pull)\phase_1_approach"
    trials = [f for f in os.listdir(folder) if f.endswith(".csv")]

    for trial in trials:

        trial_name = trial.partition("_downsampled")[0]
        # ==== USER PATHS ====
        images_folder, csv_path, output_video_path = get_paths(trial_name)

        # Visualize Inhand Camera and Ground Truth Actions
        combine_inhand_camera_and_actions(images_folder, csv_path, output_video_path)


if __name__ == '__main__':

    main()