import cv2
import pandas as pd
import glob
import os
import numpy as np

# ==== USER PATHS ====
images_folder = "/media/alejo/IL_data/01_IL_bagfiles/experiment_1_(pull)/trial_1/robot/lfd_bag_palm_camera/camera_frames/gripper_rgb_palm_camera_image_raw"
csv_path = "/media/alejo/IL_data/03_IL_preprocessed/experiment_1_(pull)/phase_1_approach/trial_1_downsampled_aligned_data_(phase_1_approach).csv"
output_video_path = "/media/alejo/IL_data/03_IL_preprocessed/experiment_1_(pull)/phase_1_approach/trial_1.mp4"


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

    # Read vectors
    v_x = row["delta_pos_x"].values[0]
    v_y = row["delta_pos_y"].values[0]

    # Rotation angle in degrees
    angle_deg = 60 + 180
    angle_rad = np.radians(angle_deg)

    # Rotate vector
    v_x_rot = v_x * np.cos(angle_rad) - v_y * np.sin(angle_rad)
    v_y_rot = v_x * np.sin(angle_rad) + v_y * np.cos(angle_rad)

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
    #        DRAW Vx (RED)
    # ===============================
    scale = 500
    end_x_vx = int(cx + v_x_rot * scale)
    end_y_vx = cy  # only horizontal component

    cv2.arrowedLine(
        img,
        (cx, cy),
        (end_x_vx, end_y_vx),
        (0, 0, 255),    # red (BGR)
        thickness=2,
        tipLength=0.25
    )

    # ===============================
    #        DRAW Vy (GREEN)
    # ===============================
    end_x_vy = cx
    end_y_vy = int(cy - v_y_rot * scale)  # invert y

    cv2.arrowedLine(
        img,
        (cx, cy),
        (end_x_vy, end_y_vy),
        (0, 255, 0),    # green
        thickness=2,
        tipLength=0.25
    )

    # ===============================
    #     DRAW RESULTANT VECTOR (YELLOW)
    # ===============================
    end_x = int(cx + v_x_rot * scale)
    end_y = int(cy - v_y_rot * scale)

    cv2.arrowedLine(
        img,
        (cx, cy),
        (end_x, end_y),
        (0, 255, 255),   # yellow
        thickness=4,
        tipLength=0.3
    )

    # ===============================
    #          DRAW Vz (BLUE)
    # ===============================
    v_z = row["delta_pos_z"].values[0]
    

    # Bottom-right corner anchor point
    margin = 80
    start_x_vz = 0 + margin    # shift left so it fits fully
    start_y_vz = 0 + margin

    # Center dot
    cv2.circle(img, (start_x_vz, start_y_vz), 3, (255, 255, 255), -1)

    end_x_vz = int(start_x_vz + v_z * scale)
    end_y_vz = start_y_vz  # horizontal only

    cv2.arrowedLine(
        img,
        (start_x_vz, start_y_vz),
        (end_x_vz, end_y_vz),
        (255, 0, 0),   # blue (BGR)
        thickness=3,
        tipLength=0.25
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
