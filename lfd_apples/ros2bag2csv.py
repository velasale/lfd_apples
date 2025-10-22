
import os
import sqlite3
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import ast
import re
import array
import itertools
from scipy.ndimage import gaussian_filter, median_filter

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

import cv2
import numpy as np

plt.rcParams.update({
    "text.usetex": False,          # don't need full LaTeX, use mathtext
    "font.family": "serif",        # use a serif font
    "mathtext.fontset": "cm",      # Computer Modern (classic LaTeX font)
    "font.size": 12,               # adjust global font size
})



# ----------------- FORWARD KINEMATICS ----------------- #
def dh_transform(a, alpha, d, theta):
    """Compute the Denavit-Hartenberg transformation matrix."""
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    # TF = Matrix([[cos(q),-sin(q), 0, a],
    #         [sin(q)*cos(alpha), cos(q)*cos(alpha), -sin(alpha), -sin(alpha)*d],
    #         [sin(q)*sin(alpha), cos(q)*sin(alpha),  cos(alpha),  cos(alpha)*d],
    #         [   0,  0,  0,  1]])

    # "DH Modified Convention"
    return np.array([
                    [ct, -st, 0, a],
                    [st * ca, ct * ca, -sa, -sa * d],
                    [st * sa, ct * sa,  ca,  ca * d],
                    [0, 0, 0, 1]
                        ])

    # "DH Standard Convention"
    # return np.array([
    #     [ct, -st * ca,  st * sa, a * ct],
    #     [st,  ct * ca, -ct * sa, a * st],
    #     [0,      sa,      ca,     d   ],
    #     [0,      0,       0,      1   ]
    # ])  


def fr3_fk(joint_angles):
    """Compute the forward kinematics for the Franka Emika Panda robot."""

    # Heads UP: Franka joints are saved in this order in the topic /franka/joint_states:
    # fr3_joint1
    # fr3_joint3
    # fr3_joint6
    # fr3_joint7
    # fr3_joint2
    # fr3_joint4
    # fr3_joint5

    # DH parameters for Franka Emika Panda
    dh_params = [
        (0,         0,          0.333,  joint_angles[0]),
        (0,         -np.pi/2,   0,      joint_angles[1]),
        (0,         np.pi/2,    0.316,  joint_angles[2]),
        (0.0825,    np.pi/2,    0,      joint_angles[3]),
        (-0.0825,   -np.pi/2,   0.384,  joint_angles[4]),
        (0,         np.pi/2,    0,      joint_angles[5]),
        (0.088,     np.pi/2,    0,      joint_angles[6]),
        (0,         0,          0.107,  0),
        (0,         0,          0.227,  0)      # end-effector (tool)
    ]

    T = np.eye(4)
    for a, alpha, d, theta in dh_params:
        T = T @ dh_transform(a, alpha, d, theta)
        # print('\n', a, alpha, d, theta)
        # print(T)  # Debug: print each transformation step


    return T

# ----------------- HANDY FUNCTIONS ----------------- #
def parse_position(s):
    # Convert bytes → string if needed
    if isinstance(s, (bytes, bytearray)):
        s = s.decode("utf-8", errors="ignore")
    else:
        s = str(s)

    # Extract inside [...]
    match = re.search(r"\[([^\]]+)\]", s)
    if not match:
        return np.array([])  # fallback if parsing fails

    numbers = match.group(1)
    return np.fromstring(numbers, sep=",")


def parse_array(x):
    # If it's already a Python array (from `array` module)
    if isinstance(x, array.array):
        return list(x)

    # If it's bytes, decode to string
    if isinstance(x, (bytes, bytearray)):
        x = x.decode("utf-8", errors="ignore")

    # If it's string like "array('h', [1006, 1006, 1007, 300])"
    if isinstance(x, str):
        match = re.search(r"\[(.*?)\]", x)      
               

        if match:
            # Check if values are floats or ints
            values = match.group(1).split(",")
            if '.' in values[0]:
                # print('float')
                return [float(v) for v in match.group(1).split(",")]
                        
            else:
                # print('int')
                return [int(v) for v in match.group(1).split(",")]                           
            
        else:
            # last resort: try literal_eval
            try:
                return list(ast.literal_eval(x))
            except Exception:
                return []
    
    # fallback
    return []


# ------------------ PLOTTING FUNCTIONS ----------------- #
def plot_3dpose(df, engagement_time=None, disposal_time=None):
    """Plot 3D positions from a DataFrame containing a geometry_msgs/Pose message."""

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  

    # Position
    x = np.array(df['_pose._position._x'].to_list(), dtype=float)
    y = np.array(df['_pose._position._y'].to_list(), dtype=float)
    z = np.array(df['_pose._position._z'].to_list(), dtype=float)      
    t = np.array(df["elapsed_time"].to_list(), dtype=float)

    # Orientation (quaternion)
    qx = np.array(df['_pose._orientation._x'].to_list(), dtype=float)
    qy = np.array(df['_pose._orientation._y'].to_list(), dtype=float)
    qz = np.array(df['_pose._orientation._z'].to_list(), dtype=float)
    qw = np.array(df['_pose._orientation._w'].to_list(), dtype=float)
    quats = np.column_stack((qx, qy, qz, qw))
    rot = R.from_quat(quats)

    # Split trajectory before and after engagement time
    if engagement_time is not None:
        mask_before = t < engagement_time
        mask_between = (t >= engagement_time) & (t < disposal_time)
        mask_after = t >= disposal_time

        ax.plot(x[mask_before], y[mask_before], z[mask_before],
                label='approach', color='orange', linewidth=2)
        ax.plot(x[mask_between], y[mask_between], z[mask_between],
                label='pick', color='green', linewidth=2)
        ax.plot(x[mask_after], y[mask_after], z[mask_after],
                label='disposal', color='purple', linewidth=2)
    else:
        ax.plot(x, y, z, label='End-Effector Path', color='black', linewidth=2)
    

    # Plot unit frames along the path every `step` points
    scale = 0.25  # length of the axes
    step = max(1, len(x) // 5)  # plot at most 20 frames   
    for xi, yi, zi, r in zip(x[::step], y[::step], z[::step], rot[::step]):
        # Local frame axes
        x_axis = r.apply([1, 0, 0]) * scale
        y_axis = r.apply([0, 1, 0]) * scale
        z_axis = r.apply([0, 0, 1]) * scale

        # Draw quivers
        ax.quiver(xi, yi, zi, x_axis[0], x_axis[1], x_axis[2], color='r', length=scale)
        ax.quiver(xi, yi, zi, y_axis[0], y_axis[1], y_axis[2], color='g', length=scale)
        ax.quiver(xi, yi, zi, z_axis[0], z_axis[1], z_axis[2], color='b', length=scale)

    # Add labels at start and end
    ax.text(x[0], y[0], z[0], "start", color='black', fontsize=10)
    ax.text(x[-1], y[-1], z[-1], "end", color='black', fontsize=10)

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    max_range = 0.8

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])

    ax.plot(x, y, label = 'shadow', zdir='z', zs=z_middle - max_range/2, color='black', alpha=0.5)

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    ax.set_title('3D End-Effector Position')
    ax.legend()    
    

def plot_wrench(df):
    """Plot forces and torques from a DataFrame containing a geometry_msgs/Wrench message."""
    # Position unfiltered signals
    fx = np.array(df['_wrench._force._x'].to_list(), dtype=float)
    fy = np.array(df['_wrench._force._y'].to_list(), dtype=float)
    fz = np.array(df['_wrench._force._z'].to_list(), dtype=float)      
    tx = np.array(df['_wrench._torque._x'].to_list(), dtype=float)
    ty = np.array(df['_wrench._torque._y'].to_list(), dtype=float)
    tz = np.array(df['_wrench._torque._z'].to_list(), dtype=float)      

    wrench = [fx, fy, fz, tx, ty, tz]
    t = np.array(df["elapsed_time"].to_list(), dtype=float)

    # Apply median filter to smooth the signals
    wrench_filtered = [gaussian_filter(w, 100) for w in wrench]
    
    # Tangential forces
    tangential_force = np.sqrt(wrench_filtered[0]**2 + wrench_filtered[1]**2)

    # --- Data sanity check ---
    zero_x, zero_y, zero_z = data_sanity_check(df)
    zero_t = t[zero_x]
    print(f'Zero force timestamps (s): {zero_t}')   


    # --- Plotting ---
    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

    axs[0].plot(t, wrench[0], color="r", alpha=0.3)
    axs[0].plot(t, wrench_filtered[0], label="Fx filtered", color="r")
    axs[0].plot(t, wrench[1], color="g", alpha=0.3)
    axs[0].plot(t, wrench_filtered[1], label="Fy filtered", color="g")
    axs[0].plot(t, wrench[2], color="b", alpha=0.3)
    axs[0].plot(t, wrench_filtered[2], label="Fz filtered", color="b")
    axs[0].set_ylabel("Force [N]")
    axs[0].legend()
    axs[0].set_ylim([-20, 20])
    axs[0].grid(True)

    # --- Vertical dashed lines at zero_t ---
    for zt in zero_t:    
        for ax in axs:
            ax.axvline(x=zt, color="k", linestyle="--", alpha=0.5)  # black dashed line

    axs[1].plot(t, tangential_force, color="brown", label="Tangential Force √(Fx²+Fy²)")
    axs[1].plot(t, wrench_filtered[2], color="b", label="Normal Force Fz")    
    axs[1].set_ylabel("Force [N]")
    axs[1].legend()
    axs[1].set_ylim([-20, 20])
    axs[1].grid(True)

    axs[2].plot(t, wrench[3], color="r", alpha=0.3)
    axs[2].plot(t, wrench_filtered[3], label="Tx filtered", color="r")
    axs[2].plot(t, wrench[4], color="g", alpha=0.3)
    axs[2].plot(t, wrench_filtered[4], label="Ty filtered", color="g")
    axs[2].plot(t, wrench[5], color="b", alpha=0.3)
    axs[2].plot(t, wrench_filtered[5], label="Tz filtered", color="b")
    axs[2].set_ylabel("Torque [Nm]")
    axs[2].set_xlabel("Elapsed Time [s]")   
    axs[2].legend()
    axs[2].set_ylim([-5, 5])
    axs[2].grid(True)
    
    plt.xlim([0,50])
    plt.tight_layout()    


def plot_pressure(df):
    # Convert raw array field into list of ints
    df["_data"] = df["_data"].apply(parse_array)

    colors = itertools.cycle(('#0072B2', '#E69F00', '#000000'))

    # Expand into new columns
    df[["p1", "p2", "p3", "tof"]] = pd.DataFrame(df["_data"].tolist(), index=df.index)

    # Ensure they are numeric numpy arrays
    p1 = df["p1"].astype(float).to_numpy() / 10  # convert to kPa
    p2 = df["p2"].astype(float).to_numpy() / 10  # convert to kPa 
    p3 = df["p3"].astype(float).to_numpy()  / 10  # convert to kPa
    tof = df["tof"].astype(float).to_numpy() / 10  # convert to cm
    t  = df["elapsed_time"].astype(float).to_numpy()

    # Apply median filter to smooth the signals
    tof_filtered = gaussian_filter(tof, 3)

    # Plot pressures vs elapsed_time
    fig, ax1 = plt.subplots(figsize=(10,6))
    # Left axis: Pressure
    ax1.plot(t, p1, label="suction cup a", color = next(colors))
    ax1.plot(t, p2, label="suction cup b", color = next(colors), linestyle='--')
    ax1.plot(t, p3, label="suction cup c", color = next(colors), linestyle='-.')
    ax1.set_xlabel("Elapsed Time [s]")
    ax1.set_ylabel("Air Pressure [kPa]")
    ax1.set_ylim([0,110])
    ax1.grid()

    # Right axis: Time-of-flight
    ax2 = ax1.twinx()
    ax2.plot(t, tof_filtered, label="tof filtered", color='blue')    
    ax2.plot(t, tof, color='blue', alpha=0.3)    
    ax2.set_ylabel("Time-of-Flight [cm]")
    ax2.set_ylim([0,31])
    # ax2.grid()

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.xlim([0,50])
    
    plt.title("Air Pressure Signals vs Elapsed Time")       
    
    plt.tight_layout()

# Check consecutive indexes
def find_consecutive_indexes(zero_indices):
    consecutive = []
    for i in range(len(zero_indices[0]) - 1):
        if zero_indices[0][i+1] == zero_indices[0][i] + 1:
            consecutive.append(zero_indices[0][i])

    just_first_indexes = []
    for j in consecutive:
        if j - 1 not in consecutive:
            just_first_indexes.append(j)

    return just_first_indexes


def data_sanity_check(df):
    """Check for zero wrench data points and print their indexes."""

    fx = np.array(df['_wrench._force._x'].to_list(), dtype=float)
    fy = np.array(df['_wrench._force._y'].to_list(), dtype=float)
    fz = np.array(df['_wrench._force._z'].to_list(), dtype=float)      
    tx = np.array(df['_wrench._torque._x'].to_list(), dtype=float)
    ty = np.array(df['_wrench._torque._y'].to_list(), dtype=float)
    tz = np.array(df['_wrench._torque._z'].to_list(), dtype=float)      

    zero_fx = find_consecutive_indexes(np.where(fx == 0))
    zero_fy = find_consecutive_indexes(np.where(fy == 0))
    zero_fz = find_consecutive_indexes(np.where(fz == 0))
    zero_tx = find_consecutive_indexes(np.where(tx == 0))
    zero_ty = find_consecutive_indexes(np.where(ty == 0))
    zero_tz = find_consecutive_indexes(np.where(tz == 0))


    if len(zero_fx) > 0 or len(zero_fy) > 0 or len(zero_fz) > 0:
        print(f'\n \033[93m !!! HEADS UP !!!\033[0m Wrench data has zero values at these indexes: \n \
              fx: {zero_fx} \n  fy: {zero_fy} \n  fz: {zero_fz} \n tx: {zero_tx} \n  ty: {zero_ty} \n  tz: {zero_tz}')
    else:
        print('\nWrench data sanity check passed: no zero values found.')    
      

    return zero_fx, zero_fy, zero_fz   


# ------------------ MAIN CLASSES & FUNCTIONS ----------------- #

def message_to_dict(msg):
    """Recursively flatten a ROS message into a dict."""
    if hasattr(msg, "__slots__"):
        result = {}
        for slot, slot_type in zip(msg.__slots__, msg.SLOT_TYPES):
            val = getattr(msg, slot)
            if hasattr(val, "__slots__"):  # nested
                nested = message_to_dict(val)
                for k, v in nested.items():
                    result[f"{slot}.{k}"] = v
            elif isinstance(val, (list, tuple)):
                for i, v in enumerate(val):
                    if hasattr(v, "__slots__"):
                        nested = message_to_dict(v)
                        for k, nv in nested.items():
                            result[f"{slot}[{i}].{k}"] = nv
                    else:
                        result[f"{slot}[{i}]"] = v
            else:
                result[slot] = val
        return result
    else:
        return {"data": msg}


def extract_images_from_bag(db3_file_path, output_dir="camera_frames", save_avi=True, fps=30):
    """
    Automatically detect camera topics (containing 'camera' or 'image_raw') in a ROS2 .db3 bag
    and extract frames for each one.
    """
    os.makedirs(output_dir, exist_ok=True)
    conn = sqlite3.connect(db3_file_path)
    cursor = conn.cursor()

    # Find all topics in the bag
    cursor.execute("SELECT id, name, type FROM topics")
    topics = cursor.fetchall()

    # Filter topics that look like camera topics
    camera_topics = [t for t in topics if "camera" in t[1].lower() or "image_raw" in t[1].lower()]
    if not camera_topics:
        print("⚠️ No camera topics found in bag.")
        conn.close()
        return

    for topic_id, topic_name, topic_type in camera_topics:
        print(f"\n🔹 Found camera topic: {topic_name} ({topic_type})")
        msg_class = get_message(topic_type)
        cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id=?", (topic_id,))
        rows = cursor.fetchall()
        if len(rows) == 0:
            print(f"⚠️ No messages in topic '{topic_name}'")
            continue

        # Create folder per topic
        topic_folder = os.path.join(output_dir, topic_name.strip("/").replace("/", "_"))
        os.makedirs(topic_folder, exist_ok=True)

        start_time = rows[0][0]
        video_writer = None

        for i, (timestamp, data) in enumerate(rows):
            msg = deserialize_message(data, msg_class)
            height, width = msg.height, msg.width
            encoding = msg.encoding.lower()
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)

            if encoding in ["rgb8", "bgr8"]:
                img = np_arr.reshape((height, width, 3))
            elif encoding == "mono8":
                img = np_arr.reshape((height, width))
            else:
                print(f"⚠️ Skipping frame {i}: unsupported encoding '{encoding}'")
                continue

            if encoding == "rgb8":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            t_sec = (timestamp - start_time) / 1e9
            filename = os.path.join(topic_folder, f"frame_{i:05d}_{t_sec:.6f}.jpg")
            cv2.imwrite(filename, img)

            # Initialize AVI writer
            if save_avi and video_writer is None:
                avi_path = os.path.join(topic_folder, f"{topic_name.strip('/').replace('/', '_')}.avi")
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                video_writer = cv2.VideoWriter(avi_path, fourcc, fps, (width, height))
                print(f"🎥 Saving AVI video to {avi_path} at {fps} FPS")

            if save_avi:
                img_color = img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                video_writer.write(img_color)

        if save_avi and video_writer is not None:
            video_writer.release()
        print(f"✅ Done extracting topic '{topic_name}' to '{topic_folder}'")

    conn.close()


class Trial:
    def __init__(self, db3_file_path, csv_dir="csv_topics"):
        self.filepath = db3_file_path
        self.topics_names = []
        self.csv_dir = csv_dir
        os.makedirs(self.csv_dir, exist_ok=True)

        self.engagement_time = 0.0
        self.disposal_time = 0.0
        self.first_timestamp = 0.0

        # If CSVs already exist → load them
        if self._csvs_exist():
            print("📂 Loading topics from CSVs...")
            self._load_from_csv()
        else:
            print("🗄️ Parsing bag file and saving CSVs...")
            self._load_topics()
            self.save_to_files(self.csv_dir)

        # Parameters
        self.PARAMETER = 5.32
        self.ENGAMENT_THRESHOLD = 600.0

    def _csvs_exist(self):
        """Check if there are any CSVs in the output dir."""
        return any(fname.endswith(".csv") for fname in os.listdir(self.csv_dir))

    def _make_attr_name(self, topic_name):
        return topic_name.strip("/").replace("/", "_")

    
    def get_first_timestamp(self, cursor):

        # Get topic ids and names
        cursor.execute("SELECT id, name FROM topics")
        topics = cursor.fetchall()

        first_timestamps = {}

        for topic_id, topic_name in topics:
            cursor.execute(
                "SELECT timestamp FROM messages WHERE topic_id=? ORDER BY timestamp ASC LIMIT 1",
                (topic_id,),
            )
            result = cursor.fetchone()
            if result:
                first_timestamps[topic_name] = result[0]

        # Find the smallest timestamp and corresponding topic
        earliest_topic = min(first_timestamps, key=first_timestamps.get)
        earliest_time = first_timestamps[earliest_topic]

        self.first_timestamp = earliest_time

        print("\n⏱️ First message timestamps per topic:")
        for topic, ts in first_timestamps.items():
            print(f"\n  {topic}: {ts}")

        print("\n🕒 Earliest message across all topics:")
        print(f"  Topic: {earliest_topic}")
        print(f"  Timestamp (ns): {earliest_time}")
        print(f"  Timestamp (s):  {self.first_timestamp / 1e9:.6f}")
    
    def _load_topics(self):
        """Load topics from the .db3 file and keep them in memory as DataFrames."""
        conn = sqlite3.connect(self.filepath)
        cursor = conn.cursor()

        cursor.execute("SELECT id, name, type FROM topics")
        topics = cursor.fetchall()

        self.get_first_timestamp(cursor)

        topic_map = {t[0]: (t[1], t[2]) for t in topics}

        for topic_id, (topic_name, topic_type) in topic_map.items():
            try:
                msg_class = get_message(topic_type)
            except Exception as e:
                print(f"⚠️ Skipping {topic_name}: could not load {topic_type} ({e})")
                continue

            cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id=?", (topic_id,))
            rows = cursor.fetchall()

            records = []
            for timestamp, data in rows:
                msg = deserialize_message(data, msg_class)
                msg_dict = message_to_dict(msg)
                msg_dict["timestamp"] = timestamp
                records.append(msg_dict)

            if records:
                df = pd.DataFrame(records)
                t0 = df["timestamp"].iloc[0]
                df["elapsed_time"] = (df["timestamp"] - self.first_timestamp) / 1e9

                attr_name = self._make_attr_name(topic_name)
                setattr(self, attr_name, df)
                self.topics_names.append(attr_name)

        conn.close()

    def _load_from_csv(self):
        """Load topics from saved CSVs."""
        for fname in os.listdir(self.csv_dir):
            if fname.endswith(".csv"):
                attr_name = fname.replace(".csv", "")
                df = pd.read_csv(os.path.join(self.csv_dir, fname))
                setattr(self, attr_name, df)
                self.topics_names.append(attr_name)

    def list_topics(self):
        print("Available topics:")
        for t in self.topics_names:
            print("  -", t)

    def __repr__(self):
        attrs = [a for a in dir(self) if not a.startswith("_") and isinstance(getattr(self, a), pd.DataFrame)]
        return f"<Trial file={os.path.basename(self.filepath)}, topics={attrs}>"

    def save_to_files(self, output_dir):
        """Save each topic DataFrame to CSV and JSON."""
        os.makedirs(output_dir, exist_ok=True)

        for attr_name in self.topics_names:
            df = getattr(self, attr_name)
            if not isinstance(df, pd.DataFrame):
                continue

            # --- Save CSV ---
            csv_path = os.path.join(output_dir, f"{attr_name}.csv")
            if not os.path.exists(csv_path):
                df.to_csv(csv_path, index=False)
                print(f"💾 Saved CSV: {csv_path}")

            # # --- Save JSON ---
            # json_path = os.path.join(output_dir, f"{attr_name}.json")
            # if not os.path.exists(json_path):
            #     df.to_json(json_path, orient="records", lines=True)
            #     print(f"💾 Saved JSON: {json_path}")

    def get_engagement_time(self):
        
        """Estimate the time when the gripper engaged based on pressure data."""
        if not hasattr(self, "microROS_sensor_data"):
            print("⚠️ No microROS_sensor_data topic found.")
            return None     
        df = self.microROS_sensor_data.copy()
        df["_data"] = df["_data"].apply(parse_array)
        df[["p1", "p2", "p3", "other"]] = pd.DataFrame(df["_data"].tolist(), index=df.index)
        p1 = df["p1"].astype(float).to_numpy()
        p2 = df["p2"].astype(float).to_numpy()
        p3 = df["p3"].astype(float).to_numpy()
        t  = df["elapsed_time"].astype(float).to_numpy()    

        engaged_indices = np.where((p1 < self.ENGAMENT_THRESHOLD) &
                                   (p2 < self.ENGAMENT_THRESHOLD) &
                                   (p3 < self.ENGAMENT_THRESHOLD))[0]
        if len(engaged_indices) == 0:
            print("⚠️ No engagement detected based on pressure threshold.")
            return None 
        self.engagement_time = t[engaged_indices[0]]
        print(f"✅ Engagement detected at t = {self.engagement_time:.2f} s")    
        
        return self.engagement_time


    def get_disposal_time(self):
        
        """Estimate the time when the gripper engaged based on pressure data."""
        if not hasattr(self, "microROS_sensor_data"):
            print("⚠️ No microROS_sensor_data topic found.")
            return None     
        df = self.microROS_sensor_data.copy()
        df["_data"] = df["_data"].apply(parse_array)
        df[["p1", "p2", "p3", "other"]] = pd.DataFrame(df["_data"].tolist(), index=df.index)
        p1 = df["p1"].astype(float).to_numpy()
        p2 = df["p2"].astype(float).to_numpy()
        p3 = df["p3"].astype(float).to_numpy()
        t  = df["elapsed_time"].astype(float).to_numpy()    

        disposal_indices = np.where((p1 > self.ENGAMENT_THRESHOLD) &
                                   (p2 > self.ENGAMENT_THRESHOLD) &
                                   (p3 > self.ENGAMENT_THRESHOLD) &
                                   (t > self.engagement_time))[0]
        if len(disposal_indices) == 0:
            print("⚠️ No engagement detected based on pressure threshold.")
            return None 
        self.disposal_time = t[disposal_indices[0]]
        print(f"✅ Disposal detected at t = {self.disposal_time:.2f} s")    
        
        return self.engagement_time



def extract_data_and_plot(bag_folder, trial_folder):
    
    bag_file = bag_folder + "/" + trial_folder + "/lfd_bag_main/lfd_bag_main_0.db3"
    csv_dir = bag_folder + "/" + trial_folder + "/lfd_bag_main/bag_csvs"

    trial = Trial(bag_file, csv_dir)
    trial.list_topics()   # 🔹 prints all topics    
    trial.get_engagement_time()  # 🔹 estimates time of engagement from pressure data
    trial.get_disposal_time()  # 🔹 estimates time of engagement from pressure data
   
    output_frames = os.path.join(bag_folder, trial_folder, "lfd_bag_palm_camera", "camera_frames")     
    extract_images_from_bag(
        db3_file_path=os.path.join(bag_folder,trial_folder, "lfd_bag_palm_camera","lfd_bag_palm_camera_0.db3"),
        output_dir=output_frames
    )

    output_frames = os.path.join(bag_folder, trial_folder, "lfd_bag_fixed_camera", "camera_frames")     
    extract_images_from_bag(
        db3_file_path=os.path.join(bag_folder, trial_folder, "lfd_bag_fixed_camera","lfd_bag_fixed_camera_0.db3"),
        output_dir=output_frames
    )

    # --- EEF POSE --- 
    # Access topics directly
    print(trial.joint_states.head())
    print(trial.franka_robot_state_broadcaster_current_pose.head())
    plot_3dpose(trial.franka_robot_state_broadcaster_current_pose, trial.engagement_time, trial.disposal_time)
    
    # --- EEF WRENCH ---
    print(trial.franka_robot_state_broadcaster_external_wrench_in_stiffness_frame.head())
    plot_wrench(trial.franka_robot_state_broadcaster_external_wrench_in_stiffness_frame)

    # --- PRESSURE SIGNALS ---
    plot_pressure(trial.microROS_sensor_data)    
    

    plt.show()
    




if __name__ == "__main__":

    bag_folder = "/home/alejo/lfd_bags/experiment_1"    
    trial_folder = "trial_78"
    extract_data_and_plot(bag_folder, trial_folder)
    
    