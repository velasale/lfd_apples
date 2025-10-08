
import os
import sqlite3
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import ast
import re
import array

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

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
def plot_3dpose(df):
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

    ax.plot(x, y, z, label='End-Effector Path', color = 'black')
    

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
    ax.text(x[0], y[0], z[0], "Start", color='black', fontsize=10, weight='bold')
    ax.text(x[-1], y[-1], z[-1], "End", color='black', fontsize=10, weight='bold')

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

    ax.plot(x, y, label = 'Shadow', zdir='z', zs=min(z), color='black', alpha=0.5)

    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Z (meters)')
    ax.set_title('3D End-Effector Position')
    ax.legend()    
    

def plot_wrench(df):
    """Plot forces and torques from a DataFrame containing a geometry_msgs/Wrench message."""
    # Position
    fx = np.array(df['_wrench._force._x'].to_list(), dtype=float)
    fy = np.array(df['_wrench._force._y'].to_list(), dtype=float)
    fz = np.array(df['_wrench._force._z'].to_list(), dtype=float)      
    tx = np.array(df['_wrench._torque._x'].to_list(), dtype=float)
    ty = np.array(df['_wrench._torque._y'].to_list(), dtype=float)
    tz = np.array(df['_wrench._torque._z'].to_list(), dtype=float)      
    t = np.array(df["elapsed_time"].to_list(), dtype=float)

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axs[0].plot(t, fx, label="Fx", color="r")
    axs[0].plot(t, fy, label="Fy", color="g")
    axs[0].plot(t, fz, label="Fz", color="b")
    axs[0].set_ylabel("Force [N]")
    axs[0].legend()
    axs[0].set_ylim([-12, 12])

    axs[1].plot(t, tx, label="Tx", color="r")
    axs[1].plot(t, ty, label="Ty", color="g")
    axs[1].plot(t, tz, label="Tz", color="b")
    axs[1].set_ylabel("Torque [Nm]")
    axs[1].set_xlabel("Time [s]")   
    axs[1].legend()
    axs[1].set_ylim([-3, 3])
    

    plt.tight_layout()


def plot_pressure(df):
    # Convert raw array field into list of ints
    df["_data"] = df["_data"].apply(parse_array)

    # Expand into new columns
    df[["p1", "p2", "p3", "other"]] = pd.DataFrame(df["_data"].tolist(), index=df.index)

    # Ensure they are numeric numpy arrays
    p1 = df["p1"].astype(float).to_numpy()
    p2 = df["p2"].astype(float).to_numpy()
    p3 = df["p3"].astype(float).to_numpy()
    t  = df["elapsed_time"].astype(float).to_numpy()

    # Plot pressures vs elapsed_time
    plt.figure(figsize=(10,6))
    plt.plot(t, p1, label="Pressure 1")
    plt.plot(t, p2, label="Pressure 2")
    plt.plot(t, p3, label="Pressure 3")

    plt.xlabel("Elapsed Time [s]")
    plt.ylabel("Pressure")
    plt.title("Pressure Signals vs Elapsed Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()




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



class Trial:
    def __init__(self, db3_file_path, csv_dir="csv_topics"):
        self.filepath = db3_file_path
        self.topics_names = []
        self.csv_dir = csv_dir
        os.makedirs(self.csv_dir, exist_ok=True)

        # If CSVs already exist → load them
        if self._csvs_exist():
            print("📂 Loading topics from CSVs...")
            self._load_from_csv()
        else:
            print("🗄️ Parsing bag file and saving CSVs...")
            self._load_topics()
            self.save_to_files()

        self.PARAMETER = 5.32

    def _csvs_exist(self):
        """Check if there are any CSVs in the output dir."""
        return any(fname.endswith(".csv") for fname in os.listdir(self.csv_dir))

    def _make_attr_name(self, topic_name):
        return topic_name.strip("/").replace("/", "_")

    def _load_topics(self):
        """Load topics from the .db3 file and keep them in memory as DataFrames."""
        conn = sqlite3.connect(self.filepath)
        cursor = conn.cursor()

        cursor.execute("SELECT id, name, type FROM topics")
        topics = cursor.fetchall()
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
                df["elapsed_time"] = (df["timestamp"] - t0) / 1e9

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

    def save_to_files(self, output_dir="bag_exports"):
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





if __name__ == "__main__":

    bag_file = "/home/alejo/franka_bags/franka_joint_bag_replay/franka_joint_bag_0.db3"
    trial = Trial(bag_file, csv_dir="bag_csvs")

    trial.list_topics()   # 🔹 prints all topics    
      

    # --- EEF POSE --- 
    # Access topics directly
    print(trial.joint_states.head())
    print(trial.franka_robot_state_broadcaster_current_pose.head())
    plot_3dpose(trial.franka_robot_state_broadcaster_current_pose)
    
    # --- EEF WRENCH ---
    print(trial.franka_robot_state_broadcaster_external_wrench_in_stiffness_frame.head())
    plot_wrench(trial.franka_robot_state_broadcaster_external_wrench_in_stiffness_frame)

    # --- PRESSURE SIGNALS ---
    plot_pressure(trial.microROS_sensor_data)
    

    plt.show()