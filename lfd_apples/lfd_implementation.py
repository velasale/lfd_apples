#!/usr/bin/env python3
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from controller_manager_msgs.srv import SwitchController, LoadController, ListControllers
import subprocess
import time
import os
import matplotlib.pyplot as plt
import pandas as pd

from lfd_apples.listen_franka import main as listen_main
from lfd_apples.listen_franka import start_recording_bagfile, stop_recording_bagfile, save_metadata, find_next_trial_number 
from lfd_apples.ros2bag2csv import extract_data_and_plot, parse_array, fr3_jacobian

from std_msgs.msg import Int16MultiArray
from std_srvs.srv import SetBool
from geometry_msgs.msg import PoseStamped, TwistStamped, Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from lfd_apples.lfd_vision import extract_pooled_latent_vector
from ultralytics import YOLO
import cv2
import pickle
import joblib
import numpy as np




class LFDController(Node):

    def __init__(self):
        super().__init__('move_to_home_and_freedrive')

        # Topic Subscribers
        self.distance_sub = self.create_subscription(
            Int16MultiArray, 'microROS/sensor_data', self.gripper_sensors_callback, 10)
        self.eef_pose_sub = self.create_subscription(
            PoseStamped, '/franka_robot_state_broadcaster/current_pose', self.eef_pose_callback, 10)
        self.palm_camera_sub = self.create_subscription(
            Image, 'gripper/rgb_palm_camera/image_raw', self.palm_camera_callback, 10)       
        self.joint_states_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_states_callback, 10)  # Dummy subscriber to ensure joint states are available
        
        # Topic Publishers
        self.vel_pub = self.create_publisher(
            TwistStamped, '/cartesian_velocity_controller/command', 10)
        
        # --- Timer for high-rate velocity ramping ---
        self.timer_period = 0.03  # 500 Hz
        self.create_timer(self.timer_period, self.publish_smoothed_velocity)

        # --- Velocity ramping variables ---
        self.current_cmd = TwistStamped()
        self.target_cmd = TwistStamped()        

        self.current_linear_accel  = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.current_angular_accel = {'x': 0.0, 'y': 0.0, 'z': 0.0}

        self.SCALING_FACTOR = 0.01
        self.max_linear_accel   = 0.1 * self.SCALING_FACTOR   # m/s²
        self.max_angular_accel  = 0.2 * self.SCALING_FACTOR    # rad/s²

        self.max_linear_jerk    = 0.5 * self.SCALING_FACTOR    # m/s³
        self.max_angular_jerk   = 1.0 * self.SCALING_FACTOR    # rad/s³   

        self.last_cmd_time = self.get_clock().now()

        # Action Client
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            'fr3_arm_controller/follow_joint_trajectory'
        )
        
        # Controller names
        self.arm_controller = 'fr3_arm_controller'
        self.gravity_controller = 'gravity_compensation_example_controller'
        self.eef_velocity_controller = 'cartesian_velocity_controller'

        # Switch controller client
        self.switch_client = self.create_client(SwitchController, '/controller_manager/switch_controller')
        self.switch_client.wait_for_service()

        # Load controller client
        self.load_client = self.create_client(LoadController, '/controller_manager/load_controller')
        self.load_client.wait_for_service()

        self.joint_names = [f"fr3_joint{i+1}" for i in range(7)]
        self.joint_states = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.trajectory_points = []       

        # Home position with eef pose starting on - x
        self.home_positions = [0.6414350870607822,
                               -1.5320604540253377,
                               0.4850253317447517,
                               -2.474376823551583,
                               0.9726833812685999,
                               2.1330229376987626,
                               -1.0721952822461973]
        
        # Home position with eef pose starting on + x
        # self.home_positions = [1.8144304752349854,
        #                        -1.0095794200897217,
        #                        -0.8489214777946472,
        #                        -2.585618019104004,
        #                        0.9734971523284912,
        #                        2.7978947162628174,
        #                        -2.0960772037506104]

        self.disposal_joints_positions = []

        # Flags
        self.approach_accomplished = False
        self.contact_accomplished = False
        self.pick_accomplished = False
        self.data_ready = False
        self.running_lfd_approach = False
        self.running_lfd_contact = False
        self.running_lfd_pick = False        

        # Thresholds
        self.tof_threshold = 50                 # units in mm
        self.air_pressure_threshold = 600       # units in hPa

        # Computer Vision
        self.bridge = CvBridge()
        self.raw_image = None
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pt_path = os.path.join(script_dir, "resources", "best_segmentation.pt")
        self.yolo_model = YOLO(pt_path)
        self.yolo_latent_layer = 12

        # State Space Vectors
        self.KEEP_ACTIONS_MEMORY = False
        if self.KEEP_ACTIONS_MEMORY:
            self.t_2_data = np.zeros(142)       
            self.t_1_data = np.zeros(142)
        else:
            self.t_2_data = np.zeros(136)       
            self.t_1_data = np.zeros(136)
            

        self.t_data = []
        self.tof = np.zeros(1)
        self.latent_image = np.zeros(127)
        self.eef_pose = np.zeros(7)
        self.Y = np.zeros(6)

        # Load learned model
        phase = 'phase_1_approach'
        model = 'mlp'
        timesteps = '2_timesteps'       
        BASE_DIR = '/home/alejo/Documents/DATA'
        # BASE_DIR = '/media/alejo/IL_data'
        model_path = os.path.join(BASE_DIR, '05_IL_learning/experiment_1_(pull)', phase, timesteps)
        model_name = model + '_experiment_1_(pull)_' + phase + '_' + timesteps + '.joblib'
        with open(os.path.join(model_path, model_name), "rb") as f:
            # self.lfd_model = pickle.load(f)
            self.lfd_model = joblib.load(f)
        self.lfd_X_mean = np.load(os.path.join(model_path, model + '_Xmean_experiment_1_(pull)_' + phase + '_' + timesteps + '.npy'))
        self.lfd_X_std = np.load(os.path.join(model_path, model + '_Xstd_experiment_1_(pull)_' + phase + '_' + timesteps + '.npy'))   
        self.lfd_Y_mean = np.load(os.path.join(model_path, model + '_Ymean_experiment_1_(pull)_' + phase + '_' + timesteps + '.npy'))
        self.lfd_Y_std = np.load(os.path.join(model_path, model + '_Ystd_experiment_1_(pull)_' + phase + '_' + timesteps + '.npy'))   

        # DataFrame to store normalized states over time
        self.lfd_states_df = pd.DataFrame()
        self.lfd_actions_df = pd.DataFrame()      


       
     
                
    def move_to_home(self):
        self.get_logger().info('Waiting for action server...')
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server not available')
            return False                 

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = self.home_positions
        point.time_from_start.sec = 5
        goal_msg.trajectory.points.append(point)

        self.get_logger().info('Sending trajectory goal...')
        send_goal_future = self.action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return False

        self.get_logger().info('Goal accepted, waiting for result...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        self.get_logger().info(f'Motion complete with error code: {result.error_code}')
        return True

    def ensure_controller_loaded(self, controller_name):
        """Load a controller if not already loaded."""
        # --- Step 1: check if controller is already loaded
        list_client = self.create_client(ListControllers, '/controller_manager/list_controllers')
        list_client.wait_for_service()
        list_req = ListControllers.Request()
        future = list_client.call_async(list_req)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()

        if any(c.name == controller_name for c in resp.controller):
            self.get_logger().info(f"Controller '{controller_name}' is already loaded, skipping load.")
            return

        # --- Step 2: load controller if not loaded
        load_req = LoadController.Request()
        load_req.name = controller_name
        future = self.load_client.call_async(load_req)
        rclpy.spin_until_future_complete(self, future)
        load_resp = future.result()
        if load_resp.ok:
            self.get_logger().info(f"Controller '{controller_name}' loaded successfully.")
        else:
            self.get_logger().error(f"Failed to load controller '{controller_name}'.")

    def configure_controller(self, controller_name):
        """Configure controller to inactive if it's not already inactive."""
        list_client = self.create_client(ListControllers, '/controller_manager/list_controllers')
        list_client.wait_for_service()
        future = list_client.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()

        ctrl_state = None
        for c in resp.controller:
            if c.name == controller_name:
                ctrl_state = c.state
                break

        if ctrl_state == 'inactive':
            self.get_logger().info(f"Controller {controller_name} already inactive, skipping.")
            return

        # Otherwise, set to inactive
        try:
            subprocess.run(['ros2', 'control', 'set_controller_state', controller_name, 'inactive'], check=True)
            self.get_logger().info(f'Controller {controller_name} configured to inactive.')
        except subprocess.CalledProcessError:
            self.get_logger().error(f'Failed to configure {controller_name} to inactive.')
        
    def swap_controller(self, stop_controller: str, start_controller: str, settle_time: float = 1.0):
        """
        Safely switch from one ROS2 controller to another (Franka-safe).
        stop_controller: controller name to deactivate
        start_controller: controller name to activate
        settle_time: seconds to wait between switches (default 1.5s)
        """
        # --- Step 1: Deactivate current controller ---
        if stop_controller:
            self.get_logger().info(f"Deactivating controller: {stop_controller}")
            switch_req = SwitchController.Request()
            switch_req.deactivate_controllers = [stop_controller]
            switch_req.activate_controllers = []
            switch_req.strictness = 2  # BEST_EFFORT

            future = self.switch_client.call_async(switch_req)
            rclpy.spin_until_future_complete(self, future)
            resp = future.result()

            if not resp or not resp.ok:
                self.get_logger().warn(f"⚠️ Failed to deactivate {stop_controller}. Continuing anyway.")
            else:
                self.get_logger().info(f"✅ {stop_controller} deactivated successfully.")

        # --- Step 2: Wait for safety ---
        self.get_logger().info(f"Waiting {settle_time:.1f}s before activating {start_controller}...")
        time.sleep(settle_time)

        # --- Step 3: Load and configure next controller ---
        self.ensure_controller_loaded(start_controller)
        self.configure_controller(start_controller)

        # --- Step 4: Activate next controller ---
        self.get_logger().info(f"Activating controller: {start_controller}")
        activate_req = SwitchController.Request()
        activate_req.deactivate_controllers = []
        activate_req.activate_controllers = [start_controller]
        activate_req.strictness = 2  # BEST_EFFORT

        future = self.switch_client.call_async(activate_req)
        rclpy.spin_until_future_complete(self, future)
        resp = future.result()

        if not resp or not resp.ok:
            self.get_logger().error(f"❌ Failed to activate {start_controller}.")
            return False

        self.get_logger().info(f"✅ {start_controller} activated successfully.")
        return True

    def replay_joints(self, csv_path, downsample_factor=1, speed_factor=1.0, ramp_time=1.0):

        self.get_logger().info(f"Loading trajectory from {csv_path}")
        df = pd.read_csv(csv_path)

        # Convert '_position' column and downsample
        df_joints = pd.DataFrame(
            df["_position"].apply(parse_array).apply(lambda x: x[:7]).tolist(),
            columns=[f"fr3_joint{i+1}" for i in range(7)]
        ).iloc[::downsample_factor].reset_index(drop=True)

        # Debug print
        self.get_logger().info(f"Raw points loaded: {len(df_joints)}")

        # --- clear previous trajectory points ---
        self.trajectory_points = []

        # Original data recorded at 1 kHz → dt = 0.001 s
        # Adjusted for downsampling and speed scaling
        original_dt = 0.040 # Original time step in seconds
        dt = original_dt * downsample_factor / speed_factor
        time_from_start = ramp_time

        for _, row in df_joints.iterrows():
            point = JointTrajectoryPoint()
            point.positions = row.tolist()
            time_from_start += dt
            point.time_from_start.sec = int(time_from_start)
            point.time_from_start.nanosec = int((time_from_start - int(time_from_start)) * 1e9)
            self.trajectory_points.append(point)

        total_duration = time_from_start
        self.get_logger().info(f"Prepared {len(self.trajectory_points)} trajectory points, total_duration={total_duration:.3f}s")


        # --- Smooth ramp from current position to first point ---
        try:
            joint_state_msg = rclpy.wait_for_message('/joint_states', JointState, node=self, timeout=2.0)
            current_positions = list(joint_state_msg.position[:7])
            self.get_logger().info("Got current joint state for smooth ramp.")
        except Exception:
            current_positions = self.home_positions
            self.get_logger().warn("Could not get /joint_states, using home positions for ramp.")

        first_point_positions = self.trajectory_points[0].positions

        # Generate ramp points
        ramp_points = []
        ramp_steps = max(int(ramp_time / dt), 2)  # at least 2 steps
        for i in range(1, ramp_steps + 1):
            alpha = i / ramp_steps
            ramp_pos = [
                (1 - alpha) * cur + alpha * target
                for cur, target in zip(current_positions, first_point_positions)
            ]
            point = JointTrajectoryPoint()
            point.positions = ramp_pos
            point.velocities = [0.0] * 7
            point.time_from_start.sec = int(alpha * ramp_time)
            point.time_from_start.nanosec = int((alpha * ramp_time - int(alpha * ramp_time)) * 1e9)
            ramp_points.append(point)

        # Insert ramp points at the beginning
        self.trajectory_points = ramp_points + self.trajectory_points
        self.get_logger().info(f"Inserted {len(ramp_points)} ramp points for smooth start.")

        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Action server not available")
            return        

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names
        goal_msg.trajectory.points = self.trajectory_points

        # Let the controller settle after activation
        time.sleep(1.0)

        self.get_logger().info("Sending trajectory via FollowJointTrajectory action...")
        send_goal_future = self.action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected!")
            return

        self.get_logger().info("Trajectory goal accepted, waiting for result...")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        self.get_logger().info(f"Trajectory execution finished: {result}")

    def run_lfd_approach(self, n_timesteps=2):       

        self.get_logger().info("Starting run_lfd_approach: publishing twists from palm_camera_callback until TOF < threshold.")
        self.running_lfd_approach = True

        while rclpy.ok() and self.running_lfd_approach:

            if self.tof.item() < self.tof_threshold:
                self.get_logger().info(f"TOF threshold reached: {self.tof} < {self.tof_threshold}")                
                self.send_stop_message()
                self.running_lfd_approach = False

            rclpy.spin_once(self,timeout_sec=0.0)
            time.sleep(0.001)
        
        self.send_stop_message()
        self.running_lfd_approach = False

    def send_stop_message(self):
        """
        # Ensure a final zero-twist is sent to stop motion
        
        :param self: Description
        """

        self.target_cmd.twist = Twist()  # zero target
        self.current_cmd.header.stamp = self.get_clock().now().to_msg()
        self.vel_pub.publish(self.current_cmd)


    def compute_joint_velocities(self, desired_ee_velocity: np.ndarray, joint_positions: list):
        """
        desired_ee_velocity: 6-element NumPy array
        joint_positions: list of 7 floats
        returns: 7-element NumPy array
        """
        jacobian = fr3_jacobian(joint_positions)
        joint_velocities = np.linalg.pinv(jacobian) @ desired_ee_velocity
        self.get_logger().info(f'Joint velocities: {joint_velocities}')

    # === Sensor Topics Callback ====
    def gripper_sensors_callback(self, msg: Int16MultiArray):

        # get current msg
        self.scA = float(msg.data[0])
        self.scB = float(msg.data[1])
        self.scC = float(msg.data[2])
        self.tof = float(msg.data[3])

        self.tof = np.array([self.tof])

        # Update sensors average since last action
                

    def palm_camera_callback(self, msg: Image):
        """
        We'll use this callback to publish the twist commands
        
        :param self: Description
        :param msg: Description
        :type msg: Image
        """

        if self.running_lfd_approach:
            
            # ====================== PROCESS DATA ======================
            # --- Process Image ---
            self.raw_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Get latent space features
            pooled_vector, feat_map = extract_pooled_latent_vector(
                self.raw_image,
                self.yolo_model,
                layer_index=self.yolo_latent_layer
            )
            self.latent_image = pooled_vector.tolist()
            # --- Combine data ---
            self.t_data = np.concatenate([self.tof, self.latent_image, self.eef_pose])            
            # Combine data, with past steps            
            self.X = np.concatenate([self.t_2_data, self.t_1_data, self.t_data])
            # Normalize data                        
            self.X_norm = (self.X - self.lfd_X_mean) / self.lfd_X_std
            # --- Append as row in DataFrame ---
            self.lfd_states_df = pd.concat([
                self.lfd_states_df, 
                # pd.DataFrame([self.t_data_norm])
                pd.DataFrame([self.X_norm])
            ], ignore_index=True)
            
            # ======================= Predict Actions ======================
            # Predict normalized actions
            self.Y_norm = self.lfd_model.predict(self.X_norm.reshape(1, -1))
            # Denormalize actions
            self.Y = self.Y_norm * self.lfd_Y_std + self.lfd_Y_mean
                        
            scaling_factor = 1.0
            self.Y = self.Y * scaling_factor            
            y = self.Y[0]
            # --- Append as row in DataFrame ---
            self.lfd_actions_df = pd.concat([
                self.lfd_actions_df, 
                # pd.DataFrame([self.t_data_norm])
                pd.DataFrame([y])

            ], ignore_index=True)
            self.get_logger().info(f'Target actions: {y}')       

            # Send actions (twist)        
            self.target_cmd.twist.linear.x = float(y[0])
            self.target_cmd.twist.linear.y = float(y[1])
            self.target_cmd.twist.linear.z = float(y[2])
            self.target_cmd.twist.angular.x = float(y[3])
            self.target_cmd.twist.angular.y = float(y[4])
            self.target_cmd.twist.angular.z = float(y[5])

            self.get_logger().info(f"pal camera callback sending topic")
            self.Y = np.array([self.target_cmd.twist.linear.x,
                                self.target_cmd.twist.linear.y,
                                self.target_cmd.twist.linear.z,
                                self.target_cmd.twist.angular.x,
                                self.target_cmd.twist.angular.y,
                                self.target_cmd.twist.angular.z], dtype=float)
            
            self.compute_joint_velocities(self.Y, self.joint_states)


            # Add actions to State Space to pass them to the next time steps
            self.t_2_data = self.t_1_data

            if self.KEEP_ACTIONS_MEMORY:
                self.t_1_data = np.concatenate([self.t_data, self.Y])
            else:
                self.t_1_data = self.t_data
            
            
        
    def publish_smoothed_velocity(self):
        if not self.running_lfd_approach:
            return

        dt = (self.get_clock().now() - self.last_cmd_time).nanoseconds * 1e-9
        self.last_cmd_time = self.get_clock().now()

        if dt <= 0.0:
            return

        # --- Linear axes ---
        for axis in ['x', 'y', 'z']:
            v_cur = getattr(self.current_cmd.twist.linear, axis)
            v_tgt = getattr(self.target_cmd.twist.linear, axis)
            a_cur = self.current_linear_accel[axis]

            # Desired acceleration
            a_des = (v_tgt - v_cur) / dt
            a_des = np.clip(a_des, -self.max_linear_accel, self.max_linear_accel)

            # Jerk-limited acceleration update
            da = a_des - a_cur
            da = np.clip(da, -self.max_linear_jerk * dt,
                            self.max_linear_jerk * dt)

            a_new = a_cur + da
            v_new = v_cur + a_new * dt

            self.current_linear_accel[axis] = a_new
            setattr(self.current_cmd.twist.linear, axis, v_new)

        # --- Angular axes ---
        for axis in ['x', 'y', 'z']:
            v_cur = getattr(self.current_cmd.twist.angular, axis)
            v_tgt = getattr(self.target_cmd.twist.angular, axis)
            a_cur = self.current_angular_accel[axis]

            a_des = (v_tgt - v_cur) / dt
            a_des = np.clip(a_des, -self.max_angular_accel, self.max_angular_accel)

            da = a_des - a_cur
            da = np.clip(da, -self.max_angular_jerk * dt,
                            self.max_angular_jerk * dt)

            a_new = a_cur + da
            v_new = v_cur + a_new * dt

            self.current_angular_accel[axis] = a_new
            setattr(self.current_cmd.twist.angular, axis, v_new)

        self.current_cmd.header.stamp = self.get_clock().now().to_msg()
        self.vel_pub.publish(self.current_cmd)


    def eef_pose_callback(self, msg: PoseStamped):

        # get pose
        self.eef_pos_x = msg.pose.position.x
        self.eef_pos_y = msg.pose.position.y
        self.eef_pos_z = msg.pose.position.z
        self.eef_ori_x = msg.pose.orientation.x
        self.eef_ori_y = msg.pose.orientation.y
        self.eef_ori_z = msg.pose.orientation.z
        self.eef_ori_w = msg.pose.orientation.w

        self.eef_pose = np.array([self.eef_pos_x,
                                  self.eef_pos_y,
                                  self.eef_pos_z,
                                  self.eef_ori_x,
                                  self.eef_ori_y,
                                  self.eef_ori_z,
                                  self.eef_ori_w])

        # Update sensors average since last action
    

    def joint_states_callback(self, msg: JointState):

        self.joint_states = list(msg.position[:7])

        # Dummy callback to ensure joint states are available



def check_data_plots(BAG_DIR, trial_number, inhand_camera_bag=True):

    print("Extracting data and generating plots...")

    try:
        plt.ion()  # <-- interactive mode ON
        # extract_data_and_plot(os.path.join(BAG_DIR, TRIAL), "")
        extract_data_and_plot(BAG_DIR, trial_number, inhand_camera_bag)
        print("✅ Data extraction complete.")

        # Prompt user to hit Enter to close figures
        input("\n\033[1;32m - Plots generated. Check how things look and press ENTER to close all figures.\033[0m\n")
        
        # Close all matplotlib figures
        plt.close('all')

        # Prompt user to take notes in JSON file if needed
        input("\n\033[1;32m - Annotate the json file if needed, and press ENTER to continue with next demo.\033[0m\n")


    except Exception as e:
        print(f"❌ Error during data extraction: {e}")


def main():

    rclpy.init()

    node = LFDController()    

    BASE_DIR = '/home/alejo/Documents/DATA'
    BAG_MAIN_DIR = '06_IL_implementation/bagfiles'
    EXPERIMENT = "experiment_1_(pull)"
    BAG_FILEPATH = os.path.join(BASE_DIR, BAG_MAIN_DIR, EXPERIMENT)
    os.makedirs(BAG_FILEPATH, exist_ok=True)
    
    batch_size = 10
    node.get_logger().info(f"Starting lfd implementation with robot, session of {batch_size} demos.")

    for demo in range(batch_size):

        node.get_logger().info("\033[1;32m ---------- Press Enter to start lfd implementation trial {}/10 ----------\033[0m".format(demo+1))
        input()  # Wait for user to press Enter
      
        # ------------ Step 0: Initial configuration ----------------
        TRIAL = find_next_trial_number(BAG_FILEPATH, prefix="trial_")
        os.makedirs(os.path.join(BAG_FILEPATH, TRIAL), exist_ok=True)
        # HUMAN_BAG_FILEPATH = os.path.join(BAG_FILEPATH, TRIAL, 'human')
        ROBOT_BAG_FILEPATH = os.path.join(BAG_FILEPATH, TRIAL)               
        

        # ------------ Step 1: Move robot to home position ---------
        node.get_logger().info("Moving to home position...")
        while not node.move_to_home():
            pass

        # Enable freedrive mode and record demonstration                
        node.swap_controller(node.arm_controller, node.gravity_controller)
        time.sleep(1.0)        
        # Start recording                
        node.get_logger().info("Human Demo. Free-drive arm close to apple.")        

        input("\n\033[1;32m - Press ENTER when you are done.\033[0m\n")    


        # Enable cartesian velocity controller
        node.get_logger().info("Switching to Cartesian velocity controller...")
        node.swap_controller(node.gravity_controller, node.eef_velocity_controller)     
        time.sleep(1.0)        
        # node.swap_controller(node.arm_controller, node.eef_velocity_controller)

        # Start recording
        input("\n\033[1;32m3 - Place apple on the proxy. Press ENTER when done.\033[0m\n")
        robot_rosbag_list = start_recording_bagfile(ROBOT_BAG_FILEPATH)


        # -------------- Step 2: Run lfd controller ----------------        
        input("\n\033[1;32m4 - Press Enter to start ROBOT lfd implementation.\033[0m\n")        
        node.run_lfd_approach()      

        # Save states to CSV        
        csv_path = os.path.join(ROBOT_BAG_FILEPATH, 'lfd_recorded_data.csv')
        node.lfd_states_df.to_csv(csv_path, index=False, header=False)
        node.get_logger().info(f"LFD approach data saved to {csv_path}")
        csv_path = os.path.join(ROBOT_BAG_FILEPATH, 'lfd_actions.csv')
        node.lfd_actions_df.to_csv(csv_path, index=False, header=False)
        node.get_logger().info(f"LFD approach actions saved to {csv_path}")
        
        

        # -------------- Step 3: Dispose apple ----------------
        input("\n\033[1;32m5 - Press Enter to dispose apple.\033[0m\n")
        # Dispose apple
        node.swap_controller(node.eef_velocity_controller, node.arm_controller)        

        # Stop recording
        stop_recording_bagfile(robot_rosbag_list)

        # Check data
        node.get_logger().info("Check ROBOT demo data")         
        check_data_plots(ROBOT_BAG_FILEPATH, TRIAL)
        node.get_logger().info(f"Robot lfd implementation {demo+1}/10 done.")
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
