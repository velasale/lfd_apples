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
import yaml
from pathlib import Path

import pandas as pd

# Custom imports
from lfd_apples.listen_franka import main as listen_main
from lfd_apples.listen_franka import start_recording_bagfile, stop_recording_bagfile, save_metadata, find_next_trial_number 
from lfd_apples.ros2bag2csv import extract_data_and_plot, parse_array, fr3_jacobian, fr3_fk
from lfd_apples.lfd_vision import extract_pooled_latent_vector

# ROS2 imports
from std_msgs.msg import Int16MultiArray
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import SetBool, Trigger
from geometry_msgs.msg import PoseStamped, TwistStamped, Twist
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


# CV Bridge and YOLO imports
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

import pickle
import joblib
import numpy as np
print("NumPy version:", np.__version__)

from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt



class EMA():
    '''
    Exponential Moving Average
    It is causal, first order low-pass filter
    '''

    def __init__(self, alpha):
        self.alpha = alpha
        self.y = None

    def update(self, x):
        if self.y is None:
            self.y = x          # initialize with first sample
        else:
            self.y = self.alpha * x + (1.0 - self.alpha) * self.y
        return self.y


class LFDController(Node):

    def __init__(self):
        super().__init__('move_to_home_and_freedrive')

        self.initialize_ros_topics()
        self.initialize_ros_service_clients()
        self.initialize_ros_controller_properties()    
        self.initialize_rviz_trail()  
        self.initialize_ros_timers()
        self.initialize_flags()

        self.initialize_state_variables()
        self.initialize_debugging_mode_variables()        
        self.initialize_signal_variables_and_thresholds()
        self.initialize_arm_poses()      
        
        self.initialize_ml_models()

        # --- Velocity ramping variables ---
        # self.current_linear_accel  = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        # self.current_angular_accel = {'x': 0.0, 'y': 0.0, 'z': 0.0}

        # self.SCALING_FACTOR = 0.5e-3
        # self.max_linear_accel   = 0.1 * self.SCALING_FACTOR   # m/s²
        # self.max_angular_accel  = 0.2 * self.SCALING_FACTOR    # rad/s²

        # self.max_linear_jerk    = 0.1 * self.SCALING_FACTOR    # m/s³
        # self.max_angular_jerk   = 1.0 * self.SCALING_FACTOR    # rad/s³   


        self.last_cmd_time = self.get_clock().now()   

        self.joint_names = [f"fr3_joint{i+1}" for i in range(7)]
        self.joint_states = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.trajectory_points = []       
     
        

    # === Initialization functions ===
    def initialize_ros_topics(self):

        # Topic Subscribers
        self.distance_sub = self.create_subscription(
            Int16MultiArray, 'microROS/sensor_data', self.gripper_sensors_callback, 10)
        self.eef_pose_sub = self.create_subscription(
            PoseStamped, '/franka_robot_state_broadcaster/current_pose', self.eef_pose_callback, 10)
        self.palm_camera_sub = self.create_subscription(
            Float32MultiArray, 'lfd/latent_image', self.palm_camera_callback, 10)       
        self.joint_states_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_states_callback, 10)  # Dummy subscriber to ensure joint states are available
        
        # Topic Publishers
        self.vel_pub = self.create_publisher(
            TwistStamped, '/cartesian_velocity_controller/command', 10)
        self.delta_pub = self.create_publisher(
            Float32MultiArray, '/cartesian_delta', 10)       
        self.servo_pub = self.create_publisher(
            TwistStamped, '/servo_node_lfd/delta_twist_cmds', 10)
        self.tcp_marker_pub = self.create_publisher(
            Marker, '/lfd/tcp_trail', 10) 


    def initialize_ros_service_clients(self):

        # Service Clients
        self.servo_node_client = self.create_client(Trigger, '/servo_node_lfd/start_servo')
        while not self.servo_node_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for servo node service...')


    def initialize_ros_controller_properties(self):

        # Action Client
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            'fr3_arm_controller/follow_joint_trajectory'
        )
        
        # Controller names
        self.arm_controller = 'fr3_arm_controller'
        self.twist_controller = 'fr3_twist_controller'
        self.gravity_controller = 'gravity_compensation_example_controller'
     
        # Switch controller client
        self.switch_client = self.create_client(SwitchController, '/controller_manager/switch_controller')
        self.switch_client.wait_for_service()

        # Load controller client
        self.load_client = self.create_client(LoadController, '/controller_manager/load_controller')
        self.load_client.wait_for_service()

        # --- Velocity ramping variables ---
        self.current_cmd = TwistStamped()
        self.target_cmd = TwistStamped()        

        # Controller gain for delta
        # Converts deltas into m/s and rad/s
        # In our case deltas were obtained for delta_times = 1msec, hence gain = 1/0.001 = 1000
        self.DELTA_GAIN = 2250


    def initialize_rviz_trail(self):
        self.tcp_trail_marker = Marker()
        self.tcp_trail_marker.header.frame_id = "fr3_link0"  # base frame
        self.tcp_trail_marker.ns = "tcp_trail"
        self.tcp_trail_marker.id = 0
        self.tcp_trail_marker.type = Marker.POINTS
        self.tcp_trail_marker.action = Marker.ADD

        # Point size
        self.tcp_trail_marker.scale.x = 0.01
        self.tcp_trail_marker.scale.y = 0.01

        # Color (red, opaque)
        self.tcp_trail_marker.color.g = 1.0
        self.tcp_trail_marker.color.a = 1.0

        self.tcp_trail_marker.points = []

        # Trail sampling
        self.trail_step = 0
        self.TRAIL_EVERY_N_STEPS = 5


    def initialize_state_variables(self):
        '''
        Space Representation for lfd inference        
        '''
        
        # State Space 
        self.ncols_tof = 1
        self.ncols_latent_image = 64
        self.ncols_eef_pose = 7        
        self.ncols_air_pressure = 3
        self.ncols_joint_positions = 7
        self.ncols_joint_velocities = 7
        self.ncols_joint_efforts = 7
        self.ncols_wrench = 6

        # Action Space
        self.ncols_eef_velocity = 6
        self.ncols_approach = self.ncols_tof + self.ncols_latent_image + self.ncols_joint_positions

        self.ncols_contact = self.ncols_tof + self.ncols_air_pressure + self.ncols_wrench
        self.ncols_pick = self.ncols_tof + self.ncols_air_pressure + self.ncols_wrench        

        self.KEEP_ACTIONS_MEMORY = False
        previous_timesteps_ncols = self.ncols_approach  
        if self.KEEP_ACTIONS_MEMORY:
            previous_timesteps_ncols += self.ncols_eef_velocity  # Add action space dimensions         
        self.t_2_data = np.zeros(previous_timesteps_ncols)       
        self.t_1_data = np.zeros(previous_timesteps_ncols)            

        # Initialize state space arrays
        self.t_data = []
        self.tof = np.zeros(self.ncols_tof)
        self.latent_image = np.zeros(self.ncols_latent_image)        
        self.eef_pose = np.zeros(self.ncols_eef_pose)
        self.joint_positions = np.zeros(self.ncols_joint_positions)

        self.Y = np.zeros(self.ncols_eef_velocity)
        self.Y_base_frame = np.zeros(self.ncols_eef_velocity)

        # DataFrames to store normalized states over time
        self.lfd_states_df = pd.DataFrame()
        self.lfd_actions_df = pd.DataFrame()      


    def initialize_debugging_mode_variables(self):
        '''
        These variables are to be used while DEBUG mode is on
        In this mode, the model is not used, instead a prerecorded demo is replayed
        
        :param self: Description
        '''

        # Debugging variables
        self.DEBUGGING_MODE = False
        self.DEBUGGING_STEP = 0

        # --- Data for debugging ---
        # Replay sequence of actions from a previous demo
        # Twist actions given at the base frame
        # demos_folder = '/home/alejo/Documents/DATA/02_IL_preprocessed_(aligned_and_downsampled)/experiment_1_(pull)'
        # demo_trial = 'trial_4_downsampled_aligned_data.csv'

        # Twist actions given at the eef frame
        self.DEBUG_TRIAL = 'trial_4'
        eef_demos_folder = '/home/alejo/Documents/DATA/03_IL_preprocessed_(transformed_to_eef)/experiment_1_(pull)'
        eef_demo_trial = self.DEBUG_TRIAL + '_downsampled_aligned_data_transformed.csv'
        
        debugging_demo_csv = os.path.join(eef_demos_folder, eef_demo_trial)
        self.debugging_demo_pd = pd.read_csv(debugging_demo_csv)

        # Load action names from yaml config
        data_columns_path = config_path = Path(__file__).parent / "config" / "lfd_data_columns.yaml"
        with open(data_columns_path, "r") as f:
            cfg = yaml.safe_load(f)    
        
        self.action_cols = cfg['action_cols']       

        # If using actions at the base frame, rename with the following:
        # self.action_cols = ['delta_pos_x', 'delta_pos_y', 'delta_pos_z', 'delta_ori_x', 'delta_ori_y', 'delta_ori_z']

        self.n_action_cols = len(self.action_cols)     # These are the ouputs (actions)

        # Load actions
        self.debugging_actions_pd = self.debugging_demo_pd[self.action_cols]


    def initialize_signal_variables_and_thresholds(self):        

        # Thresholds
        self.TOF_THRESHOLD = 50                 # units in mm
        self.AIR_PRESSURE_THRESHOLD = 600       # units in hPa        
       
        # Gripper Signal Variables
        self.ema_alpha = 0.5
        self.ema_scA = EMA(self.ema_alpha)
        self.ema_scB = EMA(self.ema_alpha)
        self.ema_scC = EMA(self.ema_alpha)
        self.ema_tof = EMA(self.ema_alpha)


    def initialize_arm_poses(self):            
      
        self.HOME_POSITIONS = self.debugging_demo_pd[['pos_joint_1',
                                                     'pos_joint_2',
                                                     'pos_joint_3', 
                                                     'pos_joint_4',
                                                     'pos_joint_5',
                                                     'pos_joint_6',
                                                     'pos_joint_7']].iloc[0].tolist()

        self.goal_pose = [-0.031,
                          0.794,
                          0.474]

        self.DISPOSAL_POSITIONS = []


    def initialize_flags(self):
        
        # Flags
        self.approach_accomplished = False
        self.contact_accomplished = False
        self.pick_accomplished = False
        self.data_ready = False
        self.running_lfd_approach = False
        self.running_lfd_contact = False
        self.running_lfd_pick = False        


    def initialize_ros_timers(self):        
        
        # --- Timer for high-rate velocity ramping ---
        self.timer_period = 0.001  # 500 Hz
        self.create_timer(self.timer_period, self.publish_smoothed_velocity)

        # --- Timer to recreate incoming palm camera with fake hardware ---
        self.create_timer(0.034, self.incoming_cam_sim)


    def initialize_ml_models(self):
        
        # Load learned model
        phase = 'phase_1_approach'
        model = 'mlp'
        timesteps = '0_timesteps'               
        self.load_model(phase, model, timesteps)              


    # === ROS controller functions ===
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


    # === TBD =====

    def load_model(self, phase, model, timesteps):        

        BASE_DIR = '/home/alejo/Documents/DATA'
        # BASE_DIR = '/media/alejo/IL_data'
        model_path = os.path.join(BASE_DIR, '06_IL_learning/experiment_1_(pull)', phase, timesteps)
        model_name = model + '_experiment_1_(pull)_' + phase + '_' + timesteps + '.joblib'
        with open(os.path.join(model_path, model_name), "rb") as f:
            # self.lfd_model = pickle.load(f)
            self.lfd_model = joblib.load(f)
        self.lfd_X_mean = np.load(os.path.join(model_path, model + '_Xmean_experiment_1_(pull)_' + phase + '_' + timesteps + '.npy'))
        self.lfd_X_std = np.load(os.path.join(model_path, model + '_Xstd_experiment_1_(pull)_' + phase + '_' + timesteps + '.npy'))   
        self.lfd_Y_mean = np.load(os.path.join(model_path, model + '_Ymean_experiment_1_(pull)_' + phase + '_' + timesteps + '.npy'))
        self.lfd_Y_std = np.load(os.path.join(model_path, model + '_Ystd_experiment_1_(pull)_' + phase + '_' + timesteps + '.npy'))   


    def move_to_home(self):
        self.get_logger().info('Waiting for action server...')
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server not available')
            return False                 

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = self.HOME_POSITIONS
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
            current_positions = self.HOME_POSITIONS
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


    def update_tcp_trail(self):
        if not self.running_lfd_approach:
            return

        self.trail_step += 1
        if self.trail_step % self.TRAIL_EVERY_N_STEPS != 0:
            return

        p = Point()
        p.x = self.eef_pos_x_from_f3k
        p.y = self.eef_pos_y_from_f3k
        p.z = self.eef_pos_z_from_f3k

        self.tcp_trail_marker.points.append(p)
        self.tcp_trail_marker.header.stamp = self.get_clock().now().to_msg()

        self.tcp_marker_pub.publish(self.tcp_trail_marker)


    def run_lfd_approach(self, n_timesteps=2):       

        self.get_logger().info("Starting run_lfd_approach: publishing twists from palm_camera_callback until TOF < threshold.")
        self.running_lfd_approach = True

        # Rviz: Clear previous trail       
        self.tcp_trail_marker.points.clear()
        self.trail_step = 0

        while rclpy.ok() and self.running_lfd_approach:
            
            # self.tof = np.array([1000])  # Dummy value for testing
            self.tof = 1000.0

            if self.tof < self.TOF_THRESHOLD:
                self.get_logger().info(f"TOF threshold reached: {self.tof} < {self.TOF_THRESHOLD}")                
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

        # Apply EMA filtering to match conditions durint model training
        self.scA = self.ema_scA.update(self.scA)
        self.scB = self.ema_scB.update(self.scB)
        self.scC = self.ema_scC.update(self.scC)
        self.tof = self.ema_tof.update(self.tof)

        self.tof_for_state = np.array([self.tof])


    def incoming_cam_sim(self):
        """
        Simulate incoming latent image from palm camera by publishing dummy data.
        This is for testing purposes only.
        """

        if self.running_lfd_approach and self.DEBUGGING_MODE:

            # --- Get next action from debugging demo ---
            # Get current step
            
            if self.DEBUGGING_STEP >= len(self.debugging_actions_pd):
                self.get_logger().info("Debugging demo actions exhausted.")
                self.send_stop_message()
                self.running_lfd_approach = False
                return
            else:
            
                action_row = self.debugging_actions_pd.iloc[self.DEBUGGING_STEP]
                y = action_row.values
                
                self.get_logger().info(f"Debugging mode - Step {self.DEBUGGING_STEP}, Action: {y}")

                # If using deltas, multiply by scaling factor
                # Send actions (twist)        
                self.target_cmd.twist.linear.x = float(y[0]) * self.DELTA_GAIN
                self.target_cmd.twist.linear.y = float(y[1]) * self.DELTA_GAIN
                self.target_cmd.twist.linear.z = float(y[2]) * self.DELTA_GAIN    
                self.target_cmd.twist.angular.x = float(y[3]) * self.DELTA_GAIN 
                self.target_cmd.twist.angular.y = float(y[4]) * self.DELTA_GAIN 
                self.target_cmd.twist.angular.z = float(y[5]) * self.DELTA_GAIN 
                            
                self.DEBUGGING_STEP += 1           


    def palm_camera_callback(self, msg: Float32MultiArray):
        """
        We'll use this callback to publish Twist commands everytime a new latent image arrives.
        Image arrives as a Float32MultiArray with 64 elements
        
        :param self: Description
        :param msg: Description
        :type msg: Image
        """

        if self.running_lfd_approach and not self.DEBUGGING_MODE:
            
            # --- Extract latent image ---
            self.latent_image = msg.data

            # --- Combine data ---
            self.t_data = np.concatenate([self.tof, self.latent_image, self.joint_states])            
            # Combine data, with past steps            
            self.X = np.concatenate([self.t_2_data, self.t_1_data, self.t_data])
            self.X = np.concatenate([self.t_data])
            
            
            # --- Normalize data ---                        
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
            self.Y = self.Y.squeeze()    


            # Transform actions (eef cartesian velocities) to base frame                        
            position = self.eef_pose[:3]
            orientation = self.eef_pose[3:]
            position_skew_matrix = np.array([[0, -position[2], position[1]],
                                         [position[2], 0, -position[0]],
                                         [-position[1], position[0], 0]])
            # Rotation matrix from quaternion   base <- eef
            R_base_eef = R.from_quat(orientation).as_matrix()   
            v_eef = self.Y[:3]  # Assuming the model outputs in base frame for now
            w_eef = self.Y[3:]  # Assuming the model outputs in base frame for now

            # Check array sizes and shapes
            v_eef = np.array(v_eef, dtype=float).reshape(3,)
            w_eef = np.array(w_eef, dtype=float).reshape(3,)
            R_base_eef = np.array(R_base_eef, dtype=float).reshape(3, 3)
            position_skew_matrix = np.array(position_skew_matrix, dtype=float).reshape(3, 3)

            v_base = R_base_eef @ v_eef + position_skew_matrix @ R_base_eef @ w_eef
            w_base = R_base_eef @ w_eef

            self.Y_base_frame = np.array([v_base[0], v_base[1], v_base[2], w_base[0], w_base[1], w_base[2]])

            # Scale down the velocities
            self.DELTA_GAIN = 1       # deltas were obtained at 1khz
            self.Y_base_frame = self.Y_base_frame * self.DELTA_GAIN       

            y = self.Y
            # --- Append as row in DataFrame ---
            self.lfd_actions_df = pd.concat([
                self.lfd_actions_df, 
                # pd.DataFrame([self.t_data_norm])
                pd.DataFrame([y])

            ], ignore_index=True)
            self.get_logger().info(f'Target actions in eef frame: {y}')       
            y_base = self.Y_base_frame
            self.get_logger().info(f'Target actions in base frame: {y_base}') 

            # Adjust velocities with feedback from goal pose


            # Send actions (twist)                   
            self.target_cmd.twist.linear.x = float(y_base[0]) + self.eef_pos_x_error * self.DELTA_GAIN
            self.target_cmd.twist.linear.y = float(y_base[1]) + self.eef_pos_y_error * self.DELTA_GAIN
            self.target_cmd.twist.linear.z = float(y_base[2]) + self.eef_pos_z_error * self.DELTA_GAIN
            self.target_cmd.twist.angular.x = float(y_base[3])
            self.target_cmd.twist.angular.y = float(y_base[4])
            self.target_cmd.twist.angular.z = float(y_base[5])

            self.get_logger().info(f"palm camera callback sending topic")

            # Add actions to State Space to pass them to the next time steps
            self.t_2_data = self.t_1_data

            if self.KEEP_ACTIONS_MEMORY:
                self.t_1_data = np.concatenate([self.t_data, self.Y])
            else:
                self.t_1_data = self.t_data
                        
        if self.running_lfd_approach and self.DEBUGGING_MODE:

            # --- Get next action from debugging demo ---
            # Get current step
            
            if self.DEBUGGING_STEP >= len(self.debugging_actions_pd):
                self.get_logger().info("Debugging demo actions exhausted.")
                self.send_stop_message()
                self.running_lfd_approach = False
                return
            else:
            
                action_row = self.debugging_actions_pd.iloc[self.DEBUGGING_STEP]
                y = action_row.values
                
                self.get_logger().info(f"Debugging mode - Step {self.DEBUGGING_STEP}, Action: {y}")

                # Send actions (twist)        
                # If using deltas, multiply by scaling factor to conver to speed units               
                self.target_cmd.twist.linear.x = float(y[0]) * self.DELTA_GAIN
                self.target_cmd.twist.linear.y = float(y[1]) * self.DELTA_GAIN
                self.target_cmd.twist.linear.z = float(y[2]) * self.DELTA_GAIN    
                self.target_cmd.twist.angular.x = float(y[3]) * self.DELTA_GAIN
                self.target_cmd.twist.angular.y = float(y[4]) * self.DELTA_GAIN
                self.target_cmd.twist.angular.z = float(y[5]) * self.DELTA_GAIN

                # self.target_cmd.twist.linear.x = 0.0 #float(y[0]) * self.DELTA_GAIN
                # self.target_cmd.twist.linear.y = 0.0 #float(y[1]) * self.DELTA_GAIN
                # self.target_cmd.twist.linear.z = 0.0 #float(y[2]) * self.DELTA_GAIN    
                # self.target_cmd.twist.angular.x = 0.1 #float(y[3]) * self.DELTA_GAIN
                # self.target_cmd.twist.angular.y = 0.0 #float(y[4]) * self.DELTA_GAIN
                # self.target_cmd.twist.angular.z = 0.0 #float(y[5]) * self.DELTA_GAIN
            
                self.DEBUGGING_STEP += 1        
            
        
    def publish_smoothed_velocity(self):

        if not self.running_lfd_approach:
            return

        dt = (self.get_clock().now() - self.last_cmd_time).nanoseconds * 1e-9
        self.last_cmd_time = self.get_clock().now()

        if dt <= 0.0:
            return

        # # --- Linear axes ---
        # for axis in ['x', 'y', 'z']:
        #     v_cur = getattr(self.current_cmd.twist.linear, axis)
        #     v_tgt = getattr(self.target_cmd.twist.linear, axis)
        #     a_cur = self.current_linear_accel[axis]

        #     # Desired acceleration
        #     a_des = (v_tgt - v_cur) / dt
        #     a_des = np.clip(a_des, -self.max_linear_accel, self.max_linear_accel)

        #     # Jerk-limited acceleration update
        #     da = a_des - a_cur
        #     da = np.clip(da, -self.max_linear_jerk * dt,
        #                     self.max_linear_jerk * dt)

        #     a_new = a_cur + da
        #     v_new = v_cur + a_new * dt

        #     self.current_linear_accel[axis] = a_new
        #     setattr(self.current_cmd.twist.linear, axis, v_new)

        # # --- Angular axes ---
        # for axis in ['x', 'y', 'z']:
        #     v_cur = getattr(self.current_cmd.twist.angular, axis)
        #     v_tgt = getattr(self.target_cmd.twist.angular, axis)
        #     a_cur = self.current_angular_accel[axis]

        #     a_des = (v_tgt - v_cur) / dt
        #     a_des = np.clip(a_des, -self.max_angular_accel, self.max_angular_accel)

        #     da = a_des - a_cur
        #     da = np.clip(da, -self.max_angular_jerk * dt,
        #                     self.max_angular_jerk * dt)

        #     a_new = a_cur + da
        #     v_new = v_cur + a_new * dt

        #     self.current_angular_accel[axis] = a_new
        #     setattr(self.current_cmd.twist.angular, axis, v_new)
        
        self.current_cmd.twist = self.target_cmd.twist  # Uncomment to override smoothing
        self.current_cmd.header.frame_id = "fr3_hand_tcp"

        self.current_cmd.header.stamp = self.get_clock().now().to_msg()
        self.servo_pub.publish(self.current_cmd)


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
        
        self.eef_pos_x_error = self.goal_pose[0] - self.eef_pos_x
        self.eef_pos_y_error = self.goal_pose[1] - self.eef_pos_y
        self.eef_pos_z_error = self.goal_pose[2] - self.eef_pos_z

        # Update sensors average since last action
    

    def joint_states_callback(self, msg: JointState):

        self.joint_states = list(msg.position[:7])


        # FK to obtain EEF pose
        q = self.joint_states #.to_numpy(dtype=float)                
        T, Ts = fr3_fk(q)
        position = T[:3, 3]
        self.eef_pos_x_from_f3k = position[0]
        self.eef_pos_y_from_f3k = position[1]
        self.eef_pos_z_from_f3k = position[2]    

        # Update TCP trail
        self.update_tcp_trail()

        # Dummy callback to ensure joint states are available



def check_data_plots(BAG_DIR, trial_number, inhand_camera_bag=True):

    print("Extracting data and generating plots...")

    try:
        plt.ion()  # <-- interactive mode ON
        # extract_data_and_plot(os.path.join(BAG_DIR, TRIAL), "")
        extract_data_and_plot(BAG_DIR, trial_number, inhand_camera_bag, implementation_stage=True)
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
    BAG_MAIN_DIR = '07_IL_implementation/bagfiles'
    EXPERIMENT = "experiment_1_(pull)"
    BAG_FILEPATH = os.path.join(BASE_DIR, BAG_MAIN_DIR, EXPERIMENT)
    os.makedirs(BAG_FILEPATH, exist_ok=True)
    
    batch_size = 10
    node.get_logger().info(f"Starting lfd implementation with robot, session of {batch_size} demos.")    


    for demo in range(batch_size):

        node.get_logger().info("\033[1;32m\n---------- Press ENTER key to start lfd implementation trial {}/10 ----------\033[0m".format(demo+1))
        input()  
      
        # ------------ Step 0: Initial configuration ----------------
        TRIAL = find_next_trial_number(BAG_FILEPATH, prefix="trial_")
        os.makedirs(os.path.join(BAG_FILEPATH, TRIAL), exist_ok=True)
        # HUMAN_BAG_FILEPATH = os.path.join(BAG_FILEPATH, TRIAL, 'human')
        ROBOT_BAG_FILEPATH = os.path.join(BAG_FILEPATH, TRIAL)              

        save_metadata(os.path.join(BAG_FILEPATH, TRIAL, "metadata_" + TRIAL))         

        # ------------ Step 1: Move robot to home position ---------
        node.get_logger().info("\n\033[1;32m\nSTEP 1: Arm moving to home position... wait\033[0m\n")
        while not node.move_to_home():
            pass

        input("\n\033[1;32m\nSTEP 2: Place apple on the proxy. \nPress ENTER key when done.\033[0m\n")    

        # ------------ Step 2: Allow free drive if needed ---------
        node.swap_controller(node.arm_controller, node.gravity_controller)
        time.sleep(1.0)                
        node.get_logger().info("\n\033[1;32m\nSTEP 3: Free-drive arm until apple in camera FOV. \nPress ENTER key when you are done.\033[0m\n")        
        input()    
       
        # ------------ Step 3: Enable Servo Node Cartesian velocity controller ---------
        node.swap_controller(node.gravity_controller, node.twist_controller)
        time.sleep(1.0)  

        node.get_logger().info("Enabling Moveit2 Servo Node for twist controller...")
        req = Trigger.Request()        
        node.servo_node_client.call_async(req)

        time.sleep(1.0)        

        # Start bag recording        
        robot_rosbag_list = start_recording_bagfile(ROBOT_BAG_FILEPATH)

        # -------------- Step 4: Run lfd controller ----------------        
        input("\n\033[1;32m\nSTEP 4: Press ENTER key to start ROBOT lfd implementation.\033[0m\n")        
        node.DEBUGGING_MODE = True
        node.DEBUGGING_STEP = 0
        node.run_lfd_approach()      

        # Save states to CSV        
        csv_path = os.path.join(ROBOT_BAG_FILEPATH, 'lfd_recorded_data.csv')
        node.lfd_states_df.to_csv(csv_path, index=False, header=False)
        node.get_logger().info(f"LFD approach data saved to {csv_path}")
        csv_path = os.path.join(ROBOT_BAG_FILEPATH, 'lfd_actions.csv')
        node.lfd_actions_df.to_csv(csv_path, index=False, header=False)
        node.get_logger().info(f"LFD approach actions saved to {csv_path}")               

        # -------------- Step 5: Dispose apple ----------------
        input("\n\033[1;32m\nSTEP 5: Press ENTER key to dispose apple.\033[0m\n")
        
        node.swap_controller(node.twist_controller, node.arm_controller)
        time.sleep(1.0)  

        # Stop bag recording
        stop_recording_bagfile(robot_rosbag_list)

        # Check data
        node.get_logger().info("Check ROBOT demo data")         
        check_data_plots(ROBOT_BAG_FILEPATH, TRIAL)
        node.get_logger().info(f"Robot lfd implementation {demo+1}/10 done.")
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
