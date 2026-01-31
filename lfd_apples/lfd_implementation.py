#!/usr/bin/env python3

# System imports
import subprocess
import time
import os
from pathlib import Path
import torch
from collections import deque
from ament_index_python.packages import get_package_share_directory
import sys, termios, tty, threading

# Custom imports
from lfd_apples.listen_franka import main as listen_main
from lfd_apples.listen_franka import start_recording_bagfile, stop_recording_bagfile, save_metadata, find_next_trial_number 
from lfd_apples.ros2bag2csv import extract_data_and_plot, parse_array, fr3_jacobian, fr3_fk
from lfd_apples.lfd_vision import extract_pooled_latent_vector
from lfd_apples.lfd_learning import VelocityMLP, DatasetForLearning, resolve_columns, get_phase_columns, expand_features_over_time
from lfd_apples.lfd_lstm import LSTMRegressor

# ROS2 imports
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import SetBool, Trigger
from geometry_msgs.msg import PoseStamped, TwistStamped, Twist
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from controller_manager_msgs.srv import SwitchController, LoadController, ListControllers

# CV Bridge and YOLO imports
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

# File type imports
import pickle
import joblib
import yaml
import pandas as pd
import json
import datetime

import numpy as np
print("NumPy version:", np.__version__)

from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt


class KeyboardListener:
    def __init__(self):
        self.esc_pressed = False
        self.thread = threading.Thread(target=self.listen, daemon=True)
        self.thread.start()

    def listen(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.read(1)
                if ch == '\x1b':  # ESC
                    self.esc_pressed = True
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


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

    def __init__(self, MODEL_PARAMS):
        super().__init__('move_to_home_and_freedrive')

        self.MODEL_PARAMS = MODEL_PARAMS

        self.initialize_state_variables()
        self.initialize_debugging_mode_variables('trial_1')        
        self.initialize_signal_variables_and_thresholds()
        self.initialize_arm_poses()      
        
        self.initialize_ml_models()
        
        self.initialize_ros_topics()
        self.initialize_ros_service_clients()
        self.initialize_ros_controller_properties()    
        self.initialize_rviz_trail()  
        self.initialize_ros_timers()
        self.initialize_flags()               

        self.last_cmd_time = self.get_clock().now()   
        self.joint_names = [f"fr3_joint{i+1}" for i in range(7)]
        self.joint_states = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.trajectory_points = []       
             

    # === Initialization functions ===
    def initialize_ros_topics(self):

        # ROS Topic Subscribers
        self.distance_sub = self.create_subscription(
            Int16MultiArray,
            'microROS/sensor_data',
            self.gripper_sensors_callback,
            10)
        self.eef_pose_sub = self.create_subscription(
            PoseStamped,
            '/franka_robot_state_broadcaster/current_pose',
            self.eef_pose_callback,
            10)
        self.palm_camera_sub = self.create_subscription(
            Float32MultiArray,
            'lfd/latent_image',
            self.palm_camera_callback,
            10)       
        self.joint_states_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_states_callback,
            10)  # Dummy subscriber to ensure joint states are available
        self.bbox_center_sub = self.create_subscription(
            Int16MultiArray,
            'lfd/bbox_center',
            self.bbox_callback, 10)  
        
        
        # ROS Topic Publishers
        self.vel_pub = self.create_publisher(
            TwistStamped,
            '/cartesian_velocity_controller/command',
            10)
        self.delta_pub = self.create_publisher(
            Float32MultiArray,
            '/cartesian_delta',
            10)       
        self.servo_pub = self.create_publisher(
            TwistStamped,
            '/servo_node_lfd/delta_twist_cmds',
            10)
        self.tcp_marker_pub = self.create_publisher(
            Marker,
            '/lfd/tcp_trail',
            10) 
        self.apple_marker_pub = self.create_publisher(
            Marker,
            '/lfd/apple',
            10) 


    def initialize_ros_service_clients(self):

        # Service Clients
        self.servo_node_client = self.create_client(
            Trigger,
            '/servo_node_lfd/start_servo')
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
        # Converts 'm' and 'rad' deltas into m/s and rad/s
        # In our case, delta_eef_pose was calculated for delta_times = 0.001 sec, 
        # hence, we use a gain of 1000 (1/0.001sec) to convert m to m/sec.
        self.DELTA_GAIN = 1000
        # self.DELTA_GAIN = 1500    # I used this one for command interface = position

        # PID GAINS
        # If using deltas, multiply by scaling factor
        # Send actions (twist)       
        self.PI_GAIN = 0.0                    
        self.POSITION_KP = 1.25  # 2.0
        self.POSITION_KI = 0.025

        PIXELS = 320
        DISTANCE_IN_M = 0.1   
        self.PIXEL_TO_METER_RATE = DISTANCE_IN_M / PIXELS      # approx 20cm per 320 pixels

        self.position_kp_z = 0.2    # 0.1


    def initialize_rviz_trail(self):

        self.tcp_trail_marker = Marker()
        self.tcp_trail_marker.header.frame_id = "fr3_link0"  # base frame
        self.tcp_trail_marker.ns = "tcp_trail"
        self.tcp_trail_marker.id = 0
        self.tcp_trail_marker.type = Marker.POINTS
        self.tcp_trail_marker.action = Marker.ADD

        # Point size
        self.tcp_trail_marker.scale.x = 0.005
        self.tcp_trail_marker.scale.y = 0.005

        # Color (red, opaque)
        self.tcp_trail_marker.color.g = 1.0
        self.tcp_trail_marker.color.a = 1.0

        self.tcp_trail_marker.points = []

        # Trail sampling
        self.trail_step = 0
        self.TRAIL_EVERY_N_STEPS = 2
    

    def place_rviz_apple(self):

        # ======================= SHAPE =======================
        self.apple_marker = Marker()

        self.apple_marker.header.frame_id = "fr3_link0"
        self.apple_marker.header.stamp = self.get_clock().now().to_msg()
        self.apple_marker.ns = 'apple'
        self.apple_marker.id = 0
        self.apple_marker.type = Marker.SPHERE
        self.apple_marker.action = Marker.ADD

        self.apple_marker.pose.position.x = float(self.apple_pose_base.values[0])
        self.apple_marker.pose.position.y = float(self.apple_pose_base.values[1])
        self.apple_marker.pose.position.z = float(self.apple_pose_base.values[2])

        self.apple_marker.pose.orientation.w = 1.0

        # Sphere size (diameter!)
        self.apple_marker.scale.x = 0.08
        self.apple_marker.scale.y = 0.08
        self.apple_marker.scale.z = 0.08

        # Color (red apple)
        self.apple_marker.color.r = 1.0
        self.apple_marker.color.g = 0.0
        self.apple_marker.color.b = 0.0
        self.apple_marker.color.a = 1.0

        # ======================== TEXT ======================
        text_marker = Marker()

        text_marker.header.frame_id = 'fr3_link0'
        text_marker.header.stamp = self.get_clock().now().to_msg()

        text_marker.ns = 'trial_id'
        text_marker.id = 1

        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD

        # Position: slightly above the apple
        text_marker.pose.position.x = self.apple_marker.pose.position.x
        text_marker.pose.position.y = self.apple_marker.pose.position.y
        text_marker.pose.position.z = self.apple_marker.pose.position.z + 0.06

        text_marker.pose.orientation.w = 1.0

        # Text content
        text_marker.text = self.DEBUG_TRIAL

        # Text height (meters, NOT scale.x/y)
        text_marker.scale.z = 0.05

        # Color (white text)
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0

        # ===================== PUBLISH MARKERS =====================
        self.apple_marker_pub.publish(self.apple_marker)
        self.apple_marker_pub.publish(text_marker)
  

    def initialize_state_variables(self):
        '''
        Space Representation for lfd inference        
        '''
        self.apple_pose_base = pd.Series(
            [1.0, 1.0, 1.0],
            index=['apple._x._base', 'apple._y._base', 'apple._z._base']
            )

        model_name = self.MODEL_PARAMS['MODEL']              

        # --- Adjust some parameters depending on model ---
        if model_name != 'lstm':
            self.MODEL_PARAMS['HIDDEN_DIM'] = '__'
            self.MODEL_PARAMS['NUM_LAYERS'] = '__'
            SEQ_LEN = -1
            timesteps = self.MODEL_PARAMS['TIMESTEPS']
        else:
            SEQ_LEN = self.MODEL_PARAMS['SEQ_LEN']
            timesteps = '0_timesteps'


        # --- Load yaml with State and Action Space variables ---
        data_columns_path = config_path = Path(__file__).parent / "config" / "lfd_data_columns.yaml"
        with open(data_columns_path, "r") as f:
            cfg = yaml.safe_load(f)    
        
        self.ACTION_NAMES = cfg['action_cols']              
        self.N_ACTIONS = len(self.ACTION_NAMES)   

        # States   
        PHASE = 'phase_1_approach'
        self.APPROACH_STATE_NAMES, self.APPROACH_STATE_NAME_KEYS = get_phase_columns(cfg, PHASE)
        ouput_set = set(self.ACTION_NAMES)
        self.APPROACH_STATE_NAMES = [
            c for c in self.APPROACH_STATE_NAMES
            if c not in ouput_set
        ]
        n_time_steps = int(timesteps.split('_timesteps')[0])
        if not model_name == 'lstm' and n_time_steps>0:            
            self.APPROACH_STATE_NAMES = expand_features_over_time(self.APPROACH_STATE_NAMES, n_time_steps)                  
        
        # TODO: These numbers could be automatically extracted from YAML FILE
        self.NCOLS_TOF = 1
        self.NCOLS_LATENT_IMAGE = 64
        self.NCOLS_BBOX = 2
        self.NCOLS_EEF_POSE = 7        
        self.NCOLS_AIR_PRESSURE = 3
        self.NCOLS_JOINT_POSITIONS = 7
        self.NCOLS_JOINT_VELOCITIES = 7
        self.NCOLS_JOINT_EFFORTS = 7
        self.NCOLS_WRENCH = 6

        self.N_APPROACH_STATES = len(self.APPROACH_STATE_NAMES)
        self.ncols_contact = self.NCOLS_TOF + self.NCOLS_AIR_PRESSURE + self.NCOLS_WRENCH
        self.ncols_pick = self.NCOLS_TOF + self.NCOLS_AIR_PRESSURE + self.NCOLS_WRENCH        

        # Action Space
        self.N_ACTIONS = 6
        self.KEEP_ACTIONS_MEMORY = False
        previous_timesteps_ncols = self.N_APPROACH_STATES  
        if self.KEEP_ACTIONS_MEMORY:
            previous_timesteps_ncols += self.N_ACTIONS  # Add action space dimensions         
        self.t_2_data = np.zeros(previous_timesteps_ncols)       
        self.t_1_data = np.zeros(previous_timesteps_ncols)            

        # Initialize state space arrays
        self.state = []
        self.tof = np.zeros(self.NCOLS_TOF)
        self.latent_image = np.zeros(self.NCOLS_LATENT_IMAGE)        
        self.bbox = np.zeros(self.NCOLS_BBOX)
        self.eef_pose = np.zeros(self.NCOLS_EEF_POSE)
        self.joint_positions = np.zeros(self.NCOLS_JOINT_POSITIONS)

        self.Y = np.zeros(self.N_ACTIONS)
        self.Y_base_frame = np.zeros(self.N_ACTIONS)

        # DataFrames to store normalized states over time
        self.lfd_states_df = pd.DataFrame()
        self.lfd_actions_df = pd.DataFrame()              

        self.state_buffer = deque(maxlen = SEQ_LEN)


    def initialize_debugging_mode_variables(self, trial='trial_1'):
        '''
        These variables are to be used while DEBUG mode is on
        In this mode, the model is not used; instead, actions from a pre-recorded demo are replayed
        
        :param self: Description
        '''

        self.DEBUG_TRIAL = trial

        # Debugging variables
        self.DEBUGGING_MODE = False
        self.approach_debugging_step = 0
        self.contact_debugging_step = 0
        self.pick_debugging_step = 0

        # --- Data for debugging ---
        # Replay sequence of actions from a previous demo
        # Twist actions given at the base frame
        # demos_folder = '/home/alejo/Documents/DATA/02_IL_preprocessed_(aligned_and_downsampled)/experiment_1_(pull)'
        # demo_trial = 'trial_4_downsampled_aligned_data.csv'        

        # Twist actions at EEF frame (TCP frame)
        # 1 - Approach Phase
        approach_eef_demos_folder = '/home/alejo/Documents/DATA/04_IL_preprocessed_(cropped_per_phase)/experiment_1_(pull)/phase_1_approach'
        approach_eef_demo_trial = self.DEBUG_TRIAL + '_downsampled_aligned_data_transformed_(phase_1_approach).csv'        
        approach_debugging_demo_csv = os.path.join(approach_eef_demos_folder, approach_eef_demo_trial)
        self.approach_debugging_demo_pd = pd.read_csv(approach_debugging_demo_csv)

        # 2 - Near-Contact Phase
        contact_eef_demos_folder = '/home/alejo/Documents/DATA/04_IL_preprocessed_(cropped_per_phase)/experiment_1_(pull)/phase_2_contact'
        contact_eef_demo_trial = self.DEBUG_TRIAL + '_downsampled_aligned_data_transformed_(phase_2_contact).csv'        
        contact_debugging_demo_csv = os.path.join(contact_eef_demos_folder, contact_eef_demo_trial)
        self.contact_debugging_demo_pd = pd.read_csv(contact_debugging_demo_csv)

        # 3 - Pick Phase
        pick_eef_demos_folder = '/home/alejo/Documents/DATA/04_IL_preprocessed_(cropped_per_phase)/experiment_1_(pull)/phase_3_pick'
        pick_eef_demo_trial = self.DEBUG_TRIAL + '_downsampled_aligned_data_transformed_(phase_3_pick).csv'        
        pick_debugging_demo_csv = os.path.join(pick_eef_demos_folder, pick_eef_demo_trial)
        self.pick_debugging_demo_pd = pd.read_csv(pick_debugging_demo_csv)

        # Load apple pose
        self.apple_pose_base = self.approach_debugging_demo_pd[['apple._x._base', 'apple._y._base', 'apple._z._base']].iloc[0]
        self.apple_pose_tcp = self.approach_debugging_demo_pd[['apple._x._ee', 'apple._y._ee', 'apple._z._ee']].iloc[0]
                

        # Load actions
        self.approach_debugging_actions_pd = self.approach_debugging_demo_pd[self.ACTION_NAMES]
        self.contact_debugging_actions_pd = self.contact_debugging_demo_pd[self.ACTION_NAMES]
        self.pick_debugging_actions_pd = self.pick_debugging_demo_pd[self.ACTION_NAMES]


    def initialize_signal_variables_and_thresholds(self):        

        # Thresholds
        # TOF threshold is used to switch from 'approach' to 'contact' phase
        self.TOF_THRESHOLD = 60                 # units in mm
        # Air pressure threshold is used to tell when a suction cup has engaged.
        self.AIR_PRESSURE_THRESHOLD = 600       # units in hPa        
       
        # Gripper Signal Variables with Exponential Moving Average
        self.ema_alpha = 0.5
        self.ema_scA = EMA(self.ema_alpha)
        self.ema_scB = EMA(self.ema_alpha)
        self.ema_scC = EMA(self.ema_alpha)
        self.ema_tof = EMA(self.ema_alpha)
        
        self.ema_img = EMA(alpha = 0.2)


    def initialize_arm_poses(self):            
      
        self.HOME_POSITIONS = self.approach_debugging_demo_pd[['pos_joint_1',
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
        self.running_lfd = False
        self.running_lfd_approach = False
        self.running_lfd_contact = False
        self.running_lfd_pick = False        


    def initialize_ros_timers(self):        
        
        # --- Timer for high-rate velocity ramping ---
        self.timer_period = 0.001  # 500 Hz
        self.create_timer(self.timer_period, self.publish_smoothed_velocity)

        # --- Timer to recreate incoming palm camera with fake hardware ---
        # self.create_timer(0.034, self.incoming_cam_sim)


    def initialize_ml_models(self):        

        model_name = self.MODEL_PARAMS['MODEL']

        if model_name != 'lstm':
            self.MODEL_PARAMS['hidden_dim'] = '__'
            self.MODEL_PARAMS['num_layers'] = '__'
            SEQ_LEN = -1
            TIMESTEPS = self.MODEL_PARAMS['TIMESTEPS']
        else:
            SEQ_LEN = self.MODEL_PARAMS['SEQ_LEN']
            TIMESTEPS = '0_timesteps'

        # --- Model Path ---     
        PHASE='phase_1_approach'      
        self.BASE_PATH = '/home/alejo/Documents/DATA'        
        self.MODEL_PATH = os.path.join(self.BASE_PATH, '06_IL_learning/experiment_1_(pull)', PHASE, TIMESTEPS)       


        if model_name in ['rf', 'mlp', 'mlp_torch']:
            model_filename = model_name + '_experiment_1_(pull)_' + PHASE + '_' + TIMESTEPS + '.joblib'
            with open(os.path.join(self.MODEL_PATH, model_filename), "rb") as f:
                # self.lfd_model = pickle.load(f)
                self.LFD_MODEL = joblib.load(f)
            self.X_MEAN = np.load(os.path.join(self.MODEL_PATH, model_name + '_Xmean_experiment_1_(pull)_' + PHASE + '_' + TIMESTEPS + '.npy'))
            self.X_STD = np.load(os.path.join(self.MODEL_PATH, model_name + '_Xstd_experiment_1_(pull)_' + PHASE + '_' + TIMESTEPS + '.npy'))   
            self.Y_MEAN = np.load(os.path.join(self.MODEL_PATH, model_name + '_Ymean_experiment_1_(pull)_' + PHASE + '_' + TIMESTEPS + '.npy'))
            self.Y_STD = np.load(os.path.join(self.MODEL_PATH, model_name + '_Ystd_experiment_1_(pull)_' + PHASE + '_' + TIMESTEPS + '.npy'))   

        if model_name == "lstm":        

            # APPROACH MODEL
            # Note: Subfolder depends on the states used during training
            self.APPROACH_STATE_NAME_KEYS = self.APPROACH_STATE_NAME_KEYS[:-1]
            input_keys_subfolder_name = "__".join(self.APPROACH_STATE_NAME_KEYS)
            model_subfolder = os.path.join(self.MODEL_PATH, input_keys_subfolder_name)            

            prefix = str(self.MODEL_PARAMS['NUM_LAYERS']) + '_layers_' + str(self.MODEL_PARAMS['HIDDEN_DIM']) + '_dim_' + str(SEQ_LEN) + "_seq_lstm_"
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        

            # --- Load statistics ---
            self.X_MEAN = torch.tensor(
                np.load(os.path.join(self.MODEL_PATH, model_subfolder, prefix + f"_Xmean_experiment_1_(pull)_{PHASE}_{TIMESTEPS}.npy")),
                dtype=torch.float32,
                device=self.device
            )

            self.X_STD = torch.tensor(
                np.load(os.path.join(self.MODEL_PATH, model_subfolder, prefix + f"_Xstd_experiment_1_(pull)_{PHASE}_{TIMESTEPS}.npy")),
                dtype=torch.float32,
                device=self.device
            )

            self.Y_MEAN = torch.tensor(
                np.load(os.path.join(self.MODEL_PATH, model_subfolder, prefix + f"_Ymean_experiment_1_(pull)_{PHASE}_{TIMESTEPS}.npy")),
                dtype=torch.float32,
                device=self.device
            )

            self.Y_STD = torch.tensor(
                np.load(os.path.join(self.MODEL_PATH, model_subfolder, prefix + f"_Ystd_experiment_1_(pull)_{PHASE}_{TIMESTEPS}.npy")),
                dtype=torch.float32,
                device=self.device
            )      
            
            # --- Load Model ---
            self.LFD_MODEL = LSTMRegressor(
                input_dim = self.N_APPROACH_STATES,   # number of features
                hidden_dim = self.MODEL_PARAMS['HIDDEN_DIM'],
                output_dim = self.N_ACTIONS,
                num_layers = self.MODEL_PARAMS['NUM_LAYERS'],
                pooling='last'
            )

            # Move model to device
            self.LFD_MODEL.to(self.device)
            self.LFD_MODEL.load_state_dict(torch.load(os.path.join(self.MODEL_PATH, model_subfolder, prefix + "model.pth")))

            # Set to evaluation mode
            self.LFD_MODEL.eval()
   

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

       
    def update_tcp_trail(self):
        if not self.running_lfd:
            return

        self.trail_step += 1
        if self.trail_step % self.TRAIL_EVERY_N_STEPS != 0:
            return

        p = Point()
        p.x = self.eef_pos_x_from_fk
        p.y = self.eef_pos_y_from_fk
        p.z = self.eef_pos_z_from_fk

        self.tcp_trail_marker.points.append(p)
        self.tcp_trail_marker.header.stamp = self.get_clock().now().to_msg()

        self.tcp_marker_pub.publish(self.tcp_trail_marker)


    # === Data saving functions ===
    def save_metadata(self, filename):
        """
        Create json file and save it with the same name as the bag file
        @param filename:
        @return:
        """
        
        # --- Open default metadata template for the experiment
        pkg_share = get_package_share_directory('lfd_apples')
        template_path = os.path.join(pkg_share, 'data', 'implementation_metadata_template.json')

        with open(template_path, 'r') as template_file:
            trial_info = json.load(template_file)
        
        # Date
        trial_info['general']['date'] = str(datetime.datetime.now())

        # Update proxy info
        # apple_id = input("Type the apple id: ")  # Wait for user to press Enter
        trial_info['proxy']['apple']['id'] = self.apple_id
        # spur_id = input("Type the spur id: ")  # Wait for user to press Enter
        trial_info['proxy']['spur']['id'] = self.spur_id
        trial_info['proxy']['apple']['pose']['position']['x'] =  self.apple_pose_base[0]
        trial_info['proxy']['apple']['pose']['position']['y'] =  self.apple_pose_base[1]
        trial_info['proxy']['apple']['pose']['position']['z'] =  self.apple_pose_base[2]
        trial_info['proxy']['apple']['frame_id'] = 'base'
        

        # Update controllers info
        trial_info['controllers']['delta gain'] = self.DELTA_GAIN
        trial_info['controllers']['approach']['data based'] = self.MODEL_PARAMS
        trial_info['controllers']['approach']['PI']['P'] = self.POSITION_KP
        trial_info['controllers']['approach']['PI']['I'] = self.POSITION_KI
        trial_info['controllers']['approach']['PI']['PI gain'] = self.PI_GAIN
        trial_info['controllers']['approach']['pixel to meter'] = self.PIXEL_TO_METER_RATE
        trial_info['controllers']['approach']['states'] = self.APPROACH_STATE_NAME_KEYS
        trial_info['controllers']['approach']['actions'] = self.ACTION_NAMES

        # Update gripper weight
        # Gripper weight = 1190 g, Rim weight = 160 g, Gripper + Rim weight = 1350 g
        trial_info['robot']['gripper']['weight'] =  "1190 g"       


        # --- Save metadata in file    
        with open(filename + '.json', "w") as outfile:
            json.dump(trial_info, outfile, indent=4)

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

        if self.running_lfd_approach:
            
            if self.tof < self.TOF_THRESHOLD:
                self.get_logger().info(f"TOF threshold reached: {self.tof} < {self.TOF_THRESHOLD}")                  
                self.running_lfd = False
                self.running_lfd_approach = False
                self.send_stop_message()

            if self.tof < 100:
                self.get_logger().info(f"TOF reached: {self.tof} < {self.TOF_THRESHOLD} turning off PI controller")      
                self.PI_GAIN = 0.0


    def incoming_cam_sim(self):
        """
        Simulate incoming latent image from palm camera by publishing dummy data.
        This is for testing purposes only.
        """

        if self.running_lfd and self.DEBUGGING_MODE:

            # --- Get next action from debugging demo ---
            
            # ============= APPROACH =============
            if self.running_lfd_approach:

                if self.approach_debugging_step >= len(self.approach_debugging_actions_pd):
                    self.get_logger().info(f'Approach steps:{self.approach_debugging_step}')
                    self.get_logger().info("Debugging demo actions exhausted.")
                    self.send_stop_message()
                    # self.running_lfd_approach = False
                    self.running_lfd_approach = False
                    self.lfd_contact = True
                    return
                else:
                
                    action_row = self.approach_debugging_actions_pd.iloc[self.approach_debugging_step]
                    y = action_row.values
                    
                    self.get_logger().info(f"Approach Debugging mode - Step {self.approach_debugging_step}, Action: {y}")
                    

                    # =========================  HEADS UP ===========================
                    # If using bbox pixels from camera without prior, otherwise comment to use the prior knowledge
                    self.PIXEL_TO_METER_RATE = 0.3/320      # approx 20cm per 320 pixels
                    self.eef_pos_x_error = self.bbox_center_at_tcp[0] * self.PIXEL_TO_METER_RATE
                    self.eef_pos_y_error = self.bbox_center_at_tcp[1] * self.PIXEL_TO_METER_RATE
                    # ================================================================

                    self.sum_pos_x_error += self.eef_pos_x_error
                    self.sum_pos_y_error += self.eef_pos_y_error

                    self.get_logger().info(f"apple pose tcp: {self.p_apple_tcp}")

                    self.target_cmd.twist.linear.x = 0.0 * float(y[0]) * self.DELTA_GAIN \
                                                        + self.POSITION_KP * self.eef_pos_x_error \
                                                        + self.POSITION_KI * self.sum_pos_x_error
                    self.target_cmd.twist.linear.y = 0.0 * float(y[1]) * self.DELTA_GAIN \
                                                        + self.POSITION_KP * self.eef_pos_y_error \
                                                        + self.POSITION_KI * self.sum_pos_y_error
                    self.target_cmd.twist.linear.z = 1.0 * float(y[2]) * self.DELTA_GAIN \
                                                        + 0.0 * self.position_kp_z * self.eef_pos_z_error   
                    
                    self.target_cmd.twist.angular.x = 1.0 * float(y[3]) * self.DELTA_GAIN 
                    self.target_cmd.twist.angular.y = 1.0 * float(y[4]) * self.DELTA_GAIN 
                    self.target_cmd.twist.angular.z = 1.0 * float(y[5]) * self.DELTA_GAIN 
                                
                    self.approach_debugging_step += 1           

            # ============= CONTACT =============
            elif self.lfd_contact:
            
                if self.contact_debugging_step >= len(self.contact_debugging_actions_pd) and self.lfd_contact:
                    self.get_logger().info(f'Contact steps:{self.contact_debugging_step}')
                    self.get_logger().info("Debugging demo actions exhausted.")
                    self.send_stop_message()
                    # self.running_lfd_approach = False                
                    self.lfd_contact = False
                    self.lfd_pick = True
                    return
                else:
                
                    action_row = self.contact_debugging_actions_pd.iloc[self.contact_debugging_step]
                    y = action_row.values
                    
                    self.get_logger().info(f"Contact Debugging mode - Step {self.contact_debugging_step}, Action: {y}")

                    # If using deltas, multiply by scaling factor
                    # Send actions (twist)                           
                    self.target_cmd.twist.linear.x = float(y[0]) * self.DELTA_GAIN 
                    self.target_cmd.twist.linear.y = float(y[1]) * self.DELTA_GAIN 
                    self.target_cmd.twist.linear.z = float(y[2]) * self.DELTA_GAIN 
                    self.target_cmd.twist.angular.x = float(y[3]) * self.DELTA_GAIN 
                    self.target_cmd.twist.angular.y = float(y[4]) * self.DELTA_GAIN 
                    self.target_cmd.twist.angular.z = float(y[5]) * self.DELTA_GAIN 
                                
                    self.contact_debugging_step += 1          

            # ============= PICK =============
            elif self.lfd_pick:

                if self.pick_debugging_step >= len(self.pick_debugging_actions_pd):
                    self.get_logger().info(f'Pick steps:{self.pick_debugging_step}')
                    self.get_logger().info("Debugging demo actions exhausted.")
                    self.send_stop_message()
                    self.running_lfd = False                
                    self.lfd_pick = False
                
                    return
                else:
                
                    action_row = self.pick_debugging_actions_pd.iloc[self.pick_debugging_step]
                    y = action_row.values
                    
                    self.get_logger().info(f"Pick Debugging mode - Step {self.pick_debugging_step}, Action: {y}")

                    # If using deltas, multiply by scaling factor
                    # Send actions (twist)                           
                    self.target_cmd.twist.linear.x = float(y[0]) * self.DELTA_GAIN 
                    self.target_cmd.twist.linear.y = float(y[1]) * self.DELTA_GAIN 
                    self.target_cmd.twist.linear.z = float(y[2]) * self.DELTA_GAIN  
                    self.target_cmd.twist.angular.x = float(y[3]) * self.DELTA_GAIN 
                    self.target_cmd.twist.angular.y = float(y[4]) * self.DELTA_GAIN 
                    self.target_cmd.twist.angular.z = float(y[5]) * self.DELTA_GAIN 
                                
                    self.pick_debugging_step += 1          
                       
    
    def bbox_callback(self, msg: Int16MultiArray):
        '''
        Center of Bounding Box from Yolo node
        
        :param self: Description
        :param msg: Description
        :type msg: Int16MultiArray
        '''

        self.bbox_center_at_tcp = msg.data

        # =========================  HEADS UP ===========================
        # If using bbox pixels from camera without prior, otherwise comment to use the prior knowledge       
        self.bbox_pos_x_error = self.bbox_center_at_tcp[0] * self.PIXEL_TO_METER_RATE
        self.bbox_pos_y_error = self.bbox_center_at_tcp[1] * self.PIXEL_TO_METER_RATE
        # ================================================================
           

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
        
        # self.eef_pos_x_error = self.goal_pose[0] - self.eef_pos_x
        # self.eef_pos_y_error = self.goal_pose[1] - self.eef_pos_y
        # self.eef_pos_z_error = self.goal_pose[2] - self.eef_pos_z

        # Update sensors average since last action
    

    def joint_states_callback(self, msg: JointState):

        self.joint_states = list(msg.position[:7])

        # FK to obtain EEF pose at the base frame
        q = self.joint_states #.to_numpy(dtype=float)                
        T, Ts = fr3_fk(q)
        p_tcp_base = T[:3, 3]
        R_base_tcp = T[:3, :3]
        
        self.eef_pos_x_from_fk = p_tcp_base[0]
        self.eef_pos_y_from_fk = p_tcp_base[1]
        self.eef_pos_z_from_fk = p_tcp_base[2]    

        # Update TCP trail
        self.update_tcp_trail()

        # Distance between eef and apple
        self.distance_at_base_frame = self.apple_pose_base.values.squeeze() - p_tcp_base

        # Apple position in TCP frame
        self.p_apple_tcp = R_base_tcp.T @ (self.distance_at_base_frame)

        # Update distance from known target location
        self.eef_pos_x_error = self.p_apple_tcp[0]
        self.eef_pos_y_error = self.p_apple_tcp[1]
        self.eef_pos_z_error = self.p_apple_tcp[2]


    def palm_camera_callback(self, msg: Float32MultiArray):
        """
        We'll use this callback to publish Twist commands everytime a new latent image arrives.        
        
        :param self: Description
        :param msg: Latent Features from 15th layer of YOLO v8 cnn. 
        :type msg: Float32MultiArray with 64 elements
        """

        if self.running_lfd and not self.DEBUGGING_MODE:            

            if self.running_lfd_approach:

                # ====================== Build State ==========================
                # --- Extract latent image ---
                self.latent_image = np.array(msg.data, dtype=np.float32)
                self.ema_img.update(self.latent_image)
                # self.img_time_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9    # time in seconds

                # --- Combine data ---
                # self.state = np.concatenate([self.tof_for_state, self.latent_image, self.p_apple_tcp])    
                self.state = np.concatenate([self.tof_for_state, self.ema_img.y])    
                            
                # Combine data, with past steps if model is 'rf' or 'mlp'            
                # self.X = np.concatenate([self.t_2_data, self.t_1_data, self.state])
                self.state_buffer.append(self.state)        # Update buffer

                if not len(self.state_buffer) == self.MODEL_PARAMS['SEQ_LEN']:
                    pass
                else:
                    self.X = np.stack(self.state_buffer)

                    self.X_tensor = torch.tensor(self.X, dtype=torch.float32, device = self.device)
                    print('X Tensor shape:', self.X_tensor.shape)
                
                    # --- Normalize State ---                        
                    self.X_norm = (self.X_tensor - self.X_MEAN) / self.X_STD
                    print('X Tensor normalized shape:', self.X_norm.shape)

                    # # --- Append as row in DataFrame ---
                    # self.lfd_states_df = pd.concat([
                    #     self.lfd_states_df, 
                    #     # pd.DataFrame([self.t_data_norm])
                    #     pd.DataFrame([self.X_norm])
                    # ], ignore_index=True)
                
                    # ======================= Predict Actions ======================
                    # Predict normalized actions
                    # self.Y_norm = self.LFD_MODEL.predict(self.X_norm.reshape(1, -1))
                    self.Y_norm = self.LFD_MODEL(self.X_norm)
                    print('Y Tensor normalized shape:', self.Y_norm.shape)

                    # Denormalize actions
                    self.Y = self.Y_norm * self.Y_STD + self.Y_MEAN
                    self.Y = self.Y.squeeze()  
                    
                    # Move tensor to cpu
                    self.Y = self.Y.detach().cpu().numpy()
                
                    # y = self.Y
                    # # --- Append as row in DataFrame ---
                    # self.lfd_actions_df = pd.concat([
                    #     self.lfd_actions_df, 
                    #     # pd.DataFrame([self.t_data_norm])
                    #     pd.DataFrame([y])
                    #     ], ignore_index=True)
                    # self.get_logger().info(f'Target actions in eef frame: {y}')                     

                    # Adjust velocities with feedback from goal pose               
                    self.sum_pos_x_error += self.bbox_pos_x_error
                    self.sum_pos_y_error += self.bbox_pos_y_error

                    # Linear Velocities                   
                    self.target_cmd.twist.linear.x = 1.0 * float(self.Y[0]) * self.DELTA_GAIN \
                                                        + self.PI_GAIN * self.POSITION_KP * self.bbox_pos_x_error \
                                                        + self.PI_GAIN * self.POSITION_KI * self.sum_pos_x_error

                    self.target_cmd.twist.linear.y = 1.0 * float(self.Y[1]) * self.DELTA_GAIN \
                                                        + self.PI_GAIN * self.POSITION_KP * self.bbox_pos_y_error \
                                                        + self.PI_GAIN * self.POSITION_KI * self.sum_pos_y_error

                    self.target_cmd.twist.linear.z = 1.0 * float(self.Y[2]) * self.DELTA_GAIN

                    # Angular Velocities
                    self.target_cmd.twist.angular.x = 1.0 * float(self.Y[3]) * self.DELTA_GAIN
                    self.target_cmd.twist.angular.y = 1.0 * float(self.Y[4]) * self.DELTA_GAIN
                    self.target_cmd.twist.angular.z = 1.0 * float(self.Y[5]) * self.DELTA_GAIN

                    self.get_logger().info(f'Target actions in eef frame: {self.Y}')         
                    self.get_logger().info(f"palm camera callback sending topic")

                    # Combine data, with past steps if model is 'rf' or 'mlp'    
                    # Add actions to State Space to pass them to the next time steps
                    # self.t_2_data = self.t_1_data
                    # if self.KEEP_ACTIONS_MEMORY:
                    #     self.t_1_data = np.concatenate([self.state, self.Y])
                    # else:
                    #     self.t_1_data = self.state
                        
        if self.running_lfd and self.DEBUGGING_MODE:
            
            # Simply run the incoming_cam_simulated function that handles all twists
            self.incoming_cam_sim()


    # === Action Functions ====
    def run_lfd_approach(self):       

        self.get_logger().info("Starting run_lfd_approach: publishing twists from palm_camera_callback until TOF < threshold.")
        self.running_lfd = True
        self.running_lfd_approach = True
        self.sum_pos_x_error = 0.0
        self.sum_pos_y_error = 0.0
        self.PI_GAIN = 1.0      

        # Rviz: Clear previous trail       
        self.tcp_trail_marker.points.clear()
        self.trail_step = 0

        # Stop from keyboard
        key = KeyboardListener()

        while rclpy.ok() and self.running_lfd and not key.esc_pressed:
            
            # This runs depending on the flags           
            # 'palm camera callback' has the logic

            rclpy.spin_once(self,timeout_sec=0.0)
            time.sleep(0.001)
        
        self.send_stop_message()
        self.running_lfd = False

        
    def publish_smoothed_velocity(self):

        if not self.running_lfd:
            return

        dt = (self.get_clock().now() - self.last_cmd_time).nanoseconds * 1e-9
        self.last_cmd_time = self.get_clock().now()

        if dt <= 0.0:
            return
     
        
        self.current_cmd.twist = self.target_cmd.twist  
        self.current_cmd.header.frame_id = "fr3_hand_tcp"

        self.current_cmd.header.stamp = self.get_clock().now().to_msg()
        self.servo_pub.publish(self.current_cmd)


    def send_stop_message(self):
        """
        # Ensure a final zero-twist is sent to stop motion
        
        :param self: Description
        """

        self.target_cmd.twist = Twist()  # zero target
        self.current_cmd.header.frame_id = "fr3_hand_tcp"

        self.current_cmd.header.stamp = self.get_clock().now().to_msg()
        self.servo_pub.publish(self.current_cmd)


    def compute_joint_velocities(self, desired_ee_velocity: np.ndarray, joint_positions: list):
        """
        desired_ee_velocity: 6-element NumPy array
        joint_positions: list of 7 floats
        returns: 7-element NumPy array
        """
        jacobian = fr3_jacobian(joint_positions)
        joint_velocities = np.linalg.pinv(jacobian) @ desired_ee_velocity
        self.get_logger().info(f'Joint velocities: {joint_velocities}')


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

    MODEL_PARAMS = {'MODEL': 'lstm',
                    'SEQ_LEN': 30,
                    'NUM_LAYERS': 2,
                    'HIDDEN_DIM': 1024
                    }  
    
    node = LFDController(MODEL_PARAMS)    

    BASE_DIR = '/home/alejo/Documents/DATA'
    BAG_MAIN_DIR = '07_IL_implementation/bagfiles'

    EXPERIMENT = "experiment_1_(pull)/approach"

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
               

        # ------------ Step 1: Move robot to home position ---------
        node.get_logger().info("\n\033[1;32m\nSTEP 1: Arm moving to home position... wait\033[0m\n")
        while not node.move_to_home():
            pass

        input("\n\033[1;32m\nSTEP 2: Place apple on the proxy. \nPress ENTER key when done.\033[0m\n")    

        # ------------ Step 2: Allow free drive if needed ---------
        node.swap_controller(node.arm_controller, node.gravity_controller)
        time.sleep(0.5)                

        node.get_logger().info("\n\033[1;32m\nSTEP 3: Record apple position by bringing gripper close to it (cups gently apple). \nPress ENTER key when you are done.\033[0m\n")        
        input()   
        node.swap_controller(node.gravity_controller, node.arm_controller)
        time.sleep(0.5)          

        # Save proxy state        
        node.apple_pose_base[:] = [node.eef_pos_x, node.eef_pos_y, node.eef_pos_z]
        node.apple_id = input("Type the apple id: ")  # Wait for user to press Enter
        node.spur_id = input("Type the spur id: ")  # Wait for user to press Enter        
        node.place_rviz_apple()

        node.swap_controller(node.arm_controller, node.gravity_controller)
        time.sleep(0.5)               

        node.get_logger().info("\n\033[1;32m\nSTEP 3: Free-drive arm until apple in camera FOV. \nPress ENTER key when you are done.\033[0m\n")        
        input()    
       
        # ------------ Step 3: Enable Servo Node Cartesian velocity controller ---------
        node.swap_controller(node.gravity_controller, node.twist_controller)
        time.sleep(0.5)  

        node.get_logger().info("Enabling Moveit2 Servo Node for twist controller...")
        req = Trigger.Request()        
        node.servo_node_client.call_async(req)

        time.sleep(1.0)        

        # Start bag recording        
        robot_rosbag_list = start_recording_bagfile(ROBOT_BAG_FILEPATH)

        # -------------- Step 4: Run lfd controller ----------------        
        input("\n\033[1;32m\nSTEP 4: Press ENTER key to start ROBOT lfd implementation.\033[0m\n")     
        node.DEBUGGING_MODE = False
        if node.DEBUGGING_MODE: node.initialize_debugging_mode_variables   

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

        node.save_metadata(os.path.join(BAG_FILEPATH, TRIAL, "metadata_" + TRIAL))  

        # Check data
        node.get_logger().info("Check ROBOT demo data")         
        check_data_plots(ROBOT_BAG_FILEPATH, TRIAL)
        node.get_logger().info(f"Robot lfd implementation {demo+1}/10 done.")
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
