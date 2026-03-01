#!/usr/bin/env python3

# System imports
import subprocess
import time
import os
from pathlib import Path
import torch
from collections import deque
from ament_index_python.packages import get_package_share_directory
import sys, termios, tty, threading, select

# Custom imports
from lfd_apples.listen_franka import main as listen_main
from lfd_apples.listen_franka import start_recording_bagfile, stop_recording_bagfile, save_metadata, find_next_trial_number 
from lfd_apples.ros2bag2csv import extract_data_and_plot, parse_array, fr3_jacobian, fr3_fk
from lfd_apples.lfd_vision import extract_pooled_latent_vector
from lfd_apples.lfd_learning import get_phase_columns, expand_features_over_time
from lfd_apples.lfd_lstm import LSTMRegressor

# ROS2 imports
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray, Float32MultiArray, Bool
from std_srvs.srv import SetBool, Trigger
from geometry_msgs.msg import PoseStamped, TwistStamped, Twist, WrenchStamped, Point
from sensor_msgs.msg import Image, JointState
from visualization_msgs.msg import Marker
from control_msgs.action import FollowJointTrajectory
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



# --- Handy Objects ---
class KeyboardListener:
    def __init__(self):
        self.esc_pressed = False
        self._stop = False
        self.thread = threading.Thread(target=self.listen, daemon=True)
        self.thread.start()

    def listen(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        # cbreak: char-by-char input WITHOUT breaking the terminal
        tty.setcbreak(fd)

        try:
            while not self._stop and not self.esc_pressed:
                # Non-blocking check for input
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                if rlist:
                    ch = sys.stdin.read(1)
                    if ch == '\x1b':  # ESC
                        self.esc_pressed = True
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def stop(self):
        self._stop = True


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


class LoadPhaseController():
    '''
    Convenient Class to load all the lstm model properties
    '''

    def __init__(self, MODEL_PARAMS, device):

        self.SEQ_LEN = MODEL_PARAMS['SEQ_LEN']
        self.PHASE = MODEL_PARAMS['PHASE']
        self.TIMESTEPS = '0_timesteps'
        self.HIDDEN_DIM = MODEL_PARAMS['HIDDEN_DIM']
        self.NUM_LAYERS = MODEL_PARAMS['NUM_LAYERS']

        self.device = device

        self.load_info_from_yaml()
        self.define_paths()
        self.load_model_statistics()
        self.load_model()


    def load_info_from_yaml(self):
        # --- Load yaml with State and Action Space variables ---
        data_columns_path = config_path = Path(__file__).parent / "config" / "lfd_data_columns.yaml"
        with open(data_columns_path, "r") as f:
            cfg = yaml.safe_load(f)    
        
        self.ACTION_NAMES = cfg['action_cols']              
        self.N_ACTIONS = len(self.ACTION_NAMES)   

        # States   
        self.STATE_NAMES, self.STATE_NAME_KEYS = get_phase_columns(cfg, self.PHASE)
        self.N_STATES = len(self.STATE_NAMES) - self.N_ACTIONS


    def define_paths(self):
        # --- Model Path ---             
        self.BASE_PATH = '/home/alejo/Documents/DATA'        
        self.MODEL_PATH = os.path.join(self.BASE_PATH, '06_IL_learning/experiment_1_(pull)', self.PHASE, self.TIMESTEPS)  

        self.STATE_NAME_KEYS = self.STATE_NAME_KEYS[:-1]
        input_keys_subfolder_name = "__".join(self.STATE_NAME_KEYS)
        self.model_subfolder = os.path.join(self.MODEL_PATH, input_keys_subfolder_name)      

        self.prefix = str(self.NUM_LAYERS) + '_layers_' + str(self.HIDDEN_DIM) + '_dim_' + str(self.SEQ_LEN) + "_seq_lstm_"
              

    def load_model_statistics(self):

        # --- Load statistics ---
        self.X_MEAN = torch.tensor(
            np.load(os.path.join(self.MODEL_PATH, 
                                 self.model_subfolder,
                                 self.prefix + f"_Xmean_experiment_1_(pull)_{self.PHASE}_{self.TIMESTEPS}.npy")),
            dtype=torch.float32,
            device=self.device
        )

        self.X_STD = torch.tensor(
            np.load(os.path.join(self.MODEL_PATH,
                                 self.model_subfolder,
                                 self.prefix + f"_Xstd_experiment_1_(pull)_{self.PHASE}_{self.TIMESTEPS}.npy")),
            dtype=torch.float32,
            device=self.device
        )

        self.Y_MEAN = torch.tensor(
            np.load(os.path.join(self.MODEL_PATH,
                                 self.model_subfolder,
                                 self.prefix + f"_Ymean_experiment_1_(pull)_{self.PHASE}_{self.TIMESTEPS}.npy")),
            dtype=torch.float32,
            device=self.device
        )

        self.Y_STD = torch.tensor(
            np.load(os.path.join(self.MODEL_PATH,
                                 self.model_subfolder,
                                 self.prefix + f"_Ystd_experiment_1_(pull)_{self.PHASE}_{self.TIMESTEPS}.npy")),
            dtype=torch.float32,
            device=self.device
        )      


    def load_model(self):

        # --- Load Model ---
        self.MODEL = LSTMRegressor(
            input_dim = self.N_STATES,   # number of features
            hidden_dim = self.HIDDEN_DIM,
            output_dim = self.N_ACTIONS,
            num_layers = self.NUM_LAYERS,
            pooling='last'
        )

        # Move model to device
        self.MODEL.to(self.device)
        self.MODEL.load_state_dict(torch.load(os.path.join(self.MODEL_PATH,
                                                           self.model_subfolder,
                                                           self.prefix + "model.pth")))

        # Set to evaluation mode
        self.MODEL.eval()


    def normalize_x(self, x):

        # print('X-Tensor shape:', x)
        x_n = (x - self.X_MEAN)/self.X_STD
        # print('normalized X-Tensor shape:', x_n)
        return x_n
    

    def denormalize_y(self, y):
        return y*self.Y_STD + self.Y_MEAN
    

    def forward(self, x):
        x_n = self.normalize_x(x)
        y_n = self.MODEL(x_n)
        # print('normalized Y-Tensor shape:', y_n)

        y_dn = self.denormalize_y(y_n)

        y_dn = y_dn.squeeze()

        # Move tensor to cpu
        y_dn = y_dn.detach().cpu().numpy()
        
        return y_dn



# --- Main Class ---
class LFDController(Node):

    def __init__(self, APPROACH_MODEL_PARAMS, CONTACT_MODEL_PARAMS, PICK_MODEL_PARAMS):
        super().__init__('move_to_home_and_freedrive')

        self.APPROACH_MODEL_PARAMS = APPROACH_MODEL_PARAMS
        self.CONTACT_MODEL_PARAMS = CONTACT_MODEL_PARAMS
        self.PICK_MODEL_PARAMS = PICK_MODEL_PARAMS

        self.initialize_state_and_models()
        self.initialize_debugging_mode_variables('trial_1')        
        self.initialize_signal_variables_and_thresholds()
        self.initialize_arm_poses()            
                
        self.initialize_ros_topics()
        self.initialize_ros_service_clients()
        self.initialize_ros_controller_properties()    
        self.initialize_rviz_trail()  
        self.initialize_flags()               

        self.last_cmd_time = self.get_clock().now()   
        self.joint_names = [f"fr3_joint{i+1}" for i in range(7)]
        self.joint_states = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.trajectory_points = []       
             

    # === Initializing functions ===
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
        self.eef_wrench_sub = self.create_subscription(
            WrenchStamped,
            '/franka_robot_state_broadcaster/external_wrench_in_stiffness_frame',
            self.eef_wrench_callback,
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
        self.tgt_twist_pub = self.create_publisher(
            TwistStamped,
            '/lfd/delta_twist_target',
            10)               
        self.tcp_marker_pub = self.create_publisher(
            Marker,
            '/lfd/tcp_trail',
            10) 
        self.apple_marker_pub = self.create_publisher(
            Marker,
            '/lfd/apple',
            10) 
        self.apple_probing_pub = self.create_publisher(
            Bool,
            'lfd/apple_probing_apple',
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
        
        # Controller namess
        self.arm_controller = 'fr3_arm_controller'
        self.gravity_controller = 'gravity_compensation_example_controller'
        self.joint_velocity_controller = 'fr3_joint_velocity_controller'
     
     
        # Switch controller client
        self.switch_client = self.create_client(SwitchController, '/controller_manager/switch_controller')
        self.switch_client.wait_for_service()

        # Load controller client
        self.load_client = self.create_client(LoadController, '/controller_manager/load_controller')
        self.load_client.wait_for_service()

        # --- Twist variables ---
        self.target_cmd = TwistStamped()        
        self.target_cmd.twist.linear.x = 0.0
        self.target_cmd.twist.linear.y = 0.0
        self.target_cmd.twist.linear.z = 0.0
        self.target_cmd.twist.angular.x = 0.0
        self.target_cmd.twist.angular.y = 0.0
        self.target_cmd.twist.angular.z = 0.0
        self.tgt_twist_pub.publish(self.target_cmd)       

        # Controller gain for delta
        # Converts 'm' and 'rad' deltas into m/s and rad/s
        # In our case, delta_eef_pose was calculated for delta_times = 0.001 sec, 
        # hence, we use a gain of 1000 (1/0.001sec) to convert m to m/sec.
        self.DELTA_GAIN = 1000
        # self.DELTA_GAIN = 1500    # I used this one for command interface = position

        # PID GAINS
        # If using deltas, multiply by scaling factor
        # Send actions (twist)       
        self.INITIAL_PI_GAIN = 0.0                    
        self.POSITION_KP = 1.25  # 2.0
        self.POSITION_KI = 0.010

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
  

    def initialize_state_and_models(self):
        '''
        Space Representation for lfd inference        
        '''
        self.apple_pose_base = pd.Series(
            [1.0, 1.0, 1.0],
            index=['apple._x._base', 'apple._y._base', 'apple._z._base']
            )

        self.apple_pose_min_dist = 1e3

        # State and Actions book-keeping
        self.lfd_states_df = pd.DataFrame()
      

        # Phase state buffers
        self.approach_state_buffer = deque(maxlen = self.APPROACH_MODEL_PARAMS['SEQ_LEN'])
        self.contact_state_buffer = deque(maxlen = self.CONTACT_MODEL_PARAMS['SEQ_LEN'])
        self.pick_state_buffer = deque(maxlen = self.PICK_MODEL_PARAMS['SEQ_LEN'])

        # Models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
        self.APPROACH_CONTROLLER = LoadPhaseController(self.APPROACH_MODEL_PARAMS, self.device)
        self.CONTACT_CONTROLLER = LoadPhaseController(self.CONTACT_MODEL_PARAMS, self.device)
        self.PICK_CONTROLLER = LoadPhaseController(self.PICK_MODEL_PARAMS, self.device)        

        # Actions
        self.current_twist_linear_cmd_base = np.array([0.0, 0.0, 0.0])
        self.current_twist_angular_cmd_base = np.array([0.0, 0.0, 0.0])


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
        self.ACTION_NAMES = self.APPROACH_CONTROLLER.ACTION_NAMES
        self.approach_debugging_actions_pd = self.approach_debugging_demo_pd[self.ACTION_NAMES]
        self.contact_debugging_actions_pd = self.contact_debugging_demo_pd[self.ACTION_NAMES]
        self.pick_debugging_actions_pd = self.pick_debugging_demo_pd[self.ACTION_NAMES]


    def initialize_signal_variables_and_thresholds(self):        

        # Thresholds
        # TOF threshold is used to switch from 'approach' to 'contact' phase
        self.TOF_THRESHOLD = 50                 # units in mm
        # Air pressure threshold is used to tell when a suction cup has engaged.
        self.AIR_PRESSURE_THRESHOLD = 600       # units in hPa        
       
        # Gripper Signal Variables with Exponential Moving Average
        self.ema_alpha = 0.75
        self.ema_scA = EMA(self.ema_alpha)
        self.ema_scB = EMA(self.ema_alpha)
        self.ema_scC = EMA(self.ema_alpha)
        self.ema_tof = EMA(self.ema_alpha)
        
        self.ema_img = EMA(alpha = 0.75)

        self.ema_wrench = EMA(alpha = 0.06)     # Sampling freq is 1Khz, so we want to smooth over approx 0.5 seconds of data, which corresponds to alpha = 0.07

        # Number of suction cups engaged
        self.scA_previous_state = False
        self.scB_previous_state = False
        self.scC_previous_state = False
        self.scups_engaged = 0

        self.prev_time = None

        self.sum_vel_x_error = 0.0
        self.sum_vel_y_error = 0.0
        self.sum_vel_z_error = 0.0


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
        self.flag_PI = False

        self.running_lfd = False
        self.running_lfd_approach = False
        self.running_lfd_contact = False
        self.running_lfd_pick = False   
        self.waiting_for_fingers_to_close = False     
        self.transition_to_pick = False


    # def initialize_ros_timers(self):        
        
    #     # --- Timer for high-rate velocity ramping ---
    #     self.timer_period = 0.0025  # 500 Hz
    #     self.create_timer(self.timer_period, self.publish_smoothed_velocity)

    #     # --- Timer to recreate incoming palm camera with fake hardware ---
    #     # self.create_timer(0.034, self.incoming_cam_sim)
           

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
        point.accelerations = [0.0] * len(self.joint_names)
        point.velocities = [0.0] * len(self.joint_names)

        point.time_from_start.sec = 8
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


    def apple_probing_flag(self, value: bool):
        
        self.probing_flag = Bool()        
        self.probing_flag.data = value
        self.apple_probing_pub.publish(self.probing_flag)


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
        trial_info['controllers']['approach']['data based'] = self.APPROACH_MODEL_PARAMS
        trial_info['controllers']['approach']['states'] = self.APPROACH_CONTROLLER.STATE_NAME_KEYS
        trial_info['controllers']['approach']['actions'] = self.ACTION_NAMES

        trial_info['controllers']['contact']['data based'] = self.CONTACT_MODEL_PARAMS
        trial_info['controllers']['contact']['states'] = self.CONTACT_CONTROLLER.STATE_NAME_KEYS
        trial_info['controllers']['contact']['actions'] = self.ACTION_NAMES

        trial_info['controllers']['pick']['data based'] = self.PICK_MODEL_PARAMS
        trial_info['controllers']['pick']['states'] = self.PICK_CONTROLLER.STATE_NAME_KEYS
        trial_info['controllers']['pick']['actions'] = self.ACTION_NAMES


        # Update gripper weight
        # Gripper weight = 1190 g, Rim weight = 160 g, Gripper + Rim weight = 1350 g
        trial_info['robot']['gripper']['weight'] =  "1190 g"       


        # Save results metrics
        trial_info['results']['approach metrics']['final pose'] =  self.p_apple_tcp_enf_of_approach.flatten().tolist()


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

        # Put into numpy arrays
        self.scups_state = np.array([self.scA, self.scB, self.scC])        
        self.tof_state = np.array([self.tof])       

        
        # State machine triggers: Approach --> Contact --> Pick
        if self.running_lfd_approach:
            
            if self.tof < self.TOF_THRESHOLD:
                self.get_logger().warning(f"TOF threshold reached: {self.tof:.2f} < {self.TOF_THRESHOLD} --> CONTACT")     
                self.running_lfd_approach = False                                

                self.scups_engaged = 0
                self.running_lfd_contact = True


            if self.tof < 100 and not self.flag_PI:
                self.get_logger().warning(f"TOF: {self.tof:.2f} < 100, switching PI off")
                self.flag_PI = True
                self.PI_GAIN = 0.0


        if self.running_lfd_contact:
            # self.get_logger().info(f"CONTACT Controller running: {self.scups_engaged} suction cups engaged")   

            # Count engaged suction cups
            if self.scups_engaged > 1:
                self.get_logger().warning(f"More than 2 cups engaged --> ACTIVATING FINGERS")    
                self.running_lfd_contact = False
                self.waiting_for_fingers_to_close = True                              
                self.send_stop_message()
            
            # Increase Count
            if self.scA < self.AIR_PRESSURE_THRESHOLD and not self.scA_previous_state:
                self.scA_previous_state = True
                self.scups_engaged +=1
                self.get_logger().warning(f" {self.scups_engaged} suction cups engaged")   
            if self.scB < self.AIR_PRESSURE_THRESHOLD and not self.scB_previous_state:
                self.scB_previous_state = True
                self.scups_engaged +=1
                self.get_logger().warning(f" {self.scups_engaged} suction cups engaged")   
            if self.scC < self.AIR_PRESSURE_THRESHOLD and not self.scC_previous_state:
                self.scC_previous_state = True
                self.scups_engaged +=1
                self.get_logger().warning(f" {self.scups_engaged} suction cups engaged")   

            # Decrease Count
            if self.scA > self.AIR_PRESSURE_THRESHOLD and self.scA_previous_state:
                self.scA_previous_state = False
                self.scups_engaged -=1
                self.get_logger().warning(f" {self.scups_engaged} suction cups engaged")   
            if self.scB > self.AIR_PRESSURE_THRESHOLD and self.scB_previous_state:
                self.scB_previous_state = False
                self.scups_engaged -=1
                self.get_logger().warning(f" {self.scups_engaged} suction cups engaged")   
            if self.scC > self.AIR_PRESSURE_THRESHOLD and self.scC_previous_state:
                self.scC_previous_state = False
                self.scups_engaged -=1
                self.get_logger().warning(f" {self.scups_engaged} suction cups engaged")   
        

        if self.waiting_for_fingers_to_close:
            # Initialize timer when entering the state
            if not hasattr(self, 'fingers_timer_start'):
                self.fingers_timer_start = self.get_clock().now()
                self.get_logger().info("Starting fingers close timer...")

            # Compute elapsed time
            elapsed = (self.get_clock().now() - self.fingers_timer_start).nanoseconds * 1e-9  # seconds

            WAIT_FOR_FINGERS_DURATION = 0.25  # seconds, adjust as needed
            
            self.target_cmd.twist.linear.z = +0.00

            self.get_logger().warning(f"Closing Fingers --> {elapsed}") 

            if elapsed > WAIT_FOR_FINGERS_DURATION:
                self.get_logger().warning("More than 2 cups engaged --> WAITING FOR FINGERS TO CLOSE DONE")    
                self.waiting_for_fingers_to_close = False              
                self.transition_to_pick = True
                self.send_stop_message()
                del self.fingers_timer_start  # reset timer for next time
                        

        if self.transition_to_pick:
            # Initialize timer when entering the state
            if not hasattr(self, 'transition_to_pick_timer_start'):
                self.transition_to_pick_timer_start = self.get_clock().now()
                self.get_logger().info("Starting transition to pick...")
            

            self.target_cmd.twist.linear.z = -0.00


            # Compute elapsed time
            elapsed = (self.get_clock().now() - self.transition_to_pick_timer_start).nanoseconds * 1e-9  # seconds

            WAIT_FOR_TRANSITION_TO_PICK_DURATION = 0.25  # seconds, adjust as needed
            self.get_logger().warning(f"Transition To Pick --> {elapsed}") 

            if elapsed > WAIT_FOR_TRANSITION_TO_PICK_DURATION:
                self.get_logger().warning("Transitioning to Pick with a simple twist in -z ")    
                self.transition_to_pick = False              
                self.running_lfd_pick = True
                del self.transition_to_pick_timer_start  # reset timer for next time


        if self.running_lfd_pick:          

            self.get_logger().info(f"PICK controller running")   
            
            # if condition TBD:
            #   self.running_lfd_pick = False
            #   self.running_lfd = False


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
        
        # TCP velocities
        t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        p = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

        q = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])

        if self.prev_time is not None:
            dt = t - self.prev_time
            if dt > 1e-4:
                self.v_base = (p - self.prev_p) / dt

                # Optional angular velocity
                R_prev = R.from_quat(self.prev_q)
                R_now  = R.from_quat(q)
                R_err = R_prev.inv() * R_now
                self.omega_base = R_err.as_rotvec() / dt

        self.R_now = R.from_quat(q).as_matrix()

        self.prev_p = p
        self.prev_q = q
        self.prev_time = t        
    

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


        if self.running_lfd_approach:

            euclidean_distance = np.linalg.norm(self.p_apple_tcp)

            if euclidean_distance < self.apple_pose_min_dist:

                self.apple_pose_min_dist = euclidean_distance
                self.p_apple_tcp_enf_of_approach = self.p_apple_tcp


    def eef_wrench_callback(self, msg: WrenchStamped):
    
        self.eef_fx = msg.wrench.force.x
        self.eef_fy = msg.wrench.force.y
        self.eef_fz = msg.wrench.force.z
        self.eef_tx = msg.wrench.torque.x
        self.eef_ty = msg.wrench.torque.y
        self.eef_tz = msg.wrench.torque.z

        self.wrench_state = np.array([self.eef_fx, self.eef_fy, self.eef_fz, self.eef_tx, self.eef_ty, self.eef_tz])

        # Online EMA filtering of wrench data to reduce noise
        self.wrench_state = self.ema_wrench.update(self.wrench_state)


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
                self.latent_img = np.asarray(self.ema_img.y, dtype=np.float32)   
                # self.img_time_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9    # time in seconds

                # --- Combine data ---
                # self.state = np.concatenate([self.tof_state, self.latent_image, self.p_apple_tcp])    
                self.state = np.concatenate([self.tof_state, self.latent_img, self.p_apple_tcp])    
                
                self.approach_state_buffer.append(self.state)        # Update buffer

                if not len(self.approach_state_buffer) == self.APPROACH_MODEL_PARAMS['SEQ_LEN']:
                    pass
                else:
                    self.X = np.stack(self.approach_state_buffer)
                    self.X_tensor = torch.tensor(self.X, dtype=torch.float32, device = self.device)
                    # Predictions in tcp frame
                    self.Y = self.APPROACH_CONTROLLER.forward(self.X_tensor)
                
                    # Save state rows in DataFrame
                    self.lfd_states_df = pd.concat([self.lfd_states_df, pd.DataFrame([self.X[-1]])], ignore_index=True)

                    # Linear Twist                   
                    self.target_cmd.twist.linear.x = 1.0 * float(self.Y[0]) * self.DELTA_GAIN 
                    self.target_cmd.twist.linear.y = 1.0 * float(self.Y[1]) * self.DELTA_GAIN
                    self.target_cmd.twist.linear.z = 1.0 * float(self.Y[2]) * self.DELTA_GAIN
                    # Angular Twist
                    self.target_cmd.twist.angular.x = 1.0 * float(self.Y[3]) * self.DELTA_GAIN
                    self.target_cmd.twist.angular.y = 1.0 * float(self.Y[4]) * self.DELTA_GAIN
                    self.target_cmd.twist.angular.z = 1.0 * float(self.Y[5]) * self.DELTA_GAIN
                 

                    twist_array = np.array([self.target_cmd.twist.linear.x,
                                            self.target_cmd.twist.linear.y,
                                            self.target_cmd.twist.linear.z,
                                            self.target_cmd.twist.angular.x,
                                            self.target_cmd.twist.angular.y,
                                            self.target_cmd.twist.angular.z
                                            ])

                    actions_str = np.array2string(
                        twist_array,
                        precision=6,
                        suppress_small=True,
                        separator=' ',
                        max_line_width=np.inf
                    )                    
                    self.get_logger().info(f'APPROACH actions: {actions_str}')
                   
                                

            if self.running_lfd_contact:

                # Combine data and update sequence buffer
                self.state = np.concatenate([self.tof_state, self.scups_state, self.p_apple_tcp])    
                state_str = np.array2string(self.state, precision=2, separator=',', suppress_small=True)                
                self.get_logger().info(f"CONTACT state: {state_str}")                
                self.contact_state_buffer.append(self.state)

                # Wait until buffer filled
                if not len(self.contact_state_buffer) == self.CONTACT_MODEL_PARAMS['SEQ_LEN']:
                    pass
                else:
                    self.X = np.stack(self.contact_state_buffer)
                    self.X_tensor = torch.tensor(self.X, dtype=torch.float32, device = self.device)
                    # Predictions in tcp frame
                    self.Y = self.CONTACT_CONTROLLER.forward(self.X_tensor)

                    # Save state rows in DataFrame
                    self.lfd_states_df = pd.concat([self.lfd_states_df, pd.DataFrame([self.X[-1]])], ignore_index=True)                  

                    # Linear Twist
                    self.target_cmd.twist.linear.x = 1.0 * float(self.Y[0]) * self.DELTA_GAIN
                    self.target_cmd.twist.linear.y = 1.0 * float(self.Y[1]) * self.DELTA_GAIN
                    self.target_cmd.twist.linear.z = 1.0 * float(self.Y[2]) * self.DELTA_GAIN
                    # Angular Twist
                    self.target_cmd.twist.angular.x = 1.0 * float(self.Y[3]) * self.DELTA_GAIN
                    self.target_cmd.twist.angular.y = 1.0 * float(self.Y[4]) * self.DELTA_GAIN
                    self.target_cmd.twist.angular.z = 1.0 * float(self.Y[5]) * self.DELTA_GAIN

                    twist_array = np.array([self.target_cmd.twist.linear.x,
                                            self.target_cmd.twist.linear.y,
                                            self.target_cmd.twist.linear.z,
                                            self.target_cmd.twist.angular.x,
                                            self.target_cmd.twist.angular.y,
                                            self.target_cmd.twist.angular.z
                                            ])

                    actions_str = np.array2string(
                        twist_array,
                        precision=6,
                        suppress_small=True,
                        separator=' ',
                        max_line_width=np.inf
                    )

                    self.get_logger().info(f'CONTACT actions: {actions_str}')
                            


            if self.running_lfd_pick:

                # Combine data and update sequence buffer
                # tof, air_pressure, wrench, apple_prior
                # self.state = np.concatenate([self.tof_state, self.scups_state, self.wrench_state, self.p_apple_tcp])    
                self.state = np.concatenate([self.wrench_state, self.p_apple_tcp])   
                state_str = np.array2string(self.state, precision=6, separator=',', suppress_small=True)                
                self.get_logger().info(f"PICK state: {state_str}")   

                self.pick_state_buffer.append(self.state)

                # Wait until buffer filled
                if not len(self.pick_state_buffer) == self.PICK_MODEL_PARAMS['SEQ_LEN']:
                    pass
                else:
                    self.X = np.stack(self.pick_state_buffer)
                    self.X_tensor = torch.tensor(self.X, dtype=torch.float32, device = self.device)
                    # Predictions in tcp frame
                    self.Y = self.PICK_CONTROLLER.forward(self.X_tensor)

                    # Save state and action rows in DataFrames
                    self.lfd_states_df = pd.concat([self.lfd_states_df, pd.DataFrame([self.X[-1]])], ignore_index=True)                  

                    # Linear Twist
                    self.target_cmd.twist.linear.x = 1.0 * float(self.Y[0]) * self.DELTA_GAIN
                    self.target_cmd.twist.linear.y = 1.0 * float(self.Y[1]) * self.DELTA_GAIN
                    self.target_cmd.twist.linear.z = 1.0 * float(self.Y[2]) * self.DELTA_GAIN
                    # Angular Twist
                    self.target_cmd.twist.angular.x = 1.0 * float(self.Y[3]) * self.DELTA_GAIN
                    self.target_cmd.twist.angular.y = 1.0 * float(self.Y[4]) * self.DELTA_GAIN
                    self.target_cmd.twist.angular.z = 1.0 * float(self.Y[5]) * self.DELTA_GAIN

                    twist_array = np.array([self.target_cmd.twist.linear.x,
                                            self.target_cmd.twist.linear.y,
                                            self.target_cmd.twist.linear.z,
                                            self.target_cmd.twist.angular.x,
                                            self.target_cmd.twist.angular.y,
                                            self.target_cmd.twist.angular.z
                                            ])
                    
                    actions_str = np.array2string(
                        twist_array,
                        precision=6,
                        suppress_small=True,
                        separator=' ',
                        max_line_width=np.inf
                    )
                    self.get_logger().info(f'PICK actions: {actions_str}')
                          

            # Publish target twist. These are then ramped by 'lfd_action_smoother_node.py'         
            now = self.get_clock().now()
            self.target_cmd.header.stamp = now.to_msg()
            self.target_cmd.header.frame_id = "fr3_hand_tcp"
            self.tgt_twist_pub.publish(self.target_cmd)



        if self.running_lfd and self.DEBUGGING_MODE:
            
            # Simply run the incoming_cam_simulated function that handles all twists
            self.incoming_cam_sim()


    # === Action Functions ====
    def run_lfd_controller(self):       

        self.get_logger().info("Starting run_lfd_controller.")
        self.running_lfd = True
        self.running_lfd_approach = True

        # self.sum_pos_x_error = 0.0
        # self.sum_pos_y_error = 0.0        

        # self.PI_GAIN = self.INITIAL_PI_GAIN     

        # Clear previous eef trail in RVIZ       
        self.tcp_trail_marker.points.clear()
        self.trail_step = 0

        # Emergency Stop from keyboard
        key = KeyboardListener()

        while rclpy.ok() and self.running_lfd and not key.esc_pressed:
                   
            # This runs depending on the flags           
            # 'palm camera callback' has the logic

            rclpy.spin_once(self,timeout_sec=0.0)
            time.sleep(0.01)
        
        self.send_stop_message()
        self.running_lfd = False

        
    def send_stop_message(self):
        """
        # Ensure a final zero-twist is sent to stop motion        
        :param self: Description
        """

        self.target_cmd.twist = Twist()  # zero target
        now = self.get_clock().now()
        self.target_cmd.header.stamp = now.to_msg()
        self.target_cmd.header.frame_id = "fr3_hand_tcp"
        self.tgt_twist_pub.publish(self.target_cmd)


   

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
    YELLOW = "\033[1;33m"
    RESET = "\033[0m"

    APPROACH_MODEL_PARAMS = {'MODEL': 'lstm',
                             'PHASE': 'phase_1_approach',
                             'NUM_LAYERS': 2,
                             'HIDDEN_DIM': 1024,
                             'SEQ_LEN': 30                                                         
                            }  
    
    CONTACT_MODEL_PARAMS = {'MODEL': 'lstm',
                            'PHASE': 'phase_2_contact',
                            'NUM_LAYERS': 2,
                            'HIDDEN_DIM': 64,
                            'SEQ_LEN': 5                                             
                            }  
    
    PICK_MODEL_PARAMS = {'MODEL': 'lstm',
                         'PHASE': 'phase_3_pick',
                         'NUM_LAYERS': 2,
                         'HIDDEN_DIM': 1024,
                         'SEQ_LEN': 50                        
                        }  
    
    node = LFDController(APPROACH_MODEL_PARAMS, CONTACT_MODEL_PARAMS, PICK_MODEL_PARAMS)    

    BASE_DIR = '/home/alejo/Documents/DATA'
    BAG_MAIN_DIR = '07_IL_implementation/bagfiles'

    EXPERIMENT = "experiment_1_(pull)/approach"

    BAG_FILEPATH = os.path.join(BASE_DIR, BAG_MAIN_DIR, EXPERIMENT)
    os.makedirs(BAG_FILEPATH, exist_ok=True)

    SLEEP_TIME = 0.25
    
    batch_size = 10
    node.get_logger().info(f"Starting lfd implementation with robot, session of {batch_size} demos.")    


    for demo in range(batch_size):

        node.get_logger().info(f"{YELLOW}\n\n---------- Press ENTER key to start lfd implementation trial {demo+1}/10 ----------\n{RESET}")
        input()  
      
        # ------------ Step 0: Initial configuration ----------------
        TRIAL = find_next_trial_number(BAG_FILEPATH, prefix="trial_")
        os.makedirs(os.path.join(BAG_FILEPATH, TRIAL), exist_ok=True)
        # HUMAN_BAG_FILEPATH = os.path.join(BAG_FILEPATH, TRIAL, 'human')
        ROBOT_BAG_FILEPATH = os.path.join(BAG_FILEPATH, TRIAL)              

        node.initialize_state_and_models()
        node.initialize_signal_variables_and_thresholds()
        node.initialize_flags()               
               

        # ------------ Step 1: Move robot to home position ---------
        node.get_logger().info(f"{YELLOW}\n\nSTEP 1: Arm moving to home position... wait.\n{RESET}")
        while not node.move_to_home():
            pass


        # ------------ Step 2: Place apple and record position ------       
        node.get_logger().info(f"{YELLOW}\n\nSTEP 2: Place apple on the proxy. \nPress ENTER key when done.\n{RESET}")    
        input()
       
        node.swap_controller(node.arm_controller, node.gravity_controller)
        time.sleep(SLEEP_TIME)                

        node.apple_probing_flag(True)
        node.get_logger().info(f"{YELLOW}\n\nSTEP 3: Record apple position by bringing gripper close to it (cups gently apple).\nPress ENTER key when you are done.\n{RESET}")        
        input()           
        node.swap_controller(node.gravity_controller, node.arm_controller)
        time.sleep(SLEEP_TIME)          

        # Save proxy state        
        node.apple_pose_base[:] = [node.eef_pos_x, node.eef_pos_y, node.eef_pos_z]
        node.apple_id = input("Type the apple id: ")    # Wait for user to press Enter
        node.spur_id = input("Type the spur id: ")      # Wait for user to press Enter        
        node.place_rviz_apple()

        node.swap_controller(node.arm_controller, node.gravity_controller)
        time.sleep(SLEEP_TIME)               


        # ------------- Step 3: Place arm at a feasbile initial pose --------
        node.get_logger().info(f"{YELLOW}\n\nSTEP 3: Free-drive arm until apple in camera FOV. \nPress ENTER key when you are done.\n{RESET}")        
        input()    
       
        # Moveit 2 - Servo node
        node.apple_probing_flag(False)
        node.swap_controller(node.gravity_controller, node.joint_velocity_controller)
        time.sleep(SLEEP_TIME)          
        node.send_stop_message()
        req = Trigger.Request()        
        node.servo_node_client.call_async(req)        
        node.send_stop_message()
        node.get_logger().info("Moveit2 Servo Node enabled")
        time.sleep(SLEEP_TIME)        


        # Start recording bag
        robot_rosbag_list = start_recording_bagfile(ROBOT_BAG_FILEPATH)


        # -------------- Step 4: Run lfd controller ----------------        
        node.get_logger().info(f"{YELLOW}\n\nSTEP 4: Press ENTER key to start ROBOT lfd implementation.\n{RESET}")     
        input()

        node.DEBUGGING_MODE = False
        if node.DEBUGGING_MODE: node.initialize_debugging_mode_variables   

        node.run_lfd_controller()      

        # Save states to CSV        
        csv_path = os.path.join(ROBOT_BAG_FILEPATH, 'lfd_states.csv')
        node.lfd_states_df.to_csv(csv_path, index=False, header=False)
        node.get_logger().info(f"LFD approach data saved to {csv_path}")
        

        # -------------- Step 5: Dispose apple ----------------
        node.get_logger().info(f"{YELLOW}\n\nSTEP 5: Press ENTER key to dispose apple.\n{RESET}")
        input()
        
        node.swap_controller(node.joint_velocity_controller, node.arm_controller)
        time.sleep(SLEEP_TIME)  

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
