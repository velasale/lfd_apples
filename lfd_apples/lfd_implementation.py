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
from lfd_apples.ros2bag2csv import extract_data_and_plot, parse_array

from std_msgs.msg import Int16MultiArray
from std_srvs.srv import SetBool
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from lfd_apples.lfd_vision import extract_pooled_latent_vector
from ultralytics import YOLO
import cv2


class MoveToHomeAndFreedrive(Node):

    def __init__(self):
        super().__init__('move_to_home_and_freedrive')

        # Subscribers
        self.distance_sub = self.create_subscription(
            Int16MultiArray, 'microROS/sensor_data', self.gripper_sensors_callback, 10)
        self.eef_pose_sub = self.create_subscription(
            PoseStamped, '/franka_robot_state_broadcaster/current_pose', self.eef_pose_callback, 10)
        self.palm_camera_sub = self.create_subscription(
            Image, 'gripper/rgb_palm_camera/image_raw', self.palm_camera_callback, 10)        

        
        # Load learned model
        self.model = 0
        self.model_mean = 0
        self.model_std = 0

        # Action client for joint trajectory
        self.action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/fr3_arm_controller/follow_joint_trajectory'
        )

        # Controller names
        self.arm_controller = 'fr3_arm_controller'
        self.gravity_controller = 'gravity_compensation_example_controller'

        # Switch controller client
        self.switch_client = self.create_client(SwitchController, '/controller_manager/switch_controller')
        self.switch_client.wait_for_service()

        # Load controller client
        self.load_client = self.create_client(LoadController, '/controller_manager/load_controller')
        self.load_client.wait_for_service()

        self.joint_names = [f"fr3_joint{i+1}" for i in range(7)]
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

        # Flags
        self.approach_accomplished = False
        self.contact_accomplished = False
        self.pick_accomplished = False
        self.data_ready = False

        # Image
        self.bridge = CvBridge()
        self.raw_image = None
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pt_path = os.path.join(script_dir, "resources", "best_segmentation.pt")
        self.yolo_model = YOLO(pt_path)
        self.yolo_latent_layer = 12

        # State Space Vectors
        self.t_2_data = []
        self.t_1_data = []
        self.t_data = []
     
                
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

    def run_lfd_model(self, n_timesteps=2):       
               
        if not self.pick_accomplished:

            # --- Step 1: Organize data ---
            # Phase_1 approach data stream
            
            self.t_2_data = self.t_1_data            
            self.t_1_data = self.t_data            
            self.t_data = [self.tof, self.latent_image]

            # filter new_data
            
            # Combine data, from past steps            
            self.X = self.t_2_data + self.t_1_data + self.t_data
            # Normalize data


            # --- Step 2: Feed model with data ---
            # Y_pred = self.model.predict(X_test_norm)

            # Send actions to the arm            

            # Loop

            return False
        
        elif self.pick_accomplished:
            return True


    # === Sensor Topics Callback ====
    def gripper_sensors_callback(self, msg: Int16MultiArray):

        # get current msg
        self.scA = msg.data[0]
        self.scB = msg.data[1]
        self.scC = msg.data[2]
        self.tof = msg.data[3]

        # Update sensors average since last action
        # to be done


    def self_camera_callback(self, msg: Image):

        self.raw_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Get latent space features
        pooled_vector, feat_map = extract_pooled_latent_vector(
            self.raw_image,
            self.yolo_model,
            layer_index=self.yolo_latent_layer
        )

        self.latent_image = pooled_vector.tolist()


    def eef_pose_callback(self, msg: PoseStamped):

        # get pose
        eef_pos_x = msg.pose.position.x
        eef_pos_y = msg.pose.position.y
        eef_pos_z = msg.pose.position.z
        eef_ori_x = msg.pose.orientation.x
        eef_ori_y = msg.pose.orientation.y
        eef_ori_z = msg.pose.orientation.z
        eef_ori_w = msg.pose.orientation.w

        # Update sensors average since last action
    

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

    node = MoveToHomeAndFreedrive()    

    BAG_MAIN_DIR = "/media/alejo/New Volume/06_IL_implementation/bagfiles"
    EXPERIMENT = "experiment_1_(pull)"
    BAG_FILEPATH = os.path.join(BAG_MAIN_DIR, EXPERIMENT)
    os.makedirs(BAG_FILEPATH, exist_ok=True)
    
    batch_size = 10
    node.get_logger().info(f"Starting lfd implementation with robot session of {batch_size} demos.")

    for demo in range(batch_size):

        node.get_logger().info("\033[1;32m ---------- Press Enter to start demonstration {}/10 ----------\033[0m".format(demo+1))
        input()  # Wait for user to press Enter
      
        # ------------ Step 0: Initial configuration ----------------
        TRIAL = find_next_trial_number(BAG_FILEPATH, prefix="trial_")
        os.makedirs(os.path.join(BAG_FILEPATH, TRIAL), exist_ok=True)
        # HUMAN_BAG_FILEPATH = os.path.join(BAG_FILEPATH, TRIAL, 'human')
        ROBOT_BAG_FILEPATH = os.path.join(BAG_FILEPATH, TRIAL)               
        
        # ------------ Step 1: LFD implementation with Robot --------
        node.get_logger().info("Moving to home position...")
        while not node.move_to_home():
            pass

        # Start recording
        input("\n\033[1;32m3 - Place apple on the proxy. Press ENTER when done.\033[0m\n")
        robot_rosbag_list = start_recording_bagfile(ROBOT_BAG_FILEPATH)

        # lfd robot implementation        
        input("\n\033[1;32m4 - Press Enter to start ROBOT demonstration.\033[0m\n")        
        while not node.run_lfd_model():
            pass

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
