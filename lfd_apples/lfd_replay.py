#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import pandas as pd
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from lfd_apples.ros2bag2csv import parse_array
import numpy as np

from lfd_apples.lfd_moveit_test import MoveToHomeAndFreedrive


class FrankaReplay(Node):
    def __init__(self, csv_path, downsample_factor=20, speed_factor=0.5):
        super().__init__('franka_replay_action')

        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/fr3_arm_controller/follow_joint_trajectory'
        )

        self.get_logger().info(f"Loading trajectory from {csv_path}")
        df = pd.read_csv(csv_path)

        # Convert '_position' column and downsample
        df_joints = pd.DataFrame(
            df["_position"].apply(parse_array).apply(lambda x: x[:7]).tolist(),
            columns=[f"fr3_joint{i+1}" for i in range(7)]
        ).iloc[::downsample_factor].reset_index(drop=True)

        if len(df_joints) < 2:
            raise ValueError("Trajectory must have at least 2 points")

        self.joint_names = [f"fr3_joint{i+1}" for i in range(7)]
        self.trajectory_points = []

        # Original data recorded at 1 kHz → dt = 0.001 s
        # Adjusted for downsampling and speed scaling
        dt = 1.0 * downsample_factor / speed_factor
        time_from_start = 0.0

        for _, row in df_joints.iterrows():
            point = JointTrajectoryPoint()
            point.positions = row.tolist()
            time_from_start += dt
            point.time_from_start.sec = int(time_from_start)
            point.time_from_start.nanosec = int((time_from_start - int(time_from_start)) * 1e9)
            self.trajectory_points.append(point)

        self.get_logger().info(f"Prepared {len(self.trajectory_points)} trajectory points")
        self.send_transition_to_start(df_joints.iloc[0].tolist())

    # === Smooth transition from home to first CSV point ===
    def send_transition_to_start(self, start_pose):
        self.get_logger().info("Preparing smooth move from home to trajectory start...")
        self._action_client.wait_for_server()

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names

        # Get current joint positions from the home pose (you could also pass these in)
        # Here we use the known home position used earlier
        home_positions = [0.0, -1.57, 0.0, -2.71, 0.0, 2.6, 0.69]

        start_point = JointTrajectoryPoint()
        start_point.positions = start_pose
        start_point.time_from_start.sec = 5  # 5 seconds for smooth transition

        goal_msg.trajectory.points.append(
            JointTrajectoryPoint(positions=home_positions, time_from_start=rclpy.duration.Duration(seconds=0).to_msg())
        )
        goal_msg.trajectory.points.append(start_point)

        self.get_logger().info("Sending smooth transition to trajectory start...")
        future = self._action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Transition goal rejected!")
            rclpy.shutdown()
            return

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info("Transition complete, starting main trajectory...")

        # After reaching the start point, send the full trajectory
        self.send_trajectory()

    def send_trajectory(self):
        self.get_logger().info("Waiting for action server for trajectory...")
        self._action_client.wait_for_server()

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names
        goal_msg.trajectory.points = self.trajectory_points

        self.get_logger().info("Sending trajectory via FollowJointTrajectory action...")
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected!")
            rclpy.shutdown()
            return
        self.get_logger().info("Trajectory goal accepted, waiting for result...")
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f"Trajectory execution finished: {result}")
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    # 1️⃣ Move to home safely
    home_node = MoveToHomeAndFreedrive()
    if home_node.move_to_home():
        home_node.destroy_node()
    else:
        home_node.destroy_node()
        rclpy.shutdown()
        return

    # 2️⃣ Replay CSV trajectory with smooth transition
    lfd_demo_csv = '/home/alejo/lfd_bags/experiment_1/trial_1/lfd_bag_main/bag_csvs/joint_states.csv'
    replay_node = FrankaReplay(
        csv_path=lfd_demo_csv,
        downsample_factor=20,  # 1000 Hz → 50 Hz
        speed_factor=0.2       # 20% of real-time speed
    )
    rclpy.spin(replay_node)
    replay_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
