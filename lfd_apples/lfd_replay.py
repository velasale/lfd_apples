#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import pandas as pd
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from lfd_apples.ros2bag2csv import parse_array

class FrankaReplay(Node):
    def __init__(self, csv_path):
        super().__init__('franka_replay')
        self.publisher = self.create_publisher(
            JointTrajectory,
            '/fr3_arm_controller/joint_trajectory',
            10
        )

        # Load CSV with columns: time, joint1,...,joint7
        self.get_logger().info(f"Loading trajectory from {csv_path}")
        df = pd.read_csv(csv_path)

        # Ensure correct joint names for Franka
        joint_names = [
            "fr3_joint1", "fr3_joint2", "fr3_joint3",
            "fr3_joint4", "fr3_joint5", "fr3_joint6",
            "fr3_joint7"
        ]

        traj_msg = JointTrajectory()
        traj_msg.joint_names = joint_names

        
        # Convert '_position' column (Series of lists) into a DataFrame
        df_joints = pd.DataFrame(df["_position"].apply(parse_array).tolist(),
                         columns=[
                             "fr3_joint1","fr3_joint2","fr3_joint3",
                             "fr3_joint4","fr3_joint5","fr3_joint6","fr3_joint7"
                         ])

        # print(df_joints)

        # Build trajectory points
        time = 0.0
        dt = 0.05  # 50 ms per step → 20 Hz playback (tune as needed)
        for _, row in df_joints.iterrows():
            point = JointTrajectoryPoint()

            point.positions = [
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                row[5],
                row[6]                
            ]
            
            time += dt
            point.time_from_start.sec = int(time)
            point.time_from_start.nanosec = int((time % 1) * 1e9)
            traj_msg.points.append(point)

        self.get_logger().info("Publishing trajectory...")
        self.publisher.publish(traj_msg)
        self.get_logger().info("Trajectory sent ✅")


def main(args=None):
    rclpy.init(args=args)

    lfd_demo_csv =  '/home/alejo/franka_bags/franka_joint_bag_1/bag_exports/NS_1_joint_states.csv'

    node = FrankaReplay(lfd_demo_csv)  # <-- set your CSV path
    rclpy.spin_once(node, timeout_sec=2.0)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
