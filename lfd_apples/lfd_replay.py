#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import pandas as pd
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from lfd_apples.ros2bag2csv import parse_array
import time
import numpy as np
# from moveit.planning import MoveItPy

print(np.pi)



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

# class MoveToStart(Node):
#     def __init__(self):
#         super().__init__('move_to_start')
#         self.publisher = self.create_publisher(JointTrajectory, '/fr3_arm_controller/joint_trajectory', 10)

#     def go_to_position(self):

#         group = MoveGroupCommander("panda_arm")
#         group.set_joint_value_target([
#                                         0.0,
#                                         -np.pi/4,
#                                         0.0,
#                                         -3.0*np.pi/4,
#                                         0.0,
#                                         np.pi/2,
#                                         np.pi/4
#                                     ])


#         # Optional: control speed/acceleration scaling
#         group.set_max_velocity_scaling_factor(0.2)  # 20% speed
#         group.set_max_acceleration_scaling_factor(0.2)

#         # Plan and execute
#         plan = group.plan()
#         group.execute(plan[1])  # Execute planned trajectory
#         self.get_logger().info("Sent robot to start position!")


# def move_robot_to_start():
#     rclpy.init()
#     node = MoveToStart()

#     # Give publisher time to connect
#     time.sleep(1.0)

#     node.go_to_position()

#     # Allow time for execution
#     time.sleep(6.0)

#     node.destroy_node()
#     rclpy.shutdown()


def main(args=None):
    rclpy.init()
    # node = MoveToStart()

    # Give publisher time to connect
    time.sleep(1.0)
    
    # lfd_demo_csv =  '/home/alejo/franka_bags/franka_joint_bag_hdemo_2/bag_csvs/joint_states.csv'
    lfd_demo_csv = '/home/alejo/lfd_bags/experiment_1/trial_8/lfd_bag_main/bag_csvs/joint_states.csv'

    node = FrankaReplay(lfd_demo_csv)  # <-- set your CSV path
    rclpy.spin_once(node, timeout_sec=2.0)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
