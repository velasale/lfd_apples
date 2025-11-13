#!/usr/bin/env python3
import argparse
import time
import numpy as np
from pylibfranka import ControllerMode, JointPositions, Robot

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="192.168.1.11")
    args = parser.parse_args()

    print(f"Connecting to robot at {args.ip}...")
    robot = Robot(args.ip)
    print("Connection successful.")

    try:
        print("Running automatic error recovery...")
        robot.automatic_error_recovery()
        print("Error recovery completed.")

        print("Starting gentle safe joint motion...")

        control = robot.start_joint_position_control(ControllerMode.CartesianImpedance)

        robot_state, duration = control.readOnce()
        initial_q = robot_state.q_d if hasattr(robot_state, "q_d") else robot_state.q

        time_elapsed = 0.0
        motion_finished = False

        while not motion_finished:
            robot_state, duration = control.readOnce()
            time_elapsed += duration.to_sec()

            # update function creates a smooth S-curve trajectory:
            delta_angle = np.pi/16 * (1 - np.cos((np.pi/3.0) * time_elapsed))

            q_des = [
                initial_q[0],
                initial_q[1],
                initial_q[2],
                initial_q[3] + delta_angle * 0.3,   # smaller scaling
                initial_q[4] + delta_angle * 0.3,
                initial_q[5],
                initial_q[6] + delta_angle * 0.3,
            ]

            jp = JointPositions(q_des)

            if time_elapsed >= 3.0:
                jp.motion_finished = True
                motion_finished = True

            control.writeOnce(jp)

        print("Motion OK, recovery verified.")
        robot.stop()

    except Exception as e:
        print(f"Recovery failed: {e}")
        try: robot.stop()
        except: pass

if __name__ == "__main__":
    main()
