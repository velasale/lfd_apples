#!/usr/bin/env python3

# Copyright (c) 2025 Franka Robotics GmbH
# Use of this source code is governed by the Apache-2.0 license, see LICENSE

import argparse
import time

import numpy as np

from pylibfranka import ControllerMode, JointPositions, Robot


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="localhost", help="Robot IP address")
    args = parser.parse_args()

    # Connect to robot
    robot = Robot(args.ip)

    try:       
        print("Starting freedrive... Move the arm gently.")
        # Run torque control loop

         # Start torque control
        active_control = robot.start_torque_control()

        while True:
            # Read robot state and duration
            robot_state, duration = active_control.readOnce()

            # Here you can process the robot_state if needed
            # For freedrive, we typically do not apply any torques

            # Sleep for the duration to maintain control loop timing
            time.sleep(duration.to_sec())   

    except KeyboardInterrupt:
        print("Freedrive interrupted by user.")
    except Exception as e:
        print(f"Error occurred: {e}")
        if robot is not None:
            robot.stop()
        return -1
    finally:
        print("Exiting freedrive mode.")

if __name__ == "__main__":
    main()