#!/usr/bin/env python3

from pylibfranka import Robot

def main():
    robot = Robot("192.168.1.11")   # <-- update IP if needed
    print("Running automatic error recovery...")
    robot.automatic_error_recovery()
    print("Robot is back in normal operation mode.")

if __name__ == "__main__":
    main()
