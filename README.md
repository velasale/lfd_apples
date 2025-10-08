# LFD Apples - Gripper Control

## Things You'll Need

- **Robot:** Franka Research
- **Gripper:** micro-ROS enabled

---

## GRIPPER Setup

### 1. Create a Wi-Fi Hotspot
Turn your laptop into a Wi-Fi hotspot so it can communicate with the gripper. Replace `wlp108s0f0` with your wireless interface name (check with `nmcli device status` or `ip link`).

```bash
sudo nmcli device wifi hotspot ifname wlp108s0f0 ssid alejos password harvesting
```

### 2. Run microROS agent

Start the micro-ROS agent on the same machine that runs the hotspot:

```bash
ros2 run micro_ros_agent micro_ros_agent udp4 --port 8888
```

## ARM setup

### 1. Recording Demonstrations
This step is meant for recording the trajectory of the arm as the human drives it. It requires the following nodes to be running:

#### lfd_trial.py node
This node controls the air valve, and fingers. It is meant to ease the demonstrator's overhead by automatically switching the air ON/OFF, and CLOSE/OPEN the fingers. 
- The air is switched ON when the distance signal from the Time-Of-Flight sensor is less than 50mm (from the target). 
- Fingers are deployed when the air-pressure readings from each suction cup go below the engagement threshold (~600hPa).

```bash
ros2 run lfd_apples lfd_trial
```

#### listen_franka.py node
This node subscribes to the arm's and gripper's ros2 topics and saves them in a bagfile

```bash
ros2 run lfd_apples listen_franka
```

#### Free drive node
I use this example to simply free drive the arm while doing the demo.

```bash
ros2 launch franka_bringup example.launch.py controller_name:=move_to_start_example_controller
```



### 2. Replay Demonstrations
This step is to make the robot replay the demonstration without human intervention. Hence, wrench topic is cleaner and free from human introduced forces while driving the arm.
All topics are saved in a new bagfile, which is now a 'pristine' demonstration.


#### Move it node
This node is to use move it as the robot controller
```bash
ros2 launch franka_fr3_moveit_config moveit.launch.py robot_ip:=192.168.1.11
```

#### lfd_replay.py node
This node is reads the joint positions from the human demonstration, sends it to the arm controller.

```bash
ros2 run lfd_apples lfd_replay
```

#### listen_franka.py node
This node subscribes to the arm's and gripper's ros2 topics and saves them in a bagfile

```bash
ros2 run lfd_apples listen_franka
```