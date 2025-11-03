# LFD Apples - Gripper Control

## Things You'll Need

- **Robot:** Franka Research
- **Gripper:** micro-ROS enabled

---

## GRIPPER

### lfd_gripper.launch.py 
This launch file performs the following actions: 

1) Turns your laptop into a **Wi-Fi hotspot**.
2) Runs **microROS agent** to handle the communication with the ESP32 on the gripper's side.
3) Runs a ROS2 node to control the air valve, and fingers. It reduces the demonstrator's overhead by automatically switching the air ON/OFF, and CLOSE/OPEN the fingers.
4) Runs a node to publish the **In-Hand camera** image.
5) Runs a node to publish the **Fixed camera** image


```bash
ros2 launch lfd_apples lfd_gripper.launch.py ssid:=my_hotspot password:=mypassword palm_camera_device_num:=4 fixed_camera_device_num:=5

```


Notes: 
* Replace `wlp108s0f0` in the launch file with your wireless interface name. You can check this with `nmcli device status` or `ip link`.
* Use the **ssid / password** that were previously uploaded to the ESP32 board. By default these values are `alejos` / `harvesting`.
* Check the camera device number with `v4l2-ctl --list-devices`.

## ARM

### 1. Recording Demonstrations
This step is meant for recording the trajectory of the arm as the human drives it. It requires the following nodes to be running:


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


## TROUBLESHOOTING

### Franka Arm
Failed to lock the realtime publisher issue  
[solution 1](https://github.com/frankarobotics/franka_ros2/issues/105): edit *franka_robot_broadcaster.hpp* by adding the following right before including the *realtime_publisher* (near line 24):
```c
#define NON_POLLING TRUE
```   
