# LFD Apples - Gripper Control

## Things You'll Need

- **Robot:** Franka Research
- **Gripper:** micro-ROS enabled

---

## Quick Setup

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