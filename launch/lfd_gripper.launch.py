from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction

def generate_launch_description():
    return LaunchDescription([
        # Step 1: Start Wi-Fi hotspot
        ExecuteProcess(
            cmd=['sudo', 'nmcli', 'device', 'wifi', 'hotspot', 'ifname', 'wlp108s0f0', 'ssid', 'alejos', 'password', 'harvesting'],
            output='screen',
        ),

        # Step 2: Start micro-ROS agent after a 5-second delay
        TimerAction(
            period=5.0,  # seconds to wait
            actions=[
                ExecuteProcess(
                    cmd=['ros2', 'run', 'micro_ros_agent', 'micro_ros_agent', 'udp4', '--port', '8888'],
                    output='screen',
                )
            ]
        )
    ])
