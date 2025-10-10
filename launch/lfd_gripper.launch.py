from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare launch arguments for SSID and password
    ssid_arg = DeclareLaunchArgument(
        'ssid',
        default_value='alejos',
        description='Wi-Fi SSID for the hotspot'
    )
    password_arg = DeclareLaunchArgument(
        'password',
        default_value='harvesting',
        description='Wi-Fi password for the hotspot'
    )

    # Use LaunchConfiguration to get the values
    ssid = LaunchConfiguration('ssid')
    password = LaunchConfiguration('password')

    return LaunchDescription([
        ssid_arg,
        password_arg,

        # Start Wi-Fi hotspot using parameters
        ExecuteProcess(
            cmd=['sudo', 'nmcli', 'device', 'wifi', 'hotspot',
                 'ifname', 'wlp108s0f0',
                 'ssid', ssid,
                 'password', password],
            output='screen',
        ),

        # Start micro-ROS agent after a 5-second delay
        TimerAction(
            period=5.0,
            actions=[
                ExecuteProcess(
                    cmd=['ros2', 'run', 'micro_ros_agent', 'micro_ros_agent', 'udp4', '--port', '8888'],
                    output='screen',
                )
            ]
        )
    ])
