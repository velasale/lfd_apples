from launch_ros.actions import Node
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

    # --- Declare camera device number arg ---
    camera_device_arg = DeclareLaunchArgument(
        'palm_camera_device_num',
        default_value='2',
        description='Camera device number for the palm camera'
    )

    # Use LaunchConfiguration to get the values
    ssid = LaunchConfiguration('ssid')
    password = LaunchConfiguration('password')
    camera_device_num = LaunchConfiguration('palm_camera_device_num')


    return LaunchDescription([
        ssid_arg,
        password_arg,
        camera_device_arg,

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
        ),

        TimerAction(
            period=8.0,
            actions=[
                Node(
                    package='lfd_apples',
                    executable='lfd_automatic_gripper',
                    name='automatic_gripper',
                    output='screen',
                )
            ]
        ),

        # --- Palm camera node after 10s ---
        TimerAction(
            period=10.0,
            actions=[
                Node(
                    package='lfd_apples',              # or your actual package name
                    executable='lfd_inhand_camera',     # make sure this matches your entry point
                    name='gripper_palm_camera_publisher',
                    output='screen',
                    parameters=[{'palm_camera_device_num': camera_device_num}],
                )
            ]
        )
        
    ])
