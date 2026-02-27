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
    palm_camera_device_arg = DeclareLaunchArgument(
        'palm_camera_device_num',
        default_value='2',
        description='Camera device number for the palm camera'
    )

    # --- Declare camera device number arg ---
    fixed_camera_device_arg = DeclareLaunchArgument(
        'fixed_camera_device_num',
        default_value='4',
        description='Camera device number for the fixed camera'
    )




    # Use LaunchConfiguration to get the values
    ssid = LaunchConfiguration('ssid')
    password = LaunchConfiguration('password')
    palm_camera_device_num = LaunchConfiguration('palm_camera_device_num')
    fixed_camera_device_num = LaunchConfiguration('fixed_camera_device_num')


    return LaunchDescription([
        ssid_arg,
        password_arg,
        palm_camera_device_arg,
        fixed_camera_device_arg,



        # Start Wi-Fi hotspot using parameters
        ExecuteProcess(
            cmd=['nmcli', 'device', 'wifi', 'hotspot',
                 'ifname', 'wlp108s0f0',
                 'ssid', ssid,
                 'password', password],
            output='screen',
        ),
      
        ExecuteProcess(
                    cmd=['ros2', 'run', 'micro_ros_agent', 'micro_ros_agent', 'udp4', '--port', '8888'],
                    output='screen',
                ),



        # # Start micro-ROS agent after a 5-second delay
        # TimerAction(
        #     period=1.0,
        #     actions=[
        #         ExecuteProcess(
        #             cmd=['ros2', 'run', 'micro_ros_agent', 'micro_ros_agent', 'serial', '--dev', '/dev/ttyUSB0', '--baudrate', '921600'],
        #             output='screen',
        #         )
        #     ]
        # ),

    
        Node(
                package='lfd_apples',
                executable='lfd_automatic_gripper',
                name='automatic_gripper',
                output='screen',
            ),


        # In-Hand Camera
        Node(
                package='lfd_apples',              # or your actual package name
                executable='lfd_inhand_camera',     # make sure this matches your entry point
                name='gripper_palm_camera_publisher',
                output='screen',
                parameters=[
                    {'camera_device_num': palm_camera_device_num},
                    {'topic_name': 'gripper/rgb_palm_camera/image_raw'},
                    {'frame_id': 'gripper_palm_camera_optical_link'}                        
                    ],
            ),
     
        # Fixed Camera
        Node(
                package='lfd_apples',              # or your actual package name
                executable='lfd_inhand_camera',      # make sure this matches your entry point
                name='fixed_camera_publisher',
                output='screen',
                parameters=[
                    {'camera_device_num': fixed_camera_device_num},
                    {'topic_name': 'fixed/rgb_camera/image_raw'},
                    {'frame_id': 'fixed_camera_optical_link'}                        
                    ],
            ),


        # Yolo Node
        Node(
                package='lfd_apples',
                executable='yolo_latent_node',
                name='yolo_node',
                # output='screen'
        ),        
            
    ])
