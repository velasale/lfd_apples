import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution
)
from launch_ros.parameter_descriptions import ParameterValue


def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)

    try:
        with open(absolute_file_path, "r") as file:
            return yaml.safe_load(file)
    except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
        return None



def generate_launch_description():
      
    # Configurable parameters
    robot_ip = '192.168.1.11'       #LaunchConfiguration('robot_ip')
    use_fake_hardware = 'false'     #LaunchConfiguration('use_fake_hardware')
    fake_sensor_commands = 'false'  #LaunchConfiguration('fake_sensor_commands')
    namespace = ''                  #LaunchConfiguration('namespace')
    load_gripper = 'true'           #LaunchConfiguration('load_gripper')
    ee_id = 'franka_hand'           #LaunchConfiguration('ee_id')

    # Planning_context
    franka_xacro_file = os.path.join(
        get_package_share_directory('franka_description'),
        'robots', 'fr3', 'fr3.urdf.xacro'
    )

    # ====== MOVEIT 2  SERVO NODE ========

    # Get parameters for the servo node
    servo_yaml = load_yaml("lfd_apples", "config/lfd_servo.yaml")
    servo_params = {"moveit_servo": servo_yaml}
       

    servo_node = Node(
        package="moveit_servo",
        executable="servo_node_main",
        name="servo_node",
        namespace=namespace,
        output="screen",
        parameters=[
            robot_description,
            servo_params,
        ],
    )

    return LaunchDescription([servo_node])
