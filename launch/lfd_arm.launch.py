import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)

    try:
        with open(absolute_file_path, "r") as file:
            return yaml.safe_load(file)
    except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
        return None



def generate_launch_description():

    # ====== MOVEIT 2  SERVO NODE ========
    # Get parameters for the servo node
    servo_yaml = load_yaml("lfd_apples", "config/lfd_servo.yaml")
    servo_params = {"moveit_servo": servo_yaml}

    


    servo_node = Node(
        package="moveit_servo",
        executable="servo_node_main",
        name="servo_node",
        output="screen",
        parameters=[
            servo_params,
        ],
    )

    return LaunchDescription([servo_node])
