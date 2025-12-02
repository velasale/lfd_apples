# Gripper aliases
alias fingersIn='ros2 service call /microROS/move_stepper std_srvs/srv/SetBool "{data: false}"'
alias fingersOut='ros2 service call /microROS/move_stepper std_srvs/srv/SetBool "{data: true}"'
alias suctionOn='ros2 service call /microROS/toggle_valve std_srvs/srv/SetBool "{data: true}"'
alias suctionOff='ros2 service call /microROS/toggle_valve std_srvs/srv/SetBool "{data: false}"'

# Imitation Learning aliases
# taskset used to assign CPU cores
alias arm='taskset -c 1,2,3,4 ros2 launch franka_fr3_moveit_config moveit.launch.py robot_ip:=192.168.1.11'
alias human_demo='taskset -c 5,6,7,8 ros2 run lfd_apples lfd_data_collection'
alias gripper='ros2 launch lfd_apples lfd_gripper.launch.py palm_camera_device_num:=6 fixed_camera_device_num:=4'
alias sensors='ros2 topic echo /microROS/sensor_data'
alias lfd_layout='terminator -l lfd_trials'
alias reset_arm='ros2 run lfd_apples franka_recover'
