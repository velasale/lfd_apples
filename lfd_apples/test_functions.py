import numpy as np
from .ros2bag2csv import fr3_fk   # import function to test

def test_fk_known_pose():
    # Known joint angles and expected end-effector position for franka arm
    # eef in this case is the suctio gripper, and the tp is located at 22.7cm from the flange
    
    joint_angles = np.array([
        -0.110428,
        -0.137232,
         0.105849,
        -2.850474,
        -0.007657,
         2.694830,
         0.778153
    ])

    expected_position = np.array([0.3546, -0.0071, 0.0065])
    T = fr3_fk(joint_angles)
    position = T[:3, 3]

    np.testing.assert_allclose(position, expected_position, rtol=1e-3, atol=1e-3)


# Note: To run the test simply type:
# pytest-3 -v src/franka_tasks/franka_tasks/test_functions.py
