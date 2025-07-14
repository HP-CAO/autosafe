import numpy as np
import math
ACTION_BOUND = 20.0 #todo double check action bound, to normalize the safe action space to [-1, 1], here we do it in the policy network due to the gravity compensation

F = np.array([[ 0.0248  ,  0.0438 ,   0.0247 ,   0.0437 ,  -0.1761 ,  -0.0279  , -0.3900  ,  0.3906  ,  0.6271   ,-0.0126  ,  0.0126  ,  0.0216] ,
                       [-0.0248  , -0.0438 ,   0.0247 ,   0.0437 ,  -0.1761 ,  -0.0279  , -0.3900  , -0.3906  , -0.6271   ,-0.0126  , -0.0126  , -0.0216] ,
                       [-0.0248  , -0.0438 ,  -0.0247 ,  -0.0437 ,  -0.1761 ,  -0.0279  ,  0.3900  , -0.3906  ,  0.6271   , 0.0126  , -0.0126  ,  0.0216] ,
                       [ 0.0248  ,  0.0438 ,  -0.0247 ,  -0.0437 ,  -0.1761 ,  -0.0279  ,  0.3900  ,  0.3906  , -0.6271   , 0.0126  ,  0.0126  , -0.0216] ])  # feedback law for the safe controller

MATRIX_P = np.array([[  0.1028 ,   0.1091 ,  -0.0000 ,  -0.0000 ,  -0.0000 ,  -0.0000 ,   0.0000  ,  0.4080 ,   0.0000 ,   0.0000 ,   0.0041,   -0.0000],
                     [  0.1091 ,   0.1897 ,  -0.0000 ,  -0.0000 ,  -0.0000 ,   0.0000 ,   0.0000  ,  0.7117 ,   0.0000 ,   0.0000 ,   0.0071,   -0.0000],
                     [ -0.0000 ,  -0.0000 ,   0.1028 ,   0.1091 ,   0.0000 ,   0.0000 ,  -0.4080  , -0.0000 ,   0.0000 ,  -0.0041 ,  -0.0000,   -0.0000],
                     [ -0.0000 ,  -0.0000 ,   0.1091 ,   0.1897 ,   0.0000 ,   0.0000 ,  -0.7117  , -0.0000 ,   0.0000 ,  -0.0071 ,  -0.0000,   -0.0000],
                     [ -0.0000 ,  -0.0000 ,   0.0000 ,   0.0000 ,   0.1874 ,   0.0109 ,  -0.0000  , -0.0000 ,   0.0000 ,   0.0000 ,  -0.0000,   -0.0000],
                     [ -0.0000 ,   0.0000 ,   0.0000 ,   0.0000 ,   0.0109 ,   0.0043 ,  -0.0000  ,  0.0000 ,   0.0000 ,  -0.0000 ,  -0.0000,    0.0000],
                     [  0.0000 ,   0.0000 ,  -0.4080 ,  -0.7117 ,  -0.0000 ,  -0.0000 ,   6.2021  ,  0.0000 ,  -0.0000 ,   0.0626 ,   0.0000,    0.0000],
                     [  0.4080 ,   0.7117 ,  -0.0000 ,  -0.0000 ,  -0.0000 ,   0.0000 ,   0.0000  ,  6.2021 ,   0.0000 ,   0.0000 ,   0.0626,    0.0000],
                     [  0.0000 ,   0.0000 ,   0.0000 ,   0.0000 ,   0.0000 ,   0.0000 ,  -0.0000  ,  0.0000 ,  19.9204 ,  -0.0000 ,   0.0000,    0.2012],
                     [  0.0000 ,   0.0000 ,  -0.0041 ,  -0.0071 ,   0.0000 ,  -0.0000 ,   0.0626  ,  0.0000 ,  -0.0000 ,   0.0012 ,  -0.0000,    0.0000],
                     [  0.0041 ,   0.0071 ,  -0.0000 ,  -0.0000 ,  -0.0000 ,  -0.0000 ,   0.0000  ,  0.0626 ,   0.0000 ,  -0.0000 ,   0.0012,   -0.0000],
                     [ -0.0000 ,  -0.0000 ,  -0.0000 ,  -0.0000 ,  -0.0000 ,   0.0000 ,   0.0000  ,  0.0000 ,   0.2012 ,   0.0000 ,  -0.0000,    0.0040],])

SAFE_CONTROLLER_STEP = 10  # think of distance|safe value based switching
SAFE_CONTROLLER_ACTIVATE_THRESHOLD = 1.0

SYSTEM_A = np.array([[1.0000, 0.0333, 0, 0],
                     [0.6465, 1.5268, 2.1666, 0.4020],
                     [0, 0, 1.0000, 0.0333],
                     [-1.5151, -1.2348, -4.3123, 0.0577]])

SYSTEM_B = np.array([[0, 0.0334, 0, -0.0783]])

def activate_safe_controller_condition(state, activate_threshold=SAFE_CONTROLLER_ACTIVATE_THRESHOLD):
    safe_value = np.array(state).transpose() @ MATRIX_P @ np.array(state)
    if safe_value >= activate_threshold:
        return True
    else:
        return False


def safety_violation_condition(state):
    x = state[0]
    theta = state[2]
    if abs(x) >= 0.9 or abs(theta) >= 0.8:
        return True
    else:
        return False


def get_init_condition_in_safety_envelope(n_points_per_dim=1):
    eigen_values, eigen_vectors = np.linalg.eig(MATRIX_P)  # get eigen value and eigen vector

    Q = eigen_vectors

    initial_condition_list = []

    for i in range(n_points_per_dim):
        # angle_1 = i * math.pi / n_points_per_dim
        angle_1 = np.random.uniform(0, math.pi)
        y0 = math.sqrt(1 / eigen_values[0]) * math.cos(angle_1)
        vector_in_3d = math.sin(angle_1)

        if vector_in_3d == 0:
            y1 = 0
            y2 = 0
            y3 = 0
            s = Q @ np.array([y0, y1, y2, y3]).transpose()
            # print(s.transpose() @ P_matrix_4 @ s)
            initial_condition_list.append([s[0], s[1], s[2], s[3]])
            continue

        for k in range(n_points_per_dim):
            # angle_2 = k * math.pi / n_points_per_dim
            angle_2 = np.random.uniform(0, math.pi)
            y1 = vector_in_3d * math.sqrt(1 / eigen_values[1]) * math.cos(angle_2)
            vector_in_2d = vector_in_3d * math.sin(angle_2)

            if vector_in_2d == 0:
                y2 = 0
                y3 = 0
                s = Q @ np.array([y0, y1, y2, y3]).transpose()
                # print(s.transpose() @ P_matrix_4 @ s)
                initial_condition_list.append([s[0], s[1], s[2], s[3]])
                continue

            for j in range(n_points_per_dim):
                # angle_3 = j * math.pi / n_points_per_dim
                angle_3 = np.random.uniform(0, math.pi)
                y2 = vector_in_2d * math.sqrt(1 / eigen_values[2]) * math.cos(angle_3)
                y3 = vector_in_2d * math.sqrt(1 / eigen_values[3]) * math.sin(angle_3)
                s = Q @ np.array([y0, y1, y2, y3]).transpose()
                # print(s.transpose() @ MATRIX_P @ s)
                initial_condition_list.append([s[0], s[1], s[2], s[3]])

    # print(f"Generating {len(initial_condition_list)} conditions for training ...")
    return initial_condition_list


class ModelbasedAgent:
    def __init__(self, feedback_law=F):
        self.feedback_law = feedback_law
        self.gravity_compensation = 0.25 * 0.027 * 9.8 # for each propeller

    def get_action(self, tracking_error):
        # tracking_error env state - env control goal
        action_abs = self.feedback_law @ (tracking_error) + self.gravity_compensation
        return  action_abs



