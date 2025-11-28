import numpy as np
import math
import cvxpy as cp
ACTION_BOUND = 1.0  #todo double check action bound, to normalize the safe action space to [-1, 1], here we do it in the policy network due to the gravity compensation
# GRAVITY_COMPENSATION = (0.25 * 0.027 * 9.8) / ACTION_BOUND
F = np.array([[ 0.0248  ,  0.0438 ,   0.0247 ,   0.0437 ,  -0.1761 ,  -0.0279  , -0.3900  ,  0.3906  ,  0.6271   ,-0.0126  ,  0.0126  ,  0.0216] ,
                       [-0.0248  , -0.0438 ,   0.0247 ,   0.0437 ,  -0.1761 ,  -0.0279  , -0.3900  , -0.3906  , -0.6271   ,-0.0126  , -0.0126  , -0.0216] ,
                       [-0.0248  , -0.0438 ,  -0.0247 ,  -0.0437 ,  -0.1761 ,  -0.0279  ,  0.3900  , -0.3906  ,  0.6271   , 0.0126  , -0.0126  ,  0.0216] ,
                       [ 0.0248  ,  0.0438 ,  -0.0247 ,  -0.0437 ,  -0.1761 ,  -0.0279  ,  0.3900  ,  0.3906  , -0.6271   , 0.0126  ,  0.0126  , -0.0216] ])/ACTION_BOUND  # feedback law for the safe controller

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

SYSTEM_A = np.array([
    [1., 0.02, 0., 0., 0, 0., 0., 0., 0., 0., 0., 0. ],
    [0., 1., 0., 0., 0., 0., 0., 0.196, 0., 0.,0., 0.] ,
    [0., 0., 1., 0.02, 0., 0., 0., 0., 0., 0., 0., 0. ],
    [0., 0., 0., 1., 0., 0., -0.196, 0., 0., 0.,0., 0.],
    [0., 0., 0., 0., 1., 0.02, 0., 0., 0., 0., 0., 0. ],
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0 ],
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.02,0., 0. ],
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.02, 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.02],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,0., 0. ],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,  0., 1 ]])

SYSTEM_B = np.array([[0.         , 0.        ,  0.        ,  0.      ],
    [ 0.         , 0.        ,  0.        ,  0.      ],
     [0.         , 0.        ,  0.        ,  0.      ],
    [ 0.         , 0.        ,  0.        ,  0.      ],
     [0.         , 0.        ,  0.        ,  0.      ],
     [0.74074074  , 0.74074074 ,  0.74074074 ,  0.74074074],
     [0.         , 0.        ,  0.        ,  0.      ],
     [0.         , 0.        ,  0.        ,  0.      ],
     [0.         , 0.        ,  0.        ,  0.      ],
     [40.10305602  ,40.10305602 ,-40.10305602 ,-40.10305602],
     [-40.10305602  ,40.10305602 ,40.10305602 ,-40.10305602],
     [-23.15814035  ,23.15814035 ,-23.15814035 , 23.15814035]
])

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
        self.matrix_P = MATRIX_P

    def get_action(self, tracking_error):
        # tracking_error env state - env control goal
        action_abs = self.feedback_law @ (tracking_error)
        action_abs = np.clip(action_abs, -1, 1)
        return  action_abs

    def safety_switch_on(self, tracking_error):
        safety_value = tracking_error @ MATRIX_P @ tracking_error.T
        return safety_value > 1



def cbf_filter_dt(x, u_nom, A=SYSTEM_A, B=SYSTEM_B, P=MATRIX_P,
                             u_min=-1.0, u_max=1.0,
                             V_max=1.0):
    """
    Lyapunov-based CBF filter for discrete-time linear system:
        x_{k+1} = A x_k + B u_k

    Safety constraint:
        V_next = x_{k+1}^T P x_{k+1} <= V_max
    where typically V_max = 1.0 if you've scaled P accordingly.

    Returns:
        u_safe: filtered control
        certified: True if u_nom already satisfied Lyapunov constraint
                   (QP returned u_nom), False otherwise
        V_next_nom: V(x_{k+1}) under nominal control
        V_next_safe: V(x_{k+1}) under filtered control
    """
    x = np.asarray(x).reshape(-1, 1)
    A = np.asarray(A)
    B = np.asarray(B).reshape(-1, 4)   # ensure (n,1)
    P = np.asarray(P)
    u_nom = u_nom * ACTION_BOUND
    u_nom = np.array(u_nom).reshape(4, 1)
    n = A.shape[0]
    assert A.shape == (n, n)
    assert B.shape == (n, 4)
    assert P.shape == (n, n)

    # Nominal next state and Lyapunov value (for diagnostics)
    # x_next_nom = A @ x + B @ u_nom
    # V_now = float(x.T @ P @ x)
    # V_next_nom = float(x_next_nom.T @ P @ x_next_nom)

    # Optimization variable
    u = cp.Variable((4,1))

    # Next state in terms of u
    x_next = A @ x + B @ u

    # Objective: stay close to u_nom
    objective = cp.Minimize(cp.sum_squares(u - u_nom))

    # Lyapunov safety constraint: V_next <= V_max
    lyap_constraint = cp.quad_form(x_next, P) <= V_max

    # Input bounds
    constraints = [lyap_constraint, u >= u_min, u <= u_max]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, warm_start=True, verbose=False)

    if u.value is None:
        # Infeasible or solver failed: fallback to clipped nominal
        u_safe = np.clip(u_nom, u_min, u_max).reshape(-1)
        certified = False
        # x_next_safe = A @ x + B.flatten() * u_safe
        # V_next_safe = float(x_next_safe.T @ P @ x_next_safe)
    else:
        u_safe =u.value
        # x_next_safe = A @ x + B.flatten() * u_safe
        # V_next_safe = float(x_next_safe.T @ P @ x_next_safe)
        u_safe = u_safe.reshape(-1)
        certified = abs(u_safe - u_nom) < 1e-6

    return u_safe / ACTION_BOUND, certified
