import numpy as np
import math
import cvxpy as cp

ACTION_BOUND = 1.0 #todo double check action bound, to normalize the safe action space to [-1, 1]

F = np.array([[0.1531, -708.6539,  -0.0965]]) / ACTION_BOUND # feedback law for the safe controller

MATRIX_P = np.array([[0.0, -0.0024, -0.000],
                     [-0.0024, 7.7754, 0.0010],
                     [-0.0000, 0.0010, 0.000]]) * 1.0e+05

SAFE_CONTROLLER_STEP = 10  # think of distance|safe value based switching
SAFE_CONTROLLER_ACTIVATE_THRESHOLD = 1.0

SYSTEM_A = np.array([[0.9983, -22.9537, -0.0001],
                     [0., 0.9976, 0.0],
                     [0., 0, 0.9672]])

SYSTEM_B = np.array([[0, 0., 0.1639]])

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
    def __init__(self, feedback_law=F, action_bound=ACTION_BOUND, delta_min=0.0):
        self.feedback_law = feedback_law
        self.matrix_P = MATRIX_P
        self.delta_min = delta_min

    def get_action(self, tracking_error):
        # x, x_dot, theta, theta_dot = tracking_error
        # F = np.squeeze(self.feedback_law)
        # action_abs_1 = F[0] * x + F[1] * x_dot + F[2] * theta + F[3] * theta_dot
        action_abs = np.dot(self.feedback_law, tracking_error)
        # Normalize to [-1, 1] # the action bound is determined when calculating the feedback law
        action_abs = np.clip(action_abs, -1, 1)
        # print(f"action_abs: {action_abs}, tracking_error: {tracking_error}")
        return  action_abs

    def safety_switch_on(self, tracking_error):
        safety_value = tracking_error @ self.matrix_P @ tracking_error.T
        return safety_value > (1 - self.delta_min)



def cbf_filter_dt(x, u_nom, A=SYSTEM_A, B=SYSTEM_B, P=MATRIX_P,
                             u_min=-1.0, u_max=1.0,
                             gamma=0.1):
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
    x = np.asarray(x).reshape(-1)
    A = np.asarray(A)
    B = np.asarray(B).reshape(-1, 1)   # ensure (n,1)
    P = np.asarray(P)
    u_nom = u_nom * ACTION_BOUND
    n = A.shape[0]
    assert A.shape == (n, n)
    assert B.shape == (n, 1)
    assert P.shape == (n, n)

    # CBF bound: V(x_{k+1}) <= gamma + (1-gamma)*V(x_k)
    V_now = float(x @ P @ x)
    V_max_cbf = gamma + (1.0 - gamma) * V_now

    u = cp.Variable()
    x_next = A @ x + B.flatten() * u
    objective = cp.Minimize(cp.square(u - u_nom))
    lyap_constraint = cp.quad_form(x_next, P, assume_PSD=True) <= V_max_cbf
    constraints = [lyap_constraint, u >= u_min, u <= u_max]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, warm_start=True, verbose=False)

    if u.value is None:
        # Infeasible: fallback to safe LQR policy
        u_safe = float(np.clip(float(np.dot(F.flatten() * ACTION_BOUND, x)), u_min, u_max))
        certified = False
    else:
        u_safe = float(u.value)
        certified = abs(u_safe - u_nom) < 1e-6

    return u_safe / ACTION_BOUND, certified
