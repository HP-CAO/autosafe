import numpy as np
import math
import cvxpy as cp

ACTION_BOUND = 20.0 #todo double check action bound, to normalize the safe action space to [-1, 1]

F = np.array([[25.9995, 19.4241, 75.9886, 13.8553]]) / ACTION_BOUND # feedback law for the safe controller

MATRIX_P = np.array([[13.3425, 6.73778, 16.2166, 3.47318],
                     [6.73778, 3.94828, 9.69035, 2.09032],
                     [16.2166, 9.69035, 25.9442, 5.31439],
                     [3.47318, 2.09032, 5.31439, 1.16344]])

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
    def __init__(self, feedback_law=F, action_bound=ACTION_BOUND):
        self.feedback_law = feedback_law
        self.matrix_P = MATRIX_P


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
        safety_value = tracking_error @ MATRIX_P @ tracking_error.T
        return safety_value > 1


def cbf_data_linear_dt(x, A, B, p_max, th_max, gamma):
    """
    Discrete-time CBF constraints for linear cart-pole.
    Returns a list of (a_i, b_i) defining inequalities: a_i * u >= b_i.

    x: current state (shape (4,) or (4,1))
    A, B: discrete-time model matrices
    p_max: max |p|
    th_max: max |theta|
    gamma: CBF gain (>0)
    """
    x = np.asarray(x).reshape(-1)  # ensure shape (4,)

    constraints = []
    B = B.reshape(-1, 1)  # ensure shape (4,)
    # Helper: compute (a, b) for a given c, d
    def make_constraint(c, d):
        c = np.asarray(c).reshape(1, -1)  # 1x4
        h = float(c @ x + d)  # current barrier value (not strictly needed below)
        a = float(c @ B)  # scalar: c^T B
        b = float(-c @ (A @ x) + (1.0 - gamma) * (c @ x) - gamma * d)
        # print(f"make_constraint: c={c.flatten()}, d={d}, h={h}, a={a}, b={b}")
        return a, b, h

    # h_p+ = p_max - p  => c = [-1, 0, 0, 0], d = p_max
    a_p_plus, b_p_plus, h_p_plus = make_constraint([-1.0, -0.5, 0.0, 0.0], p_max)
    constraints.append((a_p_plus, b_p_plus))

    # h_p- = p_max + p  => c = [1, 0, 0, 0], d = p_max
    a_p_minus, b_p_minus, h_p_minus = make_constraint([1.0, 0.5, 0.0, 0.0], p_max)
    constraints.append((a_p_minus, b_p_minus))

    # h_th+ = th_max - th  => c = [0, 0, -1, 0], d = th_max
    a_th_plus, b_th_plus, h_th_plus = make_constraint([0.0, 0.0, -1.0, -0.5], th_max)
    constraints.append((a_th_plus, b_th_plus))

    # h_th- = th_max + th  => c = [0, 0, 1, 0], d = th_max
    a_th_minus, b_th_minus, h_th_minus = make_constraint([0.0, 0.0, 1.0, 0.5], th_max)
    constraints.append((a_th_minus, b_th_minus))
    return constraints

# p_max=0.5, th_max=0.785
def cbf_filter_dt(x, u_nom, A=SYSTEM_A, B=SYSTEM_B,
                  p_max=0.05, th_max=0.785,
                  gamma=0.6,
                  u_min=-20.0, u_max=20.0):
    """
    One-step discrete-time CBF-QP filter for a linear cart-pole model.

    x: current state (4,)
    u_nom: nominal control (scalar)
    A, B: discrete-time model matrices
    p_max, th_max: safety bounds
    gamma: CBF gain in discrete-time condition
    u_min, u_max: actuator limits
    """
    cons = cbf_data_linear_dt(x, A, B, p_max, th_max, gamma)

    u_nom = u_nom * ACTION_BOUND  # scale back to absolute control

    u = cp.Variable()  # scalar control
    objective = cp.Minimize(cp.square(u - u_nom))

    constraints = [u >= u_min, u <= u_max]
    for (a_i, b_i) in cons:
        # if input doesn't affect that constraint, skip it
        if abs(a_i) > 1e-6:
            constraints.append(a_i * u >= b_i)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

    if u.value is None:
        # QP failed: fallback to clipped nominal control
        return float(np.clip(u_nom, u_min, u_max)) / ACTION_BOUND, False
    else:
        return float(u.value) / ACTION_BOUND, True