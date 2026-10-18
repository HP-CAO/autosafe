import numpy as np
import math
import cvxpy as cp
# from scipy.signal import cont2discrete
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


def get_init_condition_in_safety_envelope(n_samples=1):
    """
    Sample n_samples points on the 4D safety envelope boundary s^T P s = 1.

    Uses the same spherical-coordinate approach as the quadrotor version:
    draw 3 uniform random angles, map to a unit vector via the standard
    4-sphere map, then scale by 1/sqrt(λ_i) so that s^T P s = 1 exactly.
    """
    P_sym = (MATRIX_P + MATRIX_P.T) / 2
    eigen_values, eigen_vectors = np.linalg.eigh(P_sym)
    Q = eigen_vectors
    lam = eigen_values
    n_dim = MATRIX_P.shape[0]

    initial_condition_list = []
    for _ in range(n_samples):
        # angles[0..n_dim-3] in [0, pi], angles[n_dim-2] in [0, 2*pi]
        angles = np.random.uniform(0, math.pi, n_dim - 1)
        angles[-1] *= 2

        z = np.empty(n_dim)
        sin_prod = 1.0
        for k in range(n_dim - 1):
            z[k] = sin_prod * math.cos(angles[k])
            sin_prod *= math.sin(angles[k])
        z[-1] = sin_prod

        y = z / np.sqrt(lam)
        s = Q @ y
        initial_condition_list.append(s.tolist())

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
    a_p_plus, b_p_plus, h_p_plus = make_constraint([-1.0, -0.1, 0.0, 0.0], p_max)
    constraints.append((a_p_plus, b_p_plus))

    # h_p- = p_max + p  => c = [1, 0, 0, 0], d = p_max
    a_p_minus, b_p_minus, h_p_minus = make_constraint([1.0, 0.1, 0.0, 0.0], p_max)
    constraints.append((a_p_minus, b_p_minus))

    # h_th+ = th_max - th  => c = [0, 0, -1, 0], d = th_max
    a_th_plus, b_th_plus, h_th_plus = make_constraint([0.0, 0.0, -1.0, -0.1], th_max)
    constraints.append((a_th_plus, b_th_plus))

    # h_th- = th_max + th  => c = [0, 0, 1, 0], d = th_max
    a_th_minus, b_th_minus, h_th_minus = make_constraint([0.0, 0.0, 1.0, 0.1], th_max)
    constraints.append((a_th_minus, b_th_minus))
    return constraints
#
#
# # p_max=0.5, th_max=0.785
# def cbf_filter_dt(x, u_nom, A=SYSTEM_A, B=SYSTEM_B,
#                   p_max=0.5, th_max=0.785,
#                   gamma=0.9,
#                   u_min=-20.0, u_max=20.0):
#     """
#     One-step discrete-time CBF-QP filter for a linear cart-pole model.
#
#     x: current state (4,)
#     u_nom: nominal control (scalar)
#     A, B: discrete-time model matrices
#     p_max, th_max: safety bounds
#     gamma: CBF gain in discrete-time condition, if gamma = 1.0 => h(x_{k+1}) >= 0
#     u_min, u_max: actuator limits
#     """
#
#     cons = cbf_data_linear_dt(x, A, B, p_max *0.1 , th_max*0.1, gamma)
#
#     u_nom = u_nom * ACTION_BOUND  # scale back to absolute control
#
#     u = cp.Variable()  # scalar control
#     objective = cp.Minimize(cp.square(u - u_nom))
#
#     constraints = [u >= u_min, u <= u_max]
#
#     for (a_i, b_i) in cons:
#         # if input doesn't affect that constraint, skip it
#         if abs(a_i) > 1e-6:
#             constraints.append(a_i * u >= b_i)
#
#     prob = cp.Problem(objective, constraints)
#     prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
#
#     if u.value is None:
#         # QP failed: fallback to clipped nominal control
#         return float(np.clip(u_nom, u_min, u_max)) / ACTION_BOUND, False
#     else:
#         return float(np.clip(u.value, u_min, u_max)) / ACTION_BOUND, True


def test_visualize_safety_envelope(n=3):
    """
    Test get_init_condition_in_safety_envelope by projecting the n^3 sampled
    4D boundary points onto 2D planes. With enough samples the scatter approximates
    the 2D projection of the safety envelope.
    """
    import matplotlib.pyplot as plt

    init_conditions = np.array(get_init_condition_in_safety_envelope(n_samples=n))
    n_total = len(init_conditions)

    projections = [
        (0, 2, 'x (position)', r'$\theta$ (angle)'),
        (1, 3, r'$\dot{x}$ (velocity)', r'$\dot{\theta}$ (angular vel.)'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    def draw_ellipse(ax, M, **kwargs):
        ev, evec = np.linalg.eigh(M)
        t = np.linspace(0, 2 * np.pi, 500)
        pts = evec @ np.diag(1.0 / np.sqrt(ev)) @ np.array([np.cos(t), np.sin(t)])
        ax.plot(pts[0], pts[1], **kwargs)

    for ax, (i, j, xlabel, ylabel) in zip(axes, projections):
        A_idx, B_idx = [i, j], [k for k in range(4) if k not in (i, j)]
        P_AA = MATRIX_P[np.ix_(A_idx, A_idx)]
        P_BB = MATRIX_P[np.ix_(B_idx, B_idx)]
        P_AB = MATRIX_P[np.ix_(A_idx, B_idx)]
        S = P_AA - P_AB @ np.linalg.inv(P_BB) @ P_AB.T  # Schur complement = true 2D projection

        ax.scatter(init_conditions[:, i], init_conditions[:, j],
                   s=8, alpha=0.5, c='steelblue',
                   label=f'Sampled ICs (n={n}, N={n_total})')
        draw_ellipse(ax, S,    color='crimson',   linewidth=2, linestyle='-',  label='Schur complement (true 2D projection)')
        draw_ellipse(ax, P_AA, color='darkorange', linewidth=2, linestyle='--', label='Cross-section ($x_{other}=0$)')

        if i == 0 and j == 2:
            for xb in [-0.9, 0.9]:
                ax.axvline(xb, color='gray', linestyle=':', linewidth=1, alpha=0.7)
            for yb in [-0.8, 0.8]:
                ax.axhline(yb, color='gray', linestyle=':', linewidth=1, alpha=0.7,
                           label='Safety bounds' if yb == -0.8 else None)

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'Projected safety envelope: {xlabel} vs {ylabel}', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('safety_envelope_2d.pdf', bbox_inches='tight')
    plt.show()

    print(f"Sampled {n_total} points with n={n}")
    violations = sum(1 for s in init_conditions if s @ MATRIX_P @ s > 1.0 + 1e-6)
    print(f"Safety envelope violations: {violations} / {n_total}")


def cbf_filter_dt(x, u_nom, A=SYSTEM_A, B=SYSTEM_B, P=MATRIX_P,
                             u_min=-20.0, u_max=20.0,
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
    # enforces h(x_{k+1}) >= (1-gamma)*h(x_k) for h(x) = 1 - x^T P x
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

    return np.clip(u_safe / ACTION_BOUND, -1, 1), certified


if __name__ == '__main__':
    test_visualize_safety_envelope(n=100000)