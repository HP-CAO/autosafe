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


def get_init_condition_in_safety_envelope(n_samples=1):
    """
    Sample n_samples points on the 12-dim safety envelope boundary s^T P s = 1.

    NOTE: unlike the 4D cartpole version where n_points_per_dim gives n^3 total
    points via nested loops, the 12D nested-loop analog would produce n^11 points
    (e.g. 10^11 for n=10), which is impractical.  This function therefore takes
    n_samples as the direct total count of points.

    Each sample draws (n_dim-1) uniform random angles, maps them to a unit vector
    via the standard n-sphere spherical-coordinate map, then scales by 1/sqrt(λ_i)
    so that s^T P s = 1 exactly.
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
        angles[-1] *= 2                       # last angle covers [0, 2*pi]

        z = np.empty(n_dim)
        sin_prod = 1.0
        for k in range(n_dim - 1):
            z[k] = sin_prod * math.cos(angles[k])
            sin_prod *= math.sin(angles[k])
        z[-1] = sin_prod                      # accumulated sin product, no angle left

        y = z / np.sqrt(lam)                  # y^T Λ y = ||z||^2 = 1
        s = Q @ y
        initial_condition_list.append(s.tolist())

    return initial_condition_list


def test_visualize_safety_envelope(n=500):
    """
    Test get_init_condition_in_safety_envelope by projecting the n sampled 12D
    boundary points onto 2D planes.  Overlays the Schur-complement ellipse (true
    2D projection) and the cross-section ellipse for comparison.
    """
    import matplotlib.pyplot as plt

    init_conditions = np.array(get_init_condition_in_safety_envelope(n_samples=n))

    state_labels = [
        'x', r'$\dot{x}$', 'y', r'$\dot{y}$', 'z', r'$\dot{z}$',
        r'$\phi$', r'$\theta$', r'$\psi$',
        r'$\dot{\phi}$', r'$\dot{\theta}$', r'$\dot{\psi}$',
    ]

    projections = [
        (0, 2),   # x vs y
        (6, 7),   # phi vs theta
        (4, 5),   # z vs vz
        (8, 11),  # psi vs psi_dot
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    def draw_ellipse(ax, M, **kwargs):
        ev, evec = np.linalg.eigh(M)
        t = np.linspace(0, 2 * np.pi, 500)
        pts = evec @ np.diag(1.0 / np.sqrt(ev)) @ np.array([np.cos(t), np.sin(t)])
        ax.plot(pts[0], pts[1], **kwargs)

    for ax, (i, j) in zip(axes, projections):
        A_idx = [i, j]
        B_idx = [k for k in range(12) if k not in A_idx]
        P_AA = MATRIX_P[np.ix_(A_idx, A_idx)]
        P_BB = MATRIX_P[np.ix_(B_idx, B_idx)]
        P_AB = MATRIX_P[np.ix_(A_idx, B_idx)]
        S = P_AA - P_AB @ np.linalg.inv(P_BB) @ P_AB.T  # true 2D projection

        ax.scatter(init_conditions[:, i], init_conditions[:, j],
                   s=8, alpha=0.5, c='steelblue',
                   label=f'Sampled ICs (N={n})')
        draw_ellipse(ax, S,    color='crimson',    linewidth=2, linestyle='-',
                     label='Schur complement (true 2D projection)')
        draw_ellipse(ax, P_AA, color='darkorange',  linewidth=2, linestyle='--',
                     label='Cross-section (other dims=0)')

        ax.set_xlabel(state_labels[i], fontsize=11)
        ax.set_ylabel(state_labels[j], fontsize=11)
        ax.set_title(f'{state_labels[i]} vs {state_labels[j]}', fontsize=12)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('safety_envelope_2d_quadrotor.pdf', bbox_inches='tight')
    plt.show()

    violations = sum(1 for s in init_conditions if s @ MATRIX_P @ s > 1.0 + 1e-6)
    print(f"Sampled {n} points — safety envelope violations: {violations} / {n}")


class ModelbasedAgent:
    def __init__(self, feedback_law=F, delta_min=0.0):
        self.feedback_law = feedback_law
        self.matrix_P = MATRIX_P
        self.delta_min = delta_min

    def get_action(self, tracking_error):
        # tracking_error env state - env control goal
        action_abs = self.feedback_law @ (tracking_error)
        action_abs = np.clip(action_abs, -1, 1)
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

    # CBF bound: V(x_{k+1}) <= gamma + (1-gamma)*V(x_k)
    V_now = float(x.T @ P @ x)
    V_max_cbf = gamma + (1.0 - gamma) * V_now

    u = cp.Variable((4, 1))
    x_next = A @ x + B @ u
    objective = cp.Minimize(cp.sum_squares(u - u_nom))
    lyap_constraint = cp.quad_form(x_next, P, assume_PSD=True) <= V_max_cbf
    constraints = [lyap_constraint, u >= u_min, u <= u_max]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, warm_start=True, verbose=False)

    if u.value is None:
        # Infeasible: fallback to safe LQR policy
        u_safe = np.clip((F @ x.reshape(-1)) * ACTION_BOUND, u_min, u_max).reshape(-1)
        certified = False
    else:
        u_safe = u.value.reshape(-1)
        certified = np.all(np.abs(u_safe - u_nom.reshape(-1)) < 1e-6)

    return u_safe / ACTION_BOUND, certified


if __name__ == '__main__':
    test_visualize_safety_envelope(n=1000000)
