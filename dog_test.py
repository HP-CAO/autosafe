import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# =============================================================================
# Model: Agent definition
# =============================================================================
class Agent:
    def __init__(
            self,
            x0,
            goal,
            obstacles,
            dt: float = 0.1,
            horizon: int = 20,
            w_goal: float = 1.0,
            w_obstacle: float = 100.0,
            w_control=(0.1, 0.1, 0.5),
            x_bounds=(0.0, 6.0),
            y_bounds=(0.0, 6.0),
            safety_margin: float = 0.2
    ):
        """
        x0           : initial state [x, y, theta]
        goal         : goal position [x, y]
        obstacles    : list of (pos, radius)
        dt           : timestep
        horizon      : planning horizon
        w_goal       : goal cost weight
        w_obstacle   : obstacle cost weight
        w_control    : (w_vx, w_vy, w_omega)
        x_bounds     : (min_x, max_x)
        y_bounds     : (min_y, max_y)
        """
        self.x0 = np.array(x0, float)
        self.goal = np.array(goal, float)
        self.obstacles = [(np.array(p, float), r) for p, r in obstacles]
        self.dt = dt
        self.horizon = horizon
        self.w_goal = w_goal
        self.w_obstacle = w_obstacle
        self.w_control = np.array(w_control, float)
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds

        # Safety margin for obstacle avoidance
        self.safety_margin = safety_margin

        # to store last plan
        self.planned_controls = None  # shape (horizon,3)
        self.planned_trajectory = None  # shape (horizon+1,3)

    # def step(self, state, control):
    #     """ Integrator: x,y,theta <- apply vx,vy,omega """
    #     x, y, theta = state
    #     vx, vy, omega = control
    #     return np.array([
    #         x + vx * self.dt,
    #         y + vy * self.dt,
    #         theta + omega * self.dt
    #     ], float)

    def step(self, state, control):
        """Integrate planar SE(2) pose using **body-frame** (vx, vy, ω)."""
        x, y, theta = state
        vx_b, vy_b, omega = control  # body-frame commands

        # body → world rotation
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        vx_w = cos_t * vx_b - sin_t * vy_b
        vy_w = sin_t * vx_b + cos_t * vy_b

        # angle wrap to [-pi, pi]
        theta_new = (theta + omega * self.dt + np.pi) % (2 * np.pi) - np.pi  # keep ω in body frame

        return np.array([
            x + vx_w * self.dt,
            y + vy_w * self.dt,
            theta_new
        ], dtype=float)

    def cost_sequence(self, controls):
        """ Return total cost for flattened or (horizon,3) controls. """
        U = np.array(controls, float).reshape(self.horizon, 3)
        s = self.x0.copy()
        total = 0.0
        discount = 1.0
        w_goal_run = 0.5
        w_boundary = 50.0

        for vx, vy, omega in U:
            # integrate *first* so we measure cost on the successor
            s_next = self.step(s, (vx, vy, omega))
            px, py = s_next[:2]

            # 1. running goal cost (small pull)
            total += discount * w_goal_run * np.sum((s_next[:2] - self.goal) ** 2)

            # 2. quadratic obstacle buffer
            for op, r in self.obstacles:
                d = np.linalg.norm(s_next[:2] - op) - (r + self.safety_margin)
                if d < 0:  # inside buffer
                    total += discount * self.w_obstacle * d ** 2

            # 3. soft boundary penalty (quadratic outside box)
            bx0, bx1 = self.x_bounds
            by0, by1 = self.y_bounds
            q = 0.0
            q += max(bx0 - px, 0.0) ** 2 + max(px - bx1, 0.0) ** 2
            q += max(by0 - py, 0.0) ** 2 + max(py - by1, 0.0) ** 2
            total += discount * w_boundary * q

            # 4. control effort (axis-wise)
            total += discount * np.sum(
                self.w_control * np.array([vx, vy, omega]) ** 2
            )

            # advance
            s = s_next
            discount *= 0.98  # discount factor for future costs

        # 5. terminal goal cost (major pull)
        total += self.w_goal * np.sum((s[:2] - self.goal) ** 2)
        return total

    def simulate_sequence(self, controls):
        """ Roll out and return states for given controls. """
        U = np.array(controls, float).reshape(self.horizon, 3)
        traj = [self.x0.copy()]
        s = self.x0.copy()
        for u in U:
            s = self.step(s, u)
            traj.append(s.copy())
        return np.vstack(traj)

    def set_plan(self, controls):
        """ Store last plan for plotting or inspection. """
        self.planned_controls = np.array(controls, float).reshape(self.horizon, 3)
        self.planned_trajectory = self.simulate_sequence(self.planned_controls)
        return self.planned_controls, self.planned_trajectory


# =============================================================================
# Solver: Standard CEM
# =============================================================================
class CEMController:
    def __init__(
            self,
            agent: Agent,
            N: int = 500,
            Ke: int = 50,
            K: int = 5,
            sigma_init: float = 1.0,
            epsilon: float = 1e-6,
            vx_max: float = 1.0,
            vy_max: float = 1.0,
            omega_max: float = 0.3,
            alpha_mu: float = 0.7,
            alpha_sigma: float = 0.3,
    ):
        self.agent = agent
        self.N = N
        self.Ke = Ke
        self.K = K
        self.sigma_init = sigma_init
        self.epsilon = epsilon
        self.vx_max = vx_max
        self.vy_max = vy_max
        self.omega_max = omega_max
        self.alpha_mu = alpha_mu
        self.alpha_sigma = alpha_sigma

    def plan(self, warm_start=None, return_stats=False):
        H = self.agent.horizon
        dim = 3 * H

        # initialize
        mu = warm_start.flatten() if warm_start is not None else np.zeros(dim)
        sigma = np.eye(dim) * (self.sigma_init ** 2)

        for _ in range(self.K):
            # sample & enforce bounds
            S = np.random.multivariate_normal(mu, sigma, self.N)
            S = S.reshape(self.N, H, 3)
            S[..., 0] = np.clip(S[..., 0], -self.vx_max, self.vx_max)
            S[..., 1] = np.clip(S[..., 1], -self.vy_max, self.vy_max)
            S[..., 2] = np.clip(S[..., 2], -self.omega_max, self.omega_max)
            flat = S.reshape(self.N, dim)

            # evaluate
            costs = np.array([self.agent.cost_sequence(f) for f in flat])

            # elite update
            elites = flat[costs.argsort()[:self.Ke]]
            new_mu = elites.mean(axis=0)
            new_sigma = np.cov(elites, rowvar=False) + self.epsilon * np.eye(dim)

            # smooth
            mu = self.alpha_mu * new_mu + (1 - self.alpha_mu) * mu
            sigma = self.alpha_sigma * new_sigma + (1 - self.alpha_sigma) * sigma

        plan = mu.reshape(H, 3)
        # final clip
        plan[:, 0] = np.clip(plan[:, 0], -self.vx_max, self.vx_max)
        plan[:, 1] = np.clip(plan[:, 1], -self.vy_max, self.vy_max)
        plan[:, 2] = np.clip(plan[:, 2], -self.omega_max, self.omega_max)

        if return_stats:
            return plan, self.agent.cost_sequence(plan)
        return plan


# =============================================================================
# Meta-solver: Bregman-CEM
# =============================================================================
class BregmanCEM:
    def __init__(
            self,
            agent,
            num_cem: int = 3,
            **cem_kwargs
    ):
        """
        agent      : Agent instance
        num_cem    : number of parallel CEM workers
        cem_kwargs : forwarded to each CEMController
        """
        self.agent = agent
        self.num_cem = num_cem
        self.cem_kwargs = cem_kwargs

        # Initialize CEM workers
        self.workers = [
            CEMController(agent, **cem_kwargs)
            for _ in range(num_cem)
        ]

        # Bregman-CEM parameters
        self.beta = cem_kwargs.get("beta", 0.1)
        self.trust_region = cem_kwargs.get("trust_region", 5.0)
        self.respawn_frequency = cem_kwargs.get("respawn_frequency", 5)

        # Centroid and warm-starts
        self.centroid_mu = np.zeros((agent.horizon, 3), float)
        self.worker_warm_starts = [None] * num_cem

        self.counter = 0

    def plan(self):
        sequences, costs = [], []

        # Determine default warm-start (centroid) after first iteration
        default_warm = self.centroid_mu if self.counter > 0 else None

        # Execute each worker with its own warm-start
        for i, worker in enumerate(self.workers):
            warm = self.worker_warm_starts[i] if self.worker_warm_starts[i] is not None else default_warm
            seq, cost = worker.plan(warm_start=warm, return_stats=True)
            sequences.append(seq)
            costs.append(cost)
            # Clear once used
            self.worker_warm_starts[i] = None

        print(f'Iter {self.counter}: costs = {costs}')

        # Compute new centroid as weighted average of sequences
        costs = np.array(costs)
        z = (costs - costs.mean()) / (costs.std() + 1e-8)
        weights = np.exp(-self.beta * z)
        weights /= weights.sum()
        centroid_mu_new = sum(w * s for w, s in zip(weights, sequences))
        alpha_mu = self.cem_kwargs.get("alpha_mu", 0.7)
        self.centroid_mu = alpha_mu * centroid_mu_new + (1 - alpha_mu) * self.centroid_mu

        # Respawn worst worker periodically
        if self.counter % self.respawn_frequency == 0:
            # if num_cem is 1, no respawn needed
            if self.num_cem == 1:
                return sequences[0]

            # Score = weight * squared distance from centroid
            scores = [
                weights[i] * np.sum((sequences[i] - self.centroid_mu) ** 2)
                for i in range(self.num_cem)
            ]
            # print(f'Iter {self.counter}: weights = {weights}, scores = {scores}')
            worst_idx = int(np.argmax(scores))

            # Perturb centroid to create new warm start
            v = np.random.normal(size=self.centroid_mu.shape)
            v /= np.linalg.norm(v) + 1e-12
            perturbed = self.centroid_mu + 1.0 * v

            # Replace worker and assign its warm start
            self.workers[worst_idx] = CEMController(self.agent, **self.cem_kwargs)
            self.worker_warm_starts[worst_idx] = perturbed

        self.counter += 1

        # Return best sequence
        best_idx = int(np.argmin(costs))
        return sequences[best_idx]


# =============================================================================
# MPC Loop
# =============================================================================
class MPCWithCEM:
    def __init__(self, agent: Agent, controller, *,
                 max_steps=50, tol=0.3, collect_traces=False):
        self.agent = agent
        self.controller = controller
        self.max_steps = max_steps
        self.tol = tol
        self.collect_traces = collect_traces

    def run(self):
        state = self.agent.x0.copy()
        traj, ctrls, traces = [state.copy()], [], []

        for i in tqdm(range(self.max_steps), desc="MPC Steps"):
            self.agent.x0 = state.copy()

            plan = self.controller.plan()  # (H,3)
            ctrls.append(plan[0])

            if self.collect_traces:
                # full horizon in world frame for plotting (H+1,3)
                traces.append(self.agent.simulate_sequence(plan))

            state = self.agent.step(state, plan[0])
            traj.append(state.copy())

            if np.linalg.norm(state[:2] - self.agent.goal) <= self.tol:
                print(f"Reached goal in {i + 1} steps.")
                break

        return np.asarray(traj), np.asarray(ctrls), traces


# =============================================================================
# Pretty plotting
# =============================================================================
def plot_episode(traj, agent: Agent, *, traces=None, show_traces=False):
    """
    traj      : (T,3) executed trajectory
    traces    : list of (H+1,3) arrays from CEM (optional)
    show_traces : if True, draw those traces lightly
    """
    # ---- Matplotlib style ---------------------------------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=120)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(ls=":", lw=0.5, alpha=0.6)

    # (optional) sampling traces
    if show_traces and traces:
        for seq in traces:
            ax.plot(seq[:, 0], seq[:, 1],
                    color="0.6", lw=0.6, alpha=0.4, zorder=1, )

    # executed trajectory
    ax.plot(traj[:, 0], traj[:, 1], "-o",
            lw=1.2, ms=3, color="#1f77b4", label="Executed", zorder=3)

    # start & goal
    ax.scatter(0.0, 0.0, marker="s", s=40, color="g", label="Start")
    ax.scatter(*agent.goal, marker="o", s=60,
               facecolors="none", edgecolors="r", lw=1.4, label="Goal")

    # goal tolerance region (shadow)
    tol_circle = plt.Circle(agent.goal, 0.5, color="r", alpha=0.15, zorder=0)
    ax.add_patch(tol_circle)

    # obstacles
    θ = np.linspace(0, 2 * np.pi, 120)
    for p, r in agent.obstacles:
        ax.plot(p[0] + r * np.cos(θ),
                p[1] + r * np.sin(θ),
                "k--", lw=1)

    ax.set_xlim(agent.x_bounds[0] - 0.2, agent.x_bounds[1] + 0.2)
    ax.set_ylim(agent.y_bounds[0] - 0.2, agent.y_bounds[1] + 0.2)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    plt.show()


# =============================================================================
# Main entry point
# =============================================================================
def main():
    x0 = [0.0, 0.0, 0.0]
    goal = [3.2, 3.2]
    obstacles = [
        ([1.0, 1.0], 0.3),
        ([2.5, 2.5], 0.4),
        ([2.5, 1.5], 0.2),
        ([1.5, 2.0], 0.2)
    ]

    agent = Agent(x0, goal, obstacles,
                  dt=0.3, horizon=15,
                  w_goal=2.0, w_obstacle=25.0,
                  w_control=(0.1, 0.1, 0.1),
                  x_bounds=(0.0, 3.5), y_bounds=(0.0, 3.5))

    cem_kwargs = dict(
        N=50, Ke=5, K=3, sigma_init=1.0, epsilon=1e-3,
        vx_max=0.5, vy_max=0.5, omega_max=0.5,
        alpha_mu=0.7, alpha_sigma=0.7
    )
    # controller = BregmanCEM(agent, num_cem=5, **cem_kwargs)
    controller = CEMController(agent, **cem_kwargs)

    mpc = MPCWithCEM(agent, controller,
                     max_steps=40, tol=0.2,
                     collect_traces=True)

    traj, ctrls, traces = mpc.run()
    plot_episode(traj, agent, traces=traces, show_traces=True)


if __name__ == "__main__":
    np.random.seed(1)
    main()
