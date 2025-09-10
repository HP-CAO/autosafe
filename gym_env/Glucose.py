"""
Disclaimer: This code is built upon RL-AR https://github.com/HaozheTian/RL-AR/tree/main
"""

import dataclasses
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import matplotlib.pyplot as plt
import math
import do_mpc
from collections import deque
from typing import Dict, Optional, Union


@dataclasses.dataclass
class RobotConfig:
    # Robot Settings, the values are measured from the real cart-pole system produce by Quanser
    # https://www.quanser.com/products/linear-servo-base-unit-inverted-pendulum/
    pass

@dataclasses.dataclass
class TaskConfig:
    # Task Settings
    action_penalty: float = 0.01
    crash_penalty: float = 0
    ini_states: list = dataclasses.field(default_factory=lambda: [0.0, 0.0, math.pi, 0.0])
    control_goal: list = dataclasses.field(default_factory=lambda: [0., 0.])  # position and angle
    max_episode_steps: int = 500
    evaluation_period: int = 5000
    num_episodes_to_run: int = 10  # number of episodes to evaluate the agent or collect trajectories
    task_reset_mode: str = 'random'
    change_dynamics: bool = False

@dataclasses.dataclass
class SimulationConfig:
    # Simulation Settings
    pass

@dataclasses.dataclass
class GlucoseConfig:
    RobotParams: RobotConfig = dataclasses.field(default_factory=RobotConfig)
    TaskParams: TaskConfig = dataclasses.field(default_factory=TaskConfig)
    SimulationParams: SimulationConfig = dataclasses.field(default_factory=SimulationConfig)


def reset_paras(model_paras: Dict, altered_paras: Dict) -> Dict:
    for key, val in altered_paras.items():
        print(f'{key}:   Model = {model_paras[key]:.2f}  |  Plant = {val:.2f}')
        model_paras[key] = val
    return model_paras


class GlucoseHistory():
    def __init__(self) -> None:
        self.obs_queue = deque(maxlen=2)

    def reset(self, glucose):
        self.obs_queue.append(100.0)
        self.obs_queue.append(glucose)
        self.time = 0

    def add_his(self, glucose):
        self.obs_queue.append(glucose)
        self.time += 1

    def get_obs(self):
        return np.array([self.obs_queue[1],
                         self.obs_queue[1] - self.obs_queue[0],
                         self.time])


class Glucose(gym.Env):
    """
    Bergman minimal model for blood glucose simulation.

    Based on the parameters in https://www.nature.com/articles/s41598-022-16535-2
    """

    def __init__(self, altered_paras: Dict = {}, render_mode: Optional[str] = None, max_steps: int = 100,
                 action_penalty: float = 2.0):
        self.action_penalty = action_penalty
        self.render_mode = render_mode
        self.max_steps = max_steps
        # model parameters estimated values
        # model_paras = {
        #     'Gb': 138, 'Ib': 7, 'n': 0.2814, 'p1': 0,
        #     'p2': 0.0142, 'p3': 15e-6, 'D0': 4, 'dt': 10
        # }
        # model parameters true values
        model_paras = {
            'Gb': 138, 'Ib': 7, 'n': 0.2, 'p1': 0,
            'p2': 0.005, 'p3': 5e-6, 'D0': 4, 'dt': 10
        }

        # custom parameters
        for key, val in reset_paras(model_paras, altered_paras).items():
            exec(f'self.{key}={val}')
        # Gym style observation space and action space
        self.observation_space = Box(
            low=np.array([0, -20, 0]),
            high=np.array([1000, 20, 200]),
            dtype=np.float32  # Data type of the observation space
        )
        self.action_space = Box(low=-1, high=1, dtype=np.float32)

        self.obs_history = GlucoseHistory()

        Gb, Ib, n, p1, p2, p3, D0, dt = tuple(model_paras.values())
        model = do_mpc.model.Model(model_type='continuous')
        # set model states
        G = model.set_variable(var_type='_x', var_name='G')
        X = model.set_variable(var_type='_x', var_name='X')
        I = model.set_variable(var_type='_x', var_name='I')
        u = model.set_variable(var_type='_u', var_name='u')
        Dt = model.set_variable(var_type='_tvp', var_name='Dt')
        # set rhs
        dG = -p1 * (G - Gb) - G * X + Dt
        dX = -p2 * X + p3 * (I - Ib)
        dI = -n * (I - Ib) + u
        model.set_rhs('G', dG)
        model.set_rhs('X', dX)
        model.set_rhs('I', dI)
        model.setup()


        self.init_state = np.array([self.Gb, 0., self.Ib], dtype=np.float32)
        self.model_based_equilibrium = np.array([self.Gb, 0., self.Ib], dtype=np.float32) # double check

        # linear_model_matrix = model.get_linear_system_matrices(xss=self.model_based_equilibrium, uss=np.array([0]))

        simulator = do_mpc.simulator.Simulator(model)
        simulator.set_param(t_step=dt)
        tvp_template_sim = simulator.get_tvp_template()

        def tvp_fun_sim(t_now):
            t = t_now
            tvp_template_sim['Dt'] = D0 * math.exp(-0.01 * t)
            return tvp_template_sim

        simulator.set_tvp_fun(tvp_fun=tvp_fun_sim)
        simulator.setup()
        self.simulator = simulator
        self.action_low = np.array([0.])
        self.action_high = np.array([2.0])
        self.action_bias = 0.5 * (self.action_high + self.action_low)
        self.action_scale = 0.5 * (self.action_high - self.action_low)
        self.risk_list = []

    def reset(self, seed: Optional[int] = 0, options=None) -> Union[np.ndarray, Dict]:
        super().reset(seed=seed)
        # initial states
        self.time_step = 0
        self.simulator.reset_history()
        self.simulator.x0 = np.copy(self.init_state).reshape(-1, 1)
        self.state = np.copy(self.init_state)

        glucose = self.init_state[0]
        self.obs_history.reset(glucose)
        obs = self.obs_history.get_obs()

        self.traj_obs = [glucose]
        self.traj_act = []
        self.risk_list = []
        return obs, {"state": self.state, "meas": np.array([glucose])}

    def step(self, action: np.ndarray) -> Union[np.ndarray, float, bool, bool, Dict]:
        self.time_step += 1
        u = action * self.action_scale + self.action_bias  # scale and shift action
        u = u.reshape(-1, 1)  # ensure u is a column vector
        state_next = self.simulator.make_step(u)
        self.state = state_next.reshape((-1,))
        G = self.state[0]

        term = (G < 10) or (G > 1000)
        trun = True if self.time_step >= self.max_steps else False
        # Magni risk function https://proceedings.mlr.press/v126/fox20a/fox20a.pdf

        risk_score = -1 * (3.35506 * (math.log(G) ** 0.8353 - 3.7932)) ** 2

        self.risk_list.append(risk_score)

        if term:
            reward = -1e3
        else:
            reward = risk_score


        if u >= 0.1:
            reward -= self.action_penalty

        self.obs_history.add_his(G)
        obs = self.obs_history.get_obs()

        self.traj_obs.append(obs)
        self.traj_act.append(u)

        # if self.render_mode == 'human' and (term or trun):
        #     self.render()

        return obs, reward, term, trun, {"state": self.state, "meas": np.array([G])}


    def vis_trajectory(self):
        traj_obs = self.get_plot_obs(np.array(self.traj_obs[1:]))
        traj_act = np.array(self.traj_act).squeeze()

        num_steps = len(traj_obs)
        t = np.arange(0, self.dt * (num_steps - 1) + 0.001, self.dt)
        t_c = np.arange(0, self.dt * (num_steps - 2) + 0.001, self.dt / 100.0)

        traj_act = traj_act[(t_c // self.dt).astype(int)]

        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].plot(t, traj_obs, label='Glucose')
        axs[0].set_ylabel('Glucose')
        axs[0].legend(loc='upper right')
        axs[1].plot(t_c, traj_act, label='Insulin')
        axs[1].set_ylabel('Insulin')
        axs[1].set_xlabel('Time (s)')
        axs[1].legend(loc='upper right')
        plt.suptitle('Glucose Simulation')
        fig.tight_layout()
        return fig


    def get_performance_score(self):
        # calculate average risk
        return np.mean(self.risk_list)


    def render(self):
        self.vis_trajectory()
        plt.show()

    def get_plot_obs(self, eps_obs: np.ndarray):
        assert eps_obs.ndim > 1, "_get_plot_obs() deals with obs with shape (N, num_obs)"
        return eps_obs[:, 0]  # only plot glucose level