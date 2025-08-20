"""
Disclaimer: This code is built upon RL-AR https://github.com/HaozheTian/RL-AR/tree/main
"""

import numpy as np
import gymnasium as gym
from attr import dataclass
from gymnasium.spaces import Box
import matplotlib.pyplot as plt
import math
import do_mpc
import casadi as ca
from collections import deque
from typing import Dict, Optional, Union
import dataclasses

@dataclasses.dataclass
class RobotConfig:
    # Robot Settings, the values are measured from the real cart-pole system produce by Quanser
    # https://www.quanser.com/products/linear-servo-base-unit-inverted-pendulum/
    pass

@dataclasses.dataclass
class TaskConfig:
    # Task Settings
    action_penalty: float = 0
    crash_penalty: float = 0
    ini_states: list = dataclasses.field(default_factory=lambda: [0.0, 0.0, math.pi, 0.0])
    control_goal: list = dataclasses.field(default_factory=lambda: [0., 0.])  # position and angle
    max_episode_steps: int = 200
    evaluation_period: int = 5000
    num_episodes_to_run: int = 10  # number of episodes to evaluate the agent or collect trajectories
    task_reset_mode: str = 'random'
    change_dynamics: bool = False


@dataclasses.dataclass
class SimulationConfig:
    # Simulation Setting
    pass


@dataclasses.dataclass
class BiGlucoseConfig:
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

class BiGlucose(gym.Env):
    """
    Extended Bi-glucose Horvorka model for blood glucose simulation.

    Based on the parameters in https://ieeexplore.ieee.org/abstract/document/10252997
    """

    def __init__(self, altered_paras: Dict = {}, render_mode: Optional[str] = None, max_steps: int = 200,
                 action_penalty: int = 3):
        self.action_penalty = action_penalty  # penalty for large action
        self.render_mode = render_mode
        self.max_steps = max_steps
        model_paras = {
            "D_G": 80, "V_G": 0.14, "k_12": 0.0968, "F_01": 0.0119, "EGP_0": 0.0213,
            "A_g": 0.8, "t_maxG": 40, "t_maxI": 55, "V_I": 0.12, "k_e": 0.138,
            "k_a1": 0.0088, "k_a2": 0.0302, "k_a3": 0.0118, "k_b1": 7.5768e-05, "k_b2": 1.4194e-05,
            "k_b3": 0.00085, "t_maxN": 20.59, "k_N": 0.735, "V_N": 23.46, "p": 0.074,
            "S_N": 19800.0, "M_g": 180.16, "BW": 68.5, "N_b": 48.13, "dt": 10}
        # custom parameters
        for key, val in reset_paras(model_paras, altered_paras).items():
            exec(f'self.{key}={np.float32(val)}')
        # Gym style observation space and action space
        self.observation_space = Box(
            low=np.array([0, -10, 0]),
            high=np.array([1000, 10, 200]),
            dtype=np.float32  # Data type of the observation space
        )
        self.action_space = Box(
            low=np.array([-1.0, -1.0]),  # Lower bounds for each dimension
            high=np.array([1.0, 1.0]),  # Upper bounds for each dimension
            dtype=np.float32  # Data type of the observation space
        )
        self.action_low = np.array([0., 0.])
        self.action_high = np.array([1., 5.])
        self.obs_history = GlucoseHistory()

        self.init_state, self.u_basal = self._solve_steady_state()

        D_G, V_G, k_12, F_01, EGP_0, A_g, t_maxG, t_maxI, V_I, k_e, \
            k_a1, k_a2, k_a3, k_b1, k_b2, k_b3, t_maxN, k_N, V_N, p, S_N, \
            M_g, BW, N_b, dt = tuple(model_paras.values())
        model = do_mpc.model.Model(model_type='continuous')
        # set model states
        Q_1 = model.set_variable(var_type='_x', var_name='Q_1')
        Q_2 = model.set_variable(var_type='_x', var_name='Q_2')
        x_1 = model.set_variable(var_type='_x', var_name='x_1')
        x_2 = model.set_variable(var_type='_x', var_name='x_2')
        x_3 = model.set_variable(var_type='_x', var_name='x_3')
        S_1 = model.set_variable(var_type='_x', var_name='S_1')
        S_2 = model.set_variable(var_type='_x', var_name='S_2')
        I = model.set_variable(var_type='_x', var_name='I')
        Z_1 = model.set_variable(var_type='_x', var_name='Z_1')
        Z_2 = model.set_variable(var_type='_x', var_name='Z_2')
        N = model.set_variable(var_type='_x', var_name='N')
        Y = model.set_variable(var_type='_x', var_name='Y')
        # set inputs
        u_I = model.set_variable(var_type='_u', var_name='u_I')
        u_N = model.set_variable(var_type='_u', var_name='u_N')
        # Set disturbance
        U_G = model.set_variable(var_type='_tvp', var_name='U_G')
        # RHS
        G = Q_1 / V_G
        F_01c = ca.if_else(G >= 4.5, F_01, F_01 * G / 4.5)
        F_R = ca.if_else(G >= 9, 0.003 * (G - 9) * V_G, 0)
        dQ_1 = -F_01c - x_1 * Q_1 + k_12 * Q_2 - F_R + EGP_0 * (1 - x_3) + 1e3 / (M_g * BW) * U_G + Y * Q_1
        dQ_2 = x_1 * Q_1 - (k_12 + x_2) * Q_2
        dx_1 = -k_a1 * x_1 + k_b1 * I
        dx_2 = -k_a2 * x_2 + k_b2 * I
        dx_3 = -k_a3 * x_3 + k_b3 * I
        dS_1 = (u_I + self.u_basal[0]) - S_1 / t_maxI
        dS_2 = S_1 / t_maxI - S_2 / t_maxI
        dI = S_2 / (V_I * t_maxI) - k_e * I
        dZ_1 = (u_N * 1e-6 + self.u_basal[1]) - Z_1 / t_maxN
        dZ_2 = Z_1 / t_maxN - Z_2 / t_maxN
        dN = -k_N * (N - N_b) + Z_2 / (V_N * t_maxN)
        dY = -p * Y + p * S_N * (N - N_b)
        model.set_rhs('Q_1', dQ_1)
        model.set_rhs('Q_2', dQ_2)
        model.set_rhs('x_1', dx_1)
        model.set_rhs('x_2', dx_2)
        model.set_rhs('x_3', dx_3)
        model.set_rhs('S_1', dS_1)
        model.set_rhs('S_2', dS_2)
        model.set_rhs('I', dI)
        model.set_rhs('Z_1', dZ_1)
        model.set_rhs('Z_2', dZ_2)
        model.set_rhs('N', dN)
        model.set_rhs('Y', dY)
        model.setup()
        simulator = do_mpc.simulator.Simulator(model)
        simulator.set_param(t_step=dt)
        tvp_template_sim = simulator.get_tvp_template()

        def tvp_fun_sim(t_now):
            t = t_now
            tvp_template_sim['U_G'] = D_G * A_g / (t_maxG ** 2) * t * ca.exp(-t / t_maxG)
            return tvp_template_sim

        simulator.set_tvp_fun(tvp_fun=tvp_fun_sim)
        simulator.setup()
        self.simulator = simulator
        self.action_bias = 0.5 * (self.action_high + self.action_low)
        self.action_scale = 0.5 * (self.action_high - self.action_low)


    def reset(self, seed: Optional[int] = 0, options=None) -> Union[np.ndarray, Dict]:
        super().reset(seed=seed)
        # initial states
        self.time_step = 0
        self.simulator.reset_history()
        self.simulator.x0 = np.copy(self.init_state).reshape(-1, 1)
        self.state = np.copy(self.init_state)

        glucose = self.init_state[0] / self.V_G * 18
        self.obs_history.reset(glucose)
        obs = self.obs_history.get_obs()

        self.traj_obs = [glucose]
        self.traj_act = []
        return obs, {"state": self.state, "meas": np.array([glucose])}

    def step(self, action: np.ndarray) -> Union[np.ndarray, float, bool, bool, Dict]:
        self.time_step += 1
        u = action * self.action_scale + self.action_bias  # scale and shift action
        u = u.reshape(-1, 1)  # ensure u is a column vector
        state_next = self.simulator.make_step(u)
        self.state = state_next.reshape((-1,))
        G = self.state[0] / self.V_G * 18

        term = (G < 10) or (G > 1000)
        # Magni risk function https://proceedings.mlr.press/v126/fox20a/fox20a.pdf
        if term:
            reward = -1e3  # very large negative reward might cause collapse of entropy
        else:
            reward = -1 * (3.35506 * (math.log(G) ** 0.8353 - 3.7932)) ** 2

        if u[0] >= 0.1:
            reward -= self.action_penalty

        trun = True if self.time_step >= self.max_steps else False

        self.obs_history.add_his(G)
        obs = self.obs_history.get_obs()
        self.traj_obs.append(obs)
        self.traj_act.append(u)
        # if self.render_mode == 'human' and (term or trun):
        #     self.render()
        return obs, reward, term, trun, {"state": self.state, "meas": np.array([G])}

    def vis_trajectory(self):
        traj_obs = np.array(self.traj_obs[1:])
        traj_act = np.array(self.traj_act)
        num_steps = len(traj_obs)

        t = np.arange(0, self.dt * (num_steps - 1) + 0.001, self.dt)
        t_c = np.arange(0, self.dt * (num_steps - 2) + 0.001, self.dt / 100.0)

        fig, axs = plt.subplots(2, 1, sharex=True)

        axs[0].plot(t, traj_obs[:, 0], label='$G$')
        axs[0].plot(t, traj_obs[:, 1], label='$G^\prime - G$')
        axs[0].plot(t, traj_obs[:, 2], label='T')
        axs[0].set_ylabel('Observations')
        axs[0].legend(loc='upper right')
        actions_labels = ['Insulin', 'Glucagon']
        for i in range(traj_act.shape[1]):
            traj_act_i = traj_act[:, i]
            traj_act_i = traj_act_i[(t_c // self.dt).astype(int)]
            if i == 1:
                axs[1].twinx()
            axs[1].plot(t_c, traj_act_i, label=actions_labels[i])
        axs[1].legend(loc='upper right')
        plt.suptitle('BiGlucose Simulation')
        fig.tight_layout()
        return fig

    def render(self):
        _ = self.vis_trajectory()
        plt.show()

    def _solve_steady_state(self):
        # a + b*k + c*k/(d+e*k) = 0
        a = self.EGP_0 - self.F_01
        b = -((self.EGP_0 * self.k_b3) / (self.k_a3 * self.k_e * self.V_I) + \
              (7.7 * self.V_G * self.k_b1) / (self.k_a1 * self.k_e * self.V_I))
        c = 7.7 * self.V_G * self.k_b1 * self.k_a2 * self.k_12
        d = self.k_a1 * self.k_a2 * self.k_12 * self.k_e * self.V_I
        e = self.k_b2 * self.k_a1
        # Ak^2 + Bk + C = 0
        A = b * e
        B = (a * e + b * d + c)
        C = a * d
        k1, k2 = (-B + np.sqrt(B ** 2 - 4 * A * C)) / (2 * A), (-B - np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
        k = max(k1, k2)

        if k < 0:
            raise ValueError("Initial state unsolvable for the BiGlucose parameters")

        Q_1 = 7.7 * self.V_G
        Q_2 = (7.7 * self.V_G * self.k_b1 * self.k_a2 * k) / \
              (self.k_a1 * self.k_a2 * self.k_12 * self.k_e * self.V_I + self.k_b2 * self.k_a1 * k)
        x_1 = self.k_b1 / (self.k_a1 * self.V_I * self.k_e) * k
        x_2 = self.k_b2 / (self.k_a2 * self.V_I * self.k_e) * k
        x_3 = self.k_b3 / (self.k_a3 * self.V_I * self.k_e) * k
        S_1 = self.t_maxI * k
        S_2 = self.t_maxI * k
        I = k / (self.V_I * self.k_e)

        x_steady_state = np.array([Q_1, Q_2, x_1, x_2, x_3, S_1, S_2, I, 0., 0., self.N_b, 0.],
                                  dtype=np.float32)
        u_steady_state = np.array([k, 0.],
                                  dtype=np.float32)

        return x_steady_state, u_steady_state