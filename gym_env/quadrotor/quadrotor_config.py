import dataclasses
from dataclasses import field
from typing import Optional


@dataclasses.dataclass
class RobotConfig:
    quad_type: str = '2d'  # 1d | 2d | 3d


@dataclasses.dataclass
class TaskConfig:
    task_type: str = 'stabilization'  # "traj_tracking" | "stabilization" The environment's task
    norm_act_scale: float = 0.1  # Scaling the [-1,1] action space around hover thrust when `normalized_action_space` is True.
    obs_goal_horizon: int = 0  # How many future goal states to append to obervation.
    rew_state_weight: float = 1.0  # Quadratic weights for state in rl reward
    rew_act_weight: float = 0.0001  # Quadratic weights for action in rl reward.
    rew_exponential: bool = True  # If to exponentiate negative quadratic cost to positive, bounded [0,1] reward.
    done_on_out_of_bound: bool = True  #If to termiante when state is out of bound.
    info_mse_metric_state_weight: Optional[dict] = None  # Quadratic weights for state in mse calculation for info dict.
    normalized_rl_action_space: bool = False
    # task
    task_info: dict = field(default_factory=lambda: ({"stabilization_goal": [2, 2, 2.5],
                                                      "stabilization_goal_tolerance": 0.05}))  # A dictionary with the information used to generate the task X and U references
    cost_fn_type: str = 'rl_reward'  # "rl_reward" | "quadratic Cost function choice for computing the reward in .step()
    ctrl_freq: int = 50  # The frequency at which the environment steps
    # episode_len_sec: int = 5  # maximum episode duration in seconds
    # initialization
    ini_states: dict \
        = field(default_factory=lambda: ({"init_x": 0, "init_x_dot": 0, "init_z": 0, "init_z_dot": 0, "init_theta": 0,
                                          "init_theta_dot": 0}))  # The initial state of the environment (z, z_dot) or (x, x_dot, z, z_dot theta, theta_dot)
    init_state_randomization_info: Optional[
        dict] = None  # A dictionary with information used to randomize the initial state.
    # domain_randomization
    prior_prop: Optional[dict] = None  # The prior inertial properties of the environment
    inertial_prop: Optional[dict] = None  # The inertial properties of the environment (M, Ixx, Iyy, Izz).
    randomized_inertial_prop: bool = False  # Whether to randomize the inertial properties
    inertial_prop_randomization_info: Optional[
        dict] = None  # A dictionary with information used to randomize the inert. prop.
    # Constraint
    constraints: Optional[dict] = None  # Dictionary to specify the constraints being used
    done_on_violation: bool = False  # Whether to return done==True on a constraint violation.
    use_constraint_penalty: bool = False  # If to use shaped reward to penalize potential constraint violation.
    constraint_penalty: float = -1.0  # Constraint penalty cost for reward shaping.
    # disturbance
    disturbances: Optional[dict] = None  # Dictionary to specify disturbances being used.
    adversary_disturbance: Optional[dict] = None  # If to use adversary/external disturbance.
    adversary_disturbance_offset: float = 0.0  # Parameterizes the offset of the adversary disturbance.
    adversary_disturbance_scale: float = 0.01  # Parameterizes magnitude of adversary disturbance.
    # interaction
    max_episode_steps: int = 500  # Maximum number of steps in an episode.
    evaluation_period: int = 10000  # Number of interaction steps before evaluation.
    num_episodes_to_run: int = 1  # umber of episodes to run in an evaluation.
    task_reset_mode: str = 'random'  # "random reset"
    change_dynamics: bool = False #
    context_horizon: int = 10  ## todo rename to be observation horizon


@dataclasses.dataclass
class SimulationConfig:
    record: bool = False  # Whether to save a video of the simulation in folder  `files/videos/`.
    gui: bool = False  # Whether to show PyBullet's GUI
    verbose: bool = False  # If to suppress environment print statetments.
    num_drones: int = 1  # The desired number of drones in the aviary.
    output_dir: Optional[str] = None  # output_dir (str, optional): Path to directory to save any env outputs.
    seed: Optional[int] = None  # seed (int, optional): Seed for the random number generato
    info_in_reset: bool = True  # Whether .reset() returns a dictionary with the environment's symbolic model.
    pyb_freq: int = 50  # The frequency at which PyBullet steps (a multiple of ctrl_freq)


@dataclasses.dataclass
class QuadrotorGymConfig:
    RobotParams: RobotConfig = dataclasses.field(default_factory=RobotConfig)
    TaskParams: TaskConfig = dataclasses.field(default_factory=TaskConfig)
    SimulationParams: SimulationConfig = dataclasses.field(default_factory=SimulationConfig)
