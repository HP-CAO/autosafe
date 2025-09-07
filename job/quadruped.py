import os
import sys
import hydra
import dataclasses

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from job.job_config import JobConfig
from agent.OffPolicyRL.SAC import SACConfig
from agent.utils import seed_everything, setup_environment, setup_logger
from tensorboardX import SummaryWriter

from gym_env.quad_gym.robot.unitree_a1 import a1 # A1 robot
from gym_env.quad_gym.simulator.setup_pybullet import setup_pybullet
from gym_env.quad_gym.robot.locomotion_controller_mpc import LocomotionController

logger = setup_logger()


@dataclasses.dataclass
class ControlConfig:
    frequency: int = 200  # hz
    initialize_from_cloud: bool = False
    action_magnitude: list = dataclasses.field(default_factory=lambda: [ 4, 4, 2, 8, 8, 4 ])

@dataclasses.dataclass
class Config():
    SACParams: SACConfig = dataclasses.field(default_factory=SACConfig) # you can choose to use another agent
    ControlParams: ControlConfig = dataclasses.field(default_factory=ControlConfig)
    JobParams: JobConfig = dataclasses.field(default_factory=JobConfig)


@hydra.main(version_base=None, config_path="../config/QuadrupedConfig", config_name='base_config')
def run(cfg: Config) -> None:
    from omegaconf import OmegaConf
    OmegaConf.set_struct(cfg, False)
    cfg.JobParams = cfg.edge["JobParams"]
    setup_environment(cfg)
    cfg.JobParams.output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    seed_everything(cfg.JobParams.seed)
    writer = SummaryWriter(os.path.join(cfg.JobParams.output_path, cfg.JobParams.job_name))
    observation_dim = 12
    shape_action = 6

    envs_cfg = cfg.envs.simulator
    robot_cfg = cfg.envs.robot
    cfg.SACParams = cfg.edge["SACParams"]
    cfg.GymParams = cfg.gym["GymParams"]
    p = setup_pybullet(envs_cfg)
    cfg.GymParams.TaskParams.max_episode_steps = 2000

    from agent.OffPolicyRL.SAC import SACAgent
    from agent.OffPolicyRL.SAC_AutoSafe import AutoSafeSAC
    from agent.model_based.model_based_design_quadruped import MATRIX_P, ModelbasedAgent
    from agent.OnPolicyRL.PPO_AutoSafe import AutoSafePPO
    from agent.OnPolicyRL.PPO import PPOAgent

    env = a1.A1(
        pybullet_client=p,
        cmd_params=robot_cfg.command,
        a1_params=robot_cfg.interface,
        gait_params=robot_cfg.gait_scheduler,
        swing_params=robot_cfg.swing_controller,
        stance_params=robot_cfg.stance_controller,
        motor_params=robot_cfg.motor_group,
        vel_estimator_params=robot_cfg.com_velocity_estimator,
    )

    mpc_controller = LocomotionController(
        robot=env,
        desired_speed=(robot_cfg.command.desired_vx, robot_cfg.command.desired_vy),
        desired_twisting_speed=robot_cfg.command.desired_wz,
        desired_com_height=robot_cfg.command.mpc_body_height,
        mpc_body_mass=robot_cfg.command.mpc_body_mass,
        mpc_body_inertia=robot_cfg.command.mpc_body_inertia,
        gait_config=robot_cfg.gait_scheduler,
        swing_config=robot_cfg.swing_controller,
        stance_config=robot_cfg.stance_controller,
        vel_estimator_config=robot_cfg.com_velocity_estimator,
    )

    env.mpc_controller=mpc_controller
    # robot.reset()

    if cfg.JobParams.experiment_name == 'sac_base':
        from job.experiments.off_policy_base import train
        n_s = MATRIX_P.shape[0]
        # ensure having same information in input by including the tracking error as the other baselines
        agent = SACAgent(cfg.SACParams, observation_dim + n_s, shape_action)
        train(agent, env, writer, cfg)

    elif cfg.JobParams.experiment_name == 'ppo_base':
        pass

    elif cfg.JobParams.experiment_name == 'sac_autosafe':
        from job.experiments.off_policy_autosafe import train
        n_s = MATRIX_P.shape[0]
        agent = AutoSafeSAC(cfg.SACParams, observation_dim + n_s, shape_action, MATRIX_P, F, case='quadruped')
        train(agent, env, writer, cfg)

    elif cfg.JobParams.experiment_name == 'ppo_autosafe':
        pass

    elif cfg.JobParams.experiment_name == 'model_based':
        from job.experiments.model_based import evaluate
        evaluate(ModelbasedAgent(), env, writer, cfg, 0)
    elif cfg.JobParams.experiment_name == 'sac_residual':

        from job.experiments.off_policy_residual import train
        n_s = MATRIX_P.shape[0]
        rl_agent = SACAgent(cfg.SACParams, observation_dim + n_s, shape_action)
        safe_agent = ModelbasedAgent()
        train(rl_agent, safe_agent, env, writer, cfg)

    elif cfg.JobParams.experiment_name == 'sac_simplex':
        from job.experiments.off_policy_simplex import train
        observation_dim = env.observation_space.shape[0]
        n_s = MATRIX_P.shape[0]
        rl_agent = SACAgent(cfg.SACParams, observation_dim + n_s, shape_action)
        safe_agent = ModelbasedAgent()
        train(rl_agent, safe_agent, env, writer, cfg)
    elif cfg.JobParams.experiment_name == 'sac_lyapunov':
        from job.experiments.off_policy_lyapunov import train
        observation_dim = env.observation_space.shape[0]
        n_s = MATRIX_P.shape[0]
        rl_agent = SACAgent(cfg.SACParams, observation_dim + n_s, shape_action)
        safe_agent = ModelbasedAgent()
        train(rl_agent, safe_agent, env, writer, cfg)
    elif cfg.JobParams.experiment_name == 'sac_lag':
        from job.experiments.off_policy_lagrangian import train
        from agent.OffPolicyRL.SAC_Lagrangian import SACAgentLag
        observation_dim = env.observation_space.shape[0]
        n_s = MATRIX_P.shape[0]
        rl_agent = SACAgentLag(cfg.SACParams, observation_dim + n_s, env.action_space.shape[0], lag_factor=0.5)
        safe_agent = ModelbasedAgent()
        train(rl_agent, safe_agent, env, writer, cfg)
    elif cfg.JobParams.experiment_name == 'sac_lam':
        from job.experiments.off_policy_lam import train
        from agent.OffPolicyRL.SAC_Lam import SACAgentLam
        observation_dim = env.observation_space.shape[0]
        n_s = MATRIX_P.shape[0]
        safe_agent = ModelbasedAgent()
        rl_agent = SACAgentLam(cfg.SACParams, observation_dim + n_s, env.action_space.shape[0], safe_agent=safe_agent)
        train(rl_agent, safe_agent, env, writer, cfg)
    else:
        print("Experiment name not recognized. Please check the configuration.")

if __name__ == '__main__':
    run()