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
import datetime

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
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{cfg.JobParams.env_name}/{cfg.JobParams.job_name}/seed_{cfg.JobParams.seed}/{timestamp}"
    cfg.JobParams.output_path = os.path.join("outputs", run_name)
    seed_everything(cfg.JobParams.seed)
    writer = SummaryWriter(cfg.JobParams.output_path)

    observation_dim = 18
    shape_action = 6

    envs_cfg = cfg.envs.simulator
    robot_cfg = cfg.envs.robot
    cfg.SACParams = cfg.edge["SACParams"]
    cfg.GymParams = cfg.gym["GymParams"]
    cfg.GymParams.TaskParams.max_episode_steps = 500

    p = setup_pybullet(envs_cfg, enable_gui=False)

    from agent.OffPolicyRL.SAC import SACAgent
    from agent.OffPolicyRL.SAC_AutoSafe import AutoSafeSAC
    from agent.model_based.model_based_design_quadruped import MATRIX_P, ModelbasedAgent, F, ACTION_BOUND

    env = a1.A1(
        pybullet_client=p,
        cmd_params=robot_cfg.command,
        a1_params=robot_cfg.interface,
        gait_params=robot_cfg.gait_scheduler,
        swing_params=robot_cfg.swing_controller,
        stance_params=robot_cfg.stance_controller,
        motor_params=robot_cfg.motor_group,
        vel_estimator_params=robot_cfg.com_velocity_estimator,
        action_bound=ACTION_BOUND
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
        agent = SACAgent(cfg.SACParams, observation_dim + n_s,shape_action)
        train(agent, env, writer, cfg)

    elif cfg.JobParams.experiment_name == 'sac_autosafe':
        from job.experiments.off_policy_autosafe import train
        n_s = MATRIX_P.shape[0]
        agent = AutoSafeSAC(cfg.SACParams, observation_dim + n_s, shape_action, MATRIX_P, F, lam_mode=cfg.SACParams.autosafe_lam_mode)
        train(agent, env, writer, cfg)
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
        
        n_s = MATRIX_P.shape[0]
        rl_agent = SACAgent(cfg.SACParams, observation_dim + n_s, shape_action)
        safe_agent = ModelbasedAgent()
        train(rl_agent, safe_agent, env, writer, cfg)
    elif cfg.JobParams.experiment_name == 'sac_lyapunov':
        from job.experiments.off_policy_lyapunov import train
        
        n_s = MATRIX_P.shape[0]
        rl_agent = SACAgent(cfg.SACParams, observation_dim + n_s, shape_action)
        safe_agent = ModelbasedAgent()
        train(rl_agent, safe_agent, env, writer, cfg)
    elif cfg.JobParams.experiment_name == 'sac_lag':
        from job.experiments.off_policy_lagrangian import train
        from agent.OffPolicyRL.SAC_Lagrangian import SACAgentLag
        
        n_s = MATRIX_P.shape[0]
        rl_agent = SACAgentLag(cfg.SACParams, observation_dim + n_s, shape_action, lag_factor=0.5)
        safe_agent = ModelbasedAgent()
        train(rl_agent, safe_agent, env, writer, cfg)
    elif cfg.JobParams.experiment_name == 'sac_lam':
        from job.experiments.off_policy_lam import train
        from agent.OffPolicyRL.SAC_Lam import SACAgentLam
        
        n_s = MATRIX_P.shape[0]
        safe_agent = ModelbasedAgent()
        # we need to include the value of lam to make the mdp stationary
        rl_agent = SACAgentLam(cfg.SACParams, observation_dim + n_s + 1, shape_action, safe_agent=safe_agent)
        train(rl_agent, safe_agent, env, writer, cfg, mode=cfg.SACParams.sac_lam_mode)
    else:
        print("Experiment name not recognized. Please check the configuration.")

if __name__ == '__main__':
    run()