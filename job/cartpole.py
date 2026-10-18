import os
import sys
import hydra
import dataclasses
import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from gym_env.cartpole.cart_pole import CartPoleEnv, CartPoleGymConfig
from job.job_config import JobConfig
from agent.OffPolicyRL.SAC import SACConfig
from agent.OnPolicyRL.PPO import PPOConfig, PPOAgent, RolloutBuffer
from agent.utils import seed_everything, setup_environment, setup_logger
from tensorboardX import SummaryWriter

logger = setup_logger()

@dataclasses.dataclass
class AllConfig:
    GymParams: CartPoleGymConfig = dataclasses.field(default_factory=CartPoleGymConfig)
    SACParams: SACConfig = dataclasses.field(default_factory=SACConfig)
    PPOParams: PPOConfig = dataclasses.field(default_factory=PPOConfig)
    JobParams: JobConfig = dataclasses.field(default_factory=JobConfig)


@hydra.main(version_base=None, config_path="../config", config_name='CartpoleBase')
def run(cfg: AllConfig) -> None:
    setup_environment(cfg)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{cfg.JobParams.env_name}/{cfg.JobParams.job_name}/seed_{cfg.JobParams.seed}/{timestamp}"
    cfg.JobParams.output_path = os.path.join("outputs", run_name)
    seed_everything(cfg.JobParams.seed)
    writer = SummaryWriter(cfg.JobParams.output_path)
    env = CartPoleEnv(cfg.GymParams)
    from agent.OffPolicyRL.SAC import SACAgent
    from agent.OffPolicyRL.SAC_AutoSafe import AutoSafeSAC
    from agent.OffPolicyRL.SAC_AutoSafe_Vary import AutoSafeSACVary
    from agent.model_based.model_based_design_cartpole import MATRIX_P, F, ModelbasedAgent

    if cfg.JobParams.experiment_name == 'sac_base':
        from job.experiments.off_policy_base import train
        observation_dim = env.observation_space.shape[0]
        n_s = MATRIX_P.shape[0]
        # ensure having same information in input by including the tracking error as the other baselines
        agent = SACAgent(cfg.SACParams, observation_dim + n_s,env.action_space.shape[0])
        train(agent, env, writer, cfg)

    elif cfg.JobParams.experiment_name == 'sac_autosafe':
        from job.experiments.off_policy_autosafe import train
        observation_dim = env.observation_space.shape[0]
        n_s = MATRIX_P.shape[0]
        agent = AutoSafeSAC(cfg.SACParams, observation_dim + n_s, env.action_space.shape[0], MATRIX_P, F,
                            lam_mode=cfg.SACParams.autosafe_lam_mode,
                            tem_min=1.0, tem_max=25.0, delta_min=cfg.SACParams.delta_min)
        train(agent, env, writer, cfg)
    elif cfg.JobParams.experiment_name == 'sac_autosafe_vary_cartpole':
        from job.experiments.off_policy_autosafe_vary_cartpole import train
        from gym_env.cartpole.cart_pole_vary import CartPoleEnvVarySafe
        env = CartPoleEnvVarySafe(cfg.GymParams)
        observation_dim = env.observation_space.shape[0] + 1 + 4 # include the varying parameter in the state
        n_s = MATRIX_P.shape[0]
        agent = AutoSafeSACVary(cfg.SACParams, observation_dim + n_s, env.action_space.shape[0], MATRIX_P, F,
                            lam_mode=cfg.SACParams.autosafe_lam_mode,
                            tem_min=1.0, tem_max=25.0, delta_min=cfg.SACParams.delta_min)
        train(agent, env, writer, cfg)

    elif cfg.JobParams.experiment_name == 'autosafe_traj_eval':
        from job.experiments.autosafe_eval_trajectory import evaluate_trajectory_autosafe
        observation_dim = env.observation_space.shape[0]
        n_s = MATRIX_P.shape[0]
        agent = AutoSafeSAC(cfg.SACParams, observation_dim + n_s, env.action_space.shape[0], MATRIX_P, F,
                            lam_mode=cfg.SACParams.autosafe_lam_mode,
                            tem_min=1.0, tem_max=25.0)
        if cfg.SACParams.model_path != "":
            a = agent.get_action(np.zeros((observation_dim + n_s,)), mode='test')  # build network
            agent.load_weights(model_path=cfg.SACParams.model_path)
        evaluate_trajectory_autosafe(agent, env, cfg, "cartpole")
    elif cfg.JobParams.experiment_name == 'model_based':
        from job.experiments.model_based import evaluate
        evaluate(ModelbasedAgent(), env, writer, cfg, 0)
    elif cfg.JobParams.experiment_name == 'sac_residual':
        from job.experiments.off_policy_residual import train
        observation_dim = env.observation_space.shape[0]
        n_s = MATRIX_P.shape[0]
        rl_agent = SACAgent(cfg.SACParams, observation_dim + n_s, env.action_space.shape[0])
        safe_agent = ModelbasedAgent()
        train(rl_agent, safe_agent, env, writer, cfg)
    elif cfg.JobParams.experiment_name == 'sac_simplex':
        from job.experiments.off_policy_simplex import train
        observation_dim = env.observation_space.shape[0]
        n_s = MATRIX_P.shape[0]
        rl_agent = SACAgent(cfg.SACParams, observation_dim + n_s, env.action_space.shape[0])
        safe_agent = ModelbasedAgent(delta_min=cfg.SACParams.delta_min)
        train(rl_agent, safe_agent, env, writer, cfg, safe_policy_steps=5)
    elif cfg.JobParams.experiment_name == 'simplex_traj_eval':
        from job.experiments.autosafe_eval_trajectory import evaluate_trajectory_simplex
        observation_dim = env.observation_space.shape[0]
        n_s = MATRIX_P.shape[0]
        rl_agent = SACAgent(cfg.SACParams, observation_dim + n_s, env.action_space.shape[0])
        safe_agent = ModelbasedAgent()
        if cfg.SACParams.model_path != "":
            a = rl_agent.get_action(np.zeros((observation_dim + n_s,)), mode='test')  # build network
            rl_agent.load_weights(model_path=cfg.SACParams.model_path)
        evaluate_trajectory_simplex(safe_agent, rl_agent, env, cfg, "cartpole")

    elif cfg.JobParams.experiment_name == 'sac_cbf':
        from job.experiments.off_policy_cbf import train
        from agent.model_based.model_based_design_cartpole import cbf_filter_dt
        observation_dim = env.observation_space.shape[0]
        n_s = MATRIX_P.shape[0]
        rl_agent = SACAgent(cfg.SACParams, observation_dim + n_s, env.action_space.shape[0])
        train(rl_agent, cbf_filter_dt, env, writer, cfg, safe_policy_steps=1)
    elif cfg.JobParams.experiment_name == 'sac_lyapunov':
        from job.experiments.off_policy_lyapunov import train
        observation_dim = env.observation_space.shape[0]
        n_s = MATRIX_P.shape[0]
        rl_agent = SACAgent(cfg.SACParams, observation_dim + n_s, env.action_space.shape[0])
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
        # we need to include the value of lam to make the mdp stationary
        rl_agent = SACAgentLam(cfg.SACParams, observation_dim + n_s + 1, env.action_space.shape[0], safe_agent=safe_agent)
        train(rl_agent, safe_agent, env, writer, cfg, mode=cfg.SACParams.sac_lam_mode)
    else:
        print("Experiment name not recognized. Please check the configuration.")



if __name__ == '__main__':
    run()