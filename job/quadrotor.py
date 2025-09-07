import os
import sys
import hydra
import dataclasses

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_env.quadrotor.quadrotor import Quadrotor, QuadrotorGymConfig
from job.job_config import JobConfig
from agent.OffPolicyRL.SAC import SACConfig
from agent.OnPolicyRL.PPO import PPOConfig, PPOAgent, RolloutBuffer
from agent.utils import seed_everything, setup_environment, setup_logger
from tensorboardX import SummaryWriter


logger = setup_logger()

@dataclasses.dataclass
class AllConfig:
    GymParams: QuadrotorGymConfig = dataclasses.field(default_factory=QuadrotorGymConfig)
    SACParams: SACConfig = dataclasses.field(default_factory=SACConfig)
    PPOParams: PPOConfig = dataclasses.field(default_factory=PPOConfig)
    JobParams: JobConfig = dataclasses.field(default_factory=JobConfig)


@hydra.main(version_base=None, config_path="../config", config_name='QuadrotorBase')
def run(cfg: AllConfig) -> None:
    setup_environment(cfg)
    cfg.JobParams.output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    seed_everything(cfg.JobParams.seed)
    writer = SummaryWriter(os.path.join(cfg.JobParams.output_path, cfg.JobParams.job_name))
    env = Quadrotor(cfg.GymParams)

    from agent.OffPolicyRL.SAC import SACAgent
    from agent.OffPolicyRL.SAC_AutoSafe import AutoSafeSAC
    from agent.model_based.model_based_design_quadrotor import MATRIX_P, F, ModelbasedAgent
    from agent.OnPolicyRL.PPO_AutoSafe import AutoSafePPO
    from agent.OnPolicyRL.PPO import PPOAgent

    if cfg.JobParams.experiment_name == 'sac_base':
        if cfg.JobParams.experiment_name == 'sac_base':
            from job.experiments.off_policy_base import train
            observation_dim = env.observation_space.shape[0]
            n_s = MATRIX_P.shape[0]
            # ensure having same information in input by including the tracking error as the other baselines
            agent = SACAgent(cfg.SACParams, observation_dim + n_s, env.action_space.shape[0])
            train(agent, env, writer, cfg)

    elif cfg.JobParams.experiment_name == 'ppo_base':
        from job.experiments.on_policy_base import train
        observation_dim = env.observation_space.shape[0]
        agent = PPOAgent(cfg.PPOParams, observation_dim, env.action_space.shape[0])
        buffer = RolloutBuffer(cfg.PPOParams.buffer_size, cfg.PPOParams.batch_size, env.observation_space.shape[0],
                               env.action_space.shape[0])
        train(agent, buffer, env, writer, cfg)

    elif cfg.JobParams.experiment_name == 'sac_autosafe':
        from job.experiments.off_policy_autosafe import train
        observation_dim = env.observation_space.shape[0]
        n_s = MATRIX_P.shape[0]
        agent = AutoSafeSAC(cfg.SACParams, observation_dim + n_s,
                            env.action_space.shape[0], MATRIX_P, F, case='quadrotor')
        train(agent, env, writer, cfg)
    elif cfg.JobParams.experiment_name == 'ppo_autosafe':
        pass
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
        safe_agent = ModelbasedAgent()
        train(rl_agent, safe_agent, env, writer, cfg)
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
    elif cfg.JobParams.experiment_name == 'sac_lam_opt':
        from job.experiments.off_policy_lam import train
        from agent.OffPolicyRL.SAC_Lam import SACAgentLam
        observation_dim = env.observation_space.shape[0]
        n_s = MATRIX_P.shape[0]
        safe_agent = ModelbasedAgent()
        # we need to include the value of lam to make the mdp stationary
        rl_agent = SACAgentLam(cfg.SACParams, observation_dim + n_s + 1, env.action_space.shape[0],
                               safe_agent=safe_agent)
        train(rl_agent, safe_agent, env, writer, cfg, mode='opt')
        train(rl_agent, safe_agent, env, writer, cfg, mode='linear')
    elif cfg.JobParams.experiment_name == 'sac_lam_linear':
        from job.experiments.off_policy_lam import train
        from agent.OffPolicyRL.SAC_Lam import SACAgentLam
        observation_dim = env.observation_space.shape[0]
        n_s = MATRIX_P.shape[0]
        safe_agent = ModelbasedAgent()
        # we need to include the value of lam to make the mdp stationary
        rl_agent = SACAgentLam(cfg.SACParams, observation_dim + n_s + 1, env.action_space.shape[0],
                               safe_agent=safe_agent)
        train(rl_agent, safe_agent, env, writer, cfg, mode='linear')
    else:
        print("Experiment name not recognized. Please check the configuration.")
if __name__ == '__main__':
    run()