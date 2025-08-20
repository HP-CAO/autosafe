import os
import sys
import hydra
import dataclasses

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_env.quadrotor.quadrotor import Quadrotor, QuadrotorGymConfig
from job.job_config import JobConfig
from agent.OffPolicyRL.SAC import SACAgent, SACConfig
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

    if cfg.JobParams.experiment_name == 'sac_base':
        from job.experiments.off_policy_base import train
        observation_dim = env.observation_space.shape[0]
        agent = SACAgent(cfg.SACParams, observation_dim, env.action_space.shape[0])
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
        from agent.OffPolicyRL.SAC_AutoSafe import AutoSafeSAC
        from job.utils.model_based_design_quadrotor import MATRIX_P, F
        observation_dim = env.observation_space.shape[0]
        n_s = MATRIX_P.shape[0]
        agent = AutoSafeSAC(cfg.SACParams, observation_dim + n_s,
                            env.action_space.shape[0], MATRIX_P, F, case='quadrotor')
        train(agent, env, writer, cfg)

    elif cfg.JobParams.experiment_name == 'ppo_autosafe':
        pass

    elif cfg.JobParams.experiment_name == 'model_based':
        from job.experiments.model_based import evaluate
        from job.utils.model_based_design_quadrotor import ModelbasedAgent
        evaluate(ModelbasedAgent(), env, writer, cfg, 0)
    else:
        print("Experiment name not recognized. Please check the configuration.")
if __name__ == '__main__':
    run()