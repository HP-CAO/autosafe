import os
import sys
import hydra
import dataclasses

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_env.cartpole.cart_pole import CartPoleEnv, CartPoleGymConfig
from job.job_config import JobConfig
from agent.OffPolicyRL.SAC_core import SACAgent, SACConfig
from agent.utils import seed_everything, setup_environment, setup_logger
from tensorboardX import SummaryWriter

logger = setup_logger()

@dataclasses.dataclass
class AllConfig:
    GymParams: CartPoleGymConfig = dataclasses.field(default_factory=CartPoleGymConfig)
    AgentParams: SACConfig = dataclasses.field(default_factory=SACConfig)
    JobParams: JobConfig = dataclasses.field(default_factory=JobConfig)


@hydra.main(version_base=None, config_path="../config", config_name='CartpoleBase')
def run(cfg: AllConfig) -> None:
    setup_environment(cfg)
    cfg.JobParams.output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    seed_everything(cfg.JobParams.seed)
    writer = SummaryWriter(os.path.join(cfg.JobParams.output_path, cfg.JobParams.job_name))
    env = CartPoleEnv(cfg.GymParams)

    if cfg.JobParams.experiment_name == 'base':
        from job.experiments.base import train
        observation_dim = env.observation_space.shape[0]
        agent = SACAgent(cfg.AgentParams, observation_dim, env.action_space.shape[0])
        train(agent, env, writer, cfg)
    elif cfg.JobParams.experiment_name == 'autosafe':
        from job.experiments.auto_safe import train
        from agent.OffPolicyRL.SAC_AutoSafe import AutoSafeSAC
        from job.utils.model_based_design_cartpole import MATRIX_P, F
        observation_dim = env.observation_space.shape[0]
        n_s = MATRIX_P.shape[0]
        agent = AutoSafeSAC(cfg.AgentParams, observation_dim + n_s, env.action_space.shape[0], MATRIX_P, F)
        train(agent, env, writer, cfg)
    elif cfg.JobParams.experiment_name == 'autosafe_vary': #todo for testing
        from job.experiments.auto_safe_vary import train
        from agent.OffPolicyRL.SAC_AutoSafe_Vary import AutoSafeSAC
        from job.utils.model_based_design_cartpole import MATRIX_P, F
        observation_dim = env.observation_space.shape[0]
        n_s = MATRIX_P.shape[0]
        agent = AutoSafeSAC(cfg.AgentParams, observation_dim + n_s, env.action_space.shape[0], MATRIX_P, F)
        train(agent, env, writer, cfg)
    elif cfg.JobParams.experiment_name == 'safe':
        from job.experiments.model_based_eval import evaluate
        from job.utils.model_based_design_cartpole import ModelbasedAgent
        evaluate(ModelbasedAgent(), env, writer, cfg, 0)
    else:
        print("Experiment name not recognized. Please check the configuration.")




if __name__ == '__main__':
    run()