import os
import sys
import hydra
import dataclasses

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gym_env.quadrotor.quadrotor import Quadrotor, QuadrotorGymConfig
from job.job_config import JobConfig
from agent.OffPolicyRL.SAC_core import SACAgent, SACConfig
from agent.utils import seed_everything, setup_environment, setup_logger
from tensorboardX import SummaryWriter


logger = setup_logger()

@dataclasses.dataclass
class AllConfig:
    GymParams: QuadrotorGymConfig = dataclasses.field(default_factory=QuadrotorGymConfig)
    AgentParams: SACConfig = dataclasses.field(default_factory=SACConfig)
    JobParams: JobConfig = dataclasses.field(default_factory=JobConfig)


@hydra.main(version_base=None, config_path="../../config", config_name='SACQuadrotorBase')
def run(cfg: AllConfig) -> None:
    setup_environment(cfg)
    cfg.JobParams.output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    seed_everything(cfg.JobParams.seed)
    writer = SummaryWriter(os.path.join(cfg.JobParams.output_path, cfg.JobParams.job_name))
    env = Quadrotor(cfg.GymParams)

    if cfg.JobParams.experiment_name == 'base':
        from job.interactions.base import train
        observation_dim = env.observation_space.shape[0]
        agent = SACAgent(cfg.AgentParams, observation_dim, env.action_space.shape[0])
        train(agent, env, writer, cfg)
    elif cfg.JobParams.experiment_name == 'multi_task':
        from job.interactions.multi_task import train
        observation_dim = env.observation_space.shape[0] + len(env.control_goal)
        agent = SACAgent(cfg.AgentParams, observation_dim, env.action_space.shape[0])
        train(agent, env, writer, cfg)
    elif cfg.JobParams.experiment_name == 'model_based':
        from job.interactions.multi_task_model_based import evaluate
        from job.experiments.model_based_design_quadrotor import ModelbasedAgent
        evaluate(ModelbasedAgent(), env, writer, cfg, 0)
    elif cfg.JobParams.experiment_name == 'multi_task_reg':
        from job.interactions.multi_task_reg import train
        from job.experiments.model_based_design_quadrotor import ModelbasedAgent
        observation_dim = env.observation_space.shape[0] + len(env.control_goal)
        agent = SACAgent(cfg.AgentParams, observation_dim, env.action_space.shape[0])
        model_based_agent = ModelbasedAgent()
        train(agent, env, writer, cfg, model_based_agent, lamda=0.5)
    elif cfg.JobParams.experiment_name == 'multi_task_reg_apt':
        from job.interactions.multi_task_reg_adaptive import train
        from job.experiments.model_based_design_quadrotor import ModelbasedAgent, SYSTEM_A, SYSTEM_B
        observation_dim = env.observation_space.shape[0] + len(env.control_goal)
        agent = SACAgent(cfg.AgentParams, observation_dim, env.action_space.shape[0])
        model_based_agent = ModelbasedAgent()
        train(agent, env, writer, cfg, model_based_agent, SYSTEM_A, SYSTEM_B)




if __name__ == '__main__':
    run()