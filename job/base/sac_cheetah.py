import os
import sys
import hydra
import random
import numpy as np
import dataclasses
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gym_env.mujoco.MujocoHalfcheetah import HalfCheetah, HalfCheetahGymConfig
from job.job_config import JobConfig
from agent.OffPolicyRL.archived.SAC import SACAgent, SACConfig
from hydra.core.config_store import ConfigStore
from tensorboardX import SummaryWriter
from agent.OffPolicyRL.AgentEnvInt import train_agent, collect_data, eval_agent


@dataclasses.dataclass
class AllConfig:
    GymParams: HalfCheetahGymConfig = dataclasses.field(default_factory=HalfCheetahGymConfig)
    AgentParams: SACConfig = dataclasses.field(default_factory=SACConfig)
    JobParams: JobConfig = dataclasses.field(default_factory=JobConfig)


@hydra.main(version_base=None, config_path="../../config", config_name='SACCheetah')
def run(cfg: AllConfig) -> None:
    if not cfg.JobParams.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        physical_devices = tf.config.list_physical_devices('GPU')
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            exit("GPU allocated failed")

    cfg.JobParams.output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    writer = SummaryWriter(os.path.join(cfg.JobParams.output_path, cfg.JobParams.job_name))

    env = HalfCheetah(cfg.GymParams)
    agent = SACAgent(cfg.AgentParams, env.observation_space.shape[0], env.action_space.shape[0])

    random.seed(cfg.JobParams.seed)
    np.random.seed(cfg.JobParams.seed)

    if cfg.JobParams.run_mode == 'train':
        train_agent(agent, env, writer, cfg)
    elif cfg.JobParams.run_mode == 'collect':
        collect_data(agent, env, cfg)
    else:
        eval_agent(agent, env, cfg)


if __name__ == '__main__':
    run()
