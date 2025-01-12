import logging
import os
import sys
import hydra
import random

import dataclasses
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tqdm import tqdm
from gym_env.cartpole.cart_pole import CartPoleEnv, CartPoleGymConfig
from job.job_config import JobConfig
from agent.OffPolicyRL.DDPG import DDPGAgent, DDPGConfig
from hydra.core.config_store import ConfigStore
from tensorboardX import SummaryWriter
from agent.OffPolicyRL.AgentEnvInt import collect_data
from agent.OffPolicyRL.archived.ReplayBuffer import ReplayMemory
from job.safe_rl.model_based_design import *

@dataclasses.dataclass
class AllConfig:
    GymParams: CartPoleGymConfig = dataclasses.field(default_factory=CartPoleGymConfig)
    AgentParams: DDPGConfig = dataclasses.field(default_factory=DDPGConfig)
    JobParams: JobConfig = dataclasses.field(default_factory=JobConfig)


@hydra.main(version_base=None, config_path="../../config", config_name='DDPG_Cartpole_Safe_Config')
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

    env = CartPoleEnv(cfg.GymParams)
    agent = DDPGAgent(cfg.AgentParams, env.observation_space.shape[0], env.action_space.shape[0])

    random.seed(cfg.JobParams.seed)
    np.random.seed(cfg.JobParams.seed)

    if cfg.JobParams.run_mode == 'train':
        train_agent(agent, env, writer, cfg)
    elif cfg.JobParams.run_mode == 'collect':
        collect_data(agent, env, cfg)
    else:
        eval_agent(agent, env, cfg)


def train_agent(agent, env, writer, cfg):
    logging.info("Start Training Process")
    best_performance = -100
    global_step = 0
    evaluate = True

    replay_mem = ReplayMemory(size=cfg.AgentParams.replay_buffer_size)
    progress = tqdm(total=cfg.AgentParams.total_training_steps)

    while global_step < cfg.AgentParams.total_training_steps:
        terminations = False
        obs, _ = env.reset(seed=cfg.JobParams.seed, options={
            'reset_state': get_init_condition_in_safety_envelope(1)[0]})  # reset env after evaluation

        reward_list = []

        for _ in range(cfg.GymParams.TaskParams.max_episode_steps):

            if global_step < cfg.AgentParams.learning_starts:
                action = env.action_space.sample().squeeze()
            else:
                action = agent.get_action(obs, mode='train').squeeze()

            next_obs, rewards, _, truncations, infos = env.step(action)

            if safety_violation_condition(env.state):
                terminations = True

            real_next_obs = next_obs.copy()
            replay_mem.add((obs, action, rewards, real_next_obs, terminations))
            obs = next_obs
            reward_list.append(rewards)
            global_step += 1

            if global_step > cfg.AgentParams.learning_starts:
                minibatch = replay_mem.sample(cfg.AgentParams.batch_size)
                critic_loss, q_value, actor_loss = agent.optimize(minibatch)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/q_value", q_value, global_step)
                    writer.add_scalar("losses/critic_loss", critic_loss, global_step)
                    writer.add_scalar("losses/actor_loss", actor_loss, global_step)

            if global_step % cfg.GymParams.TaskParams.evaluation_period == 0:
                evaluate = True

            if terminations or truncations:
                episodic_return = np.sum(reward_list)
                episodic_length = len(reward_list)
                progress.set_description(f"Global_step={global_step}, episodic_return={episodic_return:.2f}")
                writer.add_scalar("training/episodic_return", episodic_return, global_step)
                writer.add_scalar("training/episodic_step", episodic_length, global_step)
                writer.add_scalar("training/safety_violation", terminations * 1, global_step)
                progress.update(episodic_length)
                break

        if evaluate:
            performance = eval_agent(agent, env, cfg)
            if performance > best_performance:
                best_performance = performance
                agent.save_weights(cfg.JobParams.output_path + '/tf_model/best_model')
            writer.add_scalar("eval/performance", performance, global_step)
            progress.set_description(f"Global_step={global_step}, Performance={performance:.2f}")
            evaluate = False

    agent.save_weights(cfg.JobParams.output_path + '/tf_model/final_model')


def eval_agent(agent, env, cfg):
    performance_score_list = []
    terminations = False
    for _ in range(cfg.GymParams.TaskParams.num_episodes_to_run):
        obs, _ = env.reset(seed=cfg.JobParams.seed, options={
            'reset_state': get_init_condition_in_safety_envelope(1)[0]})  # reset env after evaluation
        for _ in range(cfg.GymParams.TaskParams.max_episode_steps):
            action = agent.get_action(obs, mode='test')
            next_obs, rewards, _, truncations, infos = env.step(action)

            if safety_violation_condition(env.state):
                terminations = True

            performance_score = env.get_performance_score(next_obs, cfg.GymParams.TaskParams.control_goal)
            obs = next_obs
            performance_score_list.append(performance_score)

            if terminations or truncations:
                break
    # The performance evaluation metric should be implemented based on the task objective.
    performance = np.sum(performance_score_list)
    return performance


if __name__ == '__main__':
    run()
