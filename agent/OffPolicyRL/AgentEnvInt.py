"""
This file defines the basic interaction pipeline between the agent and the environment.
"""

import logging
import os
import time

import numpy as np

from agent.OffPolicyRL.archived.ReplayBuffer import ReplayMemory
from tqdm import tqdm


def train_agent(agent, env, writer, cfg):
    logging.info("Start Training Process")
    best_performance = -100
    global_step = 0
    evaluate = True
    replay_mem = ReplayMemory(size=cfg.AgentParams.replay_buffer_size)
    progress = tqdm(total=cfg.AgentParams.total_training_steps)

    while global_step < cfg.AgentParams.total_training_steps:

        obs, _ = env.reset(seed=cfg.JobParams.seed, options={"mode": "train"})  # reset env after evaluation

        reward_list = []

        for _ in range(cfg.GymParams.TaskParams.max_episode_steps):
            if global_step < cfg.AgentParams.learning_starts:
                action = env.action_space.sample().squeeze()
            else:
                action = agent.get_action(obs, mode='train').squeeze()

            next_obs, rewards, terminations, _, infos = env.step(action)
            real_next_obs = next_obs.copy()
            replay_mem.add((obs, action, rewards, real_next_obs, terminations))
            obs = next_obs
            reward_list.append(rewards)
            global_step += 1

            if global_step > cfg.AgentParams.learning_starts:
                minibatch = replay_mem.sample(cfg.AgentParams.batch_size)
                training_info = agent.optimize(minibatch)

                if global_step % 100 == 0:
                    for key, value in training_info.items():
                        writer.add_scalar(f"training/{key}", value, global_step)

            if global_step % cfg.GymParams.TaskParams.evaluation_period == 0:
                evaluate = True

            if terminations:
                break

        episodic_return = np.sum(reward_list)
        episodic_length = len(reward_list)
        progress.set_description(f"Global_step={global_step}, episodic_return={episodic_return:.2f}")
        writer.add_scalar("training/episodic_return", episodic_return, global_step)
        writer.add_scalar("training/episodic_step", episodic_length, global_step)
        progress.update(episodic_length)

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
    for _ in range(cfg.GymParams.TaskParams.num_episodes_to_run):
        obs, _ = env.reset(seed=cfg.JobParams.seed, options={"reset_state": cfg.GymParams.TaskParams.ini_states})
        for _ in range(cfg.GymParams.TaskParams.max_episode_steps):
            action = agent.get_action(obs, mode='test').squeeze()
            next_obs, rewards, terminations, truncations, infos = env.step(action)
            performance_score = env.get_performance_score()
            obs = next_obs
            performance_score_list.append(performance_score)
            if terminations or truncations:
                break
    # The performance evaluation metric should be implemented based on the task objective.
    performance = np.sum(performance_score_list)
    return performance


def collect_data(agent, env, cfg):
    import pickle
    traj_save_folder = cfg.JobParams.output_path + '/trajectories'
    os.makedirs(traj_save_folder, exist_ok=True)
    save_path = traj_save_folder + f"/traj_{cfg.JobParams.job_name}.pkl"
    trajectories_list = []

    for i in range(cfg.GymParams.TaskParams.num_episodes_to_run):
        obs, _ = env.reset(seed=cfg.JobParams.seed, options={"reset_state": cfg.GymParams.TaskParams.ini_states,
                                                             "mode": "eval"})
        observation_list = []
        action_list = []
        for _ in range(cfg.GymParams.TaskParams.max_episode_steps):
            action = agent.get_action(obs, mode='test')
            next_obs, rewards, terminations, truncations, infos = env.step(action)
            observation_list.append(obs)
            action_list.append(action)
            obs = next_obs
            observation_list.append(obs)
            action_list.append(action)
            if terminations or truncations:
                break

        trajectories_list.append(dict(observations=np.array(observation_list), actions=np.array(action_list)))

    with open(save_path, "wb") as f:
        pickle.dump(trajectories_list, f)


