from agent.OffPolicyRL.ReplayBuffer import ReplayMemory
from agent.utils import log_info, setup_logger
from tqdm import tqdm
import numpy as np
logger = setup_logger()

def train(agent, env, writer, cfg):
    training_info = {}
    global_step = 0
    replay_mem = ReplayMemory(size=cfg.SACParams.replay_buffer_size)
    progress = tqdm(total=cfg.SACParams.total_training_steps)
    evaluate_flag = False

    while global_step < cfg.SACParams.total_training_steps:
        obs, _ = env.reset(seed=cfg.JobParams.seed, options={"mode": "train"})
        episodic_return = []

        for _ in range(cfg.GymParams.TaskParams.max_episode_steps):

            action = agent.get_action(obs, mode='train').squeeze()
            next_obs, rewards, terminations, truncations, infos = env.step(action)
            replay_mem.add([obs, action, rewards, next_obs, terminations])

            obs = next_obs
            episodic_return.append(rewards)
            global_step += 1

            if global_step > cfg.SACParams.learning_starts:
                mini_batch = replay_mem.sample(cfg.SACParams.batch_size)
                agent_info = agent.optimize(mini_batch)
                log_info(writer, global_step, agent_info, 'agent', period=100)

            if global_step % cfg.GymParams.TaskParams.evaluation_period == 0:
                evaluate_flag = True

            if global_step % 50000 == 0:
                agent.save_weights(cfg.JobParams.output_path + '/agent_model/')

            if terminations:
                break

        if evaluate_flag:
            evaluate(agent, env, writer, cfg, global_step)
            evaluate_flag = False

        training_info['episodic_return'] = sum(episodic_return)
        training_info['episodic_length'] = len(episodic_return)
        training_info['episodic_ave_return'] = np.mean(episodic_return)
        progress.update(training_info['episodic_length'])
        progress.set_description(f"Step: {global_step}, Return: {training_info['episodic_return']:.2f}")
        writer.add_scalar("training/episodic_return", training_info['episodic_return'], global_step)
        writer.add_scalar("training/episodic_length", training_info['episodic_length'], global_step)
        writer.add_scalar("training/episodic_ave_return", training_info['episodic_ave_return'], global_step)


def evaluate(agent, env, writer, cfg, global_step):
    logger.info("Start evaluating the agent...")
    multi_rollout_return = []
    num_rollouts = 5
    all_steps = 0
    for episode in range(cfg.GymParams.TaskParams.num_episodes_to_run):
        accumulated_r = []

        for _ in range(num_rollouts):
            obs, c_gt = env.reset(seed=cfg.JobParams.seed, options={"mode": "train"})
            episodic_return = []

            for _ in range(cfg.GymParams.TaskParams.max_episode_steps):

                action = agent.get_action(obs, mode='test').squeeze()
                next_obs, r, terminations, truncations, infos = env.step(action)
                obs = next_obs
                episodic_return.append(r)
                all_steps += 1

                if terminations:
                    break

            accumulated_r.append(sum(episodic_return))

        multi_rollout_return.append(np.mean(accumulated_r))

    return_mean = np.mean(multi_rollout_return)
    return_std = np.std(multi_rollout_return)
    normalized_return = return_mean / all_steps
    returns_cv = return_std / return_mean  # coefficient of variation (dimensionless)
    writer.add_scalar("evaluation/episodic_return", return_mean, global_step)
    writer.add_scalar("evaluation/episodic_ave_return", normalized_return, global_step)
    writer.add_scalar("evaluation/episodic_cv", returns_cv, global_step)