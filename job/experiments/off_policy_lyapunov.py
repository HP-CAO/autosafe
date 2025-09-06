import copy

from agent.OffPolicyRL.ReplayBuffer import ReplayMemory
from agent.utils import log_info, setup_logger
from tqdm import tqdm
import numpy as np
logger = setup_logger()


def get_lyapunov_reward(tracking_error_pre, tracking_error_curr, lyapunov_matrix, beta=1.0, cap=5.0):
    cur_lyapunov_value = tracking_error_curr @ lyapunov_matrix @ tracking_error_curr.T
    prev_lyapunov_value = tracking_error_pre @ lyapunov_matrix @ tracking_error_pre.T
    r = np.clip(prev_lyapunov_value - cur_lyapunov_value, -cap, cap)  # positive if V decreased
    return beta * r # encouraging decrease


def train(rl_agent, safe_agent, env, writer, cfg):
    training_info = {}
    global_step = 0
    replay_mem = ReplayMemory(size=cfg.SACParams.replay_buffer_size)
    progress = tqdm(total=cfg.SACParams.total_training_steps)
    evaluate_flag = False

    while global_step < cfg.SACParams.total_training_steps:
        obs, _ = env.reset(seed=cfg.JobParams.seed, options={"mode": "train"})
        episodic_return = []
        return_shaped = []
        tracking_error = env.state - env.model_based_equilibrium
        obs_aug = np.append(obs, tracking_error)

        for _ in range(cfg.GymParams.TaskParams.max_episode_steps):

            action = rl_agent.get_action(obs_aug, mode='train').squeeze()

            next_obs, reward, terminations, truncations, infos = env.step(action)
            tracking_error_next = env.state - env.model_based_equilibrium

            lyapunov_reward = get_lyapunov_reward(tracking_error, tracking_error_next, safe_agent.matrix_P)
            r =  reward + lyapunov_reward

            next_obs_aug = np.append(next_obs, tracking_error_next)
            tracking_error = copy.deepcopy(tracking_error_next)

            replay_mem.add([obs_aug, action, r, next_obs_aug, terminations])

            obs_aug = copy.deepcopy(next_obs_aug)
            episodic_return.append(reward)  # we log the performance-based reward to be consistent with the other baselines
            return_shaped.append(r)
            global_step += 1

            if replay_mem.get_size() > cfg.SACParams.learning_starts:
                mini_batch = replay_mem.sample(cfg.SACParams.batch_size)
                agent_info = rl_agent.optimize(mini_batch)
                log_info(writer, global_step, agent_info, 'agent', period=100)

            if global_step % cfg.GymParams.TaskParams.evaluation_period == 0:
                evaluate_flag = True

            if global_step % 50000 == 0:
                rl_agent.save_weights(cfg.JobParams.output_path + '/agent_model/')

            if terminations:
                break

        if evaluate_flag:
            evaluate(rl_agent, safe_agent, env, writer, cfg, global_step)
            evaluate_flag = False

        training_info['episodic_return'] = sum(episodic_return)
        training_info['episodic_length'] = len(episodic_return)
        training_info['episodic_ave_return'] = np.mean(episodic_return)

        training_info['episodic_return_shape'] = sum(return_shaped)
        progress.update(training_info['episodic_length'])

        progress.set_description(f"Step: {global_step}, Return: {training_info['episodic_return']:.2f}")
        writer.add_scalar("training/episodic_return", training_info['episodic_return'], global_step)
        writer.add_scalar("training/episodic_length", training_info['episodic_length'], global_step)
        writer.add_scalar("training/episodic_return_shape", training_info['episodic_return_shape'], global_step)


def evaluate(rl_agent, safe_agent, env, writer, cfg, global_step):
    logger.info("Start evaluating the agent...")
    multi_rollout_return = []
    multi_rollout_performance = []
    num_rollouts = 5

    for episode in range(cfg.GymParams.TaskParams.num_episodes_to_run):
        accumulated_r = []
        accumulate_performance_score = []

        for _ in range(num_rollouts):
            obs, _ = env.reset(seed=cfg.JobParams.seed, options={"mode": "train"})
            episodic_return = []
            tracking_error = env.state - env.model_based_equilibrium
            obs_aug = np.append(obs, tracking_error)

            for _ in range(cfg.GymParams.TaskParams.max_episode_steps):

                action = rl_agent.get_action(obs_aug, mode='test').squeeze()
                next_obs, r, terminations, truncations, infos = env.step(action)
                performance_score = env.get_performance_score()
                accumulate_performance_score.append(performance_score)
                tracking_error = env.state - env.model_based_equilibrium
                next_obs_aug = np.append(next_obs, tracking_error)
                obs_aug = copy.deepcopy(next_obs_aug)
                # here for evaluation we use the performance-based reward without lyapunov reward,
                # to be consistent with other baselines
                episodic_return.append(r)

                if terminations:
                    break

            accumulated_r.append(sum(episodic_return))

        multi_rollout_return.append(np.mean(accumulated_r))
        multi_rollout_performance.append(np.mean(accumulate_performance_score))

    performance_mean = np.mean(multi_rollout_performance)
    performance_std = np.std(multi_rollout_performance)
    return_mean = np.mean(multi_rollout_return)
    return_std = np.std(multi_rollout_return)
    returns_cv = return_std / return_mean  # coefficient of variation (dimensionless)
    writer.add_scalar("evaluation/episodic_return", return_mean, global_step)
    writer.add_scalar("evaluation/episodic_cv", returns_cv, global_step)
    writer.add_scalar("evaluation/performance_mean", performance_mean, global_step)
    writer.add_scalar("evaluation/performance_std", performance_std, global_step)