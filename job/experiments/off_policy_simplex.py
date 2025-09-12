import copy

from agent.OffPolicyRL.ReplayBuffer import ReplayMemory
from agent.utils import log_info, setup_logger
from tqdm import tqdm
import numpy as np
logger = setup_logger()

def train(rl_agent, safe_agent, env, writer, cfg, safe_policy_steps=5):
    training_info = {}
    global_step = 0
    replay_mem = ReplayMemory(size=cfg.SACParams.replay_buffer_size)
    progress = tqdm(total=cfg.SACParams.total_training_steps)
    evaluate_flag = False
    ep = 0
    accumulated_term = 0

    while global_step < cfg.SACParams.total_training_steps:
        obs, _ = env.reset(seed=cfg.JobParams.seed, options={"mode": "train"})
        episodic_return = []
        tracking_error = env.state - env.model_based_equilibrium
        obs_aug = np.append(obs, tracking_error)
        safe_steps = 0
        triggered_time= 0
        for _ in range(cfg.GymParams.TaskParams.max_episode_steps):

            if safe_agent.safety_switch_on(tracking_error):
                safe_steps = safe_policy_steps
                triggered_time += 1

            if safe_steps > 0:
                action = safe_agent.get_action(tracking_error).squeeze()
                safe_steps -= 1
            else:
                action = rl_agent.get_action(obs_aug, mode='train').squeeze()

            next_obs, rewards, terminations, trunc, infos = env.step(action)
            tracking_error = env.state - env.model_based_equilibrium

            next_obs_aug = np.append(next_obs, tracking_error)

            replay_mem.add([obs_aug, action, rewards, next_obs_aug, terminations])

            obs_aug = copy.deepcopy(next_obs_aug)
            episodic_return.append(rewards)
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
                accumulated_term += 1
                break

            if trunc:
                break

        ep += 1
        print(f"safety triggered times: {triggered_time}")

        if evaluate_flag:
            evaluate(rl_agent, safe_agent, env, writer, cfg, global_step, ep, safe_policy_steps)
            evaluate_flag = False

        training_info['episodic_return'] = sum(episodic_return)
        training_info['episodic_length'] = len(episodic_return)
        training_info['episodic_ave_return'] = np.mean(episodic_return)
        progress.update(training_info['episodic_length'])
        progress.set_description(f"Step: {global_step}, Return: {training_info['episodic_return']:.2f}")
        writer.add_scalar("training/episodic_return", training_info['episodic_return'], global_step)
        writer.add_scalar("training/episodic_length", training_info['episodic_length'], global_step)
        writer.add_scalar("training/safety_trigger_time", triggered_time, global_step)

        # episode based logging
        writer.add_scalar("training/episodic_return_ep", training_info['episodic_return'], ep)
        writer.add_scalar("training/termination_ep", accumulated_term, ep)
        writer.add_scalar("training/termination_steps", accumulated_term, global_step)

def evaluate(rl_agent, safe_agent, env, writer, cfg, global_step, global_episode, safe_policy_steps):
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

            safe_steps = 0

            for _ in range(cfg.GymParams.TaskParams.max_episode_steps):

                if safe_agent.safety_switch_on(tracking_error):
                    safe_steps = safe_policy_steps

                if safe_steps > 0:
                    action = safe_agent.get_action(tracking_error).squeeze()
                    safe_steps -= 1
                else:
                    action = rl_agent.get_action(obs_aug, mode='train').squeeze()

                next_obs, r, terminations, trunc, infos = env.step(action)
                performance_score = env.get_performance_score()
                accumulate_performance_score.append(performance_score)
                tracking_error = env.state - env.model_based_equilibrium
                next_obs_aug = np.append(next_obs, tracking_error)
                obs_aug = copy.deepcopy(next_obs_aug)
                episodic_return.append(r)

                if terminations:
                    break

                if trunc:
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
    writer.add_scalar("evaluation/episodic_return_ep", return_mean, global_episode)

    try:
        env.save_trajectory(cfg.JobParams.output_path + f'{global_step}_traj.png', total_reward=return_mean)
    except:
        pass