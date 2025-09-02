import numpy as np
from tqdm import tqdm
import copy

def train(agent, buffer, env, writer, cfg):
    training_info = {}
    global_step = 0
    progress = tqdm(total=cfg.PPOParams.total_training_steps)
    evaluation_flag = False
    while global_step < cfg.PPOParams.total_training_steps:
        # Initialize the environment
        obs, _ = env.reset(seed=cfg.JobParams.seed, options={"mode": "train"})
        episode_return = []
        tracking_error = env.state - env.model_based_equilibrium
        obs_aug = np.append(obs, tracking_error)

        while buffer.ptr < buffer.buffer_size:
            # env.render()
            action, log_prob, value = agent.get_action_and_log_prob(obs_aug)
            next_obs, reward, terminated, truncated, info = env.step(action.numpy().squeeze())
            tracking_error = env.state - env.model_based_equilibrium
            next_obs_aug = np.append(next_obs, tracking_error)

            buffer.store(
                obs_aug,
                action.numpy(),
                reward,
                terminated,
                log_prob.numpy(),
                value.numpy())

            episode_return.append(reward)
            global_step += 1

            if terminated or truncated:  # reset the environment if the episode is terminated
                obs, _ = env.reset(seed=cfg.JobParams.seed, options={"mode": "train"})
                tracking_error = env.state - env.model_based_equilibrium
                obs_aug = np.append(obs, tracking_error)
                # Log the training information
                training_info["episode_return"] = sum(episode_return)
                training_info["episode_length"] = len(episode_return)
                writer.add_scalar("training/episode_return", training_info["episode_return"], global_step)
                writer.add_scalar("training/episode_length", training_info["episode_length"], global_step)
                progress.set_description(
                    f"Episode return: {training_info['episode_return']}, Episode length: {training_info['episode_length']}")
                progress.update(training_info["episode_length"])
                episode_return = []

            else:
                obs_aug = next_obs_aug

            # Evaluate the agent
            if global_step % cfg.GymParams.TaskParams.evaluation_period == 0:
                evaluation_flag = True

        # Update the policy
        if terminated:
            next_value = 0.0  # if the episode is terminated, the value of the last state is 0
        else:
            next_value = agent.get_value(obs_aug)

        metrics = agent.optimize(buffer, next_value)
        writer.add_scalar("training/actor_loss", metrics["actor_loss"], global_step)
        writer.add_scalar("training/critic_loss", metrics["critic_loss"], global_step)
        writer.add_scalar("training/entropy_loss", metrics["entropy"], global_step)
        writer.add_scalar("training/kl_div", metrics["kl_div"], global_step)
        buffer.clear()

        if evaluation_flag:
            eval_return, eval_length = evaluate(agent, env, cfg)
            writer.add_scalar("evaluation/return", eval_return, global_step)
            writer.add_scalar("evaluation/length", eval_length, global_step)
            evaluation_flag = False


def evaluate(agent, env, cfg):
    eval_return = []
    eval_length = []


    for _ in range(cfg.GymParams.TaskParams.num_episodes_to_run):
        obs, _ = env.reset(seed=cfg.JobParams.seed, options={"mode": "train"})

        tracking_error = env.state - env.model_based_equilibrium
        obs_aug = np.append(obs, tracking_error)

        episode_return = []
        for _ in range(cfg.GymParams.TaskParams.max_episode_steps):
            action = agent.get_action(obs_aug, deterministic=True)  # use deterministic policy for evaluation
            next_obs, rewards, terminations, truncations, infos = env.step(action.numpy().squeeze())
            tracking_error = env.state - env.model_based_equilibrium
            next_obs_aug = np.append(next_obs, tracking_error)
            obs_aug = copy.deepcopy(next_obs_aug)
            episode_return.append(rewards)

            if terminations or truncations:
                eval_return.append(sum(episode_return))
                eval_length.append(len(episode_return))
                break

    return_mean = np.nanmean(eval_return)
    length_mean = np.nanmean(eval_length)

    return return_mean, length_mean