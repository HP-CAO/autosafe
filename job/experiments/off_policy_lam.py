import copy

from agent.OffPolicyRL.ReplayBuffer import ReplayMemory
from agent.utils import log_info, setup_logger
from tqdm import tqdm
import numpy as np
logger = setup_logger()
import tensorflow as tf


def pretrain_lam(rl_agent, safe_agent, env, writer, cfg):
    pretrain_buffer = ReplayMemory(size=cfg.SACParams.replay_buffer_size)
    pretraining_steps = 10000
    global_steps = 0
    training_epochs = 100
    batch_size = 128
    optimizer_lam = tf.keras.optimizers.Adam(learning_rate=1e-03)
    mse_loss_fn = tf.keras.losses.MeanSquaredError()

    while global_steps < pretraining_steps:
        obs, _ = env.reset(seed=cfg.JobParams.seed, options={"mode": "train"})
        tracking_error = env.state - env.model_based_equilibrium
        obs_aug = np.append(obs, tracking_error)

        for _ in range(cfg.GymParams.TaskParams.max_episode_steps):
            global_steps += 1
            safe_action = safe_agent.get_action(tracking_error).squeeze()
            next_obs, rewards, terminations, truncations, infos = env.step(safe_action)

            # pretraining lam to predict a value close to 1,
            pretrain_buffer.add([obs_aug, np.array([0.98])])

            if terminations:
                break
    print("Data collection is done")

    for ep in range(training_epochs):
        minibatch = pretrain_buffer.sample(batch_size)
        obs = tf.constant(minibatch[0])
        label = tf.constant(minibatch[1])
        with tf.GradientTape() as tape:
            preds = rl_agent.lam_network(obs)
            loss =  mse_loss_fn(label, preds)
        vars_lam = rl_agent.lam_network.trainable_variables
        grads = tape.gradient(loss, vars_lam)
        optimizer_lam.apply_gradients(zip(grads, vars_lam))
    print("Finishing pretraining")



def train(rl_agent, safe_agent, env, writer, cfg):
    training_info = {}
    global_step = 0
    replay_mem = ReplayMemory(size=cfg.SACParams.replay_buffer_size)
    progress = tqdm(total=cfg.SACParams.total_training_steps)
    evaluate_flag = False
    pretrain_lam(rl_agent, safe_agent, env, writer, cfg)

    while global_step < cfg.SACParams.total_training_steps:
        obs, _ = env.reset(seed=cfg.JobParams.seed, options={"mode": "train"})
        episodic_return = []
        tracking_error = env.state - env.model_based_equilibrium
        obs_aug = np.append(obs, tracking_error)
        lam_list = []

        for _ in range(cfg.GymParams.TaskParams.max_episode_steps):

            rl_action = rl_agent.get_action(obs_aug, mode='train').squeeze()
            safe_action = safe_agent.get_action(tracking_error).squeeze()
            lam = rl_agent.get_lam(obs_aug).squeeze()
            action = np.clip((1 - lam) * rl_action + lam * safe_action, -1, 1)

            next_obs, rewards, terminations, truncations, infos = env.step(action)
            tracking_error = env.state - env.model_based_equilibrium

            next_obs_aug = np.append(next_obs, tracking_error)
            replay_mem.add([obs_aug, rl_action, rewards, next_obs_aug, terminations])

            obs_aug = copy.deepcopy(next_obs_aug)
            episodic_return.append(rewards)
            lam_list.append(lam)
            global_step += 1

            if global_step > cfg.SACParams.learning_starts:
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
        training_info['episodic_ave_lam'] = np.mean(lam_list)
        progress.update(training_info['episodic_length'])
        progress.set_description(f"Step: {global_step}, Return: {training_info['episodic_return']:.2f}")
        writer.add_scalar("training/episodic_return", training_info['episodic_return'], global_step)
        writer.add_scalar("training/episodic_length", training_info['episodic_length'], global_step)
        writer.add_scalar("training/episodic_ave_lam", training_info['episodic_ave_lam'], global_step)


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
                rl_action = rl_agent.get_action(obs_aug, mode='test').squeeze()
                safe_action = safe_agent.get_action(tracking_error).squeeze()
                lam = rl_agent.get_lam(obs_aug).squeeze()
                action = np.clip((1 - lam) * rl_action + lam * safe_action, -1, 1)
                next_obs, r, terminations, truncations, infos = env.step(action)
                performance_score = env.get_performance_score()
                accumulate_performance_score.append(performance_score)
                tracking_error = env.state - env.model_based_equilibrium
                next_obs_aug = np.append(next_obs, tracking_error)
                obs_aug = copy.deepcopy(next_obs_aug)
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