import copy
from agent.model_based.model_based_design_cartpole import MATRIX_P
from agent.OffPolicyRL.ReplayBuffer import ReplayMemory
from agent.utils import log_info, setup_logger
from tqdm import tqdm
import numpy as np
logger = setup_logger()
import time
inv_p = np.linalg.inv(MATRIX_P)

def build_constraint(obstacle_pos, reference_pos):
    """Returns [a..., b] for a^T x <= b keeping main cart away from obstacle."""
    if obstacle_pos < reference_pos:  # obstacle is LEFT → x >= obstacle_pos → -x <= -obstacle_pos
        return [-1, 0, 0, 0, -obstacle_pos]
    else:                              # obstacle is RIGHT → x <= obstacle_pos
        return [1,  0, 0, 0,  obstacle_pos]

def rescale_safety_envelope(all_constraints, inv_p):
    ratio_list = []

    for constraint in all_constraints:
        constraint_normal = constraint[:-1]
        constraint_scalar = constraint[-1]
        cons_max = np.sqrt(constraint_normal.T @ inv_p @ constraint_normal)
        if constraint_scalar > 0 and cons_max > constraint_scalar:
            ratio = (cons_max / constraint_scalar) ** 2
            ratio_list.append(ratio)

    if len(ratio_list) == 0:
        return 1.0, None

    max_ratio = np.max(ratio_list)
    active_index = np.argmax(ratio_list)
    return max_ratio, active_index



def check_validity_envelope(state, safety_reference, rescale_ratio, P_matrix):
    """Check if state is inside the rescaled safety envelope e^T (r*P) e <= 1.
    If not, shift safety_reference toward state so the state lands on the boundary."""
    e = np.array(state) - np.array(safety_reference)
    V = rescale_ratio * (e @ P_matrix @ e)
    if V > 1.0:
        offset = e / np.sqrt(V)
        safety_reference = np.array(state) - offset
    return np.array([safety_reference[0], 0, 0, 0])


def train(agent, env, writer, cfg):
    training_info = {}
    global_step = 0
    replay_mem = ReplayMemory(size=cfg.SACParams.replay_buffer_size)
    progress = tqdm(total=cfg.SACParams.total_training_steps)
    evaluate_flag = False
    ep = 0
    accumulated_term = 0
    task_list = env.generate_tracking_tasks(mode='train', n_steps=cfg.GymParams.TaskParams.max_episode_steps)
    while global_step < cfg.SACParams.total_training_steps:
        task = task_list[np.random.randint(len(task_list))]
        init_states = task["init_states"]
        trajectory = task["reference_trajectory"]
        obs, _ = env.reset(seed=cfg.JobParams.seed, options={"reset_state": init_states})
        episodic_return = []
        rescale_ratio = 1.0
        task_reference = trajectory[0]
        safety_reference = trajectory[0]
        safety_tracking_error = env.state - safety_reference
        task_tracking_error = env.state - task_reference
        obs_aug = np.concatenate([obs, [rescale_ratio], safety_tracking_error, task_tracking_error])
        env.control_goal = [task_reference[0], 0]
        time_list = []
        inference_time_list = []

        for step in range(cfg.GymParams.TaskParams.max_episode_steps):

            inference_time_start = time.perf_counter()
            action = agent.get_action(obs_aug, mode='train').squeeze()
            inference_time_list.append(time.perf_counter() - inference_time_start)
            next_obs, rewards, terminations, trunc, infos = env.step(action)

            # Update task_reference first so constraints and safety_reference are consistent
            task_reference = trajectory[min(step + 1, len(trajectory) - 1)]
            env.control_goal = [task_reference[0], 0]

            start_time = time.perf_counter()
            ref_x = task_reference[0]
            abs_constraints = np.array([
                build_constraint(env.cart_2_state[0], ref_x),
                build_constraint(env.cart_3_state[0], ref_x),
                [1, 0, 0, 0, env.x_threshold],   # right wall
                [-1, 0, 0, 0, env.x_threshold],  # left wall
            ], dtype=float)

            # relative constraints for envelope rescaling: b_rel = b_abs - a^T ref
            all_constraints = abs_constraints.copy()
            all_constraints[:, -1] -= abs_constraints[:, :-1] @ task_reference

            rescale_ratio, active_index = rescale_safety_envelope(all_constraints, inv_p)
            # safety_reference = check_validity_envelope(env.state, task_reference, rescale_ratio, MATRIX_P)
            time_list.append(time.perf_counter() - start_time)
            safety_reference = task_reference
            next_safety_tracking_error = env.state - safety_reference
            next_task_tracking_error = env.state - task_reference
            next_obs_aug = np.concatenate([next_obs, [rescale_ratio], next_safety_tracking_error, next_task_tracking_error])

            replay_mem.add([obs_aug, action, rewards, next_obs_aug, terminations])

            obs_aug = copy.deepcopy(next_obs_aug)
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
                accumulated_term += 1
                break

            if trunc:
                break

        ep += 1

        print("average_time: ", np.mean(time_list), "average_inference_time: ", np.mean(inference_time_list), "percentage:", np.mean(time_list) / np.mean(inference_time_list))

        if evaluate_flag:
            evaluate(agent, env, writer, cfg, global_step, ep)
            evaluate_flag = False

        training_info['episodic_return'] = sum(episodic_return)
        training_info['episodic_length'] = len(episodic_return)
        training_info['episodic_ave_return'] = np.mean(episodic_return)
        progress.update(training_info['episodic_length'])
        progress.set_description(f"Step: {global_step}, Return: {training_info['episodic_return']:.2f}")
        writer.add_scalar("training/episodic_return", training_info['episodic_return'], global_step)
        writer.add_scalar("training/episodic_length", training_info['episodic_length'], global_step)
        # episode based logging
        writer.add_scalar("training/episodic_return_ep", training_info['episodic_return'], ep)
        writer.add_scalar("training/termination_ep", accumulated_term, ep)
        writer.add_scalar("training/termination_steps", accumulated_term, global_step)


def evaluate(agent, env, writer, cfg, global_step, global_episode):
    logger.info("Start evaluating the agent...")
    multi_rollout_return = []
    multi_rollout_performance = []
    num_rollouts = 3
    task_list = env.generate_tracking_tasks(mode='test', n_steps=cfg.GymParams.TaskParams.max_episode_steps)
    for episode in range(cfg.GymParams.TaskParams.num_episodes_to_run):
        accumulated_r = []
        accumulate_performance_score = []

        for _ in range(num_rollouts):
            task = task_list[episode % len(task_list)]
            init_states = task["init_states"]
            trajectory = task["reference_trajectory"]
            obs, _ = env.reset(seed=cfg.JobParams.seed, options={"reset_state": init_states})
            episodic_return = []
            rescale_ratio = 1.0
            task_reference = trajectory[0]
            safety_reference = check_validity_envelope(env.state, task_reference, rescale_ratio, MATRIX_P)
            safety_tracking_error = env.state - safety_reference
            task_tracking_error = env.state - task_reference
            obs_aug = np.concatenate([obs, [rescale_ratio], safety_tracking_error, task_tracking_error])
            env.control_goal = [task_reference[0], 0]


            for step in range(cfg.GymParams.TaskParams.max_episode_steps):
                action = agent.get_action(obs_aug, mode='test').squeeze()
                next_obs, r, terminations, trunc, infos = env.step(action)
                performance_score = env.get_performance_score()
                accumulate_performance_score.append(performance_score)

                # Update task_reference first so constraints and safety_reference are consistent
                task_reference = trajectory[min(step + 1, len(trajectory) - 1)]
                env.control_goal = [task_reference[0], 0]

                ref_x = task_reference[0]

                abs_constraints = np.array([
                    build_constraint(env.cart_2_state[0], ref_x),
                    build_constraint(env.cart_3_state[0], ref_x),
                    [1, 0, 0, 0, env.x_threshold],   # right wall
                    [-1, 0, 0, 0, env.x_threshold],  # left wall
                ], dtype=float)

                all_constraints = abs_constraints.copy()
                all_constraints[:, -1] -= abs_constraints[:, :-1] @ task_reference
                rescale_ratio, _ = rescale_safety_envelope(all_constraints, inv_p)
                safety_reference = check_validity_envelope(env.state, task_reference, rescale_ratio, MATRIX_P)

                next_safety_tracking_error = env.state - safety_reference
                next_task_tracking_error = env.state - task_reference
                next_obs_aug = np.concatenate([next_obs, [rescale_ratio], next_safety_tracking_error, next_task_tracking_error])
                obs_aug = next_obs_aug
                episodic_return.append(r)

                if terminations or trunc:
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