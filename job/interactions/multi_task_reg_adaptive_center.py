import math

from agent.OffPolicyRL.ReplayBuffer_core import ReplayMemory
from agent.utils import log_info, setup_logger
from tqdm import tqdm
import numpy as np

logger = setup_logger()

def train(agent, env, writer, cfg, model_based_agent, matrix_A, matrix_B):
    training_info = {}
    global_step = 0
    replay_mem = ReplayMemory(size=cfg.AgentParams.replay_buffer_size)
    progress = tqdm(total=cfg.AgentParams.total_training_steps)
    evaluate_flag = False
    task_list = env.generate_control_tasks(mode='train')
    lamda = 0.5

    while global_step < cfg.AgentParams.total_training_steps:
        random_index = np.random.choice(range(len(task_list)))
        task = task_list[random_index]
        init_states = task["init_states"]
        control_goal = task["control_goal"]
        obs, c_gt = env.reset(seed=cfg.JobParams.seed, options={"reset_state": init_states})
        env.control_goal = control_goal
        episodic_return = []
        obs = np.concatenate([obs, control_goal])
        control_goal_array = np.array([control_goal[0], 0, control_goal[1], 0])

        for _ in range(cfg.GymParams.TaskParams.max_episode_steps):

            if global_step < cfg.AgentParams.learning_starts:
                action = env.action_space.sample().squeeze()
            else:
                action = agent.get_action(obs, mode='train').squeeze()

            model_based_action = model_based_agent.get_action(tracking_error=env.state - control_goal_array).squeeze()
            model_based_action = (model_based_action / cfg.GymParams.RobotParams.action_mag).clip(-1, 1)
            applied_action = lamda * action + (1 - lamda) * model_based_action

            predicted_state = env.predict_next_state(applied_action, matrix_A, matrix_B)

            next_obs, rewards, terminations, truncations, infos = env.step(applied_action)
            lamda = update_lambda(env.state, predicted_state)

            next_obs = np.concatenate([next_obs, control_goal])
            replay_mem.add([obs, action, rewards, next_obs, terminations])
            obs = next_obs

            episodic_return.append(rewards)
            global_step += 1

            if global_step > cfg.AgentParams.learning_starts:
                mini_batch = replay_mem.sample(cfg.AgentParams.batch_size)
                agent_info = agent.optimize(mini_batch)
                log_info(writer, global_step, agent_info, 'agent', period=100)

            if global_step % cfg.GymParams.TaskParams.evaluation_period == 0:
                evaluate_flag = True

            if global_step % 50000 == 0:
                agent.save_weights(cfg.JobParams.output_path + '/agent_model/')

            if terminations:
                break

        if evaluate_flag:
            evaluate(agent, env, writer, cfg, global_step, model_based_agent, matrix_A, matrix_B)
            evaluate_flag = False

        training_info['episodic_return'] = sum(episodic_return)
        training_info['episodic_length'] = len(episodic_return)
        progress.update(training_info['episodic_length'])
        progress.set_description(f"Step: {global_step}, Return: {training_info['episodic_return']:.2f}")
        writer.add_scalar("training/episodic_return", training_info['episodic_return'], global_step)
        writer.add_scalar("training/episodic_length", training_info['episodic_length'], global_step)


def evaluate(agent, env, writer, cfg, global_step, model_based_agent, matrix_A, matrix_B):
    logger.info("Start evaluating the agent...")
    task_list = env.generate_control_tasks(mode='test')
    accumulated_return_list = []
    all_task_info_list = []

    for task in task_list:
        crash = False
        init_states = task["init_states"]
        control_goal = task["control_goal"]
        obs, c_gt = env.reset(seed=cfg.JobParams.seed, options={"reset_state": init_states})
        env.control_goal = control_goal
        env.set_control_goal_array()
        episodic_return = []
        obs = np.concatenate([obs, control_goal])
        lamda = 0.5

        for _ in range(cfg.GymParams.TaskParams.max_episode_steps):

            action = agent.get_action(obs, mode='test').squeeze()
            model_based_action = model_based_agent.get_action(tracking_error=env.state - env.control_goal_array).squeeze()
            model_based_action = (model_based_action / cfg.GymParams.RobotParams.action_mag).clip(-1, 1)
            applied_action = lamda * action + (1 - lamda) * model_based_action
            predicted_state = env.predict_next_state(applied_action, matrix_A, matrix_B)
            next_obs, r, terminations, truncations, infos = env.step(applied_action)
            lamda = update_lambda(env.state, predicted_state)
            next_obs = np.concatenate([next_obs, control_goal])
            obs = next_obs
            episodic_return.append(r)

            if terminations:
                crash = True
                break

        final_state = env.state
        task_info = {"init_states": init_states, "control_goal": control_goal,
                     "final_state": final_state, "crash": crash}

        all_task_info_list.append(task_info)
        accumulated_return_list.append(sum(episodic_return))

    task_result_plot = env.plot_task_result(all_task_info_list)
    return_mean = np.mean(accumulated_return_list)
    return_std = np.std(accumulated_return_list)
    returns_cv = return_std / return_mean  # coefficient of variation (dimensionless)
    writer.add_figure("evaluation/task_result", task_result_plot, global_step)
    writer.add_scalar("evaluation/episodic_return", return_mean, global_step)
    writer.add_scalar("evaluation/episodic_cv", returns_cv, global_step)


def update_lambda(measured_state, predicted_state):
    euclidean_distance = np.linalg.norm(predicted_state - measured_state) * 10
    lamda = 1 - math.exp(euclidean_distance * -1)
    return lamda