from agent.utils import setup_logger
import numpy as np
logger = setup_logger()

def evaluate(agent, env, writer, cfg, global_step):
    logger.info("Start evaluating the agent...")
    accumulated_return_list = []
    all_task_info_list = []
    test_episode_num = cfg.GymParams.TaskParams.num_episodes_to_run
    for _ in range(test_episode_num):
        _, c_gt = env.reset(seed=cfg.JobParams.seed)
        episodic_return = []

        for _ in range(cfg.GymParams.TaskParams.max_episode_steps):
            tracking_error = env.state - env.model_based_equilibrium
            action = agent.get_action(tracking_error).squeeze()
            _, r, terminations, truncations, infos = env.step(action)
            episodic_return.append(r)
            if terminations:
                break

        accumulated_return_list.append(sum(episodic_return))
    # env.render()
    # task_result_plot = env.plot_task_result(all_task_info_list)
    return_mean = np.mean(accumulated_return_list)
    return_std = np.std(accumulated_return_list)
    returns_cv = return_std / return_mean  # coefficient of variation (dimensionless)

    # writer.add_figure("evaluation/task_result", task_result_plot, global_step)
    writer.add_scalar("evaluation/episodic_return", return_mean, global_step)
    writer.add_scalar("evaluation/episodic_cv", returns_cv, global_step)
    print("ave_episodic_reward", return_mean)


def evaluate_quad(agent, env, mpc, writer, cfg, global_step):
    logger.info("Start evaluating the agent...")
    accumulated_return_list = []
    all_task_info_list = []
    test_episode_num = cfg.GymParams.TaskParams.num_episodes_to_run

    for _ in range(test_episode_num):
        _, c_gt = env.reset(seed=cfg.JobParams.seed)
        mpc.reset_controllers()
        env.model_based_equilibrium = np.array(mpc.ref_point)
        episodic_return = []

        for _ in range(cfg.GymParams.TaskParams.max_episode_steps * 4):
            mpc.update()
            env.state = np.array(mpc.robot_state)
            # print(env.state)
            action = agent.get_action(env.state - env.model_based_equilibrium).squeeze()
            action = None
            motor_action, _ = mpc.get_action(action=action)
            _, r, terminations, truncations, infos = env.step(motor_action)
            episodic_return.append(r)
            # if terminations:
                # break
        accumulated_return_list.append(sum(episodic_return))
    # env.render()
    # task_result_plot = env.plot_task_result(all_task_info_list)
    return_mean = np.mean(accumulated_return_list)
    return_std = np.std(accumulated_return_list)
    returns_cv = return_std / return_mean  # coefficient of variation (dimensionless)

    # writer.add_figure("evaluation/task_result", task_result_plot, global_step)
    writer.add_scalar("evaluation/episodic_return", return_mean, global_step)
    writer.add_scalar("evaluation/episodic_cv", returns_cv, global_step)
    print("ave_episodic_reward", return_mean)