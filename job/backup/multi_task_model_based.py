from agent.utils import setup_logger
import numpy as np
logger = setup_logger()

def evaluate(agent, env, writer, cfg, global_step):
    logger.info("Start evaluating the agent...")
    task_list = env.generate_control_tasks(mode='test')
    accumulated_return_list = []
    all_task_info_list = []

    for task in task_list:
        crash = False
        init_states = task["init_states"]
        control_goal = task["control_goal"]
        _, c_gt = env.reset(seed=cfg.JobParams.seed, options={"reset_state": init_states})
        env.control_goal = control_goal
        env.set_control_goal_array()
        episodic_return = []

        for _ in range(cfg.GymParams.TaskParams.max_episode_steps):
            action = agent.get_action(tracking_error=env.state - env.control_goal_array).squeeze()
            action = (action / cfg.GymParams.RobotParams.action_mag).clip(-1, 1)

            _, r, terminations, truncations, infos = env.step(action)
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

