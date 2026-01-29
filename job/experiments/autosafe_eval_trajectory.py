import copy

from agent.utils import  setup_logger
import numpy as np
logger = setup_logger()


def evaluate_trajectory_autosafe(agent, env, cfg, env_name):

    # obs, _ = env.reset(seed=cfg.JobParams.seed, options={"mode": "train", "reset_state": [-0.2, 0, 0, 0]})
    obs, _ = env.reset(seed=cfg.JobParams.seed, options={"mode": "test"})

    tracking_error = env.state - env.model_based_equilibrium
    obs_aug = np.append(obs, tracking_error)
    save_path = cfg.JobParams.output_path + f'/{env_name}_traj'
    action_list = []
    tracking_error_list = []
    lam_list = []
    reward_list = []
    for _ in range(cfg.GymParams.TaskParams.max_episode_steps):
        action, lam, tem  = agent.get_action(obs_aug, mode='test_info')

        next_obs, r, terminations, trunc, infos = env.step(action.squeeze())
        tracking_error = env.state - env.model_based_equilibrium
        next_obs_aug = np.append(next_obs, tracking_error)
        obs_aug = copy.deepcopy(next_obs_aug)

        action_list.append(action.squeeze())
        tracking_error_list.append(tracking_error)
        lam_list.append(lam.squeeze())

        reward_list.append(r)

        if terminations:
            break

        if trunc:
            break

    print("total_steps", len(reward_list), "reward sum: ", sum(reward_list))
    print("evaluation done, saving trajectory to ", save_path)
    save_data = {
                 "actions": np.array(action_list),
                 "tracking_errors": np.array(tracking_error_list),
                 "lambdas": np.array(lam_list),
                 "rewards": np.array(reward_list)
    }
    np.save(save_path, save_data)


def evaluate_trajectory_simplex(safe_agent, rl_agent, env, cfg, env_name):
    # obs, _ = env.reset(seed=cfg.JobParams.seed, options={"mode": "train", "reset_state": [-0.2, 0, 0, 0]})
    obs, _ = env.reset(seed=cfg.JobParams.seed, options={"mode": "test"})
    tracking_error = env.state - env.model_based_equilibrium
    obs_aug = np.append(obs, tracking_error)
    save_path = cfg.JobParams.output_path + f'/simplex_{env_name}_traj'
    action_list = []
    tracking_error_list = []
    lam_list = []
    reward_list = []
    safe_policy_steps = 5
    safe_steps = 0

    for _ in range(cfg.GymParams.TaskParams.max_episode_steps):

        if safe_agent.safety_switch_on(tracking_error):
            safe_steps = safe_policy_steps

        if safe_steps > 0:
            action = safe_agent.get_action(tracking_error).squeeze()
            safe_steps -= 1
        else:
            action = rl_agent.get_action(obs_aug, mode='train').squeeze()

        next_obs, rewards, terminations, trunc, infos = env.step(action)
        tracking_error = env.state - env.model_based_equilibrium
        next_obs_aug = np.append(next_obs, tracking_error)
        obs_aug = copy.deepcopy(next_obs_aug)
        action_list.append(action.squeeze())
        tracking_error_list.append(tracking_error)

        reward_list.append(rewards)

        if terminations:
            break

        if trunc:
            break

    print("total_steps", len(reward_list), "reward sum: ", sum(reward_list))
    print("evaluation done, saving trajectory to ", save_path)
    save_data = {
                 "actions": np.array(action_list),
                 "tracking_errors": np.array(tracking_error_list),
                 "lambdas": np.array(lam_list),
                 "rewards": np.array(reward_list)
    }
    np.save(save_path, save_data)