import copy
from agent.model_based.model_based_design_quadrotor import MATRIX_P
from agent.OffPolicyRL.ReplayBuffer import ReplayMemory
from agent.utils import log_info, setup_logger
from tqdm import tqdm
import numpy as np
import time

logger = setup_logger()

# Quadrotor 12D state layout:
#   [x, vx, y, vy, z, vz, roll, pitch, yaw, p, q, r]
# Position indices: x=0, y=2, z=4
POS_IDX = [0, 2, 4]

inv_p = np.linalg.inv(MATRIX_P)

# ─── Virtual box obstacle ─────────────────────────────────────────────────────
OBSTACLE_CENTER   = np.array([1.5, 1.5, 1.5])   # [x, y, z]
OBSTACLE_HALF     = 0.3                         # box half-size (0.5 m side)
OBSTACLE_ACT_DIST = 1.0                           # envelope rescaling activation radius (m)
# How far in front of the nearest obstacle face the safety reference is placed.
# b_rel = SAFETY_BUFFER > 0 is guaranteed, giving a valid rescaling ratio.
SAFETY_BUFFER     = 0.40                          # m


def _quad_pos(state):
    """Return [x, y, z] from 12D quadrotor state."""
    return np.array([state[POS_IDX[0]], state[POS_IDX[1]], state[POS_IDX[2]]])


def check_collision(state):
    """True if quadrotor centre is inside the box obstacle."""
    pos = _quad_pos(state)
    return bool(np.all(np.abs(pos - OBSTACLE_CENTER) <= OBSTACLE_HALF))


def dist_to_obstacle(state):
    """Euclidean distance from quadrotor to obstacle centre."""
    return float(np.linalg.norm(_quad_pos(state) - OBSTACLE_CENTER))


def get_obstacle_safety_reference(drone_state, goal, obs_center, half):
    """
    Return a safety reference that is strictly on the drone's side of the obstacle.

    For each position axis, if the goal lies at or past the nearest obstacle face
    (relative to the drone), the reference is pulled back to
        face_pos - SAFETY_BUFFER   (drone on the negative side)
        face_pos + SAFETY_BUFFER   (drone on the positive side)
    so that  b_rel = SAFETY_BUFFER > 0  is guaranteed for all three constraints.

    When the goal is already on the safe side the goal value is kept unchanged.
    """
    drone_pos = _quad_pos(drone_state)
    ref = goal.copy()
    for i, idx in enumerate(POS_IDX):
        if drone_pos[i] <= obs_center[i]:
            face = obs_center[i] - half          # nearest face from negative side
            ref[idx] = min(goal[idx], face - SAFETY_BUFFER)
        else:
            face = obs_center[i] + half          # nearest face from positive side
            ref[idx] = max(goal[idx], face + SAFETY_BUFFER)
    return ref


def build_obstacle_constraints(state):
    """
    Build 3 axis-aligned half-space constraints (one per axis) in 12D state space.
    For each axis the constraint faces the nearest obstacle face from the current
    quadrotor side:
        a^T state <= b   (a is a 12D unit vector, b is the face position)
    Returns shape (3, 13) array — last column is b.
    """
    pos = _quad_pos(state)
    constraints = []
    for i, idx in enumerate(POS_IDX):
        a = np.zeros(12)
        if pos[i] <= OBSTACLE_CENTER[i]:
            # drone is on the negative side → keep pos[i] <= face position
            a[idx] = 1.0
            b = OBSTACLE_CENTER[i] - OBSTACLE_HALF
        else:
            # drone is on the positive side → -pos[i] <= -(face position)
            a[idx] = -1.0
            b = -(OBSTACLE_CENTER[i] + OBSTACLE_HALF)
        constraints.append(np.append(a, b))
    return np.array(constraints, dtype=float)


def rescale_safety_envelope(all_constraints, inv_p_mat):
    """
    Find the tightest rescaling ratio r such that the scaled Lyapunov envelope
    e^T (r*P) e <= 1 fits within every half-space  a^T e <= b_rel.
    """
    ratio_list = []
    for constraint in all_constraints:
        a        = constraint[:-1]
        b_rel    = constraint[-1]
        cons_max = np.sqrt(a @ inv_p_mat @ a)
        if 0 < b_rel < cons_max:
            ratio_list.append((cons_max / b_rel) ** 2)
    if not ratio_list:
        return 1.0, None
    return float(np.max(ratio_list)), int(np.argmax(ratio_list))


def check_validity_envelope(state, safety_reference, rescale_ratio, p_matrix):
    """
    Ensure the current state lies inside the rescaled envelope e^T (r*P) e <= 1.
    If it does not, shift the position components of safety_reference so the state
    lands on the boundary.  Velocities and angles stay at their reference values
    (zero for a stabilisation goal).
    Returns a 12D reference with layout [x_ref, 0, y_ref, 0, z_ref, 0, 0, …, 0].
    """
    e = np.array(state) - np.array(safety_reference)
    V = rescale_ratio * (e @ p_matrix @ e)
    if V > 1.0:
        offset = e / np.sqrt(V)
        safety_reference = np.array(state) - offset

    # Project back: keep only position components, zero out velocities/angles
    ref_out = np.zeros(12)
    for idx in POS_IDX:
        ref_out[idx] = safety_reference[idx]
    return ref_out


# ─── Training ─────────────────────────────────────────────────────────────────

def train(agent, env, writer, cfg):
    training_info = {}
    global_step   = 0
    replay_mem    = ReplayMemory(size=cfg.SACParams.replay_buffer_size)
    progress      = tqdm(total=cfg.SACParams.total_training_steps)
    evaluate_flag = False
    ep            = 0
    accumulated_term = 0
    inference_time_list = []
    optimization_time_list = []


    while global_step < cfg.SACParams.total_training_steps:
        obs, _ = env.reset(seed=cfg.JobParams.seed, options={"mode": "train"})

        goal            = env.model_based_equilibrium.copy()   # 12D goal state
        safety_reference = goal.copy()
        rescale_ratio    = 1.0

        safety_tracking_error = env.state - safety_reference
        obs_aug = np.concatenate([obs, [rescale_ratio], safety_tracking_error, goal])

        episodic_return = []
        infer_time = []
        time_list = []

        for _ in range(cfg.GymParams.TaskParams.max_episode_steps):

            t_1 = time.perf_counter()
            action = agent.get_action(obs_aug, mode='train').squeeze()
            infer_time.append(time.perf_counter() - t_1)
            next_obs, rewards, terminations, trunc, infos = env.step(action)

            # Virtual collision check — terminate if inside the box
            if check_collision(env.state):
                print(f"Virtual collision at step {global_step}: state={env.state}, action={action}")
                terminations = True

            # Adapt safety envelope when close to the obstacle
            d = dist_to_obstacle(env.state)
            if d <= OBSTACLE_ACT_DIST:
                t2 = time.perf_counter()

                # Use a reference that is strictly on the drone's side of the obstacle
                # so that b_rel = SAFETY_BUFFER > 0 for every constraint axis.
                obs_ref = get_obstacle_safety_reference(
                    env.state, goal, OBSTACLE_CENTER, OBSTACLE_HALF
                )
                abs_constraints = build_obstacle_constraints(env.state)
                rel_constraints = abs_constraints.copy()
                rel_constraints[:, -1] -= abs_constraints[:, :-1] @ obs_ref
                rescale_ratio, _ = rescale_safety_envelope(rel_constraints, inv_p_mat=inv_p)
                safety_reference = check_validity_envelope(
                    env.state, obs_ref, rescale_ratio, MATRIX_P
                )
                time_list.append(time.perf_counter() - t2)
            else:
                rescale_ratio    = 1.0
                safety_reference = goal.copy()
            next_safety_tracking_error = env.state - safety_reference
            next_obs_aug = np.concatenate(
                [next_obs, [rescale_ratio], next_safety_tracking_error, goal]
            )

            replay_mem.add([obs_aug, action, rewards, next_obs_aug, terminations])
            obs_aug = copy.deepcopy(next_obs_aug)
            episodic_return.append(rewards)
            global_step += 1

            if global_step > cfg.SACParams.learning_starts:
                mini_batch = replay_mem.sample(cfg.SACParams.batch_size)
                t1 = time.perf_counter()
                agent_info = agent.optimize(mini_batch)
                optimization_time_list.append(time.perf_counter() - t1)
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

        print("average_time: ", np.mean(time_list), "average_inference_time: ", np.mean(infer_time), "percentage:", np.mean(time_list) / np.mean(infer_time))

        if evaluate_flag:
            evaluate(agent, env, writer, cfg, global_step, ep)
            evaluate_flag = False

        training_info['episodic_return']     = sum(episodic_return)
        training_info['episodic_length']     = len(episodic_return)
        training_info['episodic_ave_return'] = np.mean(episodic_return)
        progress.update(training_info['episodic_length'])
        progress.set_description(
            f"Step: {global_step}, Return: {training_info['episodic_return']:.2f}"
        )
        writer.add_scalar("training/episodic_return",     training_info['episodic_return'],     global_step)
        writer.add_scalar("training/episodic_length",     training_info['episodic_length'],     global_step)
        writer.add_scalar("training/episodic_return_ep",  training_info['episodic_return'],     ep)
        writer.add_scalar("training/termination_ep",      accumulated_term,                     ep)
        writer.add_scalar("training/termination_steps",   accumulated_term,                     global_step)
        writer.add_scalar("training/inference_time",
                          np.mean(inference_time_list) if inference_time_list else 0, global_step)
        writer.add_scalar("training/optimization_time",
                          np.mean(optimization_time_list) if optimization_time_list else 0, global_step)


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(agent, env, writer, cfg, global_step, global_episode):
    logger.info("Start evaluating the agent...")
    multi_rollout_return      = []
    multi_rollout_performance = []
    num_rollouts = 3

    for episode in range(cfg.GymParams.TaskParams.num_episodes_to_run):
        accumulated_r               = []
        accumulate_performance_score = []

        for _ in range(num_rollouts):
            obs, _ = env.reset(seed=cfg.JobParams.seed, options={"mode": "test"})

            goal             = env.model_based_equilibrium.copy()
            safety_reference = goal.copy()
            rescale_ratio    = 1.0

            safety_tracking_error = env.state - safety_reference
            obs_aug = np.concatenate([obs, [rescale_ratio], safety_tracking_error, goal])

            episodic_return = []

            for _ in range(cfg.GymParams.TaskParams.max_episode_steps):
                action = agent.get_action(obs_aug, mode='test').squeeze()
                next_obs, r, terminations, trunc, infos = env.step(action)

                performance_score = env.get_performance_score()
                accumulate_performance_score.append(performance_score)

                # Virtual collision check
                if check_collision(env.state):
                    terminations = True

                # Adapt safety envelope when close to the obstacle
                d = dist_to_obstacle(env.state)
                if d <= OBSTACLE_ACT_DIST:
                    obs_ref = get_obstacle_safety_reference(
                        env.state, goal, OBSTACLE_CENTER, OBSTACLE_HALF
                    )
                    abs_constraints = build_obstacle_constraints(env.state)
                    rel_constraints = abs_constraints.copy()
                    rel_constraints[:, -1] -= abs_constraints[:, :-1] @ obs_ref
                    rescale_ratio, _ = rescale_safety_envelope(rel_constraints, inv_p_mat=inv_p)
                    safety_reference = check_validity_envelope(
                        env.state, obs_ref, rescale_ratio, MATRIX_P
                    )
                else:
                    rescale_ratio    = 1.0
                    safety_reference = goal.copy()

                next_safety_tracking_error = env.state - safety_reference
                next_obs_aug = np.concatenate(
                    [next_obs, [rescale_ratio], next_safety_tracking_error, goal]
                )
                obs_aug = copy.deepcopy(next_obs_aug)
                episodic_return.append(r)

                if terminations or trunc:
                    break

            accumulated_r.append(sum(episodic_return))

        multi_rollout_return.append(np.mean(accumulated_r))
        multi_rollout_performance.append(np.mean(accumulate_performance_score))

    performance_mean = np.mean(multi_rollout_performance)
    performance_std  = np.std(multi_rollout_performance)
    return_mean      = np.mean(multi_rollout_return)
    return_std       = np.std(multi_rollout_return)
    returns_cv       = return_std / (return_mean + 1e-8)

    writer.add_scalar("evaluation/episodic_return",   return_mean,      global_step)
    writer.add_scalar("evaluation/episodic_cv",       returns_cv,       global_step)
    writer.add_scalar("evaluation/performance_mean",  performance_mean, global_step)
    writer.add_scalar("evaluation/performance_std",   performance_std,  global_step)
    writer.add_scalar("evaluation/episodic_return_ep", return_mean,     global_episode)

    try:
        env.save_trajectory(
            cfg.JobParams.output_path + f'{global_step}_traj.png',
            total_reward=return_mean,
            obstacle_center=OBSTACLE_CENTER,
            obstacle_half=OBSTACLE_HALF,
        )
    except Exception:
        pass
