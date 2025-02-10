from gym_env.quad_gym.sim_quadruped_robot import GymConfig, build_a1_ground_env
from gym_env.quad_gym.utils.mpc.MPC_Control import MPCConfig, MPC

config = GymConfig()
# config.RobotParams.motor_control_mode = 'torque'
env = build_a1_ground_env(config)
mpc = MPC(MPCConfig(), env.robot)
lin_speed = [0.6, 0, 0]
ang_speed = 0
for i in range(10):
    current_time = env.robot.GetTimeSinceReset()
    env.reset()
    mpc.reset(env.robot)

    for j in range(1000):
        mpc.update_controller_params(lin_speed, ang_speed)
        mpc.controller.update()
        action, info = mpc.controller.get_action()
        # if config.RobotParams.motor_control_mode == 'torque':
        #     action = mpc.convert2torque(action, env.robot)
        _, _, ter, trunc, _ = env.step(action)
        current_time = env.robot.GetTimeSinceReset()
