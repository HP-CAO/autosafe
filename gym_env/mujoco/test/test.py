import gymnasium as gym
from time import sleep
from gymnasium import envs

for env in envs.registry:
    print(env)

env = gym.make('HalfCheetah-v5', xml_file="./gym_env/mujoco/half_cheetah_update.xml", ctrl_cost_weight=0.1, render_mode="human")
env.reset()

for i in range(1000):
    env.render()
    env.step(env.action_space.sample())
    result = env.render()
    sleep(0.01)
