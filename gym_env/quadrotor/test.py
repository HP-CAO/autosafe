import numpy as np

from gym_env.quadrotor.quadrotor import Quadrotor
from gym_env.quadrotor.quadrotor_config import QuadrotorGymConfig
import time
from agent.ModelLearning.MlpModel import MlpConfig
mlp_config = MlpConfig()

env_params = QuadrotorGymConfig()
env_params.RobotParams.quad_type = '3d'

env = Quadrotor(env_params)
obs = env.reset()

print("action_space", env.action_space)
print("observation_space", env.observation_space)

for i in range(1000):
    action = env.action_space.sample()
    action = np.zeros_like(action)
    obs, reward, done, _, info = env.step(action)
    print(obs)
    time.sleep(1)








