import gymnasium
import numpy as np
import copy


class BaseWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super(BaseWrapper, self).__init__(env)
        self._wrapped_env = env
        self.training = True

    def train(self):
        if isinstance(self._wrapped_env, BaseWrapper):
            self._wrapped_env.train()
        self.training = True

    def eval(self):
        if isinstance(self._wrapped_env, BaseWrapper):
            self._wrapped_env.eval()
        self.training = False

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._wrapped_env, attr)

    def copy_state(self, source_env):
        pass


class RewardShift(gymnasium.RewardWrapper, BaseWrapper):
    def __init__(self, env, reward_scale=1):
        super(RewardShift, self).__init__(env)
        self._reward_scale = reward_scale

    def reward(self, reward):
        if self.training:
            return self._reward_scale * reward
        else:
            return reward


def update_mean_var_count(
        mean, var, count,
        batch_mean, batch_var, batch_count):
    """
  Imported From OpenAI Baseline
  """
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class Normalizer():
    def __init__(self, shape, clip=10.):
        self.shape = shape
        self._mean = np.zeros(shape)
        self._var = np.ones(shape)
        self._count = 1e-4
        self.clip = clip
        self.should_estimate = True

    def stop_update_estimate(self):
        self.should_estimate = False

    def update_estimate(self, data):
        if not self.should_estimate:
            return
        if len(data.shape) == self.shape:
            data = data[np.newaxis, :]
        self._mean, self._var, self._count = update_mean_var_count(
            self._mean, self._var, self._count,
            np.mean(data, axis=0), np.var(data, axis=0), data.shape[0])

    def inverse(self, raw):
        return raw * np.sqrt(self._var) + self._mean

    def inverse_torch(self, raw):
        return raw * torch.Tensor(np.sqrt(self._var)).to(raw.device) \
               + torch.Tensor(self._mean).to(raw.device)

    def filt(self, raw):
        return np.clip(
            (raw - self._mean) / (np.sqrt(self._var) + 1e-4),
            -self.clip, self.clip)

    def filt_torch(self, raw):
        return torch.clamp(
            (raw - torch.Tensor(self._mean).to(raw.device)) /
            (torch.Tensor(np.sqrt(self._var) + 1e-4).to(raw.device)),
            -self.clip, self.clip)


class NormObs(gymnasium.ObservationWrapper, BaseWrapper):
    """
  Normalized Observation => Optional, Use Momentum
  """

    def __init__(self, env, epsilon=1e-4, clipob=10.):
        super(NormObs, self).__init__(env)
        self.count = epsilon
        self.clipob = clipob
        self._obs_normalizer = Normalizer(env.observation_space.shape)

    def copy_state(self, source_env):
        # self._obs_rms = copy.deepcopy(source_env._obs_rms)
        self._obs_var = copy.deepcopy(source_env._obs_var)
        self._obs_mean = copy.deepcopy(source_env._obs_mean)

    def observation(self, observation):
        if self.training:
            self._obs_normalizer.update_estimate(observation)
        return self._obs_normalizer.filt(observation)