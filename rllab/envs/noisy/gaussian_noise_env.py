import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.noisy.generalized_noisy_env import GeneralizedNoisyEnv
from rllab.misc import autoargs
from rllab.misc.overrides import overrides

class GaussianNoiseEnv(GeneralizedNoisyEnv):


	def __init__(self, env, obs_noise=1e-1):
		super(GaussianNoiseEnv, self).__init__(env)
		self.obs_noise = obs_noise

	def inject_obs_noise(self, obs):
		noise = self.get_obs_noise_scale_factor(obs) * self.obs_noise * np.random.normal(size=obs.shape)
		return obs + noise
