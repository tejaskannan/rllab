import numpy as np
from rllab.envs.noisy.generalized_noisy_env import GeneralizedNoisyEnv
from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable


# This file has a collection of random noise environments

# Environment with Gaussian Noise
class GaussianNoiseEnv(GeneralizedNoisyEnv, Serializable):

	def __init__(self, env, mu=0, sigma=1):
		super(GaussianNoiseEnv, self).__init__(env)
		Serializable.quick_init(self, locals())
		self.mu = mu
		self.sigma = sigma


	@overrides
	def inject_obs_noise(self, obs):
		noise = (np.random.standard_normal(size=obs.shape) * self.sigma) + self.mu
		return obs + noise

# Environment with Laplace Noise
class LaplaceNoiseEnv(GeneralizedNoisyEnv, Serializable):

	def __init__(self, env, mu=0, sigma=1):
		super(LaplaceNoiseEnv, self).__init__(env)
		Serializable.quick_init(self, locals())
		self.mu = mu
		self.sigma = sigma

	@overrides
	def inject_obs_noise(self, obs):
		noise = np.random.laplace(self.mu, self.sigma, obs.shape)
		return obs + noise


# Environment with Zipfian Noise (used for high-magnitude outliers)
class ZipfNoiseEnv(GeneralizedNoisyEnv, Serializable):

	# z is the Zipf distribution parameter (must be greater than 1)
	def __init__(self, env, z=3):
		super(ZipfNoiseEnv, self).__init__(env)
		Serializable.quick_init(self, locals())
		self.z = z

	@overrides
	def inject_obs_noise(self, obs):
		noise = np.random.zipf(self.z, obs.shape)
		return obs + noise

# Environment with Uniform Noise
class UniformNoiseEnv(GeneralizedNoisyEnv, Serializable):

	def __init__(self, env, noise_factor=0.1):
		super(UniformNoiseEnv, self).__init__(env)
		Serializable.quick_init(self, locals())
		self.noise_factor = noise_factor
		self.max = 1
		self.min = -1

	@overrides
	def inject_obs_noise(self, obs):
		noise = np.random.uniform(self.min, self.max, obs.shape) * self.noise_factor
		return obs + noise


# Randomly drops entries of the observations
class DroppedObservationsEnv(GeneralizedNoisyEnv, Serializable):

	# randomly fills an entry with the placeholder value using the given probability
	def __init__(self, env, probability=0, placeholder=0, replace=False):
		super(DroppedObservationsEnv, self).__init__(env)
		Serializable.quick_init(self, locals())
		self.probability = probability
		self.placeholder = placeholder
		self.replace = replace
		self.last_correct = None


	@overrides
	def inject_obs_noise(self, obs):
		copy = np.copy(obs)
		rows = obs.shape[0]
		cols = obs.shape[1]
		random_vals = np.random.uniform(0, 1, obs.shape)

		dropped = False

		for i in range(0, rows):
			for j in range(0, cols):
				sample = random[i, j]
				if (sample < probability):
					copy[i, j] = placeholder
					dropped = True

		if self.replace:
			if dropped:
				if self.last_correct == None:
					return np.fill(obs.shape, placeholder)
				return self.last_correct
			else:
				self.last_correct = obs

		return copy

class DuplicateObservationsEnv(GeneralizedNoisyEnv, Serializable):

	def __init__(self, probability=0, placeholder=0):
		super(DroppedObservationsEnv, self).__init__(env)
		Serializable.quick_init(self, locals())
		self.probability = probability
		self.last = None
		self.placeholder = placeholder


	@overrides
	def inject_obs_noise(self, obs):
		r = random.random()
		prev = self.last
		self.last = obs

		if r >= probability:
			return obs

		if self.last == None:
			return np.fill(obs.shape, self.placeholder)

		return prev



class PoissonNoiseEnv(GeneralizedNoisyEnv, Serializable):

	# lam is the lambda parameter for the poisson distribution
	def __init__(self, env, lam=1.0):
		super(PoissonNoiseEnv, self).__init__(env)
		Serializable.quick_init(self, locals())
		self.lam = lam

	@overrides
	def inject_obs_noise(self, obs):
		noise = np.random.poisson(self.lam, obs.shape)
		return obs + noise



