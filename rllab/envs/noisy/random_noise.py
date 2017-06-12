import numpy as np
from rllab.envs.noisy.generalized_noisy_env import GeneralizedNoisyEnv
from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable


# This file has a collection of random noise environments

# Environment with Gaussian Noise
class GaussianNoiseEnv(GeneralizedNoisyEnv, Serializable):

	def __init__(self, env, mu=0, sigma=1):
		Serializable.quick_init(self, locals())
		super(GaussianNoiseEnv, self).__init__(env)
		self.mu = mu
		self.sigma = sigma

	@overrides
	def inject_obs_noise(self, obs):
		noise = (np.random.standard_normal(size=obs.shape) * self.sigma) + self.mu
		noisy_obs = noise + obs
		return noisy_obs

# Environment with Laplace Noise
class LaplaceNoiseEnv(GeneralizedNoisyEnv, Serializable):

	def __init__(self, env, mu=0, scale=1, factor=1):
		super(LaplaceNoiseEnv, self).__init__(env)
		Serializable.quick_init(self, locals())
		self.mu = mu
		self.scale = scale
		self.factor = factor

	@overrides
	def inject_obs_noise(self, obs):
		noise = np.random.laplace(self.mu, self.scale, obs.shape) * self.factor
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
		
		random_vals = np.random.uniform(0, 1, obs.shape)

		dropped = False

		if len(obs.shape) > 1:
			cols = obs.shape[1]
			for i in range(0, rows):
				for j in range(0, cols):
					sample = random_vals[i, j]
					if (sample < self.probability):
						copy[i, j] = self.placeholder
						dropped = True
		else:
			for i in range(0, rows):
					sample = random_vals[i]
					if (sample < self.probability):
						copy[i] = self.placeholder
						dropped = True

		if self.replace:
			if dropped:
				if self.last_correct == None:
					placeholder_obs = np.empty(obs.shape)
					placeholder_obs.fill(self.placeholder)
					return placeholder_obs
				return self.last_correct
			else:
				self.last_correct = obs

		return copy

class DroppedObservationsReplaceEnv(DroppedObservationsEnv, Serializable):

	def __init__(self, env, probability=0, placeholder=0):
		super(DroppedObservationsReplaceEnv, self).__init__(env, probability, placeholder, True)
		Serializable.quick_init(self, locals())



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



