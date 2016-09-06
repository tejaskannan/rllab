import numpy as np
from rllab.envs.noisy.generalized_noisy_env import GeneralizedNoisyEnv

# This file has a collection of random noise environments

# Environment with Gaussian Noise
class GaussianNoiseEnv(GeneralizedNoisyEnv):

	def __init__(self, env, mu=0, sigma=1):
		super(GaussianNoiseEnv, self).__init__(env)
		self.mu = mu
		self.sigma = sigma

	@overrides
	def inject_obs_noise(self, obs):
		noise = (np.random.standard_normal(size=obs.shape) * self.sigma) + self.mu
		return obs + noise

# Environment with Laplace Noise
class LaplaceNoiseEnv(GeneralizedNoisyEnv):

	def __init__(self, env, mu=0, sigma=1):
		super(LaplaceNoiseEnv, self).__init__(env)
		self.mu = mu
		self.sigma = sigma

	@overrides
	def inject_obs_noise(self, obs):
		noise = np.random.laplace(self.mu, self.sigma, obs.shape)
		return obs + noise


# Environment with Zipfian Noise (used for high-magnitude outliers)
class ZipfNoiseEnv(GeneralizedNoisyEnv):

	# z is the Zipf distribution parameter (must be greater than 1)
	def __init__(self, env, z=3):
		super(ZipfNoiseEnv, self).__init__(env)
		self.z = z

	@overrides
	def inject_obs_noise(self, obs):
		noise = np.random.zipf(self.z, obs.shape)
		return obs + noise

# Environment with Uniform Noise
class UniformNoiseEnv(GeneralizedNoisyEnv):

	def __init__(self, env, noise_factor=0.1):
		super(UniformNoiseEnv, self).__init__(env)
		self.noise_factor = noise_factor
		self.max = 1
		self.min = -1

	@overrides
	def inject_obs_noise(self, obs):
		noise = np.random.uniform(self.min, self.max, obs.shape) * self.noise_factor
		return obs + noise


# Randomly drops entries of the observations
class DroppedObservationsEnv(GeneralizedNoisyEnv):

	# randomly fills an entry with the placeholder value using the given probability
	def __init__(self, env, probability=0, placeholder=0):
		super(DroppedObservationsEnv, self).__init__(env)
		self.probability = probability
		self.placeholder = placeholder


	@overrides
	def inject_obs_noise(self, obs):
		copy = np.copy(obs):
		rows = obs.shape[0]
		cols = obs.shape[1]
		random_vals = np.random.uniform(0, 1, obs.shape)

		for i in range(0, rows):
			for j in range(0, cols):
				sample = random[i, j]
				if (sample < probability):
					copy[i, j] = placeholder

		return copy






