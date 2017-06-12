import numpy as np
import csv
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.proxy_env import ProxyEnv
from rllab.misc import autoargs
from rllab.misc.overrides import overrides

class GeneralizedNoisyEnv(ProxyEnv, Serializable):

    @autoargs.arg('obs_noise', type=float,
                  help='Noise added to the observations (note: this makes the '
                       'problem non-Markovian!)')
    def __init__(self, env):
        Serializable.quick_init(self, locals())
        super(GeneralizedNoisyEnv, self).__init__(env)

    def get_obs_noise_scale_factor(self, obs):
        return np.ones_like(obs)

    def inject_obs_noise(self, obs):
        """
        Inject entry-wise noise to the observation. This should not change
        the dimension of the observation. This is an abstract method that 
        must be implemented by concrete noise generators
        """
        raise NotImplementedError("No Noise Function Defined")

    def get_current_obs(self):
        obs = self._wrapped_env.get_current_obs()
        noisy_obs = self.inject_obs_noise(obs)
        return noisy_obs


    @overrides
    def reset(self):
        obs = self._wrapped_env.reset()
        return self.inject_obs_noise(obs)

    @overrides
    def step(self, action):
        next_obs, reward, done, info = self._wrapped_env.step(action)
        return Step(self.inject_obs_noise(next_obs), reward, done, **info)

