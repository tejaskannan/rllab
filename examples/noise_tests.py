import os
import sys
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.noisy.random_noise import GaussianNoiseEnv
from rllab.envs.noisy.random_noise import LaplaceNoiseEnv
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp


stub(globals())

class PolicyConfiguration:
    name = ""
    policy = None
    neurons = []
    algorithm = None

    def __init__(self, policyName, algoName, neuron1, neuron2=0):
        self.name = policyName
        self.policy = self.get_policy(policyName)
        self.algorithm = get_algorithm(algo)
        self.neurons[0] = neuron1
        self.neurons[1] = neuron2

    def get_algorithm(algo_name):
        if algo_name == "TRPO":
            return TRPO
        return None

    def get_policy(policyName):
        if policyName == "GaussianMLP":
            return GaussianMLPPolicy
        if policyName == "GaussianGRU":
            return GaussianGRUPolicy
        return None

class NoiseConfiguration:
    param_val = 0
    param_start = 0
    param_incr = 0
    param_end = 0
    noise_names = None
    index = 0
    
    def __init__(self, noise_names, param_start, param_incr, param_end):
        self.param_val = param_start
        self.param_start = param_start
        self.param_end = param_end
        self.param_incr = param_incr
        self.noise_names = noise_names
        self.index = 0

    def __next__(self):
        # iterator that returns the noise environments



def get_noisy_env(noiseName, env, parameter):
    if noiseName == "Laplace":
        return LaplaceNoiseEnv(env, sigma=parameter)
    if noiseName == "Gaussian":
        return GaussianNoiseEnv(env, sigma=parameter)
    if noiseName == "DroppedObservations":
        return DroppedObservationsEnv(env, probability=parameter)
    return None






def run_test(file_name):
    config_file = open(file_name, 'r')
    modelName = get_line(config_file) # line 1 is the model name
    log_base = get_line(config_file) # line 2 is the base directory for the logging information
    noisy_env_names = parse_noise_names(get_line(config_file)) # line 3 is the list of noise environments
    policies = parse_policy_names(get_line(config_file)) # line 4 contains policy and neural net info
    parameter_info = parse_parameter_info(get_line(config_file, False)) # line 5 contains info about parameters

    param_start = parameter_info[0]
    param_incr = parameter_info[1]
    param_end = parameter_info[2]

    policies = [["Gaussian_MLP_8", GaussianMLPPolicy, [8, 0], TRPO], 
                ["Gaussian_MLP_8_8", GaussianMLPPolicy, [8, 8], TRPO],
                ["Gaussian_GRU_8", GaussianGRUPolicy, [8, 0], TRPO]]


    for policyInfo in policies:

        policyName = policyInfo[0]
        policy = policyInfo[1]
        neurons = policyInfo[2]
        algorithm = policyInfo[3]

        for noisy_env_name in noisy_env_names:

            param_val = param_start

            while param_val <= param_end:

                gym_env = normalize(GymEnv(modelName))
                noisy_env = get_noisy_env(noisy_env_name, gym_env, param_val)
                env = normalize(noisy_env)
                log_dir = log_base + noisy_env_name + "_" + policyName + "_" + str(param_val) + "/"

                if len(neurons) == 1:
                    neuronLevels = (neurons[0],)
                else:
                    neuronLevels = (neurons[0], neurons[1])

                p = policy(
                    env_spec=env.spec,
                    hidden_sizes=neuronLevels
                )

                baseline = LinearFeatureBaseline(env_spec=env.spec)

                algo = algorithm(
                    env=env,
                    policy=p,
                    baseline=baseline,
                    batch_size=4000,
                    max_path_length=env.horizon,
                    n_itr=50,
                    discount=0.99,
                    step_size=0.01,
                    plot=True
                )

                run_experiment_lite(
                    algo.train(),
                    # Number of parallel workers for sampling
                    n_parallel=1,
                    # Only keep the snapshot parameters for the last iteration
                    snapshot_mode="last",
                    # Specifies the seed for the experiment. If this is not provided, a random seed
                    # will be used
                    seed=1,
                    plot=True,
                    log_dir=log_dir
                )

                param_val += param_incr



run_test(file_name)