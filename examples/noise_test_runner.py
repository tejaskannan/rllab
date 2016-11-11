import os
import sys
import csv
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.noisy.random_noise import GaussianNoiseEnv, LaplaceNoiseEnv
from rllab.envs.noisy.random_noise import DroppedObservationsEnv, DroppedObservationsReplaceEnv
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp


stub(globals())

def get_noisy_env(noiseName, env, parameter):
    if noiseName == "Laplace":
        return LaplaceNoiseEnv(env, factor=parameter)
    if noiseName == "Gaussian":
        return GaussianNoiseEnv(env, sigma=parameter)
    if noiseName == "DroppedObservations":
        return DroppedObservationsEnv(env, probability=parameter)
    if noiseName == "DroppedObservationsReplace":
        return DroppedObservationsReplaceEnv(env, probability=parameter)
    return None


def get_algorithm(algo_name):
    if algo_name == "TRPO":
        return TRPO
    return None

def get_policy(policyName):
    if policyName == "GaussianMLP":
        return GaussianMLPPolicy
    if policyName == "GaussianGRU":
        return GaussianGRUPolicy
    if policyName == "CategoricalMLP":
        return CategoricalMLPPolicy
    return None


def run_test_csv(file_name):
    with open(file_name, 'r') as csvFile:
        dataReader = csv.reader(csvFile, delimiter=',', quotechar='|')
        lineNum = 0
        for row in dataReader:
            if (lineNum <= 0):
                lineNum += 1
                continue

            model_name = row[0]
            algorithm = get_algorithm(row[1]) 
            policy_name = row[2]
            neuron_layer_1 = int(row[3])
            neuron_layer_2 = int(row[4])
            noise_env_name = row[5]
            param_val = float(row[6])
            param_incr = float(row[7])
            param_end = float(row[8])
            iters = int(row[9])
            log_base = row[10]

            policy = get_policy(policy_name)


            while param_val <= param_end:

                gym_env = normalize(GymEnv(model_name, record_video=False))
                noisy_env = get_noisy_env(noise_env_name, gym_env, param_val)
                env = normalize(noisy_env)
                log_dir = log_base + noise_env_name + "_" + policy_name + "_" + str(neuron_layer_1) + "_" + str(neuron_layer_2) + "_" + str(param_val) + "/"

                if neuron_layer_2 == 0:
                    neuron_levels = (neuron_layer_1,)
                else:
                    neuron_levels = (neuron_layer_1, neuron_layer_2)

                p = policy(
                    env_spec=env.spec,
                    hidden_sizes=neuron_levels
                )

                baseline = LinearFeatureBaseline(env_spec=env.spec)

                algo = algorithm(
                    env=env,
                    policy=p,
                    baseline=baseline,
                    batch_size=4000,
                    max_path_length=env.horizon,
                    n_itr=iters,
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
                param_val = round(param_val, 3)


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
file_name = os.path.join(__location__, 'mountain_car_config.csv')
run_test_csv(file_name)

