import os
import sys
import csv
from rllab.algos.trpo import TRPO
from rllab.algos.vpg import VPG
from rllab.algos.tnpg import TNPG
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
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv


# stub(globals())

def get_noisy_env(noiseName, env, value):
    if noiseName == "Laplace":
        return LaplaceNoiseEnv(env, factor=value)
    if noiseName == "Gaussian":
        return GaussianNoiseEnv(env, sigma=value)
    if noiseName == "DroppedObservations":
        return DroppedObservationsEnv(env, probability=value)
    if noiseName == "DroppedObservationsReplace":
        return DroppedObservationsReplaceEnv(env, probability=value)
    return None

def get_env(env_name):
    if env_name == "Pendulum-v0":
        return normalize(GymEnv(env_name, record_video=False, force_reset=True))
    elif env_name == "MountainCarContinuous-v0":
        return normalize(GymEnv(env_name, record_video=False, force_reset=True))
    elif env_name == "LunarLanderContinuous-v2":
        return normalize(GymEnv(env_name, record_video=False, force_reset=True))
    elif env_name == "CartpoleEnv":
        return normalize(CartpoleEnv())
    elif env_name == "DoublePendulumEnv":
        return normalize(DoublePendulumEnv())
    return None

def get_env_range(env_name):
    if env_name == "Pendulum-v0":
        return 4
    elif env_name == "MountainCarContinuous-v0":
        return 1
    elif env_name == "CarRacing-v0":
        return 2
    elif env_name == "LunarLanderContinuous-v2":
        return 1
    elif env_name == "CartpoleEnv": 
        return 10
    elif env_name == "DoublePendulumEnv":
        return 50
    return -1


def get_value_from_param(param, env):
    r = get_env_range(env)
    if r < 0:
        return param
    return float(param) * r


def get_algorithm(algo_name):
    if algo_name == "TRPO":
        return TRPO
    if algo_name == "VPG":
        return VPG
    if algo_name == "TNPG":
        return TNPG
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

                log_dir = log_base + noise_env_name + "_" + policy_name + "_" + str(neuron_layer_1) + "_" + str(neuron_layer_2) + "_" + str(param_val) + "/"
                print(log_dir)
                model_env = get_env(model_name)
                if noise_env_name != "Gaussian" or noise_env_name != "Laplace":
                    value = param_val
                else:
                    value = get_value_from_param(param_val, model_name)
                print("Parameter Value: " + str(value))
                noisy_env = get_noisy_env(noise_env_name, model_env, param_val)

                def run_task(*_):
                    env = normalize(noisy_env)

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
                        max_path_length=500,
                        n_itr=iters,
                        discount=0.99,
                        step_size=0.03,
                        plot=True
                    )
                    algo.train()

                run_experiment_lite(
                    run_task,
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
file_name = os.path.join(__location__, 'configs/lunar_lander_32_tnpg.csv')
run_test_csv(file_name)

