from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.noisy.random_noise import DroppedObservationsEnv
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy

stub(globals())

probLimit = 1
increment = 0.1
modelName = "Pendulum-v0"
neuronNum = 8
policies = [["Gaussian_MLP_8", GaussianMLPPolicy, [neuronNum, 0]], 
            ["Gaussian_MLP_8_8", GaussianMLPPolicy, [neuronNum, neuronNum]],
            ["Gaussian_GRU_8", GaussianGRUPolicy, [neuronNum, 0]]]

log_base = "/Users/Tejas/Desktop/Research/rllab/data/local/experiment/updated/"
envName = "DroppedObservations"
replace = False

for i in range(0, 2):
    for policyInfo in policies:
        
        policyName = policyInfo[0]
        policy = policyInfo[1]
        neurons = policyInfo[2]

        prob = 0
        while prob <= probLimit:

            if (i == 0 and (policyName == "Gaussian_MLP_8" or policyName == "Gaussian_MLP_8_8")):
                prob = probLimit + 1
                continue

            gym_env = normalize(GymEnv(modelName))
            env = normalize(DroppedObservationsEnv(gym_env, probability=prob, replace=replace))

            if (replace):
                log_dir = log_base + envName + "_" + policyName + "_Replaced_" + str(prob) + "/"
            else:
                log_dir = log_base + envName + "_" + policyName + "_" + str(prob) + "/"

            if neurons[1] == 0:
                neuronLevels = (neurons[0],)
            else:
                neuronLevels = (neurons[0], neurons[1])

            p = policy(
                env_spec=env.spec,
                # The neural network policy should have two hidden layers, each with 32 hidden units.
                hidden_sizes=neuronLevels
            )

            baseline = LinearFeatureBaseline(env_spec=env.spec)

            algo = TRPO(
                env=env,
                policy=p,
                baseline=baseline,
                batch_size=4000,
                max_path_length=env.horizon,
                n_itr=50,
                discount=0.99,
                step_size=0.01,
                # Uncomment both lines (this and the plot parameter below) to enable plotting
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

            prob += increment
    replace = True