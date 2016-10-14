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

sigmaLimit = 3.5
increment = 0.25
modelName = "Pendulum-v0"
neuronNum = 8
policies = [["Gaussian_MLP_8", GaussianMLPPolicy, [neuronNum, 0]], 
          ["Gaussian_MLP_8_8", GaussianMLPPolicy, [neuronNum, neuronNum]],
         ["Gaussian_GRU_8", GaussianGRUPolicy, [neuronNum, 0]]]
randomEnvs = [["Laplace", LaplaceNoiseEnv], ["Gaussian", GaussianNoiseEnv]]

log_base = "/Users/Tejas/Desktop/Research/rllab/data/local/experiment/"

for randomEnvInfo in randomEnvs:
    randomEnvName = randomEnvInfo[0]
    randomEnv = randomEnvInfo[1] 

    for policyInfo in policies:
        
        policyName = policyInfo[0]
        policy = policyInfo[1]
        neurons = policyInfo[2]

        sigma = 0
        while sigma <= sigmaLimit:


            gym_env = normalize(GymEnv(modelName))
            env = normalize(GaussianNoiseEnv(gym_env, sigma=sigma))
            log_dir = log_base + randomEnvName + "_" + policyName + "_" + str(sigma) + "/"

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
                n_itr=10,
                discount=0.99,
                step_size=0.01,
                # Uncomment both lines (this and the plot parameter below) to enable plotting
                plot=True,
                optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))

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

            sigma += increment


