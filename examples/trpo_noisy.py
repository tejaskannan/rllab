from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.noisy.random_noise import GaussianNoiseEnv
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy

stub(globals())

log_dir_base = "./data/noisy_tests/"
policies = {"Gaussian MLP" : GaussianMLPPolicy}
sigmas = [0, 0.01, 0.1, 0.5, 1]
infoFileName = "info.txt"
modelName = "Pendulum-v0"

test_txt = "./data/noisy_tests/info.txt"

index = 0
for sigma in sigmas:
    for name in policies:
        policy = policies[name]
        gym_env = normalize(GymEnv(modelName))
        env = normalize(GaussianNoiseEnv(gym_env, sigma=sigma))

        p = GaussianGRUPolicy(
            env_spec=env.spec,
            # The neural network policy should have two hidden layers, each with 32 hidden units.
            hidden_sizes=(8,)
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
            plot=True
        )


