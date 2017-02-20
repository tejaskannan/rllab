from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy
from rllab.envs.noisy.random_noise import GaussianNoiseEnv



def run_task(*_):
    original_env = normalize(GymEnv("LunarLanderContinuous-v2", record_video=False, force_reset=True))
    env = normalize(GaussianNoiseEnv(original_env, sigma=0.1))

    policy = GaussianGRUPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, )
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=500,
        n_itr=500,
        discount=0.99,
        step_size=0.03,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        plot=True,
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
    log_dir="/home/tejas/Documents/rllab/data/local/experiment/lunar_lander_v7/Gaussian_GaussianGRU_32_0_test"
)

