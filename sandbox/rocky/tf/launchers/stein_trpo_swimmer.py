from sandbox.rocky.tf.algos.trpo_stein import TRPOStein
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.envs.gym_env import GymEnv

stub(globals())

env = TfEnv(normalize(GymEnv("Swimmer-v1", record_video=False)))

policy = GaussianMLPPolicy(
    name="policy",
    env_spec=env.spec,
    hidden_sizes=(64, 64),
    adaptive_std=True,
    std_hidden_sizes=(64, 64),
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPOStein(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=50000,
    max_path_length=env.horizon,
    n_itr=100,
    discount=0.995,
    delta=0.01,
    optimizer_args=dict(
        max_epochs=1,
        alpha = 1,
        subsample_factor=0.1,
    )
)

run_experiment_lite(
    algo.train(),
    n_parallel=4,
    seed=1,
)
