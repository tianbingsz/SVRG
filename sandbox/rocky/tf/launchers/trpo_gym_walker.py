from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import FiniteDifferenceHvp
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.envs.gym_env import GymEnv

stub(globals())

env = TfEnv(normalize(GymEnv("Walker2d-v1", record_video=False,
    force_reset=True)))

policy = GaussianMLPPolicy(
    name="policy",
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 64 hidden units.
    hidden_sizes=(64, 64),
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=50000,
    max_path_length=env.horizon,
    n_itr=10,
    discount=0.995,
    step_size=0.01,

)

run_experiment_lite(
    algo.train(),
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    n_parallel=4,
    seed=0,
    # plot=True,
)
