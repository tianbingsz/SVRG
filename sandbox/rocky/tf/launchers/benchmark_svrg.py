"""
The original author for the benchmark is Yang Liu from UIUC.
We reference and borrow Yang Liu's ideas and code for our
own benchmark
create: Tiabing Xu, 7-10-2017
"""
import argparse
import datetime
from rllab.misc import logger
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.box2d.mountain_car_env import MountainCarEnv
from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.algos.pg_svrg import SVRGPG
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
import os

stub(globals())


def remove_space(s):
    return "_".join(s.strip().split())


def get_date():
    return remove_space(str(datetime.datetime.now()))


prefix_map = {
    'c': 'cartpole_swing_up',
    'd': 'double_pendulum_env',
    'car': 'cartpole',
    'mou': 'mountain_car',
    'pendulum': 'inverted_pendulum',
    'swim': 'swimmer',
    'hopper': 'hopper',
    'walker': 'walker',
    'cheetah': 'cheetah',
    'humanoid': 'humanoid',
}

env_map = {
    'c': CartpoleSwingupEnv,
    'd': DoublePendulumEnv,
    'car': CartpoleEnv,
    'mou': MountainCarEnv,
}

env_name_map = {
    'pendulum': 'InvertedPendulum-v1',
    'swim': 'Swimmer-v1',
    'hopper': 'Hopper-v1',
    'walker': 'Walker2d-v1',
    'cheetah': 'HalfCheetah-v1',
    'humanoid': 'Humanoid-v1',
}

algorithm_map = {
    'tr': 'trpo',
    'svrg': 'svrg',
}

"""
SVRG + CG
"""


def run_svrg(*_):
    envir = env_name_map[env_name]
    env = TfEnv(normalize(GymEnv(envir, record_video=False,
                                 force_reset=True)))
    policy = GaussianMLPPolicy(
        name="policy",
        env_spec=env.spec,
        hidden_sizes=(64, 64),
    )

    policy_tilde = GaussianMLPPolicy(
        name="policy_tilde",
        env_spec=env.spec,
        hidden_sizes=(64, 64),
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = SVRGPG(
        env=env,
        policy=policy,
        policy_tilde=policy_tilde,
        baseline=baseline,
        batch_size=batch_size,
        max_path_length=max_path_length,
        n_itr=n_itr,
        discount=0.995,
        delta=delta,
        optimizer_args=dict(
            batch_size=mini_batch_size,
            max_epochs=1,
            epsilon=1e-8,
            use_SGD=False,
            cg_iters=cg_iters,
            subsample_factor=subsample_factor,
            max_batch=max_batch,
        )
    )

    print("run svrg cg for env {:}".format(env_name))
    print("max_epochs {:}".format(max_epochs))
    print("cg_iters {:}".format(cg_iters))
    print("step size: {:}".format(delta))
    print("max_path_length : {:}".format(max_path_length))
    print("Num of Iterations: {:}".format(n_itr))
    print("Num of Examples: {:}".format(batch_size))
    print("sub sample rate: {:}".format(subsample_factor))
    print("batch size: {:}".format(mini_batch_size))
    print("max num batches: {:}".format(max_batch))
    return algo


"""
usage: benchmark_svrg_tr.py [-h] algo env_name random_seed batch_size
n_itr max_path_length delta mini_batch_size subsample_factor
max_batch

multiple runs

positional arguments:
  algo         algorithms
  env_name     env_ame
  random_seed
  batch_size
  n_itr        iterations

optional arguments:
  -h, --help   show this help message and exit
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="multiple runs")
    parser.add_argument("root_dir", type=str, help="log root dir")
    parser.add_argument("algo", type=str, help="algorithms")
    parser.add_argument("env_name", type=str, help="env_ame")
    parser.add_argument("random_seed", type=int)
    parser.add_argument("batch_size", type=int, default=5000)
    parser.add_argument("mini_batch_size", type=int, default=100)
    parser.add_argument("n_itr", type=int, help="iterations")
    parser.add_argument("max_path_length", type=int, default=500)
    parser.add_argument("delta", type=float, default=0.01)
    parser.add_argument("max_epochs", type=int, help="epochs")
    parser.add_argument("cg_iters", type=int, help="cg_iters")
    parser.add_argument("sample_rate", type=float, default=1.0)
    parser.add_argument("max_batch", type=int, help="max_batch")

    args = parser.parse_args()
    root_dir = args.root_dir
    n_itr = args.n_itr
    algorithm = algorithm_map[args.algo]
    env_name = args.env_name
    prefix = prefix_map[args.env_name]
    seed = int(args.random_seed)
    batch_size = int(args.batch_size)
    mini_batch_size = int(args.mini_batch_size)
    max_path_length = int(args.max_path_length)
    delta = float(args.delta)
    max_epochs = int(args.max_epochs)
    cg_iters = int(args.cg_iters)
    subsample_factor = float(args.sample_rate)
    max_batch = int(args.max_batch)

    algo = run_svrg()
    log_dir = os.path.join(root_dir,
                           "{:}_seed={:}_iter={:}_batch={:}_env={:}_{:}.{:}.{:}.{:}.{:}.{:}.{:}.{:}".format(
                               algorithm, seed, n_itr, mini_batch_size, prefix,
                               get_date(), batch_size, max_path_length, delta,
                               max_epochs, cg_iters, max_batch,
                               subsample_factor))
    print('log_dir {:}'.format(log_dir))
    run_experiment_lite(
        algo.train(),
        n_parallel=4,
        snapshot_mode="last",
        seed=seed,
        log_dir=log_dir,
    )
