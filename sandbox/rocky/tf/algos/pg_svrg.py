import time
from rllab.misc import logger
from rllab.misc import ext
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt
from sandbox.rocky.tf.optimizers.svrg import SVRGOptimizer
from sandbox.rocky.tf.misc import tensor_utils
from rllab.core.serializable import Serializable
import tensorflow as tf
import numpy as np


class SVRGPG(BatchPolopt, Serializable):
    """
    SVRG Trust Region Policy Gradient.
    """

    def __init__(
            self,
            env,
            policy,
            policy_tilde,
            baseline,
            optimizer=None,
            optimizer_args=None,
            delta=0.01,
            **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict(
                    batch_size=None,
                    max_epochs=1,
                    scale=1.0,
                )
            optimizer = SVRGOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.opt_info = None
        self.policy_tilde = policy_tilde
        self.delta = delta
        super(SVRGPG, self).__init__(
            env=env, policy=policy, baseline=baseline, **kwargs)

    @overrides
    def init_opt(self):
        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1,
        )
        advantage_var = tensor_utils.new_tensor(
            name='advantage',
            ndim=1,
            dtype=tf.float32,
        )
        dist = self.policy.distribution
        dist_tilde = self.policy_tilde.distribution

        old_dist_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] + list(shape), name='old_%s' % k)
            for k, shape in dist.dist_info_specs
        }
        old_dist_info_vars_list = [old_dist_info_vars[k]
                                   for k in dist.dist_info_keys]

        state_info_vars = {
            k: tf.placeholder(tf.float32, shape=[None] + list(shape), name=k)
            for k, shape in self.policy.state_info_specs
        }
        state_info_vars_list = [state_info_vars[k]
                                for k in self.policy.state_info_keys]

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        # todo, delete this var
        state_info_vars_tilde = {
            k: tf.placeholder(tf.float32, shape=[None] + list(shape), name=k)
            for k, shape in self.policy_tilde.state_info_specs
        }
        dist_info_vars_tilde = self.policy_tilde.dist_info_sym(obs_var,
                                                               state_info_vars_tilde)
        loglik = dist.log_likelihood_sym(action_var, dist_info_vars)
        loglik_tilde = dist_tilde.log_likelihood_sym(
            action_var, dist_info_vars_tilde)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)

        # formulate as a minimization problem
        # The gradient of the surrogate objective is the policy gradient
        surr_obj = - tf.reduce_mean(loglik * advantage_var)
        surr_tilde_obj = - tf.reduce_mean(loglik_tilde * advantage_var)
        mean_kl = tf.reduce_mean(kl)
        max_kl = tf.reduce_max(kl)

        input_list = [
            obs_var,
            action_var,
            advantage_var
        ] + state_info_vars_list + old_dist_info_vars_list

        self.optimizer.update_opt(loss=surr_obj, loss_tilde=surr_tilde_obj,
                                  target=self.policy, target_tilde=self.policy_tilde,
                                  leq_constraint=(mean_kl, self.delta), inputs=input_list)

        f_kl = tensor_utils.compile_function(
            inputs=input_list + old_dist_info_vars_list,
            outputs=[mean_kl, max_kl],
        )
        self.opt_info = dict(
            f_kl=f_kl,
        )

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            self.start_worker()
            start_time = time.time()
            num_samples = 0
            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                with logger.prefix('itr #%d | ' % itr):
                    logger.log("Obtaining new samples...")
                    paths = self.obtain_samples(itr)
                    for path in paths:
                        num_samples += len(path["rewards"])
                    logger.log("total num samples..." + str(num_samples))
                    logger.log("Processing samples...")
                    samples_data = self.process_samples(itr, paths)
                    logger.log("Logging diagnostics...")
                    self.log_diagnostics(paths)
                    logger.log("Optimizing policy...")
                    self.optimize_policy(samples_data)
                    logger.log("Saving snapshot...")
                    params = self.get_itr_snapshot(
                        itr, samples_data)  # , **kwargs)
                    if self.store_paths:
                        params["paths"] = samples_data["paths"]
                    logger.save_itr_params(itr, params)
                    logger.log("Saved")
                    logger.record_tabular('Time', time.time() - start_time)
                    logger.record_tabular(
                        'ItrTime', time.time() - itr_start_time)
                    logger.dump_tabular(with_prefix=False)
                    if self.plot:
                        self.update_plot()
                        if self.pause_for_plot:
                            input("Plotting evaluation run: Press Enter to "
                                  "continue...")
        self.shutdown_worker()

    @overrides
    def optimize_policy(self, samples_data):
        logger.log("optimizing policy")
        inputs = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k]
                          for k in self.policy.distribution.dist_info_keys]
        inputs += tuple(state_info_list) + tuple(dist_info_list)
        loss_before = self.optimizer.loss(inputs)
        self.optimizer.optimize(inputs)
        loss_after = self.optimizer.loss(inputs)
        logger.record_tabular("LossBefore", loss_before)
        logger.record_tabular("LossAfter", loss_after)

        mean_kl, max_kl = self.opt_info['f_kl'](
            *(list(inputs)))
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('MaxKL', max_kl)

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
