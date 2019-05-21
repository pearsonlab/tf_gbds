import tensorflow as tf
import numpy as np
# import edward as ed
from edward.models import RandomVariable, ExpRelaxedOneHotCategorical
from tensorflow.contrib.distributions import (Distribution,
                                              FULLY_REPARAMETERIZED)
from tensorflow.python.ops.distributions.special_math import log_ndtr


class GBDS(RandomVariable, Distribution):
    def __init__(self, params, states, extra_conds, *args, **kwargs):
        name = kwargs.get("name", "GBDS")
        with tf.name_scope(name):
            self.col = params["col"]
            self.dim = params["dim"]
            with tf.name_scope("batch_size"):
                self.B = tf.shape(states)[0]
            with tf.name_scope("trial_length"):
                self.Tt = tf.shape(states)[1]
            # self.epoch = epoch
            self.s = tf.identity(states, "states")
            self.y = tf.gather(states, self.col, axis=-1, name="positions")
            self.s_dim = params["state_dim"]
            self.extra_dim = params["extra_dim"]
            self.extra_conds = tf.identity(
                extra_conds, "extra_conditions")
            self.temperature = params["temperature"]

            # number of GMM components
            self.K = params["GMM_K"]
            # neural network to generate state-dependent goals
            self.G_NN = params["GMM_NN"]
            # neural network for latent state transition
            self.A_NN = params["A_NN"]
            self.var_list = params["GMM_NN_vars"] + params["A_NN_vars"]
            self.log_vars = params["GMM_NN_vars"] + params["A_NN_vars"]

            with tf.name_scope("goal_state_penalty"):
                # penalty on goal state escaping boundaries
                if params["g_bounds_pen"] is not None:
                    self.g_pen = tf.constant(
                        params["g_bounds_pen"], tf.float32,
                        name="goal_boundary_penalty")
                    with tf.name_scope("goal_state_boundary"):
                        # boundaries for penalty
                        if params["g_bounds"] is not None:
                            self.bounds = params["g_bounds"]
                        else:
                            self.bounds = [-1., 1.]
                else:
                    self.g_pen = None

                # penalty on the precision of GMM components
                if params["g_prec_pen"] is not None:
                    self.g_prec_pen = tf.constant(
                        params["g_prec_pen"], tf.float32,
                        name="goal_precision_penalty")

            with tf.name_scope("control_signal_noise"):
                # noise coefficient on control signals
                self.unc_eps = params["unc_eps"]
                self.eps = tf.nn.softplus(self.unc_eps, "epsilon")
                self.eps_pen = params["eps_pen"]
                if params["eps_trainable"]:
                    self.var_list += [self.unc_eps]

        if "name" not in kwargs:
            kwargs["name"] = name
        if "dtype" not in kwargs:
            kwargs["dtype"] = tf.float32
        if "reparameterization_type" not in kwargs:
            kwargs["reparameterization_type"] = FULLY_REPARAMETERIZED
        if "validate_args" not in kwargs:
            kwargs["validate_args"] = True
        if "allow_nan_stats" not in kwargs:
            kwargs["allow_nan_stats"] = False

        super(GBDS, self).__init__(*args, **kwargs)
        self._args = (params, states, extra_conds)

    def get_GMM(self, s, extra_conds):
        NN_output = self.G_NN(tf.concat([s, extra_conds], -1))
        mu = tf.reshape(
            NN_output[:, :, :(self.K * self.dim)],
            [self.B, -1, self.K, self.dim], "mu")
        lmbda = tf.reshape(
            tf.nn.softplus(NN_output[:, :, (self.K * self.dim):]),
            [self.B, -1, self.K, self.dim], "lambda")

        return mu, lmbda

    def transition(self, z, s, extra_conds):
        log_alphas = tf.identity(
            self.A_NN(tf.concat([z, s, extra_conds], -1)), "logits")

        return log_alphas

    def _log_prob(self, value):
        g_q = tf.identity(value[:, :, :self.dim], "goal_posterior")
        z_q = tf.nn.log_softmax(
            value[:, :, self.dim:], -1, "latent_state_posterior")

        G_mu, G_lambda = self.get_GMM(self.s, self.extra_conds)

        logdensity_g = 0.0
        with tf.name_scope("goals"):
            gmm_res = tf.subtract(
                tf.expand_dims(g_q[:, 1:], 2, "reshape_samples"),
                G_mu[:, :-1], "GMM_residual")
            gmm_term = -tf.reduce_sum(
                (0.5 * tf.log(2 * np.pi) - tf.log(G_lambda[:, :-1]) +
                    (gmm_res * G_lambda[:, :-1]) ** 2 / 2), -1) + z_q[:, 1:]
            logdensity_g += tf.reduce_sum(
                tf.reduce_logsumexp(gmm_term, -1), -1)

        with tf.name_scope("goal_penalty"):
            if self.g_pen is not None:
                # penalty on goal state escaping game space
                logdensity_g -= self.g_pen * tf.reduce_sum(
                    tf.nn.relu(self.bounds[0] - G_mu) ** 2, [1, 2, 3])
                logdensity_g -= self.g_pen * tf.reduce_sum(
                    tf.nn.relu(G_mu - self.bounds[1]) ** 2, [1, 2, 3])

                # penalty on GMM precision
                logdensity_g -= self.g_prec_pen * tf.reduce_sum(
                    1. / G_lambda, [1, 2, 3])

        logits = self.transition(
            z_q[:, :-1], self.s[:, :-1], self.extra_conds[:, :-1])

        logdensity_z = tf.reduce_sum(ExpRelaxedOneHotCategorical(
            self.temperature, logits=logits).log_prob(z_q[:, 1:]), -1)

        logdensity_y = 0.0
        with tf.name_scope("positions"):
            y_res = tf.subtract(self.y[:, :-1], g_q[:, 1:], "residual")
            logdensity_y -= tf.reduce_sum(
                (0.5 * tf.log(2 * np.pi) + tf.log(self.eps) +
                 y_res ** 2 / (2 * self.eps ** 2)), [1, 2])

        if self.eps_pen is not None:
            logdensity_y -= self.eps_pen * tf.reduce_sum(self.unc_eps)

        logdensity = tf.reduce_mean(tf.divide(
            tf.add_n([logdensity_g, logdensity_z, logdensity_y]),
            tf.cast(self.Tt - 1, tf.float32)))

        return logdensity
