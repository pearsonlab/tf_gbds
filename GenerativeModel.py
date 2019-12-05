import tensorflow as tf
import numpy as np
# import edward as ed
from edward.models import (RandomVariable, MultivariateNormalDiag,
                           ExpRelaxedOneHotCategorical)
from tensorflow.contrib.distributions import (Distribution,
                                              FULLY_REPARAMETERIZED)
from utils import pad_lag


class GBDS(RandomVariable, Distribution):
    def __init__(self, params, traj, ctrl_obs, npcs, *args, **kwargs):
        name = kwargs.get("name", "GBDS")
        with tf.name_scope(name):
            self.dim = params["dim"]
            self.n_npcs = params["n_npcs"]
            self.lag = params["lag"]

            self.y = tf.identity(traj, "trajectory")
            self.s = tf.identity(pad_lag(traj, self.lag), "states")
            self.ctrl_obs = tf.identity(ctrl_obs, "observed_control")
            self.npcs = tf.identity(npcs, "npcs")
            self.extra_conds = tf.concat(
                [pad_lag(self.npcs[..., (self.dim * i):(self.dim * (i + 1))],
                         self.lag) for i in range(self.n_npcs)] +
                [self.npcs[:, 1:, -self.n_npcs:]], -1, "extra_conditions")

            self.NN = params["G_NN"]
            G_NN_outputs = self.NN([self.s, self.extra_conds])
            self.G_mu = tf.identity(G_NN_outputs[0], "mu")
            self.G_lambda = tf.identity(G_NN_outputs[1], "lambda")

            self.G = MultivariateNormalDiag(
                self.G_mu[:, :-1], 1. / tf.sqrt(self.G_lambda[:, :-1]),
                name="GMM")
            self.G_sample = tf.identity(self.G.sample(), "GMM_sample")

            self.unc_Kp = params["unc_Kp"]
            self.Kp = tf.identity(params["Kp"], "Kp")

            self.var_list = self.NN.variables + [self.unc_Kp]
            self.log_vars = self.NN.variables

            with tf.name_scope("goal_state_penalty"):
                # penalty on goal state escaping boundaries
                if params["g_bounds_pen"] is not None:
                    self.g_pen = tf.constant(
                        params["g_bounds_pen"], tf.float32,
                        name="goal_boundary_penalty")
                    with tf.name_scope("boundary"):
                        # boundaries for penalty
                        if params["g_bounds"] is not None:
                            self.bounds = params["g_bounds"]
                        else:
                            self.bounds = [-1., 1.]
                else:
                    self.g_pen = None

                # # penalty on the precision of GMM components
                # if params["g_prec_pen"] is not None:
                #     self.g_prec_pen = tf.constant(
                #         params["g_prec_pen"], tf.float32,
                #         name="goal_precision_penalty")

            with tf.name_scope("kinetic_noise"):
                # noise of movement
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
        self._args = (params, traj, ctrl_obs, npcs)

    def _log_prob(self, value):
        qg = tf.identity(value, "posterior_goals")

        with tf.name_scope("goals"):
            logdensity_g = tf.reduce_mean(self.G.log_prob(qg[:, 1:]))

            with tf.name_scope("penalty"):
                if self.g_pen is not None:
                    # penalty on goal state escaping game space
                    logdensity_g -= tf.multiply(
                        self.g_pen, tf.reduce_mean(tf.reduce_sum(
                        tf.nn.relu(self.bounds[0] - self.G_mu) ** 2, -1)))
                    logdensity_g -= tf.multiply(
                        self.g_pen, tf.reduce_mean(tf.reduce_sum(
                        tf.nn.relu(self.G_mu - self.bounds[1]) ** 2, -1)))
                    logdensity_g -= tf.multiply(
                        self.g_pen, tf.reduce_mean(tf.reduce_sum(
                        tf.nn.relu(self.bounds[0] - qg) ** 2, -1)))
                    logdensity_g -= tf.multiply(
                        self.g_pen, tf.reduce_mean(tf.reduce_sum(
                        tf.nn.relu(qg - self.bounds[1]) ** 2, -1)))

                # if self.g_prec_pen is not None:
                #     # penalty on GMM precision
                #     logdensity_g -= tf.multiply(
                #         self.g_prec_pen,
                #         tf.reduce_mean(tf.reduce_max(tf.reduce_sum(
                #             1. / self.G_lambda, -1), -1)))

        with tf.name_scope("control_signal"):
            u_pred = tf.multiply(
                qg - self.y[:, :-1], self.Kp, "predicted")
            u_res = tf.subtract(
                u_pred, self.ctrl_obs[:, 1:], "residual")
            logdensity_u = -tf.reduce_mean(tf.reduce_sum(
                (0.5 * tf.log(2 * np.pi) + tf.log(self.eps) +
                 u_res ** 2 / (2 * self.eps ** 2)), -1))

            with tf.name_scope("penalty"):
                if self.eps_pen is not None:
                    logdensity_u -= self.eps_pen * tf.reduce_sum(self.unc_eps)

        return tf.add(logdensity_g, logdensity_u)
