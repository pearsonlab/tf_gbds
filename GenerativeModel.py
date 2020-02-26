import tensorflow as tf
import numpy as np
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
            # self.s = tf.identity(pad_lag(traj, self.lag), "states")
            self.u_obs = tf.identity(ctrl_obs, "observed_control")
            self.npcs = tf.identity(npcs, "npcs")
            # self.extra_conds = tf.concat(
            #     [pad_lag(self.npcs[..., (self.dim * i):(self.dim * (i + 1))],
            #              self.lag) for i in range(self.n_npcs)] +
            #     [self.npcs[:, 1:, -self.n_npcs:]], -1, "extra_conditions")
            # discount prey values for now
            self.extra_conds = tf.concat(
                [pad_lag(self.npcs[..., (self.dim * i):(self.dim * (i + 1))],
                         self.lag) for i in range(self.n_npcs)],
                -1, "extra_conditions")

            self.NN = params["G_NN"]
            self.G_mu = tf.identity(self.NN(self.extra_conds), "mu")
            self.G_lambda = tf.identity(params["G_lambda"], "lambda")
            self.G = MultivariateNormalDiag(
                self.G_mu[:, :-1], 1. / tf.sqrt(self.G_lambda), name="GMM")
            self.G_sample = tf.identity(self.G.sample(), "GMM_sample")

            self.Kp = tf.identity(params["Kp"], "Kp")
            self.Ki = tf.identity(params["Ki"], "Ki")
            self.Kd = tf.identity(params["Kd"], "Kd")
            t_coeff = self.Kp + self.Ki + self.Kd
            t1_coeff = -self.Kp - 2 * self.Kd
            t2_coeff = self.Kd
            # concatenate coefficients into a filter
            self.filt = tf.stack([t2_coeff, t1_coeff, t_coeff], axis=1,
                                 name="convolution_filter")

            self.var_list = (self.NN.variables + [params["G_lambda"]] +
                             params["PID_vars"])
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
                            self.g_bounds = params["g_bounds"]
                        else:
                            self.g_bounds = [-1., 1.]
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
                        self.g_pen,
                        tf.reduce_mean(tf.reduce_sum(tf.add_n(
                            [tf.nn.relu(self.g_bounds[0] - self.G_mu) ** 2,
                             tf.nn.relu(self.G_mu - self.g_bounds[1]) ** 2,
                             tf.nn.relu(self.g_bounds[0] - qg) ** 2,
                             tf.nn.relu(qg - self.g_bounds[1]) ** 2]), -1)))

                # if self.g_prec_pen is not None:
                #     # penalty on GMM precision
                #     logdensity_g -= tf.multiply(
                #         self.g_prec_pen,
                #         tf.reduce_mean(tf.reduce_max(tf.reduce_sum(
                #             1. / self.G_lambda, -1), -1)))

        with tf.name_scope("control_signal"):
            # error = tf.subtract(qg, self.y[:, :-1], "error")
            error = tf.subtract(self.G_sample, self.y[:, 1:-1], "error")
            u_diff = []
            # get current error signal and corresponding filter
            for i in range(self.dim):
                # pad the beginning of control signal with zero
                signal = tf.expand_dims(
                    tf.pad(error[..., i], [[0, 0], [2, 0]], name="pad_zero"),
                    -1, name="signal")
                filt = tf.reshape(self.filt[i], [-1, 1, 1], "reshape_filter")
                res = tf.nn.convolution(signal, filt, padding="VALID",
                                        name="convolve_signal")
                u_diff.append(res)
            u_diff = tf.concat([*u_diff], -1, "control_signal_change")
            # u_pred = tf.add(
            #     self.u_obs[:, :-1], u_diff, "predicted_control_signal")
            # u_res = tf.subtract(self.u_obs[:, 1:], u_pred, "residual")
            u_pred = tf.add(
                self.u_obs[:, :-2], u_diff, "predicted_control_signal")
            u_res = tf.subtract(self.u_obs[:, 1:-1], u_pred, "residual")
            logdensity_u = -tf.reduce_mean(tf.reduce_sum(
                (0.5 * tf.log(2 * np.pi) + tf.log(self.eps) +
                 u_res ** 2 / (2 * self.eps ** 2)), -1))

            with tf.name_scope("penalty"):
                if self.eps_pen is not None:
                    logdensity_u -= self.eps_pen * tf.reduce_sum(self.unc_eps)

        return tf.add(logdensity_g, logdensity_u)
