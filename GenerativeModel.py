import tensorflow as tf
import numpy as np
from edward.models import RandomVariable
from tensorflow.contrib.distributions import (Distribution,
                                              FULLY_REPARAMETERIZED)
from tf_gbds.utils import pad_extra_conds


class GBDS(RandomVariable, Distribution):

    def __init__(self, params, states, ctrl_obs, extra_conds=None,
                 *args, **kwargs):

        name = kwargs.get("name", "GBDS")
        with tf.name_scope(name):
            self.col = params["col"]
            self.dim = params["dim"]
            with tf.name_scope("batch_size"):
                self.B = tf.shape(states)[0]
            with tf.name_scope("trial_length"):
                self.Tt = tf.shape(states)[1]
            self.s = tf.identity(states, "states")
            self.y = tf.gather(states, self.col, axis=-1, name="positions")
            self.ctrl_obs = tf.gather(ctrl_obs, self.col, axis=-1,
                                      name="observed_control")

            if extra_conds is not None:
                self.extra_conds = tf.identity(
                    extra_conds, "extra_conditions")
            else:
                self.extra_conds = None

            self.params = []
            self.log_vars = []
            # number of GMM components
            self.K = params["GMM_K"]
            # neural network to generate state-dependent goals
            self.GMM_NN = params["GMM_NN"]
            self.params += self.GMM_NN.variables
            self.log_vars += self.GMM_NN.variables

            with tf.name_scope("goal_state_noise"):
                # noise coefficient on goal states
                self.unc_sigma = params["unc_sigma"]
                self.sigma = tf.nn.softplus(self.unc_sigma, "sigma")
                if params["sigma_trainable"]:
                    self.params += [self.unc_sigma]
                    self.sigma_pen = tf.constant(
                        params["sigma_pen"], tf.float32, name="sigma_penalty")
                else:
                    self.sigma_pen = None

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

            with tf.name_scope("PID_control"):
                PID_params = params["PID"]
                self.Kp = PID_params["Kp"]
                self.Ki = PID_params["Ki"]
                self.Kd = PID_params["Kd"]
                self.params += PID_params["vars"]
                # For details of PID control system, refer to
                # https://en.wikipedia.org/wiki/PID_controller
                # discrete implementation of PID control with convolution
                t_coeff = self.Kp + self.Ki + self.Kd
                t1_coeff = -self.Kp - 2 * self.Kd
                t2_coeff = self.Kd
                # concatenate coefficients into a filter
                self.L = tf.stack([t2_coeff, t1_coeff, t_coeff], axis=1,
                                  name="convolution_filter")

            with tf.name_scope("control_signal_noise"):
                # noise coefficient on control signals
                self.unc_eps = params["unc_eps"]
                self.eps = tf.nn.softplus(self.unc_eps, "epsilon")
                if params["eps_trainable"]:
                    self.params += [self.unc_eps]
                    self.eps_pen = tf.constant(
                        params["eps_pen"], tf.float32, name="epsilon_penalty")
                else:
                    self.eps_pen = None

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

        self._args = (params, states, ctrl_obs, extra_conds)

    def get_preds(self, s, y, post_g, prev_u, extra_conds=None):
        # Return one-step-ahead prediction of goal and control signal,
        # given state, current position, sample from goal posterior,
        # and previous control (and extra conditions if provided).

        with tf.name_scope("pad_extra_conds"):
            if extra_conds is not None:
                s = pad_extra_conds(s, extra_conds)

        NN_output = tf.identity(self.GMM_NN(s), "NN_output")
        all_mu = tf.reshape(
            NN_output[:, :, :(self.K * self.dim)],
            [self.B, -1, self.K, self.dim], "all_mu")

        all_lambda = tf.reshape(tf.nn.softplus(
            NN_output[:, :, (self.K * self.dim):(
                2 * self.K * self.dim)], "softplus_lambda"),
            [self.B, -1, self.K, self.dim], "all_lambda")

        all_w = tf.nn.softmax(tf.reshape(
            NN_output[:, :, (2 * self.K * self.dim):],
            [self.B, -1, self.K], "reshape_w"), -1, "all_w")

        next_g = tf.divide(
            tf.expand_dims(post_g[:, :-1], 2) + all_mu * all_lambda,
            1 + all_lambda, "next_goals")

        error = tf.subtract(post_g[:, 1:], y, "control_error")

        with tf.name_scope("convolution"):
            u_diff = []
            # get current error signal and corresponding filter
            for i in range(self.dim):
                signal = error[:, :, i]
                # pad the beginning of control signal with zero
                signal = tf.expand_dims(
                    tf.pad(signal, [[0, 0], [2, 0]], name="pad_zero"),
                    -1, name="reshape_signal")
                filt = tf.reshape(self.L[i], [-1, 1, 1], "reshape_filter")
                res = tf.nn.convolution(signal, filt, padding="VALID",
                                        name="convolve_signal")
                u_diff.append(res)

        if len(u_diff) > 1:
            u_diff = tf.concat([*u_diff], -1, "control_signal_change")
        else:
            u_diff = tf.identity(u_diff[0], "contrl_signal_change")

        u_pred = tf.add(prev_u, u_diff, "predicted_control_signal")

        return (all_mu, all_lambda, all_w, next_g, u_pred)

    def _log_prob(self, value):
        all_mu, all_lambda, all_w, g_pred, u_pred = self.get_preds(
            self.s[:, :-1], self.y[:, :-1],
            value, tf.pad(
                self.ctrl_obs[:, :-1], [[0, 0], [1, 0], [0, 0]],
                name="previous_control"), self.extra_conds)

        logdensity_g = 0.0
        with tf.name_scope("goal_states"):
            res_gmm = tf.subtract(
                tf.expand_dims(value[:, 1:], 2, "reshape_samples"), g_pred,
                "GMM_residual")
            gmm_term = tf.log(all_w + 1e-8) - tf.reduce_sum(
                (1 + all_lambda) * (res_gmm ** 2) / (2 * self.sigma ** 2), -1)
            gmm_term += (0.5 * tf.reduce_sum(tf.log(1 + all_lambda), -1) -
                         tf.reduce_sum(0.5 * tf.log(2 * np.pi) +
                                       tf.log(self.sigma), -1))
            logdensity_g += tf.reduce_sum(
                tf.reduce_logsumexp(gmm_term, -1), -1)

        with tf.name_scope("boundary_penalty"):
            if self.g_pen is not None:
                logdensity_g -= self.g_pen * tf.reduce_sum(
                    tf.nn.relu(self.bounds[0] - all_mu), [1, 2, 3])
                logdensity_g -= self.g_pen * tf.reduce_sum(
                    tf.nn.relu(all_mu - self.bounds[1]), [1, 2, 3])
                logdensity_g -= .1 * tf.reduce_sum(1. / all_lambda, [1, 2, 3])

        logdensity_u = 0.0
        with tf.name_scope("control_signal"):
            u_res = tf.subtract(self.ctrl_obs, u_pred, "residual")
            logdensity_u -= tf.reduce_sum(
                (0.5 * tf.log(2 * np.pi) + tf.log(self.eps) +
                 u_res ** 2 / (2 * self.eps ** 2)), [1, 2])

        if self.sigma_pen is not None:
            logdensity_g -= self.sigma_pen * tf.reduce_sum(self.unc_sigma)
        if self.eps_pen is not None:
            logdensity_u -= self.eps_pen * tf.reduce_sum(self.unc_eps)

        logdensity = tf.divide(
            tf.reduce_mean(tf.add(logdensity_g, logdensity_u)),
            tf.cast(self.Tt, tf.float32))

        return logdensity


class joint_GBDS(RandomVariable, Distribution):

    def __init__(self, params, states, ctrl_obs, extra_conds=None,
                 *args, **kwargs):

        name = kwargs.get("name", "joint")
        with tf.name_scope(name):
            if isinstance(params, list):
                value = kwargs.get("value", tf.zeros_like(states))
                self.agents = [GBDS(
                    p, states, ctrl_obs, extra_conds, name=p["name"],
                    value=tf.gather(value, p["col"], axis=-1))
                               for p in params]
            else:
                raise TypeError("params must be a list.")

            self.params = []
            self.log_vars = []
            for agent in self.agents:
                self.params += agent.params
                self.log_vars += agent.log_vars

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

        super(joint_GBDS, self).__init__(*args, **kwargs)

        self._args = (params, states, ctrl_obs, extra_conds)

    def _log_prob(self, value):
        return tf.add_n([agent.log_prob(tf.gather(value, agent.col, axis=-1))
                         for agent in self.agents])
