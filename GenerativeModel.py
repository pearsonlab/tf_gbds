import tensorflow as tf
import numpy as np
from edward.models import RandomVariable
from tensorflow.contrib.distributions import (Distribution,
                                              FULLY_REPARAMETERIZED)
# from tensorflow.python.ops.distributions.special_math import log_ndtr
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
            # neural network to generate state-dependent goals
            self.G_NN = params["G_NN"]
            self.params += self.G_NN.variables
            self.log_vars += self.G_NN.variables

            with tf.name_scope("g0"):
                # initial goal distribution
                g0 = params["g0"]
                self.g0_mu = tf.identity(g0["mu"], "mu")
                self.g0_lambda = tf.nn.softplus(g0["unc_lambda"], "lambda")
                self.g0_params = ([g0["mu"]] + [g0["unc_lambda"]])
                self.params += self.g0_params
                self.log_vars += self.g0_params

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

            # with tf.name_scope("control_signal_censoring"):
            #     # clipping signal
            #     self.clip = params["clip"]
            #     if self.clip:
            #         if params["clip_range"] is not None:
            #             self.clip_range = params["clip_range"]
            #         else:
            #             self.clip_range = [-1., 1.]
            #         self.clip_tol = tf.constant(
            #             params["clip_tol"], tf.float32, name="clip_tolerance")
            #         self.clip_pen = tf.constant(
            #             params["clip_pen"], tf.float32, name="clip_penalty")
            #         # self.eta = params["eta"]

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

        NN_output = tf.identity(self.G_NN(s), "NN_output")
        mu = tf.reshape(
            NN_output[:, :, :self.dim], [self.B, -1, self.dim], "mu")
        Lambda = tf.reshape(tf.nn.softplus(
            NN_output[:, :, self.dim:], "softplus_lambda"),
            [self.B, -1, self.dim], "lambda")

        next_g = tf.divide(
            post_g[:, :-1] + mu * Lambda, 1 + Lambda, "next_goals")

        error = tf.subtract(post_g, y, "control_error")

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

        return (mu, Lambda, next_g, u_pred)

    # def clip_log_prob(self, upsilon, u):
    #     """upsilon (derived from time series of y) is a censored version of
    #     a noisy control signal: \hat{u} ~ N(u, \eta^2).
    #     log p(upsilon|u, g) = log p(upsilon|u) + log(u|g)
    #     log p(upsilon|u) breaks down into three cases,
    #     namely left-clipped (upsilon_t = -1), right-clipped (upsilon_t = 1),
    #     and non-clipped (-1 < upsilon_t < 1). For the first two cases,
    #     Normal CDF is used instead of PDF due to censoring.
    #     The log density term is computed for each and then add up.
    #     """

    #     l_b = tf.add(self.clip_range[0], self.clip_tol, "lower_bound")
    #     u_b = tf.subtract(self.clip_range[1], self.clip_tol, "upper_bound")
    #     pen = self.clip_pen
    #     # eta = self.eta

    #     # def z(x, loc, scale):
    #     #     return (x - loc) / scale

    #     # def normal_logpdf(x, loc, scale):
    #     #     return -(0.5 * np.log(2 * np.pi) + tf.log(scale) +
    #     #              0.5 * tf.square(z(x, loc, scale)))

    #     # def normal_logcdf(x, loc, scale):
    #     #     return log_ndtr(z(x, loc, scale))

    #     return tf.where(tf.less_equal(upsilon, l_b, name="left_clip"),
    #                     # normal_logcdf(l_b, u, eta),
    #                     pen * tf.nn.relu(u - l_b),
    #                     tf.where(tf.greater_equal(upsilon, u_b,
    #                                               name="right_clip"),
    #                              # normal_logcdf(-u_b, -u, eta),
    #                              pen * tf.nn.relu(u_b - u),
    #                              # normal_logpdf(upsilon, u, eta)))
    #                              pen * tf.nn.relu(tf.abs(u - upsilon) -
    #                                               self.clip_tol)))

    def _log_prob(self, value):
        mu, Lambda, g_pred, u_pred = self.get_preds(
            self.s[:, 1:-1], self.y[:, :-1],
            value, tf.pad(
                self.ctrl_obs[:, :-1], [[0, 0], [1, 0], [0, 0]],
                name="previous_control"), self.extra_conds)

        logdensity_g = 0.0
        with tf.name_scope("goal_states"):
            res_g = tf.subtract(value[:, 1:], g_pred, "residual")
            logdensity_g -= tf.reduce_sum(
                tf.log(self.sigma) + 0.5 * (
                    tf.log(2 * np.pi) - tf.log(1 + Lambda) +
                    (1 + Lambda) * res_g ** 2 / self.sigma ** 2), [1, 2])

        with tf.name_scope("g0"):
            res_g0 = tf.subtract(value[:, 0], self.g0_mu, "residual")
            logdensity_g -= 0.5 * tf.reduce_sum(
                tf.log(2 * np.pi) - tf.log(self.g0_lambda) +
                self.g0_lambda * res_g0 ** 2, -1)

        with tf.name_scope("boundary_penalty"):
            if self.g_pen is not None:
                # penalty on goal state escaping game space
                logdensity_g -= self.g_pen * tf.reduce_sum(
                    tf.nn.relu(self.bounds[0] - mu), [1, 2])
                logdensity_g -= self.g_pen * tf.reduce_sum(
                    tf.nn.relu(mu - self.bounds[1]), [1, 2])
                # logdensity_g -= .1 * tf.reduce_sum(1. / Lambda, [1, 2])

        logdensity_u = 0.0
        with tf.name_scope("control_signal"):
            u_res = tf.subtract(self.ctrl_obs, u_pred, "residual")
            logdensity_u -= tf.reduce_sum(
                tf.log(self.eps) + 0.5 * (tf.log(2 * np.pi) +
                    u_res ** 2 / self.eps ** 2), [1, 2])

        if self.sigma_pen is not None:
            logdensity_g -= self.sigma_pen * tf.reduce_sum(self.unc_sigma)
        if self.eps_pen is not None:
            logdensity_u -= self.eps_pen * tf.reduce_sum(self.unc_eps)

        logdensity = tf.divide(
            tf.reduce_mean(tf.add(logdensity_g, logdensity_u)),
            tf.cast(self.Tt, tf.float32))

        return logdensity

    def sample_g0(self, _=None):
        # Sample from initial goal distribution
        with tf.name_scope("get_sample"):
            g0 = tf.add(
                tf.random_normal([self.dim], name="std_normal") /
                tf.sqrt(self.g0_lambda, name="inv_std_dev"),
                self.g0_mu, name="initial_goal")

        return g0

    def sample_goal(self, state, prev_g, extra_conds=None):
        # Generate new goal given current state and previous goal
        state = tf.reshape(state, [1, 1, -1], "reshape_state")
        with tf.name_scope("pad_extra_conds"):
            if extra_conds is not None:
                state = pad_extra_conds(state, extra_conds)

        NN_output = self.G_NN(state)
        mu = tf.reshape(
            NN_output[:, :, :self.dim], [self.dim], "mu")
        Lambda = tf.reshape(
            tf.nn.softplus(NN_output[:, :, self.dim:], "softplus_lambda"),
            [self.dim], "lambda")

        with tf.name_scope("get_sample"):
            g = tf.add(
                tf.divide(prev_g + mu * Lambda, 1 + Lambda, name="mean"),
                tf.random_normal([self.dim], name="std_normal") * tf.divide(
                    tf.reshape(self.sigma, [self.dim]),
                    tf.sqrt(1 + Lambda), name="std_dev"), name="new_goal")

        return g

    def update_ctrl(self, errors, prev_u):
        # Update control signal given errors and previous control
        u_diff = tf.reduce_sum(
            tf.multiply(errors, tf.transpose(self.L), "convolve_signal"),
            0, name="control_signal_change")
        u = tf.add(prev_u, u_diff, "new_control")

        return u


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

    def sample_g0(self, n=1):
        if n == 1:
            return tf.concat([agent.sample_g0() for agent in self.agents], 0)
        else:
            return tf.concat([tf.map_fn(agent.sample_g0, tf.zeros(n))
                              for agent in self.agents], -1)

    def update_goal(self, state, prev_g, extra_conds=None):
        return tf.concat([agent.sample_goal(
            state, tf.gather(prev_g, agent.col, axis=-1), extra_conds)
                          for agent in self.agents], 0)

    def update_ctrl(self, errors, prev_u):
        return tf.concat([agent.update_ctrl(
            tf.gather(errors, agent.col, axis=-1),
            tf.gather(prev_u, agent.col, axis=-1))
                          for agent in self.agents], 0)
