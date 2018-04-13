import tensorflow as tf
import numpy as np
# import edward as ed
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
            # number of GMM components
            self.K = params["GMM_K"]
            # neural network to generate state-dependent goals
            self.GMM_NN = params["GMM_NN"]
            self.params += self.GMM_NN.variables

            with tf.name_scope("g0"):
                # initial goal distribution
                g0 = params["g0"]
                self.g0_mu = tf.identity(g0["mu"], "mu")
                self.g0_lambda = tf.nn.softplus(g0["unc_lambda"], "lambda")
                self.g0_w = tf.nn.softmax(g0["unc_w"], name="w")
                self.g0_params = ([g0["mu"]] + [g0["unc_lambda"]] +
                                  [g0["unc_w"]])
                self.params += self.g0_params

            with tf.name_scope("goal_state_noise"):
                # noise coefficient on goal states
                if params["sigma_trainable"]:
                    self.unc_sigma = params["sigma"]
                    self.sigma = tf.multiply(
                        tf.nn.softplus(self.unc_sigma, "softplus"),
                        tf.ones([1, self.dim]), "sigma")
                    self.sigma_pen = tf.constant(
                        params["sigma_pen"], tf.float32, name="sigma_penalty")
                else:
                    self.sigma = tf.multiply(
                        params["sigma"], tf.ones([1, self.dim]), "sigma")
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

            with tf.name_scope("control_signal_censoring"):
                # clipping signal
                self.clip = params["clip"]
                if self.clip:
                    if params["clip_range"] is not None:
                        self.clip_range = params["clip_range"]
                    else:
                        self.clip_range = [-1., 1.]
                    self.clip_tol = tf.constant(
                        params["clip_tol"], tf.float32, name="clip_tolerance")
                    self.clip_pen = tf.constant(
                        params["clip_pen"], tf.float32, name="clip_penalty")
                    # self.eta = params["eta"]

            with tf.name_scope("control_signal_noise"):
                # noise coefficient on control signals
                if params["eps_trainable"]:
                    self.unc_eps = params["eps"]
                    self.eps = tf.multiply(
                        tf.nn.softplus(self.unc_eps, "softplus"),
                        tf.ones([1, self.dim]), "epsilon")
                    self.eps_pen = tf.constant(
                        params["eps_pen"], tf.float32, name="epsilon_penalty")
                else:
                    self.eps = tf.multiply(
                        params["eps"], tf.ones([1, self.dim]), "epsilon")
                    self.eps_pen = None
            with tf.name_scope("control_signal_penalty"):
                # penalty on large control error
                if params["u_error_pen"] is not None:
                    self.error_pen = tf.constant(
                        params["u_error_pen"], tf.float32,
                        name="control_error_penalty")
                    self.error_tol = tf.constant(
                        params["u_error_tol"], tf.float32,
                        name="control_error_tolerance")
                else:
                    self.error_pen = None

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

        all_mu = tf.reshape(
            self.GMM_NN(s)[:, :, :(self.K * self.dim)],
            [self.B, -1, self.K, self.dim], "all_mu")

        all_lambda = tf.reshape(tf.nn.softplus(
            self.GMM_NN(s)[:, :, (self.K * self.dim):(
                2 * self.K * self.dim)], "softplus_lambda"),
            [self.B, -1, self.K, self.dim], "all_lambda")

        all_w = tf.nn.softmax(tf.reshape(
            self.GMM_NN(s)[:, :, (2 * self.K * self.dim):],
            [self.B, -1, self.K], "reshape_w"), -1, "all_w")

        next_g = tf.divide(
            tf.expand_dims(post_g[:, :-1], 2) + all_mu * all_lambda,
            1 + all_lambda, "next_goals")

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

        return (all_mu, all_lambda, all_w, next_g, error, u_pred)

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
        _, all_lambda, all_w, g_pred, ctrl_error, u_pred = self.get_preds(
            self.s[:, 1:-1], self.y[:, :-1],
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
            logdensity_g += tf.reduce_sum(tf.reduce_logsumexp(gmm_term, -1), -1)

        with tf.name_scope("g0"):
            res_g0 = tf.subtract(tf.expand_dims(value[:, 0], 1), self.g0_mu,
                                 "g0_residual")
            g0_term = tf.log(self.g0_w + 1e-8) - tf.reduce_sum(
                self.g0_lambda * (res_g0 ** 2) / 2, -1)
            g0_term += 0.5 * tf.reduce_sum(
                tf.log(self.g0_lambda) - tf.log(2 * np.pi), -1)
            logdensity_g += tf.reduce_logsumexp(g0_term, -1)

        with tf.name_scope("boundary_penalty"):
            if self.g_pen is not None:
                # penalty on goal state escaping game space
                logdensity_g -= self.g_pen * tf.reduce_sum(
                    tf.nn.relu(self.bounds[0] - value), [1, 2])
                logdensity_g -= self.g_pen * tf.reduce_sum(
                    tf.nn.relu(value - self.bounds[1]), [1, 2])

        logdensity_u = 0.0
        with tf.name_scope("control_signal"):
            u_res = tf.subtract(self.ctrl_obs, u_pred, "residual")
            logdensity_u -= tf.reduce_sum(
                (0.5 * tf.log(2 * np.pi) + tf.log(self.eps) +
                 u_res ** 2 / (2 * self.eps ** 2)), [1, 2])
            # logdensity_u -= tf.reduce_sum(
            #     (0.5 * tf.log(2 * np.pi) + tf.log(1e-5) * tf.ones([1, self.dim]) +
            #      u_res ** 2 / (tf.ones([1, self.dim]) * 2 * 1e-5 ** 2)), [1, 2])
            # logdensity_u -= self.res_pen * tf.reduce_sum(tf.nn.relu(
            #     u_res - self.res_tol), [1, 2])
            # logdensity_u -= self.res_pen * tf.reduce_sum(tf.nn.relu(
            #     -u_res - self.res_tol), [1, 2])
        with tf.name_scope("control_error_penalty"):
            # penalty on large ctrl error
            if self.error_pen is not None:
                logdensity_u -= self.error_pen * tf.reduce_sum(
                    tf.nn.relu(ctrl_error - self.error_tol), [1, 2])
                logdensity_u -= self.error_pen * tf.reduce_sum(
                    tf.nn.relu(-ctrl_error - self.error_tol), [1, 2])

        logdensity = tf.divide(
            tf.reduce_mean(tf.add(logdensity_g, logdensity_u)),
            tf.cast(self.Tt, tf.float32))

        if self.eps_pen is not None:
            logdensity -= self.eps_pen * tf.reduce_sum(self.unc_eps)
        if self.sigma_pen is not None:
            logdensity -= self.sigma_pen * tf.reduce_sum(self.unc_sigma)

        return logdensity

    def sample_g0(self, _=None):
        # Sample from initial goal distribution
        with tf.name_scope("select_component"):
            k0 = tf.squeeze(tf.multinomial(tf.reshape(
                tf.log(self.g0_w, name="log_g0_w"), [1, -1]),
                1, name="draw_sample"), name="k0")
        with tf.name_scope("get_sample"):
            g0 = tf.add(
                (tf.random_normal([self.dim], name="std_normal") /
                 tf.sqrt(self.g0_lambda[k0], name="inv_std_dev")),
                self.g0_mu[k0], name="g0")

        return g0

    def sample_GMM(self, state, prev_g, extra_conds=None):
        # Generate new goal given current state and previous goal
        state = tf.reshape(state, [1, 1, -1], "reshape_state")
        with tf.name_scope("pad_extra_conds"):
            if extra_conds is not None:
                state = pad_extra_conds(
                    state, tf.reshape(extra_conds, [1, -1]))

        with tf.name_scope("mu"):
            all_mu = tf.reshape(
                self.GMM_NN(state)[:, :, :(self.K * self.dim)],
                [self.K, self.dim], "all_mu")
        with tf.name_scope("lambda"):
            all_lambda = tf.reshape(tf.nn.softplus(
                self.GMM_NN(state)[:, :, (self.K * self.dim):(
                    2 * self.K * self.dim)], "softplus_lambda"),
                [self.K, self.dim], "all_lambda")
        with tf.name_scope("w"):
            all_w = tf.nn.softmax(tf.reshape(
                self.GMM_NN(state)[:, :, (2 * self.K * self.dim):],
                [1, self.K], "reshape_w"), -1, "all_w")

        with tf.name_scope("select_component"):
            k = tf.squeeze(tf.multinomial(
                tf.reshape(tf.log(all_w, "log_w"), [1, -1]),
                1, name="draw_sample"), name="k")
        with tf.name_scope("get_sample"):
            g = tf.add(
                tf.divide(prev_g + all_mu[k] * all_lambda[k],
                          1 + all_lambda[k], name="mean"),
                (tf.random_normal([self.dim], name="std_normal") *
                 tf.divide(tf.squeeze(self.sigma),
                           tf.sqrt(1 + all_lambda[k]), name="std_dev")),
                name="new_goal")

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
                value = kwargs.get("value", None)
                self.agents = [GBDS(
                    p, states, ctrl_obs, extra_conds, name=p["name"],
                    value=tf.gather(value, p["col"], axis=-1))
                               for p in params]
            else:
                raise TypeError("params must be a list.")

            self.params = []
            for agent in self.agents:
                self.params += agent.params

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
        return tf.concat([agent.sample_GMM(
          state, tf.gather(prev_g, agent.col, axis=-1), extra_conds)
                          for agent in self.agents], 0)

    def update_ctrl(self, errors, prev_u):
        return tf.concat([agent.update_ctrl(
            tf.gather(errors, agent.col, axis=-1),
            tf.gather(prev_u, agent.col, axis=-1))
                          for agent in self.agents], 0)
