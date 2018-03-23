import tensorflow as tf
import numpy as np
# import edward as ed
from edward.models import RandomVariable
from tensorflow.contrib.distributions import (Distribution,
                                              FULLY_REPARAMETERIZED)
# from tensorflow.python.ops.distributions.special_math import log_ndtr
from tf_gbds.utils import pad_extra_conds


class GBDS_G(RandomVariable, Distribution):
    """A customized Edward Random Variable for goal state in
    Goal-Based Dynamical System
    """

    def __init__(self, params, states, extra_conds=None,
                 name="G", value=None, dtype=tf.float32,
                 reparameterization_type=FULLY_REPARAMETERIZED,
                 validate_args=True, allow_nan_stats=True):
        """Initialize GBDS_g random variable (batch) for an agent

        Args:
            params: A Dictionary. The parameters for the agent.
            states: A Tensor.
                The time series of game states (including all agents).
            extra_conds: A Tensor.
                The extra conditions for the trials as additional input
                to GMM neural network (e.g. trial mode, opponent).
            name: Optional name for the random variable. Default to "G".
        """

        with tf.name_scope(name):
            self.s = tf.identity(states, "states")
            self.col = params["col"]
            self.dim = params["dim"]
            with tf.name_scope("batch_size"):
                self.B = tf.shape(states)[0]
            with tf.name_scope("trial_length"):
                self.Tt = tf.shape(states)[1]

            if extra_conds is not None:
                self.extra_conds = tf.identity(
                    extra_conds, "extra_conditions")
            else:
                self.extra_conds = None

            # number of GMM components
            self.K = params["GMM_K"]
            # neural network to generate state-dependent goals
            self.GMM_NN = params["GMM_NN"]

            with tf.name_scope("g0"):
                # initial goal distribution
                g0 = params["g0"]
                self.g0_mu = tf.identity(g0["mu"], "mu")
                self.g0_lambda = tf.nn.softplus(g0["unc_lambda"], "lambda")
                self.g0_w = tf.nn.softmax(g0["unc_w"], name="w")
                self.g0_params = ([g0["mu"]] + [g0["unc_lambda"]] +
                                  [g0["unc_w"]])

            self.params = self.g0_params + self.GMM_NN.variables

            with tf.name_scope("goal_state_noise"):
                # noise coefficient on goal states
                if params["sigma_trainable"]:
                    self.sigma = tf.Variable(
                        params["sigma"] * np.ones((1, self.dim)),
                        name="sigma", dtype=tf.float32)
                    self.params += [self.sigma]
                else:
                    self.sigma = tf.constant(
                        params["sigma"] * np.ones((1, self.dim)),
                        tf.float32, name="sigma")

            with tf.name_scope("goal_state_penalty"):
                # penalty on goal state escaping boundaries
                if params["g_bounds_pen"] is not None:
                    self.pen = tf.constant(
                        params["g_bounds_pen"], tf.float32,
                        name="goal_boundary_penalty")
                    with tf.name_scope("goal_state_boundary"):
                        # boundaries for penalty
                        if params["g_bounds"] is not None:
                            self.bounds = params["g_bounds"]
                        else:
                            self.bounds = [-1., 1.]
                else:
                    self.pen = None

            # with tf.name_scope("NN_kernel_penalty"):

        super(GBDS_G, self).__init__(
            name=name, value=value, dtype=dtype,
            reparameterization_type=reparameterization_type,
            validate_args=validate_args, allow_nan_stats=allow_nan_stats)

        self._kwargs["params"] = params
        self._kwargs["states"] = states
        self._kwargs['extra_conds'] = extra_conds

    def get_preds(self, s, g, extra_conds=None):
        # Return one-step-ahead prediction of goals, given states
        # (and extra conditions if provided).

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

        next_g = tf.divide(tf.expand_dims(g, 2) + all_mu * all_lambda,
                           1 + all_lambda, "next_goals")

        return (all_mu, all_lambda, all_w, next_g)

    def _log_prob(self, value):
        with tf.name_scope("next_step_prediction"):
            _, all_lambda, all_w, g_pred = self.get_preds(
                self.s[:, :-1], value[:, :-1], self.extra_conds)

        LogDensity = 0.0
        with tf.name_scope("goal_states"):
            res_gmm = tf.subtract(
                tf.expand_dims(value[:, 1:], 2, "reshape_samples"), g_pred,
                "GMM_residual")
            gmm_term = tf.log(all_w + 1e-8) - tf.reduce_sum(
                (1 + all_lambda) * (res_gmm ** 2) / (2 * self.sigma ** 2), -1)
            gmm_term += (0.5 * tf.reduce_sum(tf.log(1 + all_lambda), -1) -
                         tf.reduce_sum(0.5 * tf.log(2 * np.pi) +
                                       tf.log(self.sigma), -1))
            LogDensity += tf.reduce_sum(tf.reduce_logsumexp(gmm_term, -1), -1)

        with tf.name_scope("g0"):
            res_g0 = tf.subtract(tf.expand_dims(value[:, 0], 1), self.g0_mu,
                                 "g0_residual")
            g0_term = tf.log(self.g0_w + 1e-8) - tf.reduce_sum(
                self.g0_lambda * (res_g0 ** 2) / 2, -1)
            g0_term += 0.5 * tf.reduce_sum(
                tf.log(self.g0_lambda) - tf.log(2 * np.pi), -1)
            LogDensity += tf.reduce_logsumexp(g0_term, -1)

        with tf.name_scope("boundary_penalty"):
            if self.pen is not None:
                # penalty on goal state escaping game space
                LogDensity -= self.pen * tf.reduce_sum(
                    tf.nn.relu(self.bounds[0] - value), [1, 2])
                LogDensity -= self.pen * tf.reduce_sum(
                    tf.nn.relu(value - self.bounds[1]), [1, 2])

        # with tf.name_scope("weight_norm_sum"):
        #     norm = 0.0
        #     for layer in self.GMM_NN.layers:
        #         if "Dense" in layer.name:
        #             norm += tf.norm(layer.kernel)  # + tf.norm(layer.bias)

        # return (tf.reduce_mean(LogDensity) / tf.cast(self.Tt, tf.float32) -
        #         1e-1 * norm)
        return tf.reduce_mean(LogDensity) / tf.cast(self.Tt, tf.float32)

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


class joint_goals(RandomVariable, Distribution):
    """Auxiliary class to join random variables of goals for all agents
    """

    def __init__(self, params, states, extra_conds=None,
                 name="goal", value=None, dtype=tf.float32,
                 reparameterization_type=FULLY_REPARAMETERIZED,
                 validate_args=True, allow_nan_stats=True):
        with tf.name_scope(name):
            if isinstance(params, list):
                self.agents = [GBDS_G(
                    p, states, extra_conds, p["name"],
                    tf.gather(value, p["col"], axis=-1)) for p in params]
            else:
                raise TypeError("params must be a list.")

            self.params = []
            for agent in self.agents:
                self.params += agent.params

        super(joint_goals, self).__init__(
            name=name, value=value, dtype=dtype,
            reparameterization_type=reparameterization_type,
            validate_args=validate_args, allow_nan_stats=allow_nan_stats)

        self._kwargs["params"] = params
        self._kwargs["states"] = states
        self._kwargs['extra_conds'] = extra_conds

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


class GBDS_U(RandomVariable, Distribution):
    """A customized Edward Random Variable for control signal in
    Goal-Based Dynamical System
    """

    def __init__(self, params, goals, positions, ctrl_obs,
                 name="U", value=None, dtype=tf.float32,
                 reparameterization_type=FULLY_REPARAMETERIZED,
                 validate_args=True, allow_nan_stats=True):
        """Initialize GBDS_u random variable (batch) for an agent

        Args:
            params: A Dictionary. The parameters for the agent.
            goals: A Tensor. The corresponding goals
                (a sample from posterior distribution of goals).
            positions: A Tensor. The observed trajectories of the agent.
            ctrl_obs: A Tensor. The observed control signal. If None,
                control signal will be computed with positions and velocity.
            name: Optional name for the random variable. Default to "U".
        """

        with tf.name_scope(name):
            self.g = tf.identity(goals, "goals")
            self.y = tf.identity(positions, "positions")
            self.ctrl_obs = tf.identity(ctrl_obs, "observed_control")
            self.col = params["col"]
            self.dim = params["dim"]
            with tf.name_scope("batch_size"):
                self.B = tf.shape(self.y)[0]
            with tf.name_scope("trial_length"):
                self.Tt = tf.shape(self.y)[1]

            self.res_tol = tf.constant(params["u_res_tol"], tf.float32,
                                       name="control_residual_tolerance")
            # self.res_tol = tf.identity(params["u_res_tol"],
            #                            "control_residual_tolerance")
            self.res_pen = tf.constant(params["u_res_pen"], tf.float32,
                                       name="control_residual_penalty")
            # self.res_pen = tf.identity(params["u_res_pen"],
            #                            "control_residual_penalty")

            with tf.name_scope("PID_control"):
                PID_params = params["PID"]
                self.Kp = PID_params["Kp"]
                self.Ki = PID_params["Ki"]
                self.Kd = PID_params["Kd"]
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

            # Replaced by res_tol and res_pen
            # with tf.name_scope("control_signal_noise"):
            #     # noise coefficient on control signals
            #     self.eps = tf.constant(
            #         params["eps"] * np.ones((1, self.dim)),
            #         dtype=tf.float32, name="epsilon")
            with tf.name_scope("control_signal_penalty"):
                # penalty on large control error
                if params["u_error_pen"] is not None:
                    self.error_pen = tf.constant(
                        params["u_error_pen"], tf.float32,
                        name="control_error_penalty")
                    # self.error_pen = tf.identity(
                    #     params["u_error_pen"], "control_error_penalty")
                    self.error_tol = tf.constant(
                        params["u_error_tol"], tf.float32,
                        name="control_error_tolerance")
                else:
                    self.error_pen = None

        super(GBDS_U, self).__init__(
            name=name, value=value, dtype=dtype,
            reparameterization_type=reparameterization_type,
            validate_args=validate_args, allow_nan_stats=allow_nan_stats)

        self._kwargs["params"] = params
        self._kwargs["goals"] = goals
        self._kwargs["positions"] = positions
        self._kwargs['ctrl_obs'] = ctrl_obs

    def get_preds(self, y, post_g, prev_u):
        # Return one-step-ahead prediction of control signal, given current
        # position, sample from goal posterior, and previous control.

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

        return (error, u_pred)

    def clip_log_prob(self, upsilon, u):
        """upsilon (derived from time series of y) is a censored version of
        a noisy control signal: \hat{u} ~ N(u, \eta^2).
        log p(upsilon|u, g) = log p(upsilon|u) + log(u|g)
        log p(upsilon|u) breaks down into three cases,
        namely left-clipped (upsilon_t = -1), right-clipped (upsilon_t = 1),
        and non-clipped (-1 < upsilon_t < 1). For the first two cases,
        Normal CDF is used instead of PDF due to censoring.
        The log density term is computed for each and then add up.
        """

        l_b = tf.add(self.clip_range[0], self.clip_tol, "lower_bound")
        u_b = tf.subtract(self.clip_range[1], self.clip_tol, "upper_bound")
        pen = self.clip_pen
        # eta = self.eta

        # def z(x, loc, scale):
        #     return (x - loc) / scale

        # def normal_logpdf(x, loc, scale):
        #     return -(0.5 * np.log(2 * np.pi) + tf.log(scale) +
        #              0.5 * tf.square(z(x, loc, scale)))

        # def normal_logcdf(x, loc, scale):
        #     return log_ndtr(z(x, loc, scale))

        return tf.where(tf.less_equal(upsilon, l_b, name="left_clip"),
                        # normal_logcdf(l_b, u, eta),
                        pen * tf.nn.relu(u - l_b),
                        tf.where(tf.greater_equal(upsilon, u_b,
                                                  name="right_clip"),
                                 # normal_logcdf(-u_b, -u, eta),
                                 pen * tf.nn.relu(u_b - u),
                                 # normal_logpdf(upsilon, u, eta)))
                                 pen * tf.nn.relu(tf.abs(u - upsilon) -
                                                  self.clip_tol)))

    def _log_prob(self, value):
        with tf.name_scope("next_step_prediction"):
            # Disregard the last time step because we donnot know
            # the next value, thus cannot calculate the error
            ctrl_error, u_pred = self.get_preds(
                self.y[:, :-1], self.g[:, 1:],
                tf.pad(value[:, :-2], [[0, 0], [1, 0], [0, 0]],
                       name="previous_control"))

        LogDensity = 0.0
        with tf.name_scope("clipping_noise"):
            if self.clip:
                LogDensity += tf.reduce_sum(
                    self.clip_log_prob(self.ctrl_obs, value[:, :-1]), [1, 2])

        with tf.name_scope("control_signal"):
            u_res = tf.subtract(value[:, :-1], u_pred, "residual")
            # LogDensity -= tf.reduce_sum(
            #     (0.5 * tf.log(2 * np.pi) + tf.log(self.eps) +
            #      u_res ** 2 / (2 * self.eps ** 2)), [1, 2])
            # LogDensity -= tf.reduce_sum(
            #     (0.5 * tf.log(2 * np.pi) + tf.log(1e-5) * tf.ones([1, self.dim]) +
            #      u_res ** 2 / (tf.ones([1, self.dim]) * 2 * 1e-5 ** 2)), [1, 2])
            LogDensity -= self.res_pen * tf.reduce_sum(tf.nn.relu(
                u_res - self.res_tol), [1, 2])
            LogDensity -= self.res_pen * tf.reduce_sum(tf.nn.relu(
                -u_res - self.res_tol), [1, 2])
        with tf.name_scope("control_error_penalty"):
            # penalty on large ctrl error
            if self.error_pen is not None:
                LogDensity -= self.error_pen * tf.reduce_sum(
                    tf.nn.relu(ctrl_error - self.error_tol), [1, 2])
                LogDensity -= self.error_pen * tf.reduce_sum(
                    tf.nn.relu(-ctrl_error - self.error_tol), [1, 2])

        return tf.reduce_mean(LogDensity) / tf.cast(self.Tt - 1, tf.float32)

    def update_ctrl(self, errors, prev_u):
        # Update control signal given errors and previous control
        u_diff = tf.reduce_sum(
            tf.multiply(errors, tf.transpose(self.L), "convolve_signal"),
            0, name="control_signal_change")
        u = tf.add(prev_u, u_diff, "new_control")

        return u


class joint_ctrls(RandomVariable, Distribution):
    """Auxiliary class to join random variables of controls for all agents
    """

    def __init__(self, params, goals, positions, ctrl_obs,
                 name="control", value=None, dtype=tf.float32,
                 reparameterization_type=FULLY_REPARAMETERIZED,
                 validate_args=True, allow_nan_stats=True):
        with tf.name_scope(name):
            if isinstance(params, list):
                self.agents = [GBDS_U(
                    p, tf.gather(goals, p["col"], axis=-1,
                                 name="%s_posterior_goals" % p["name"]),
                    tf.gather(positions, p["col"], axis=-1,
                              name="%s_positions" % p["name"]),
                    tf.gather(ctrl_obs, p["col"], axis=-1,
                              name="%s_observed_control" % p["name"]),
                    p["name"], tf.gather(value, p["col"], axis=-1))
                    for p in params]
            else:
                raise TypeError("params must be a list.")

        super(joint_ctrls, self).__init__(
            name=name, value=value, dtype=dtype,
            reparameterization_type=reparameterization_type,
            validate_args=validate_args, allow_nan_stats=allow_nan_stats)

        self._kwargs["params"] = params
        self._kwargs["goals"] = goals
        self._kwargs["positions"] = positions
        self._kwargs['ctrl_obs'] = ctrl_obs

    def _log_prob(self, value):
        return tf.add_n([agent.log_prob(tf.gather(value, agent.col, axis=-1))
                         for agent in self.agents])

    def update_ctrl(self, errors, prev_u):
        return tf.concat([agent.update_ctrl(
            tf.gather(errors, agent.col, axis=-1),
            tf.gather(prev_u, agent.col, axis=-1))
                          for agent in self.agents], 0)
