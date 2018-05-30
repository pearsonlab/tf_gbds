import tensorflow as tf
import numpy as np
# import edward as ed
from edward.models import RandomVariable
from tensorflow.contrib.distributions import (Distribution,
                                              FULLY_REPARAMETERIZED)
from tensorflow.python.ops.distributions.special_math import log_ndtr
# from utils import pad_extra_conds


def z(x, loc, scale):
    return (x - loc) / scale


def normal_logpdf(x, loc, scale):
    return -(0.5 * np.log(2 * np.pi) + tf.log(scale) +
             0.5 * tf.square(z(x, loc, scale)))


def normal_logcdf(x, loc, scale):
    return log_ndtr(z(x, loc, scale))


def clip_log_prob(upsilon, u, bounds, tol, eta):
    """upsilon (derived from time series of y) is a censored version of
    a noisy control signal:

    \hat{u} ~ N(u, \eta^2)
    log p(upsilon|u, g) = log p(upsilon|u) + log(u|g)

    log p(upsilon|u) breaks down into three cases,
    namely left-clipped (upsilon_t = lower bound),
    right-clipped (upsilon_t = upper bound),
    and non-clipped (-1 < upsilon_t < 1). For the first two cases,
    Normal CDF is used instead of PDF due to censoring.
    The log density term is computed for each and then add up.
    """
    l_b = tf.add(bounds[0], tol, "lower_bound")
    u_b = tf.subtract(bounds[1], tol, "upper_bound")

    return tf.where(tf.less_equal(upsilon, l_b, name="left_clip"),
                    normal_logcdf(upsilon, u, eta),
                    tf.where(tf.greater_equal(upsilon, u_b,
                                              name="right_clip"),
                             normal_logcdf(-upsilon, -u, eta),
                             normal_logpdf(upsilon, u, eta)))


class GBDS(RandomVariable, Distribution):
    """
    Goal-Based Dynamical System (generative model; for one agent)

    Args:
    - params: a dictionary of model parameters
        Entries include:
        - name: name of the agent
        - col: column(s) in observation that corresponds to the agent
        - dim: modeling dimensions of the agents
        - GMM_NN: neural network that parametrizes the GMM parameters
                  (means, precisions, weights) of goal states given states
        - GMM_K: number of modes in GMM
        - unc_sigma: unconstrained value (before softplus) of goal state
                     noise, sigma
        - sigma_trainable: whether sigma is trainable
        - sigma_pen: penalty on sigma (None if not trainable)
        - g_bounds: boundaries of goal state
        - g_bounds_pen: penalties on goal state leaving boundaries
        - PID: a dictionary of PID control system parameters (Kp, Ki, Kd)
        - latent_u: whether latent control is modeled
                    (if True, the dimension of posterior will be doubled)
        - unc_eps: unconstrained value (before softplus) of control signal
                   noise, epsilon
        - eps_trainable: whether epsilon is trainable
        - eps_pen: penalty on epsilon (None if not trainable)
        - clip: whether control signal is clipped/censored
        - clip_range: unclipped/uncensored range (lower and upper bounds)
        - clip_tol: tolerance of the above range
        - unc_eta: unconstrained value (before softplus) of latent control
                   standard deviation, eta
        - eta_trainable: whether eta is trainable
        - eta_pen: penalty on eta (None if not trainable)
    - states: time series of game states
              (positions and velocities for all agents)
    - ctrl_obs: observed control signal (computed from trajectories)
    - extra_conds: time series of extra conditions
                   (prey positions, velocities, and values)
    """
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
            self.latent_u = params["latent_u"]

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

            with tf.name_scope("control_signal_censoring"):
                # clipping signal
                self.clip = params["clip"]
                if self.clip:
                    if params["clip_range"] is not None:
                        self.clip_range = tf.gather(
                            params["clip_range"], self.col, name="clip_range")
                    else:
                        self.clip_range = tf.constant(
                            np.repeat(np.array([[-1., 1.]]), self.dim, 0),
                            tf.float32, name="clip_range")
                    self.clip_tol = tf.constant(
                        params["clip_tol"], tf.float32, name="clip_tolerance")

                    self.unc_eta = params["unc_eta"]
                    self.eta = tf.nn.softplus(self.unc_eta, "eta")
                    if params["eta_trainable"]:
                        self.params += [self.unc_eta]
                        self.eta_pen = tf.constant(
                            params["eta_pen"], tf.float32, name="eta_penalty")
                    else:
                        self.eta_pen = None

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
    # def get_preds(self, s, y, post_g, extra_conds=None):
        """
        Return one-step-ahead prediction of goal and control signal,
        given state, current position, sample from goal posterior,
        and previous control (and extra conditions if provided).
        """
        with tf.name_scope("pad_extra_conds"):
            if extra_conds is not None:
                s = tf.concat([s, extra_conds], -1)
                # s = pad_extra_conds(s, extra_conds)

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
                # pad the beginning of control signal with zeros
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
        # u_pred = tf.cumsum(u_diff, 1, name="predicted_control_signal")

        return (all_mu, all_lambda, all_w, next_g, u_pred)

    def _log_prob(self, value):
        if self.latent_u:
            g_q = tf.identity(value[:, :, :self.dim], "goal_posterior")
            u_q = tf.identity(value[:, :, self.dim:], "control_posterior")
        else:
            g_q = tf.identity(value, "goal_posterior")
            u_q = tf.pad(self.ctrl_obs, [[0, 0], [1, 0], [0, 0]],
                         name="control_posterior")

        all_mu, all_lambda, all_w, g_pred, u_pred = self.get_preds(
            self.s[:, :-1], self.y[:, :-1], g_q, u_q[:, :-1],
            self.extra_conds[:, :-1])
        # _, all_lambda, all_w, g_pred, u_pred = self.get_preds(
        #     self.s[:, :-1], self.y[:, :-1], g_q, self.extra_conds)

        logdensity_g = 0.0
        with tf.name_scope("goal_states"):
            gmm_res = tf.subtract(
                tf.expand_dims(g_q[:, 1:], 2, "reshape_samples"), g_pred,
                "GMM_residual")
            gmm_term = tf.log(all_w + 1e-8) - tf.reduce_sum(
                (1 + all_lambda) * (gmm_res ** 2) / (2 * self.sigma ** 2), -1)
            gmm_term += (0.5 * tf.reduce_sum(tf.log(1 + all_lambda), -1) -
                         tf.reduce_sum(0.5 * tf.log(2 * np.pi) +
                                       tf.log(self.sigma), -1))
            logdensity_g += tf.reduce_sum(
                tf.reduce_logsumexp(gmm_term, -1), -1)

            tf.summary.scalar("average_log_density", tf.reduce_mean(
                tf.reduce_logsumexp(gmm_term, -1)))

        with tf.name_scope("goal_boundary_penalty"):
            if self.g_pen is not None:
                # penalty on goal state escaping game space
                # logdensity_g -= self.g_pen * tf.reduce_sum(
                #     tf.nn.relu(self.bounds[0] - g_pred), [1, 2, 3])
                # logdensity_g -= self.g_pen * tf.reduce_sum(
                #     tf.nn.relu(g_pred - self.bounds[1]), [1, 2, 3])
                logdensity_g -= self.g_pen * tf.reduce_sum(
                    tf.nn.relu(self.bounds[0] - all_mu), [1, 2, 3]) / self.K
                logdensity_g -= self.g_pen * tf.reduce_sum(
                    tf.nn.relu(all_mu - self.bounds[1]), [1, 2, 3]) / self.K
                logdensity_g -= self.g_pen * tf.reduce_sum(
                    tf.nn.relu(self.bounds[0] - g_q), [1, 2])
                logdensity_g -= self.g_pen * tf.reduce_sum(
                    tf.nn.relu(g_q - self.bounds[1]), [1, 2])

        logdensity_u = 0.0
        with tf.name_scope("control_signal"):
            if self.latent_u:
                logdensity_u += tf.add_n(
                    [tf.reduce_sum(clip_log_prob(
                        self.ctrl_obs[:, :, i], u_q[:, 1:, i],
                        self.clip_range[i], self.clip_tol, self.eta[i]), 1)
                     for i in range(self.dim)], name="clipping")

            u_res = tf.subtract(u_q[:, 1:], u_pred, "residual")
            logdensity_u -= tf.reduce_sum(
                (0.5 * tf.log(2 * np.pi) + tf.log(self.eps) +
                 u_res ** 2 / (2 * self.eps ** 2)), [1, 2])

            tf.summary.histogram("residual", u_res)
            tf.summary.scalar("average_log_density", tf.reduce_mean(
                logdensity_u))

        if self.sigma_pen is not None:
            logdensity_g -= self.sigma_pen * tf.reduce_sum(self.unc_sigma)
        if self.eps_pen is not None:
            logdensity_u -= self.eps_pen * tf.reduce_sum(self.unc_eps)
        if self.eta_pen is not None:
            logdensity_u -= self.eta_pen * tf.reduce_sum(self.unc_eta)

        logdensity = tf.divide(
            tf.reduce_mean(tf.add(logdensity_g, logdensity_u)),
            tf.cast(self.Tt - 1, tf.float32))

        return logdensity

    def sample_GMM(self, state, prev_g, extra_conds=None):
        """
        Generate new goal given current state and previous goal
        """
        state = tf.reshape(state, [1, 1, -1], "reshape_state")
        with tf.name_scope("pad_extra_conds"):
            if extra_conds is not None:
                # Velocity is already part of the state.
                extra_conds = tf.reshape(extra_conds, [1, 1, -1])
                state = tf.concat([state, extra_conds], -1)
                # state = pad_extra_conds(state, extra_conds)

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
                tf.log(all_w, "log_w"), 1, name="draw_sample"), name="k")
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
        """
        Update control signal given errors and previous control
        """
        u_diff = tf.reduce_sum(
            tf.multiply(errors, tf.transpose(self.L), "convolve_signal"),
            0, name="control_signal_change")
        u = tf.add(prev_u, u_diff, "new_control")

        return u


class joint_GBDS(RandomVariable, Distribution):
    """
    Auxiliary class to join models of all agents in the game

    Args:
    - all_params: a list of dictionaries
                  (each contains model parameters for an agent)
    - model_dim: total number of dimensions being modeled
    - states (like GBDS)
    - ctrl_obs (like GBDS)
    - extra_conds (like GBDS)
    - latent_ctrl (like GBDS)
    """
    def __init__(self, all_params, model_dim, states, ctrl_obs,
                 extra_conds=None, latent_ctrl=False, *args, **kwargs):
        name = kwargs.get("name", "joint")
        value = kwargs.get("value", None)
        if value is None:
            raise ValueError("value cannot be None")

        if not isinstance(all_params, list):
            raise TypeError(
                "all_params must be a list but a %s is given" % type(
                    all_params))

        self.names = [params["name"] for params in all_params]
        self.cols = [params["col"] for params in all_params]
        self.model_dim = model_dim
        self.latent_u = latent_ctrl

        with tf.name_scope(name):
            all_values = []
            for name, col in zip(self.names, self.cols):
                if latent_ctrl:
                    all_values.append(tf.concat(
                        [tf.gather(value, col, axis=-1),
                         tf.gather(value, tf.add(col, self.model_dim),
                                   axis=-1)], -1, "value_%s" % name))
                else:
                    all_values.append(tf.gather(
                        value, col, axis=-1, name="value_%s" % name))

            self.agents = [GBDS(params, states, ctrl_obs, extra_conds,
                                name=name, value=value)
                           for params, name, value in zip(
                               all_params, self.names, all_values)]

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

        self._args = (all_params, model_dim, states, ctrl_obs,
                      extra_conds, latent_ctrl)

    def _log_prob(self, value):
        values = []
        for name, col in zip(self.names, self.cols):
            if self.latent_u:
                values.append(tf.concat(
                    [tf.gather(value, col, axis=-1),
                     tf.gather(value, tf.add(col, self.model_dim),
                               axis=-1)], -1, "value_%s" % name))
            else:
                values.append(tf.gather(
                    value, col, axis=-1, name="value_%s" % name))

        return tf.add_n([agent.log_prob(value)
                         for agent, value in zip(self.agents, values)])

    def update_goal(self, state, prev_g, extra_conds=None):
        return tf.concat([agent.sample_GMM(
          state, tf.gather(
              prev_g, col, axis=-1, name="prev_g_%s" % name), extra_conds)
                          for agent, col, name in zip(
                              self.agents, self.cols, self.names)], 0)

    def update_ctrl(self, errors, prev_u):
        return tf.concat([agent.update_ctrl(
            tf.gather(errors, col, axis=-1, name="error_%s" % name),
            tf.gather(prev_u, col, axis=-1, name="prev_u_%s" % name))
                          for agent, col, name in zip(
                              self.agents, self.cols, self.names)], 0)
