import tensorflow as tf
import numpy as np
from edward.models import RandomVariable
from tensorflow.contrib.distributions import (Distribution,
                                              FULLY_REPARAMETERIZED)
from tensorflow.python.ops.distributions.special_math import log_ndtr


def logsumexp(x, axis=None):
    x_max = tf.reduce_max(x, axis=axis, keep_dims=True)
    return (tf.log(tf.reduce_sum(tf.exp(x - x_max),
                                 axis=axis, keep_dims=True)) + x_max)


class GBDS_g(RandomVariable, Distribution):

    def __init__(self, GenerativeParams, y, yDim, name='GBDS_g',
                 value=None, dtype=tf.float32,
                 reparameterization_type=FULLY_REPARAMETERIZED,
                 validate_args=True, allow_nan_stats=True):

        self.y = y
        self.yDim = yDim
        self.B = tf.shape(y)[0]
        self.Tt = tf.shape(y)[1]
        self.GenerativeParams = GenerativeParams

        with tf.name_scope('get_states'):
            self.get_states = GenerativeParams['get_states']

        with tf.name_scope('velocity'):
            self.vel = tf.constant(GenerativeParams['vel'],
                                   dtype=tf.float32, name='velocity')

        with tf.name_scope('g0'):
            self.g0_mu = GenerativeParams['g0_params']['mu']
            self.g0_unc_lambda = GenerativeParams['g0_params']['unc_lambda']
            self.g0_lambda = tf.nn.softplus(self.g0_unc_lambda,
                                            name='softplus_g0_lambda')
            self.g0_unc_w = GenerativeParams['g0_params']['unc_w']
            self.g0_w = tf.nn.softmax(self.g0_unc_w,
                                      name='softmax_g0_w')
            self.g0_params = ([self.g0_mu] + [self.g0_unc_lambda] +
                              [self.g0_unc_w])

        with tf.name_scope('GMM_NN'):
            self.GMM_k = GenerativeParams['GMM_k']
            self.GMM_net = GenerativeParams['GMM_net']

        with tf.name_scope('goal_state_noise'):
            self.sigma = GenerativeParams['sigma']

        with tf.name_scope('goal_boundary_penalty'):
            with tf.name_scope('boundary'):
                self.bounds_g = 1.0
            with tf.name_scope('penalty'):
                if GenerativeParams['pen_g'] is not None:
                    self.pen_g = GenerativeParams['pen_g']
                else:
                    self.pen_g = None

        super(GBDS_g, self).__init__(
            name=name, value=value, dtype=dtype,
            reparameterization_type=reparameterization_type,
            validate_args=validate_args, allow_nan_stats=allow_nan_stats)

        self._kwargs['y'] = y
        self._kwargs['yDim'] = yDim
        self._kwargs['GenerativeParams'] = GenerativeParams

    def get_preds(self, Y, post_g):
        with tf.name_scope('states'):
            states = self.get_states(Y, max_vel=self.vel)

        with tf.name_scope('get_GMM_params'):
            with tf.name_scope('mu'):
                all_mu = tf.reshape(
                    self.GMM_net(states)[:, :, :(self.yDim * self.GMM_k)],
                    [self.B, -1, self.GMM_k, self.yDim], name='all_mu')

            with tf.name_scope('lambda'):
                all_lambda = tf.nn.softplus(tf.reshape(
                    self.GMM_net(states)[:, :, (self.yDim *
                                                self.GMM_k):(2 * self.yDim *
                                                             self.GMM_k)],
                    [self.B, -1, self.GMM_k, self.yDim],
                    name='reshape_lambda'), name='all_lambda')

            with tf.name_scope('w'):
                all_w = tf.nn.softmax(tf.reshape(
                    self.GMM_net(states)[:, :, (2 * self.yDim * self.GMM_k):],
                    [self.B, -1, self.GMM_k],
                    name='reshape_w'), dim=-1, name='all_w')

        with tf.name_scope('next_g'):
            next_g = ((tf.expand_dims(post_g, 2) + all_mu * all_lambda) /
                      (1 + all_lambda))

        return (all_mu, all_lambda, all_w, next_g)

    def _log_prob(self, value):
        with tf.name_scope('next_time_point_pred'):
            _, all_lambda, all_w, g_pred = self.get_preds(
                self.y[:, :-1], post_g=value[:, :-1])

        LogDensity = 0.0
        with tf.name_scope('goal_state_loss'):
            gmm_res_g = (tf.expand_dims(value[:, 1:], 2,
                                        name='reshape_posterior_samples') -
                         g_pred)
            gmm_term = (tf.log(all_w + 1e-8) - tf.reduce_sum(
                (1 + all_lambda) * (gmm_res_g ** 2) / (2 * self.sigma ** 2),
                axis=-1))
            gmm_term += (0.5 * tf.reduce_sum(tf.log(1 + all_lambda),
                                             axis=-1) -
                         tf.reduce_sum(tf.log(self.sigma), axis=-1) -
                         0.5 * tf.log(2 * np.pi))
            LogDensity += tf.reduce_sum(logsumexp(gmm_term, axis=-1),
                                        axis=[-2, -1])

        with tf.name_scope('g0_loss'):
            res_g0 = tf.expand_dims(value[:, 0], 1) - self.g0_mu
            g0_term = (tf.log(self.g0_w + 1e-8) - tf.reduce_sum(
                 (res_g0 ** 2) * self.g0_lambda / 2, axis=-1))
            g0_term += (0.5 * tf.reduce_sum(tf.log(self.g0_lambda), axis=-1) -
                        0.5 * tf.log(2 * np.pi))
            LogDensity += tf.reduce_sum(logsumexp(g0_term, axis=-1), axis=-1)

        with tf.name_scope('goal_penalty'):
            with tf.name_scope('boundary'):
                if self.pen_g is not None:
                    LogDensity -= (self.pen_g * tf.reduce_sum(
                        tf.nn.relu(value - self.bounds_g), axis=[1, 2]))
                    LogDensity -= (self.pen_g * tf.reduce_sum(
                        tf.nn.relu(-value - self.bounds_g), axis=[1, 2]))

        LogDensity /= tf.cast(self.Tt, tf.float32)

        with tf.name_scope('control_model'):
            LogDensity += GBDS_u(
                self.GenerativeParams, value, self.y, self.yDim, name='u',
                value=tf.zeros_like(value))._log_prob(value=None)

        return LogDensity

    def getParams(self):
        return self.GMM_net.variables + self.g0_params

    def sample_g0(self):
        with tf.name_scope('select_component'):
            k_0 = tf.squeeze(tf.multinomial(tf.reshape(
                tf.log(self.g0_w), [1, -1]), 1), name='k_0')

        with tf.name_scope('get_sample'):
            g_0 = (self.g0_mu[k_0] +
                   tf.random_normal([self.yDim], name='std_normal') /
                   tf.sqrt(self.g0_lambda[k_0], name='inv_std_dev'))

        return g_0

    def sample_GMM(self, state, curr_goal):
        state = tf.reshape(state, [1, 1, -1], name='reshape_state')
        with tf.name_scope('mu'):
            all_mu = tf.reshape(
                self.GMM_net(state)[:, :, :(self.yDim * self.GMM_k)],
                [self.GMM_k, self.yDim], name='all_mu')

        with tf.name_scope('lambda'):
            all_lambda = tf.nn.softplus(tf.reshape(
                self.GMM_net(state)[:, :, (self.yDim *
                                            self.GMM_k):(2 * self.yDim *
                                                         self.GMM_k)],
                [self.GMM_k, self.yDim],
                name='reshape_lambda'), name='all_lambda')

        with tf.name_scope('w'):
            all_w = tf.nn.softmax(tf.reshape(
                self.GMM_net(state)[:, :, (2 * self.yDim * self.GMM_k):],
                [1, self.GMM_k],
                name='reshape_w'), dim=-1, name='all_w')

        with tf.name_scope('select_component'):
            k = tf.squeeze(tf.multinomial(
                tf.reshape(tf.log(self.g0_w), [1, -1]), 1), name='k')

        with tf.name_scope('get_sample'):
            next_g = (tf.divide(curr_goal + all_mu[k] * all_lambda[k],
                      1 + all_lambda[k], name='mean') +
                      tf.random_normal([self.yDim], name='std_normal') *
                      tf.divide(tf.squeeze(self.sigma),
                                tf.sqrt(1 + all_lambda[k]), name='std_dev'))

        return next_g

    def update_ctrl(self, errors, curr_ctrl):
        return GBDS_u(self.GenerativeParams, tf.zeros_like(self.y), self.y,
                      self.yDim, name='u',
                      value=tf.zeros_like(self.y)).update_ctrl(
                      errors, curr_ctrl)


class GBDS_u(RandomVariable, Distribution):

    def __init__(self, GenerativeParams, g, y, yDim, name='GBDS_u',
                 value=None, dtype=tf.float32,
                 reparameterization_type=FULLY_REPARAMETERIZED,
                 validate_args=True, allow_nan_stats=True):

        self.g = g
        self.y = y
        self.yDim = yDim
        self.B = tf.shape(y)[0]
        self.Tt = tf.shape(y)[1]

        with tf.name_scope('velocity'):
            self.vel = tf.constant(GenerativeParams['vel'], tf.float32)

        with tf.name_scope('PID_controller_params'):
            with tf.name_scope('parameters'):
                self.Kp = GenerativeParams['PID_params']['Kp']
                self.Ki = GenerativeParams['PID_params']['Ki']
                self.Kd = GenerativeParams['PID_params']['Kd']
                self.PID_params = [self.Kp, self.Ki, self.Kd]
            with tf.name_scope('filter'):
                t_coeff = self.Kp + self.Ki + self.Kd
                t1_coeff = -self.Kp - 2 * self.Kd
                t2_coeff = self.Kd
                self.L = tf.concat([t2_coeff, t1_coeff, t_coeff], axis=1,
                                   name='filter')

        with tf.name_scope('observed_control_signal'):
            self.ctrl_obs = tf.divide(self.y[:, 1:] - self.y[:, :-1],
                self.vel, name='ctrl_obs')
        # with tf.name_scope('control_signal_censoring'):
        #     self.clip_range = GenerativeParams['clip_range']
        #     self.clip_tol = GenerativeParams['clip_tol']
        #     self.eta = GenerativeParams['eta']

        with tf.name_scope('control_signal_noise'):
            # self.eps = GenerativeParams['PID_params']['eps']
            self.pen_u = GenerativeParams['pen_u']
            self.res_tol = GenerativeParams['res_tol']

        super(GBDS_u, self).__init__(
            name=name, value=value, dtype=dtype,
            reparameterization_type=reparameterization_type,
            validate_args=validate_args, allow_nan_stats=allow_nan_stats)

        self._kwargs['g'] = g
        self._kwargs['y'] = y
        self._kwargs['yDim'] = yDim
        self._kwargs['GenerativeParams'] = GenerativeParams

    def get_preds(self, Y, post_g=None, Uprev=None):
        with tf.name_scope('control_error'):
            error = tf.subtract(post_g, Y, name='ctrl_error')
        with tf.name_scope('control_signal_change'):
            Udiff = []
            for i in range(self.yDim):
                signal = error[:, :, i]
                signal = tf.expand_dims(
                    tf.pad(signal, [[0, 0], [2, 0]], name='zero_padding'), -1)
                filt = tf.reshape(self.L[i], [-1, 1, 1])
                res = tf.nn.convolution(signal, filt, padding='VALID',
                                        name='signal_conv')
                Udiff.append(res)
            if len(Udiff) > 1:
                Udiff = tf.concat([*Udiff], axis=-1)
            else:
                Udiff = Udiff[0]
        with tf.name_scope('control_signal'):
            Upred = Uprev + Udiff

        return (error, Upred)

    def clip_log_prob(self, upsilon, u):
        u_b = self.clip_range - self.clip_tol
        l_b = -self.clip_range + self.clip_tol
        eta = self.eta

        def z(x, loc, scale):
            return (x - loc) / scale

        def normal_logpdf(x, loc, scale):
            return -(0.5 * np.log(2 * np.pi) + tf.log(scale) +
                     0.5 * tf.square(z(x, loc, scale)))

        def normal_logcdf(x, loc, scale):
            return log_ndtr(z(x, loc, scale))

        return tf.where(tf.less_equal(upsilon, l_b),
                        normal_logcdf(-1., u, eta),
                        tf.where(tf.greater_equal(upsilon, u_b),
                                 normal_logcdf(-1., -u, eta),
                                 normal_logpdf(upsilon, u, eta)))

    def _log_prob(self, value):
        with tf.name_scope('next_time_point_pred'):
            ctrl_error, Upred = self.get_preds(
                Y=self.y[:, :-1], post_g=self.g[:, 1:],
                Uprev=tf.pad(self.ctrl_obs[:, :-1], [[0, 0], [1, 0], [0, 0]]))

        LogDensity = 0.0
        with tf.name_scope('control_signal_loss'):
            # LogDensity += tf.reduce_sum(
            #     self.clip_log_prob(self.ctrl_obs, value[:, :-1]),
            #     axis=[1, 2], name='clip_noise')

            resU = self.ctrl_obs - Upred
            # LogDensity -= tf.reduce_sum(
            #     (0.5 * tf.log(2 * np.pi) + tf.log(self.eps) +
            #      resU ** 2 / (2 * self.eps ** 2)), axis=[1, 2])
            LogDensity -= tf.reduce_sum(self.pen_u * tf.nn.relu(
                resU - self.res_tol), axis=[1, 2])
            LogDensity -= tf.reduce_sum(self.pen_u * tf.nn.relu(
                -resU - self.res_tol), axis=[1, 2])

        return LogDensity / tf.cast(self.Tt, tf.float32)

    def update_ctrl(self, errors, curr_ctrl):
        ctrl_diff = tf.reduce_sum(errors * tf.transpose(self.L), axis=0,
                                  name='ctrl_diff')

        return ctrl_diff + curr_ctrl


def generate_trial(goal_model, ctrl_model, y0, yDim, trial_len):
    g = tf.reshape(goal_model.sample_g0(), [1, yDim], name='g0')
    u = tf.zeros([1, yDim], tf.float32, name='u0')
    prev_error = tf.zeros([yDim], tf.float32, name='prev_error')
    prev2_error = tf.zeros([yDim], tf.float32, name='prev2_error')
    vel = goal_model.vel
    y = tf.reshape(y0, [1, yDim], name='y0')

    for t in range(trial_len - 1):
        if t == 0:
            v_t = tf.zeros_like(y0, tf.float32, name='curr_vel')
        else:
            v_t = tf.subtract(y[t], y[t - 1], name='curr_vel')
        s_t = tf.stack([y[t], v_t], 0, name='curr_state')
        g_new = tf.reshape(goal_model.sample_GMM(s_t, g[t]), [1, yDim],
                           name='g_new')
        g = tf.concat([g, g_new], 0, name='concat_g')

        error = tf.subtract(g[t + 1] , y[t], name='curr_error')
        errors = tf.stack([error, prev_error, prev2_error], 0, name='errors')
        if ctrl_model is None:
            u_new = tf.reshape(
                goal_model.update_ctrl(errors, u[t]), [1, yDim], name='u_new')
        else:
            u_new = tf.reshape(
                ctrl_model.update_ctrl(errors, u[t]), [1, yDim], name='u_new')
        u = tf.concat([u, u_new], 0, name='concat_u')

        prev2_error = prev_error
        prev_error = error

        y_new = tf.reshape(tf.clip_by_value(y[t] + vel * tf.clip_by_value(
            u[t + 1], -1., 1., name='clip_u'), -1., 1., name='bound_y'),
                           [1, yDim], name='y_new')
        y = tf.concat([y, y_new], 0, name='concat_y')

    return y, u, g
