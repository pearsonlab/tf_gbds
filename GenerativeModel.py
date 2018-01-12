import tensorflow as tf
import numpy as np
from edward.models import RandomVariable
from tensorflow.contrib.distributions import (Distribution,
                                              FULLY_REPARAMETERIZED)
from tensorflow.python.ops.distributions.special_math import log_ndtr


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
        with tf.name_scope('control_signal_censoring'):
            self.clip_range = GenerativeParams['clip_range']
            self.clip_tol = GenerativeParams['clip_tol']
            self.eta = GenerativeParams['eta']

        with tf.name_scope('control_signal_noise'):
            self.eps = GenerativeParams['PID_params']['eps']

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
                Uprev=tf.pad(value[:, :-1], [[0, 0], [1, 0], [0, 0]]))
        
        LogDensity = 0.0
        with tf.name_scope('control_signal_loss'):
            LogDensity += tf.reduce_sum(
                self.clip_log_prob(self.ctrl_obs, value),
                axis=[1, 2], name='clip_noise')

            resU = value - Upred
            LogDensity -= tf.reduce_sum(
                (0.5 * tf.log(2 * np.pi) + tf.log(self.eps) +
                 resU ** 2 / (2 * self.eps ** 2)), axis=[1, 2])

        return LogDensity / tf.cast(self.Tt, tf.float32)
