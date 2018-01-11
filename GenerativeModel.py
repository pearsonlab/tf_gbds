import tensorflow as tf
import numpy as np
from edward.models import RandomVariable
from tensorflow.contrib.distributions import (Distribution,
                                              FULLY_REPARAMETERIZED)
from tensorflow.python.ops.distributions.special_math import log_ndtr


class GBDS_u(RandomVariable, Distribution):

    def __init__(self, GenerativeParams, y, yDim, name='GBDS_u',
                 value=None, dtype=tf.float32,
                 reparameterization_type=FULLY_REPARAMETERIZED,
                 validate_args=True, allow_nan_stats=True):

        self.y = y
        self.yDim = yDim
        self.B = tf.shape(y)[0]
        self.Tt = tf.shape(y)[1]

        with tf.name_scope('velocity'):
            self.vel = tf.constant(GenerativeParams['vel'], tf.float32)

        with tf.name_scope('observed_control_signal'):
            self.ctrl_obs = tf.divide(self.y[:, 1:] - self.y[:, :-1],
                self.vel, name='ctrl_obs')
        with tf.name_scope('control_signal_censoring'):
            self.clip_range = GenerativeParams['clip_range']
            self.clip_tol = GenerativeParams['clip_tol']
            self.eta = GenerativeParams['eta']

        super(GBDS_u, self).__init__(
            name=name, value=value, dtype=dtype,
            reparameterization_type=reparameterization_type,
            validate_args=validate_args, allow_nan_stats=allow_nan_stats)

        self._kwargs['y'] = y
        self._kwargs['yDim'] = yDim
        self._kwargs['GenerativeParams'] = GenerativeParams

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
        LogDensity = tf.reduce_sum(
            self.clip_log_prob(self.ctrl_obs, value[:, :-1]),
            axis=[1, 2], name='clip_noise')

        return LogDensity / tf.cast(self.Tt, tf.float32)
