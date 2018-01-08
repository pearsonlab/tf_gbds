import tensorflow as tf
import numpy as np
from edward.models import RandomVariable
from tensorflow.contrib.distributions import (Distribution,
                                              FULLY_REPARAMETERIZED,
                                              Normal)
from tensorflow.contrib.keras import layers, models
from tensorflow.python.ops.distributions.special_math import log_ndtr
from tf_gbds.utils import pad_extra_conds


def logsumexp(x, axis=None):
    x_max = tf.reduce_max(x, axis=axis, keep_dims=True)
    return (tf.log(tf.reduce_sum(tf.exp(x - x_max),
                                 axis=axis, keep_dims=True)) + x_max)


class GBDS_g_all(RandomVariable, Distribution):
    """A customized Random Variable of goal in Goal-Based Dynamical System
    combining both goalie agent and ball agent.
    """

    def __init__(self, GenerativeParams_goalie, GenerativeParams_ball,
                 yDim, y, extra_conds, ctrl_obs, name='GBDS_g_all',
                 value=None, dtype=tf.float32,
                 reparameterization_type=FULLY_REPARAMETERIZED,
                 validate_args=True, allow_nan_stats=True):

        """Initialize a batch of GBDS_g random variables combining both ball
            and goalie agent


        Args:
          GenerativeParams_goalie: A dictionary of parameters for the goalie
                                   agent
          GenerativeParams_ball: A dictionary of parameters for the ball agent
          Entries both include:
            - get_states: function that calculates current state from position
            - pen_eps: Penalty on control signal noise, epsilon
            - pen_sigma: Penalty on goals state noise, sigma
            - pen_g: Two penalties on goal state leaving boundaries (Can be set
                     None)
            - bounds_g: Boundaries corresponding to above penalties
            - NN_postJ_mu: Neural network that parametrizes the mean of the
                           posterior of J (i.e. mu and sigma), conditioned on
                           goals
            - NN_postJ_sigma: Neural network that parametrizes the covariance
                              of the posterior of J (i.e. mu and sigma),
                              conditioned on goals
            - yCols: Columns of Y this agent corresponds to. Used to index
                     columns in real data to compare against generated data.
            - vel: Maximum velocity of each dimension in columns belong to this
                   agent(yCols)
            - all_vel: Maximum velocity of each dimension in columns belong to
                       all agents
            - clip: Clipping signal
            - clip_range: the range of the clipping states
            - clip_tol: the tolenrance of clipping
            - GMM_net: Gaussian Mixture Model network
            - GMM_k: Number of GMM components
          yDim: Number of dimensions for the data
          y: Time series of positions
          value: The Random Variable sample of goal. Since GBDS_g_all is just
                 a likelihood node, value is just used to specify the shape
                 of g. Set it to tf.zeros_like(Y).

        """

        self.yCols_goalie = GenerativeParams_goalie['yCols']
        self.yCols_ball = GenerativeParams_ball['yCols']
        self.y = y
        self.yDim = yDim
        self.extra_conds = extra_conds
        self.ctrl_obs = ctrl_obs

        yDim_goalie = len(self.yCols_goalie)
        yDim_ball = len(self.yCols_ball)

        with tf.name_scope('observed_control'):
            if self.ctrl_obs is not None:
                ctrl_goalie = tf.gather(self.ctrl_obs, self.yCols_goalie,
                                        axis=-1, name='ctrl_obs_goalie')
                ctrl_ball = tf.gather(self.ctrl_obs, self.yCols_ball,
                                      axis=-1, name='ctrl_obs_ball')
            else:
                ctrl_goalie = None
                ctrl_ball = None

        with tf.name_scope('G_goalie'):
            self.goalie = GBDS_g(GenerativeParams_goalie, yDim_goalie, yDim,
                                 y, extra_conds, ctrl_goalie, name='G_goalie',
                                 value=tf.gather(value, self.yCols_goalie,
                                                 axis=-1))
        with tf.name_scope('G_ball'):
            self.ball = GBDS_g(GenerativeParams_ball, yDim_ball, yDim, y,
                               extra_conds, ctrl_ball, name='G_ball',
                               value=tf.gather(value, self.yCols_ball,
                                               axis=-1))

        super(GBDS_g_all, self).__init__(
            name=name, value=value, dtype=dtype,
            reparameterization_type=reparameterization_type,
            validate_args=validate_args, allow_nan_stats=allow_nan_stats)

        self._kwargs['GenerativeParams_goalie'] = GenerativeParams_goalie
        self._kwargs['GenerativeParams_ball'] = GenerativeParams_ball
        self._kwargs['y'] = y
        self._kwargs['yDim'] = yDim
        self._kwargs['extra_conds'] = extra_conds
        self._kwargs['ctrl_obs'] = ctrl_obs

    def _log_prob(self, value):
        with tf.name_scope('goalie'):
            log_prob_goalie = self.goalie.log_prob(
                tf.gather(value, self.yCols_goalie, axis=-1))
        with tf.name_scope('ball'):
            log_prob_ball = self.ball.log_prob(
                tf.gather(value, self.yCols_ball, axis=-1))

        return log_prob_goalie + log_prob_ball

    def getParams(self):
        return self.goalie.getParams() + self.ball.getParams()


class GBDS_u_all(RandomVariable, Distribution):
    """A customized Random Variable of control signal in Goal-Based Dynamical
        System combining both goalie agent and ball agent.
    """

    def __init__(self, GenerativeParams_goalie, GenerativeParams_ball,
                 g, y, ctrl_obs, yDim, name='GBDS_u_all',
                 value=None, dtype=tf.float32,
                 reparameterization_type=FULLY_REPARAMETERIZED,
                 validate_args=True, allow_nan_stats=True):

        """Initialize a batch of GBDS_u random variables combining both ball
            and goalie agent


        Args:
          GenerativeParams_goalie: A dictionary of parameters for the goalie
                                   agent
          GenerativeParams_ball: A dictionary of parameters for the ball agent
          Entries both include:
            - get_states: function that calculates current state from position
            - pen_eps: Penalty on control signal noise, epsilon
            - pen_sigma: Penalty on goals state noise, sigma
            - pen_g: Two penalties on goal state leaving boundaries (Can be set
                     None)
            - bounds_g: Boundaries corresponding to above penalties
            - NN_postJ_mu: Neural network that parametrizes the mean of the
                           posterior of J (i.e. mu and sigma), conditioned on
                           goals
            - NN_postJ_sigma: Neural network that parametrizes the covariance
                              of the posterior of J (i.e. mu and sigma),
                              conditioned on goals
            - yCols: Columns of Y this agent corresponds to. Used to index
                     columns in real data to compare against generated data.
            - vel: Maximum velocity of each dimension in columns belong to this
                   agent(yCols)
            - all_vel: Maximum velocity of each dimension in columns belong to
                       all agents
            - clip: Clipping signal
            - clip_range: the range of the clipping states
            - clip_tol: the tolenrance of clipping
            - GMM_net: Gaussian Mixture Model network
            - GMM_k: Number of GMM components
          yDim: Number of dimensions for the data
          y: Time series of positions
          g: The dependent customized Random Variable of goal in Goal-Based
             Dynamical System (GBDS_g_all)
          value: The Random Variable sample of goal. Since GBDS_u_all is just a
                 likelihood node, value is just used to specify the shape of g.
                 Set it to tf.zeros_like(Y).
        """

        self.yCols_goalie = GenerativeParams_goalie['yCols']
        self.yCols_ball = GenerativeParams_ball['yCols']
        self.y = y
        self.yDim = yDim
        self.g = g
        self.ctrl_obs = ctrl_obs

        yDim_goalie = len(self.yCols_goalie)
        yDim_ball = len(self.yCols_ball)

        with tf.name_scope('goals'):
            g_goalie = tf.gather(self.g, self.yCols_goalie, axis=-1,
                                 name='goals_goalie')
            g_ball = tf.gather(self.g, self.yCols_ball, axis=-1,
                               name='goals_ball')

        with tf.name_scope('observed_control'):
            if self.ctrl_obs is not None:
                ctrl_goalie = tf.gather(self.ctrl_obs, self.yCols_goalie,
                                        axis=-1, name='ctrl_obs_goalie')
                ctrl_ball = tf.gather(self.ctrl_obs, self.yCols_ball,
                                      axis=-1, name='ctrl_obs_ball')
            else:
                ctrl_goalie = None
                ctrl_ball = None

        with tf.name_scope('U_goalie'):
            self.goalie = GBDS_u(GenerativeParams_goalie, g_goalie, y,
                                 ctrl_goalie, yDim_goalie, name='U_goalie',
                                 value=tf.gather(value, self.yCols_goalie,
                                                 axis=-1))
        with tf.name_scope('U_ball'):
            self.ball = GBDS_u(GenerativeParams_ball, g_ball, y, ctrl_ball,
                               yDim_ball, name='U_ball',
                               value=tf.gather(value, self.yCols_ball,
                                               axis=-1))

        super(GBDS_u_all, self).__init__(
            name=name, value=value, dtype=dtype,
            reparameterization_type=reparameterization_type,
            validate_args=validate_args, allow_nan_stats=allow_nan_stats)

        self._kwargs['GenerativeParams_goalie'] = GenerativeParams_goalie
        self._kwargs['GenerativeParams_ball'] = GenerativeParams_ball
        self._kwargs['g'] = g
        self._kwargs['y'] = y
        self._kwargs['ctrl_obs'] = ctrl_obs
        self._kwargs['yDim'] = yDim

    def _log_prob(self, value):
        with tf.name_scope('goalie'):
            log_prob_goalie = self.goalie.log_prob(
                tf.gather(value, self.yCols_goalie, axis=-1))
        with tf.name_scope('ball'):
            log_prob_ball = self.ball.log_prob(
                tf.gather(value, self.yCols_ball, axis=-1))

        return log_prob_goalie + log_prob_ball

    def getParams(self):
        return self.goalie.getParams() + self.ball.getParams()


class GBDS_u(RandomVariable, Distribution):
    """A customized Random Variable of control signal in Goal-Based Dynamical
       System
    """

    def __init__(self, GenerativeParams, g, y, ctrl_obs, yDim, name='GBDS_u',
                 value=None, dtype=tf.float32,
                 reparameterization_type=FULLY_REPARAMETERIZED,
                 validate_args=True, allow_nan_stats=True):

        """Initialize a batch of GBDS_u random variables for one of the agents


        Args:
          GenerativeParams: A dictionary of parameters for the agent
          Entries include:
            - get_states: function that calculates current state from position
            - pen_eps: Penalty on control signal noise, epsilon
            - pen_sigma: Penalty on goals state noise, sigma
            - pen_g: Two penalties on goal state leaving boundaries (Can be set
                     None)
            - bounds_g: Boundaries corresponding to above penalties
            - NN_postJ_mu: Neural network that parametrizes the mean of the
                           posterior of J (i.e. mu and sigma), conditioned on
                           goals
            - NN_postJ_sigma: Neural network that parametrizes the covariance
                              of the posterior of J (i.e. mu and sigma),
                              conditioned on goals
            - yCols: Columns of Y this agent corresponds to. Used to index
                     columns in real data to compare against generated data.
            - vel: Maximum velocity of each dimension in columns belong to this
                   agent(yCols)
            - all_vel: Maximum velocity of each dimension in columns belong to
                       all agents
            - clip: Clipping signal
            - clip_range: the range of the clipping states
            - clip_tol: the tolenrance of clipping
            - GMM_net: Gaussian Mixture Model network
            - GMM_k: Number of GMM components
          g: The dependent customized Random Variable of goal in Goal-Based
             Dynamical System (GBDS_g)
          y: Time series of positions
          yDim: Number of dimensions for this agent
          value: The Random Variable sample of control signal
        """

        self.g = g
        self.y = y
        self.yDim = yDim
        self.B = tf.shape(y)[0]  # batch size
        self.latent_u = GenerativeParams['latent_u']
        self.clip = GenerativeParams['clip']

        with tf.name_scope('agent_columns'):
            # which dimensions of Y to predict
            self.yCols = GenerativeParams['yCols']
        with tf.name_scope('velocity'):
            # velocity for each observation dimension (of this agent)
            self.vel = tf.constant(GenerativeParams['vel'], tf.float32)

        with tf.name_scope('PID_controller_params'):
            with tf.name_scope('parameters'):
                # coefficients for PID controller (one for each dimension)
                # https://en.wikipedia.org/wiki/PID_controller
                # Discrete_implementation
                PID_params = GenerativeParams['PID_params']
                unc_Kp = PID_params['unc_Kp']
                # unc_Ki = PID_params['unc_Ki']
                # unc_Kd = PID_params['unc_Kd']
                # create list of PID controller parameters for easy access in
                # getParams
                # self.PID_params = [unc_Kp, unc_Ki, unc_Kd]
                self.PID_params = [unc_Kp]
                # constrain PID controller parameters to be positive
                self.Kp = tf.nn.softplus(unc_Kp, name='Kp')
                # self.Ki = tf.nn.softplus(unc_Ki, name='Ki')
                # self.Kd = tf.nn.softplus(unc_Kd, name='Kd')
            with tf.name_scope('filter'):
                # calculate coefficients to be placed in convolutional filter
                # t_coeff = self.Kp + self.Ki + self.Kd
                # t1_coeff = -self.Kp - 2 * self.Kd
                # t2_coeff = self.Kd
                t_coeff = self.Kp
                t1_coeff = -self.Kp
                t2_coeff = tf.zeros_like(self.Kp, dtype=tf.float32)
                # concatenate coefficients into filter
                self.L = tf.concat([t2_coeff, t1_coeff, t_coeff], axis=1,
                                   name='filter')

        with tf.name_scope('control_signal'):
            if ctrl_obs is not None:
                self.ctrl_obs = ctrl_obs
            else:
                self.ctrl_obs = tf.divide(
                    (tf.gather(self.y, self.yCols, axis=-1)[:, 1:] -
                     tf.gather(self.y, self.yCols, axis=-1)[:, :-1]),
                    tf.reshape(self.vel, [1, self.yDim]), name='ctrl_obs')
        with tf.name_scope('control_signal_censoring'):
            # clipping signal
            if self.clip:
                self.clip_range = GenerativeParams['clip_range']
                self.clip_tol = GenerativeParams['clip_tol']
                self.eta = GenerativeParams['eta']
        with tf.name_scope('control_signal_noise'):
            # noise coefficient on control signals
            self.unc_eps = PID_params['unc_eps']
            self.eps = tf.nn.softplus(self.unc_eps, name='eps')
        with tf.name_scope('control_signal_penalty'):
            # penalty on control error
            if GenerativeParams['pen_ctrl_error'] is not None:
                self.pen_ctrl_error = GenerativeParams['pen_ctrl_error']
            else:
                self.pen_ctrl_error = None
            # penalty on epsilon (noise of control signal)
            if GenerativeParams['pen_eps'] is not None:
                self.pen_eps = GenerativeParams['pen_eps']
            else:
                self.pen_eps = None

        super(GBDS_u, self).__init__(
            name=name, value=value, dtype=dtype,
            reparameterization_type=reparameterization_type,
            validate_args=validate_args, allow_nan_stats=allow_nan_stats)

        self._kwargs['g'] = g
        self._kwargs['y'] = y
        self._kwargs['ctrl_obs'] = ctrl_obs
        self._kwargs['yDim'] = yDim
        self._kwargs['GenerativeParams'] = GenerativeParams

    def get_preds(self, Y, training=False, post_g=None, Uprev=None):
        """Return the predicted next U, and Y for each point in Y.

        For training: provide post_g, sample from the posterior,
                      which is used to calculate the ELBO
        """

        with tf.name_scope('control_error'):
            # PID Controller for next control point
            if post_g is not None:  # calculate error from posterior goals
                error = tf.subtract(post_g, Y, name='ctrl_error')
            # else:  # calculate error from generated goals
            #     error = next_g - tf.gather(Y, self.yCols, axis=1)
        with tf.name_scope('control_signal_change'):
            Udiff = []
            # get current error signal and corresponding filter
            for i in range(self.yDim):
                signal = error[:, :, i]
                # zero pad beginning
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
        with tf.name_scope('add_noise'):
            if post_g is None:  # Add control signal noise to generated data
                Udiff += self.eps * tf.random_normal(Udiff.shape)
        with tf.name_scope('control_signal'):
            Upred = Uprev + Udiff
        with tf.name_scope('predicted_position'):
            # get predicted Y
            if self.clip:
                Ypred = (Y + (tf.reshape(self.vel, [1, self.yDim]) *
                              tf.clip_by_value(
                                  Upred, -self.clip_range, self.clip_range,
                                  name='clipped_signal')))
            else:
                Ypred = (Y + tf.reshape(self.vel, [1, self.yDim]) * Upred)

        return (error, Upred, Ypred)

    # def clip_loss(self, acc, inputs):
    #     """upsilon (derived from time series of y) is a censored version of
    #     a noisy control signal: \hat{u} ~ N(u, \eta^2).
    #     log p(upsilon|u, g) = log p(upsilon|u) + log(u|g)
    #     log p(upsilon|u) breaks down into three cases,
    #     namely left-clipped (upsilon_t = -1), right-clipped (upsilon_t = 1),
    #     and non-clipped (-1 < upsilon_t < 1). For the first two cases,
    #     Normal CDF is used instead of PDF due to censoring.
    #     The log density term is computed for each and then add up.
    #     """
    #     (U_obs, value) = inputs
    #     left_clip_ind = tf.where(tf.less_equal(
    #         U_obs, (-self.clip_range + self.clip_tol)),
    #         name='left_clip_indices')
    #     right_clip_ind = tf.where(tf.greater_equal(
    #         U_obs, (self.clip_range - self.clip_tol)),
    #         name='right_clip_indices')
    #     non_clip_ind = tf.where(tf.logical_and(
    #         tf.greater(U_obs, (-self.clip_range + self.clip_tol)),
    #         tf.less(U_obs, (self.clip_range - self.clip_tol))),
    #         name='non_clip_indices')
    #     left_clip_node = Normal(tf.gather_nd(value, left_clip_ind),
    #                             self.eta, name='left_clip_node')
    #     right_clip_node = Normal(tf.gather_nd(-value, right_clip_ind),
    #                              self.eta, name='right_clip_node')
    #     non_clip_node = Normal(tf.gather_nd(value, non_clip_ind),
    #                            self.eta, name='non_clip_node')
    #     LogDensity = 0.0
    #     LogDensity += tf.reduce_sum(
    #         left_clip_node.log_cdf(-1., name='left_clip_logcdf'))
    #     LogDensity += tf.reduce_sum(
    #         right_clip_node.log_cdf(-1., name='right_clip_logcdf'))
    #     LogDensity += tf.reduce_sum(
    #         non_clip_node.log_prob(tf.gather_nd(U_obs, non_clip_ind),
    #                                name='non_clip_logpdf'))

    #     return LogDensity

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
        """Evaluates the log-density of the GenerativeModel.
        """

        # Get predictions for next timestep (at each timestep except for last)
        # disregard last timestep bc we don't know the next value, thus, we
        # can't calculate the error
        with tf.name_scope('next_time_point_pred'):
            if self.latent_u:
                ctrl_error, Upred, _ = self.get_preds(
                    Y=tf.gather(self.y[:, :-1], self.yCols, axis=-1),
                    training=True, post_g=self.g[:, 1:],
                    Uprev=tf.pad(value[:, :-1], [[0, 0], [1, 0], [0, 0]]))
            else:
                ctrl_error, Upred, _ = self.get_preds(
                    Y=tf.gather(self.y[:, :-1], self.yCols, axis=-1),
                    training=True, post_g=self.g[:, 1:],
                    Uprev=tf.pad(self.ctrl_obs[:, :-1],
                                 [[0, 0], [1, 0], [0, 0]]))

        LogDensity = 0.0
        with tf.name_scope('control_signal_loss'):
            # calculate loss on control signal
            if self.latent_u:
                # LogDensity += tf.scan(self.clip_loss, (self.ctrl_obs, value),
                #                       initializer=0.0, name='clip_noise')
                LogDensity += tf.reduce_sum(
                    self.clip_log_prob(self.ctrl_obs, value), axis=[1, 2],
                    name='clip_noise')
                resU = value[:, 1:] - Upred
            else:
                resU = self.ctrl_obs - Upred

            LogDensity -= tf.reduce_sum(resU ** 2 / (2 * self.eps ** 2),
                                        axis=[1, 2])
            LogDensity -= (0.5 * tf.log(2 * np.pi) +
                           tf.reduce_sum(tf.log(self.eps)))
        with tf.name_scope('control_signal_penalty'):
            with tf.name_scope('control_error_penalty'):
                # penalty on ctrl error
                if self.pen_ctrl_error is not None:
                    LogDensity -= (self.pen_ctrl_error * tf.reduce_sum(
                        tf.nn.relu(ctrl_error - 1.0), axis=[1, 2]))
                    LogDensity -= (self.pen_ctrl_error * tf.reduce_sum(
                        tf.nn.relu(-ctrl_error - 1.0), axis=[1, 2]))
                    # LogDensity -= (self.pen_ctrl_error * tf.reduce_sum(
                    #     tf.norm(ctrl_error, axis=-1), axis=-1))
            with tf.name_scope('control_noise_penalty'):
                # penalty on eps
                if self.pen_eps is not None:
                    LogDensity -= self.pen_eps * tf.reduce_sum(self.unc_eps)

        return LogDensity

    def getParams(self):
        """Return the learnable parameters of the model
        """

        rets = self.PID_params  # + [self.unc_eps]

        return rets


class GBDS_g(RandomVariable, Distribution):
    """A customized Random Variable of goal in Goal-Based Dynamical System
    """

    def __init__(self, GenerativeParams, yDim, yDim_in, y, extra_conds,
                 ctrl_obs, name='GBDS_g', value=None, dtype=tf.float32,
                 reparameterization_type=FULLY_REPARAMETERIZED,
                 validate_args=True, allow_nan_stats=True):
        """Initialize a batch of GBDS_g random variables for one of the agent


        Args:
          GenerativeParams: A dictionary of parameters for the agent
          Entries include:
            - get_states: function that calculates current state from position
            - pen_eps: Penalty on control signal noise, epsilon
            - pen_sigma: Penalty on goals state noise, sigma
            - pen_g: Two penalties on goal state leaving boundaries (Can be set
                     None)
            - bounds_g: Boundaries corresponding to above penalties
            - NN_postJ_mu: Neural network that parametrizes the mean of the
                           posterior of J (i.e. mu and sigma), conditioned on
                           goals
            - NN_postJ_sigma: Neural network that parametrizes the covariance
                              of the posterior of J (i.e. mu and sigma),
                              conditioned on goals
            - yCols: Columns of Y this agent corresponds to. Used to index
                     columns in real data to compare against generated data.
            - vel: Maximum velocity of each dimension in columns belong to this
                   agent(yCols)
            - all_vel: Maximum velocity of each dimension in columns belong to
                       all agents
            - clip: Clipping signal
            - clip_range: the range of the clipping states
            - clip_tol: the tolenrance of clipping
            - GMM_net: Gaussian Mixture Model network
            - GMM_k: Number of GMM components
          yDim: Number of dimensions for the agent
          yDim_in: Number of dimensions for the data
          y: Time series of positions
          value: The Random Variable sample of goal.

        """

        self.y = y
        self.B = tf.shape(y)[0]  # batch size
        self.extra_conds = extra_conds
        self.ctrl_obs = ctrl_obs

        with tf.name_scope('dimension'):
            self.yDim_in = yDim_in  # dimension of observation input
            self.yDim = yDim

        with tf.name_scope('get_states'):
            # function that calculates states from positions
            self.get_states = GenerativeParams['get_states']

        with tf.name_scope('velocity'):
            # velocity for each observation dimension (of all agents)
            self.all_vel = tf.constant(GenerativeParams['all_vel'],
                                       dtype=tf.float32, name='velocity')

        with tf.name_scope('g0'):
            self.g0_mu = GenerativeParams['g0_params']['mu']
            self.g0_unc_lambda = GenerativeParams['g0_params']['unc_lambda']
            self.g0_lambda = tf.nn.softplus(self.g0_unc_lambda,
                                            name='softplus_g0_lambda')
            self.g0_unc_w = GenerativeParams['g0_params']['unc_w']
            self.g0_w = tf.nn.softmax(self.g0_unc_w,
                                      name='softmax_g0_w')
            self.g0_params = [self.g0_mu] + [self.g0_unc_lambda] + [self.g0_unc_w]

        with tf.name_scope('GMM_NN'):
            self.GMM_k = GenerativeParams['GMM_k']  # number of GMM components
            # GMM neural networks
            self.GMM_net = GenerativeParams['GMM_net']

        with tf.name_scope('goal_state_noise'):
            # noise coefficient on goal states
            self.unc_sigma = tf.Variable(
                initial_value=-5 * np.ones((1, self.yDim)), name='unc_sigma',
                dtype=tf.float32)
            self.sigma = tf.nn.softplus(self.unc_sigma, name='sigma')

        with tf.name_scope('goal_state_noise_penalty'):
            # penalty on sigma (noise on goal state)
            if GenerativeParams['pen_sigma'] is not None:
                self.pen_sigma = GenerativeParams['pen_sigma']
            else:
                self.pen_sigma = None

        with tf.name_scope('goal_boundary_penalty'):
            with tf.name_scope('boundary'):
                # corresponding boundaries for pen_g
                if GenerativeParams['bounds_g'] is not None:
                    self.bounds_g = GenerativeParams['bounds_g']
                else:
                    self.bounds_g = 1.0
            with tf.name_scope('penalty'):
                # penalty on goal state escaping boundaries
                if GenerativeParams['pen_g'] is not None:
                    self.pen_g = GenerativeParams['pen_g']
                else:
                    self.pen_g = None

        with tf.name_scope('control_model'):
            self.latent_u = GenerativeParams['latent_u']
            if not self.latent_u:
                self.u = GBDS_u(
                    GenerativeParams, self, self.y, self.ctrl_obs, self.yDim,
                    name='ctrl', value=tf.zeros_like(value))

        super(GBDS_g, self).__init__(
            name=name, value=value, dtype=dtype,
            reparameterization_type=reparameterization_type,
            validate_args=validate_args, allow_nan_stats=allow_nan_stats)

        self._kwargs['y'] = y
        self._kwargs['yDim'] = yDim
        self._kwargs['yDim_in'] = yDim_in
        self._kwargs['GenerativeParams'] = GenerativeParams
        self._kwargs['extra_conds'] = extra_conds
        self._kwargs['ctrl_obs'] = ctrl_obs

    def get_preds(self, Y, training=False, post_g=None,
                  gen_g=None, extra_conds=None):
        """Return the predicted next g for each point in Y.

        For training: provide post_g, sample from the posterior,
                      which is used to calculate the ELBO
        """

        if training and post_g is None:
            raise Exception(
                'Must provide sample of g from posterior during training')

        with tf.name_scope('states'):
            # get states from position
            states = self.get_states(Y, max_vel=self.all_vel)
            with tf.name_scope('pad_extra_conds'):
                if self.extra_conds is not None:
                    states = pad_extra_conds(states, self.extra_conds)

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
            # Draw next goals based on force
            if post_g is not None:  # Calculate next goals from posterior
                next_g = ((tf.expand_dims(post_g, 2) + all_mu * all_lambda) /
                          (1 + all_lambda))

        return (all_mu, all_lambda, all_w, next_g)

    def sample_g0(self):
        k_0 = tf.squeeze(tf.multinomial(
            tf.tile(tf.reshape(tf.log(self.g0_w), [1, -1]),
                    [self.B, 1]), 1), name='k_0')
        g_0 = (tf.gather(self.g0_mu, k_0, axis=0, name='mean') +
               tf.random_normal([self.B, self.yDim], name='std_normal') /
               tf.sqrt(tf.gather(self.g0_lambda, k_0, axis=0),
                       name='inv_std_dev'))

        return g_0

    def sample_GMM(self, mu, lmbda, w):
        """Sample from GMM based on highest weight component
        """

        def select_components(acc, inputs):
            sub_mu, sub_lambda, w = inputs
            z = tf.range(self.GMM_k, name='classes')
            p = tf.multinomial(tf.log(w), 1, name='draw')
            component = z[tf.cast(p[0, 0], tf.int32)]

            return sub_mu[:, component, :], sub_lambda[:, component, :]

        (mu_k, lambda_k) = tf.scan(
            select_components, [tf.transpose(mu, [1, 0, 2, 3]),
                                tf.transpose(lmbda, [1, 0, 2, 3]),
                                tf.transpose(w, [1, 0, 2])],
            initializer=(tf.zeros([self.B, self.yDim]),
                         tf.zeros([self.B, self.yDim])),
            name='select_components')
        mu_k = tf.transpose(mu_k, perm=[1, 0, 2])
        lambda_k = tf.transpose(lambda_k, perm=[1, 0, 2])
        updates = {}

        return (mu_k, lambda_k), updates

    def _log_prob(self, value):
        with tf.name_scope('next_time_point_pred'):
            _, all_lambda, all_w, g_pred = self.get_preds(
                self.y[:, :-1], training=True, post_g=value[:, :-1])

        LogDensity = 0.0
        with tf.name_scope('goal_state_loss'):
            w_brdcst = tf.expand_dims(all_w, -1, name='reshape_w')
            gmm_res_g = (tf.expand_dims(value[:, 1:], 2,
                                        name='reshape_posterior_samples') -
                         g_pred)
            gmm_term = (tf.log(w_brdcst + 1e-8) - ((1 + all_lambda) /
                        (2 * tf.reshape(self.sigma, [1, -1]) ** 2)) *
                        gmm_res_g ** 2)
            gmm_term += (0.5 * tf.log(1 + all_lambda) -
                         0.5 * tf.log(2 * np.pi) -
                         tf.reshape(tf.log(self.sigma), [1, -1]))
            LogDensity += tf.reduce_sum(logsumexp(
                tf.reduce_sum(gmm_term, axis=-1), axis=-1), axis=[-2, -1])

        with tf.name_scope('g0_loss'):
            res_g0 = tf.expand_dims(value[:, 0], 1) - self.g0_mu
            g0_term = (tf.expand_dims(tf.log(self.g0_w + 1e-8), -1) -
                       (self.g0_lambda /
                        (2 * tf.reshape(self.sigma, [1, -1]) ** 2)) *
                       res_g0 ** 2)
            g0_term += (0.5 * tf.log(self.g0_lambda) -
                        0.5 * tf.log(2 * np.pi) -
                        tf.reshape(tf.log(self.sigma), [1, -1]))
            LogDensity += tf.reduce_sum(logsumexp(
                tf.reduce_sum(g0_term, axis=-1), axis=1), axis=1)

        with tf.name_scope('goal_penalty'):
            with tf.name_scope('boundary'):
                if self.pen_g is not None:
                    # linear penalty on goal state escaping game space
                    LogDensity -= (self.pen_g * tf.reduce_sum(
                        tf.nn.relu(value - self.bounds_g), axis=[1, 2]))
                    LogDensity -= (self.pen_g * tf.reduce_sum(
                        tf.nn.relu(-value - self.bounds_g), axis=[1, 2]))
            with tf.name_scope('noise'):
                if self.pen_sigma is not None:
                    # penalty on sigma
                    LogDensity -= (self.pen_sigma *
                                   tf.reduce_sum(self.unc_sigma))

        with tf.name_scope('control_model'):
            if not self.latent_u:
                LogDensity += self.u._log_prob(value=None)

        return LogDensity

    def getParams(self):
        """Return the learnable parameters of the model
        """

        rets = self.GMM_net.variables + self.g0_params
        if not self.latent_u:
            rets += self.u.getParams()

        return rets


class GenerativeModel(object):
    """
    Interface class for generative time-series models
    """
    def __init__(self, GenerativeParams, xDim, yDim, nrng=None):

        # input variable referencing top-down or external input

        self.xDim = xDim
        self.yDim = yDim

        self.nrng = nrng

        # internal RV for generating sample
        self.Xsamp = tf.placeholder(tf.float32, shape=[None, xDim],
                                    name="Xsamp")

    def evaluateLogDensity(self):
        """
        Return a function that evaluates the density of the GenerativeModel.
        """
        raise Exception('Cannot call function of interface class')

    def getParams(self):
        """
        Return parameters of the GenerativeModel.
        """
        raise Exception('Cannot call function of interface class')

    def generateSamples(self):
        """
        generates joint samples
        """
        raise Exception('Cannot call function of interface class')

    def __repr__(self):
        return "GenerativeModel"


class LDS(GenerativeModel):
    """
    Gaussian latent LDS with (optional) NN observations:
    x(0) ~ N(x0, Q0 * Q0')
    x(t) ~ N(A x(t-1), Q * Q')
    y(t) ~ N(NN(x(t)), R * R')
    For a Kalman Filter model, choose the observation network, NN(x), to be
    a one-layer network with a linear output. The latent state has
    dimensionality n (parameter "xDim") and observations have dimensionality m
    (parameter "yDim").
    Inputs:
    (See GenerativeModel abstract class definition for a list of standard
    parameters.)
    GenerativeParams  -  Dictionary of LDS parameters
                           * A     : [n x n] linear dynamics matrix; should
                                     have eigeninitial_values with magnitude
                                     strictly less than 1
                           * QChol : [n x n] square root of the innovation
                                     covariance Q
                           * Q0Chol: [n x n] square root of the innitial
                                     innovation covariance
                           * RChol : [n x 1] square root of the diagonal of the
                                     observation covariance
                           * x0    : [n x 1] mean of initial latent state
                           * NN_XtoY_Params:
                                   Dictionary with one field:
                                   - network: a lasagne network with input
                                   dimensionality n and output dimensionality m
    """
    def __init__(self, GenerativeParams, xDim, yDim, nrng=None):

        super(LDS, self).__init__(GenerativeParams, xDim, yDim, nrng)

        # parameters
        if 'A' in GenerativeParams:
            self.A = tf.Variable(initial_value=GenerativeParams['A']
                                 .astype(np.float32), name='A')
            # dynamics matrix
        else:
            # TBD:MAKE A BETTER WAY OF SAMPLING DEFAULT A
            self.A = tf.Variable(initial_value=.5*np.diag(np.ones(xDim)
                                 .astype(np.float32)), name='A')
            # dynamics matrix

        if 'QChol' in GenerativeParams:
            self.QChol = tf.Variable(initial_value=GenerativeParams['QChol']
                                     .astype(np.float32), name='QChol')
            # cholesky of innovation cov matrix
        else:
            self.QChol = tf.Variable(initial_value=(np.eye(xDim))
                                     .astype(np.float32), name='QChol')
            # cholesky of innovation cov matrix

        if 'Q0Chol' in GenerativeParams:
            self.Q0Chol = tf.Variable(initial_value=GenerativeParams['Q0Chol']
                                      .astype(np.float32), name='Q0Chol')
            # cholesky of starting distribution cov matrix
        else:
            self.Q0Chol = tf.Variable(initial_value=(np.eye(xDim))
                                      .astype(np.float32), name='Q0Chol')
            # cholesky of starting distribution cov matrix

        if 'RChol' in GenerativeParams:
            self.RChol = tf.Variable(initial_value=np.ndarray.flatten
                                     (GenerativeParams['RChol']
                                      .astype(np.float32)), name='RChol')
            # cholesky of observation noise cov matrix
        else:
            self.RChol = tf.Variable(initial_value=np.random.randn(yDim)
                                     .astype(np.float32)/10, name='RChol')
            # cholesky of observation noise cov matrix

        if 'x0' in GenerativeParams:
            self.x0 = tf.Variable(initial_value=GenerativeParams['x0']
                                  .astype(np.float32), name='x0')
            # set to zero for stationary distribution
        else:
            self.x0 = tf.Variable(initial_value=np.zeros((xDim,))
                                  .astype(np.float32), name='x0')
            # set to zero for stationary distribution

        if 'NN_XtoY_Params' in GenerativeParams:
            self.NN_XtoY = GenerativeParams['NN_XtoY_Params']['network']
        else:
            # Define a neural network that maps the latent state into the
            # output
            gen_nn = layers.Input(batch_shape=(None, xDim))
            gen_nn_d = (layers.Dense(yDim,
                        activation="linear",
                        kernel_initializer=tf.orthogonal_initializer())
                        (gen_nn))
            self.NN_XtoY = models.Model(inputs=gen_nn, outputs=gen_nn_d)

        # set to our lovely initial initial_values
        if 'C' in GenerativeParams:
            self.NN_XtoY.set_weights([GenerativeParams['C']
                                      .astype(np.float32),
                                      self.NN_XtoY.get_weights()[1]])
        if 'd' in GenerativeParams:
            self.NN_XtoY.set_weights([self.NN_XtoY.get_weights()[0],
                                      GenerativeParams['d']
                                      .astype(np.float32)])
        # we assume diagonal covariance (RChol is a vector)
        self.Rinv = 1./(self.RChol**2)
        # tf.matrix_inverse(tf.matmul(self.RChol ,T.transpose(self.RChol)))
        self.Lambda = tf.matrix_inverse(tf.matmul(self.QChol,
                                        tf.transpose(self.QChol)))
        self.Lambda0 = tf.matrix_inverse(tf.matmul(self.Q0Chol,
                                         tf.transpose(self.Q0Chol)))

        # Call the neural network output a rate, basically to keep things
        # consistent with the PLDS class
        self.rate = self.NN_XtoY(self.Xsamp)

    def sampleX(self, _N):
        _x0 = np.asarray(self.x0.eval(), dtype=np.float32)
        _Q0Chol = np.asarray(self.Q0Chol.eval(), dtype=np.float32)
        _QChol = np.asarray(self.QChol.eval(), dtype=np.float32)
        _A = np.asarray(self.A.eval(), dtype=np.float32)

        norm_samp = np.random.randn(_N, self.xDim).astype(np.float32)
        x_vals = np.zeros([_N, self.xDim]).astype(np.float32)

        x_vals[0] = _x0 + np.dot(norm_samp[0], _Q0Chol.T)

        for ii in range(_N-1):
            x_vals[ii+1] = x_vals[ii].dot(_A.T) + norm_samp[ii+1].dot(_QChol.T)

        return x_vals.astype(np.float32)

    def sampleY(self):
        # Return a symbolic sample from the generative model.
        return self.rate + tf.matmul(tf.random_normal((tf.shape(self.Xsamp)[0],
                                     self.yDim), seed=1234),
                                     tf.transpose(tf.diag(self.RChol)))

    def sampleXY(self, _N):
        # Return numpy samples from the generative model.
        X = self.sampleX(_N)
        nprand = np.random.randn(X.shape[0], self.yDim).astype(np.float32)
        _RChol = np.asarray(self.RChol.eval(), dtype=np.float32)
        Y = self.rate.eval({self.Xsamp: X}) + np.dot(nprand, np.diag(_RChol).T)
        return [X, Y]

    def getParams(self):
        return ([self.A] + [self.QChol] + [self.Q0Chol] + [self.RChol] +
                [self.x0] + self.NN_XtoY.variables)

    def evaluateLogDensity(self, X, Y):
        # Create a new graph which computes self.rate after replacing
        # self.Xsamp with X
        Ypred = tf.contrib.graph_editor.graph_replace(self.rate,
                                                      {self.Xsamp: X})
        resY = Y-Ypred
        self.resY = resY
        resX = X[1:]-tf.matmul(X[:(tf.shape(X)[0]-1)], tf.transpose(self.A))
        self.resX = resX
        self.resX0 = X[0]-self.x0

        LogDensity = -tf.reduce_sum(0.5*tf.matmul(tf.transpose(resY),
                                                  resY)*tf.diag(self.Rinv))

        LogDensity += -tf.reduce_sum(0.5*tf.matmul(tf.transpose(resX),
                                                   resX)*self.Lambda)

        LogDensity += -0.5*tf.matmul(tf.matmul(tf.reshape(self.resX0,
                                                          [1, -1]),
                                               self.Lambda0),
                                     tf.reshape(self.resX0, [-1, 1]))

        LogDensity += (0.5*tf.reduce_sum(tf.log(self.Rinv)) *
                       (tf.cast(tf.shape(Y)[0], tf.float32)))

        LogDensity += (0.5*tf.log(tf.matrix_determinant(self.Lambda)) *
                       (tf.cast(tf.shape(Y)[0]-1, tf.float32)))

        LogDensity += 0.5*tf.log(tf.matrix_determinant(self.Lambda0))

        LogDensity += (-0.5*(self.xDim + self.yDim)*np.log(2*np.pi) *
                       tf.cast(tf.shape(Y)[0], tf.float32))

return LogDensity
