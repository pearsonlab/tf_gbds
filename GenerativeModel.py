import tensorflow as tf
import numpy as np
from edward.models import RandomVariable
from tensorflow.contrib.distributions import Distribution, Normal
from tensorflow.contrib.distributions import FULLY_REPARAMETERIZED
from tf_gbds.utils import pad_extra_conds


def logsumexp(x, axis=None):
    x_max = tf.reduce_max(x, axis=axis, keep_dims=True)
    return (tf.log(tf.reduce_sum(tf.exp(x - x_max),
                                 axis=axis, keep_dims=True)) + x_max)


class GBDS_g_all(RandomVariable, Distribution):
    """A customized Random Variable of goal in Goal-Based Dynamical System
    combining both goalie agent and ball agent.
    """

    def __init__(self, GenerativeParams_goalie, GenerativeParams_ball, yDim,
                 y, extra_conds, name='GBDS_g_all', value=None,
                 dtype=tf.float32,
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

        self.yCols_ball = GenerativeParams_ball['yCols']
        self.yCols_goalie = GenerativeParams_goalie['yCols']
        self.y = y
        self.yDim = yDim
        self.extra_conds = extra_conds

        yDim_ball = len(self.yCols_ball)
        yDim_goalie = len(self.yCols_goalie)

        self.goalie = GBDS_g(GenerativeParams_goalie, yDim_goalie, yDim, y,
                             extra_conds, name='G_goalie',
                             value=tf.gather(value, self.yCols_goalie,
                                             axis=-1))
        self.ball = GBDS_g(GenerativeParams_ball, yDim_ball, yDim, y,
                           extra_conds, name='G_ball',
                           value=tf.gather(value, self.yCols_ball, axis=-1))

        super(GBDS_g_all, self).__init__(
            name=name, value=value, dtype=dtype,
            reparameterization_type=reparameterization_type,
            validate_args=validate_args, allow_nan_stats=allow_nan_stats)

        self._kwargs['GenerativeParams_goalie'] = GenerativeParams_goalie
        self._kwargs['GenerativeParams_ball'] = GenerativeParams_ball
        self._kwargs['y'] = y
        self._kwargs['yDim'] = yDim
        self._kwargs['extra_conds'] = extra_conds

    def _log_prob(self, value):
        log_prob_ball = self.ball.log_prob(
            tf.gather(value, self.yCols_ball, axis=-1))
        log_prob_goalie = self.goalie.log_prob(
            tf.gather(value, self.yCols_goalie, axis=-1))

        return log_prob_ball + log_prob_goalie

    def getParams(self):
        return self.ball.getParams() + self.goalie.getParams()


class GBDS_u_all(RandomVariable, Distribution):
    """A customized Random Variable of control signal in Goal-Based Dynamical
        System combining both goalie agent and ball agent.
    """

    def __init__(self, GenerativeParams_goalie, GenerativeParams_ball, g, y,
                 ctrl_obs, yDim, PID_params_goalie, PID_params_ball,
                 name='GBDS_u_all', value=None, dtype=tf.float32,
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
          PID_params_goalie: A dictionary of PID controller parameters for the
                             goalie model
          PID_params_ball: A dictionary of PID controller parameters for the
                           ball model
          value: The Random Variable sample of goal. Since GBDS_u_all is just a
                 likelihood node, value is just used to specify the shape of g.
                 Set it to tf.zeros_like(Y).
        """

        self.yCols_ball = GenerativeParams_ball['yCols']
        self.yCols_goalie = GenerativeParams_goalie['yCols']
        self.y = y
        self.yDim = yDim
        self.g = g
        self.ctrl_obs = ctrl_obs

        yDim_ball = len(self.yCols_ball)
        yDim_goalie = len(self.yCols_goalie)
        g_ball = tf.gather(self.g, self.yCols_ball, axis=-1)
        g_goalie = tf.gather(self.g, self.yCols_goalie, axis=-1)

        if self.ctrl_obs is not None:
            ctrl_ball = tf.gather(self.ctrl_obs, self.yCols_ball, axis=-1)
            ctrl_goalie = tf.gather(self.ctrl_obs, self.yCols_goalie, axis=-1)
        else:
            ctrl_ball = None
            ctrl_goalie = None

        self.goalie = GBDS_u(GenerativeParams_goalie, g_goalie, y,
                             ctrl_goalie, yDim_goalie, PID_params_goalie,
                             name='U_goalie',
                             value=tf.gather(value, self.yCols_goalie,
                                             axis=-1))
        self.ball = GBDS_u(GenerativeParams_ball, g_ball, y, ctrl_ball,
                           yDim_ball, PID_params_ball, name='U_ball',
                           value=tf.gather(value, self.yCols_ball, axis=-1))

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
        self._kwargs['PID_params_goalie'] = PID_params_goalie
        self._kwargs['PID_params_ball'] = PID_params_ball

    def _log_prob(self, value):
        log_prob_ball = self.ball.log_prob(
            tf.gather(value, self.yCols_ball, axis=-1))
        log_prob_goalie = self.goalie.log_prob(
            tf.gather(value, self.yCols_goalie, axis=-1))
        return log_prob_ball + log_prob_goalie

    def getParams(self):
        return self.ball.getParams() + self.goalie.getParams()


class GBDS_u(RandomVariable, Distribution):
    """A customized Random Variable of control signal in Goal-Based Dynamical
       System
    """

    def __init__(self, GenerativeParams, g, y, ctrl_obs, yDim, PID_params,
                 name='GBDS_u', value=None, dtype=tf.float32,
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
          PID_params: A dictionary of PID controller parameters
          value: The Random Variable sample of control signal
        """

        self.g = g
        self.y = y
        self.yDim = yDim
        self.B = tf.shape(y)[0]  # batch size
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
                unc_Kp = PID_params['unc_Kp']
                unc_Ki = PID_params['unc_Ki']
                unc_Kd = PID_params['unc_Kd']
                # create list of PID controller parameters for easy access in
                # getParams
                self.PID_params = [unc_Kp, unc_Ki, unc_Kd]
                # constrain PID controller parameters to be positive
                self.Kp = tf.nn.softplus(unc_Kp, name='Kp')
                self.Ki = tf.nn.softplus(unc_Ki, name='Ki')
                self.Kd = tf.nn.softplus(unc_Kd, name='Kd')
            with tf.name_scope('filter'):
                # calculate coefficients to be placed in convolutional filter
                t_coeff = self.Kp + self.Ki + self.Kd
                t1_coeff = -self.Kp - 2 * self.Kd
                t2_coeff = self.Kd
                # concatenate coefficients into filter
                self.L = tf.concat([t2_coeff, t1_coeff, t_coeff], axis=1,
                                   name='filter')

        with tf.name_scope('control_signal'):
            if ctrl_obs is not None:
                self.ctrl_obs = ctrl_obs
            else:
                self.ctrl_obs = None
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
            # penalty on epsilon (noise on control signal)
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
        self._kwargs['PID_params'] = PID_params

    def get_preds(self, Y, training=False, post_g=None, post_U=None):
        """Return the predicted next U, and Y for each point in Y.

        For training: provide post_g, sample from the posterior,
                      which is used to calculate the ELBO
        """
        with tf.name_scope('error'):
            # PID Controller for next control point
            if post_g is not None:  # calculate error from posterior goals
                error = post_g[:, 1:] - tf.gather(Y, self.yCols, axis=-1)
            # else:  # calculate error from generated goals
            #     error = next_g - tf.gather(Y, self.yCols, axis=1)
        with tf.name_scope('control_signal_change'):
            Udiff = []
            # get current error signal and corresponding filter
            for i in range(self.yDim):
                signal = error[:, :, i]
                # zero pad beginning
                signal = tf.expand_dims(tf.concat(
                    [tf.zeros([self.B, 2]), signal], 1), -1,
                    name='zero_padding')
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
            Upred = post_U[:, :-1] + Udiff
        with tf.name_scope('predicted_position'):
            # get predicted Y
            if self.clip:
                Ypred = (tf.gather(Y, self.yCols, axis=-1) +
                         tf.reshape(self.vel, [1, self.yDim]) *
                         tf.clip_by_value(Upred, -self.clip_range,
                                          self.clip_range,
                                          name='clipped_signal'))
            else:
                Ypred = (tf.gather(Y, self.yCols, axis=-1) +
                         tf.reshape(self.vel, [1, self.yDim]) * Upred)

        return (Upred, Ypred)

    def clip_loss(self, acc, inputs):
        """upsilon (derived from time series of y) is a censored version of
        a noisy control signal: \hat{u} ~ N(u, \eta^2).
        log p(upsilon|u, g) = log p(upsilon|u) + log(u|g)
        log p(upsilon|u) breaks down into three cases,
        namely left-clipped (upsilon_t = -1), right-clipped (upsilon_t = 1),
        and non-clipped (-1 < upsilon_t < 1). For the first two cases,
        Normal CDF is used instead of PDF due to censoring.
        The log density term is computed for each and then add up.
        """
        (U_obs, value) = inputs
        left_clip_ind = tf.where(tf.less_equal(
            U_obs, (-self.clip_range + self.clip_tol)),
            name='left_clip_indices')
        right_clip_ind = tf.where(tf.greater_equal(
            U_obs, (self.clip_range - self.clip_tol)),
            name='right_clip_indices')
        non_clip_ind = tf.where(tf.logical_and(
            tf.greater(U_obs, (-self.clip_range + self.clip_tol)),
            tf.less(U_obs, (self.clip_range - self.clip_tol))),
            name='non_clip_indices')
        left_clip_node = Normal(tf.gather_nd(value, left_clip_ind),
                                self.eta, name='left_clip_node')
        right_clip_node = Normal(tf.gather_nd(-value, right_clip_ind),
                                 self.eta, name='right_clip_node')
        non_clip_node = Normal(tf.gather_nd(value, non_clip_ind),
                               self.eta, name='non_clip_node')
        LogDensity = 0.0
        LogDensity += tf.reduce_sum(
            left_clip_node.log_cdf(-1., name='left_clip_logcdf'))
        LogDensity += tf.reduce_sum(
            right_clip_node.log_cdf(-1., name='right_clip_logcdf'))
        LogDensity += tf.reduce_sum(
            non_clip_node.log_prob(tf.gather_nd(U_obs, non_clip_ind),
                                   name='non_clip_logpdf'))

        return LogDensity

    def _log_prob(self, value):
        """Evaluates the log-density of the GenerativeModel.
        """

        # Calculate real control signal
        with tf.name_scope('observed_control_signal'):
            if self.ctrl_obs is None:
                U_obs = tf.concat([tf.zeros([self.B, 1, self.yDim]),
                                   (tf.gather(self.y, self.yCols,
                                              axis=-1)[:, 1:] -
                                    tf.gather(self.y, self.yCols,
                                              axis=-1)[:, :-1]) /
                                   tf.reshape(self.vel, [1, self.yDim])], 1,
                                  name='U_obs')
            else:
                U_obs = self.ctrl_obs
        # Get predictions for next timestep (at each timestep except for last)
        # disregard last timestep bc we don't know the next value, thus, we
        # can't calculate the error
        with tf.name_scope('next_time_step_pred'):
            Upred, _ = self.get_preds(self.y[:, :-1], training=True,
                                      post_g=self.g, post_U=value)

        LogDensity = 0.0
        with tf.name_scope('control_signal_loss'):
            # calculate loss on control signal
            LogDensity += tf.scan(self.clip_loss, (U_obs, value),
                                  initializer=0.0, name='clip_noise')

            resU = value[:, 1:] - Upred
            LogDensity -= tf.reduce_sum(resU ** 2 / (2 * self.eps ** 2),
                                        axis=[1, 2])
            LogDensity -= (0.5 * tf.log(2 * np.pi) +
                           tf.reduce_sum(tf.log(self.eps)))
        with tf.name_scope('control_signal_penalty'):
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
                 name='GBDS_g', value=None, dtype=tf.float32,
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

        with tf.name_scope('dimension'):
            self.yDim_in = yDim_in  # dimension of observation input
            self.yDim = yDim
            self.y = y
            self.B = tf.shape(y)[0]  # batch size
            self.extra_conds = extra_conds

        with tf.name_scope('get_states'):
            # function that calculates states from positions
            self.get_states = GenerativeParams['get_states']

        with tf.name_scope('GMM_NN'):
            self.GMM_k = GenerativeParams['GMM_k']  # number of GMM components
            # GMM neural networks
            self.GMM_net = GenerativeParams['GMM_net']

        with tf.name_scope('goal_state_penalty'):
            # penalty on sigma (noise on goal state)
            if GenerativeParams['pen_sigma'] is not None:
                self.pen_sigma = GenerativeParams['pen_sigma']
            else:
                self.pen_sigma = None

        with tf.name_scope('boundary_penalty'):
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

        with tf.name_scope('velocity'):
            # velocity for each observation dimension (of all agents)
            self.all_vel = tf.constant(GenerativeParams['all_vel'],
                                       dtype=tf.float32, name='velocity')

        with tf.name_scope('goal_state_noise'):
            # noise coefficient on goal states
            self.unc_sigma = tf.Variable(
                initial_value=-5 * np.ones((1, self.yDim)), name='unc_sigma',
                dtype=tf.float32)
            self.sigma = tf.nn.softplus(self.unc_sigma, name='sigma')

        super(GBDS_g, self).__init__(
            name=name, value=value, dtype=dtype,
            reparameterization_type=reparameterization_type,
            validate_args=validate_args, allow_nan_stats=allow_nan_stats)

        self._kwargs['y'] = y
        self._kwargs['yDim'] = yDim
        self._kwargs['yDim_in'] = yDim_in
        self._kwargs['GenerativeParams'] = GenerativeParams
        self._kwargs['extra_conds'] = extra_conds

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
                    [self.B, -1, self.GMM_k, self.yDim], name='reshape_mu')

            with tf.name_scope('lambda'):
                all_lambda = tf.nn.softplus(tf.reshape(
                    self.GMM_net(states)[:, :, (self.yDim *
                                                self.GMM_k):(2 * self.yDim *
                                                             self.GMM_k)],
                    [self.B, -1, self.GMM_k, self.yDim],
                    name='reshape_lambda'), name='softplus_lambda')

            with tf.name_scope('w'):
                all_w = tf.nn.softmax(tf.reshape(
                    self.GMM_net(states)[:, :, (2 * self.yDim * self.GMM_k):],
                    [self.B, -1, self.GMM_k],
                    name='reshape_w'), dim=-1, name='softmax_w')

        with tf.name_scope('next_g'):
                # Draw next goals based on force
            if post_g is not None:  # Calculate next goals from posterior
                next_g = ((tf.reshape(post_g[:, :-1],
                                      [self.B, -1, 1, self.yDim]) +
                           all_mu * all_lambda) / (1 + all_lambda))

        return (all_mu, all_lambda, all_w, next_g)

    def sample_GMM(self, mu, lmbda, w):
        """Sample from GMM based on highest weight component
        """

        def select_components(acc, inputs):
            sub_mu, sub_lambda, w = inputs
            z = tf.range(self.K, name='classes')
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
        with tf.name_scope('get_params_g_pred'):
            all_mu, all_lambda, all_w, g_pred = self.get_preds(
                self.y[:, :-1], training=True, post_g=value)

        LogDensity = 0.0
        with tf.name_scope('goal_state_loss'):
            w_brdcst = tf.reshape(all_w, [self.B, -1, self.GMM_k, 1],
                                  name='reshape_w')
            gmm_res_g = (tf.reshape(value[:, 1:], [self.B, -1, 1, self.yDim],
                                    name='reshape_sample') - g_pred)
            gmm_term = (tf.log(w_brdcst + 1e-8) - ((1 + all_lambda) /
                        (2 * tf.reshape(self.sigma, [1, -1]) ** 2)) *
                        gmm_res_g ** 2)
            gmm_term += (0.5 * tf.log(1 + all_lambda) -
                         0.5 * tf.log(2 * np.pi) -
                         tf.log(tf.reshape(self.sigma, [1, 1, 1, -1])))
            LogDensity += tf.reduce_sum(logsumexp(
              tf.reduce_sum(gmm_term, axis=-1), axis=-1), axis=[-2, -1])

        with tf.name_scope('goal_and_control_penalty'):
            if self.pen_g is not None:
                # linear penalty on goal state escaping game space
                LogDensity -= (self.pen_g * tf.reduce_sum(
                    tf.nn.relu(all_mu - self.bounds_g), axis=[1, 2, 3]))
                LogDensity -= (self.pen_g * tf.reduce_sum(
                    tf.nn.relu(-all_mu - self.bounds_g), axis=[1, 2, 3]))
            if self.pen_sigma is not None:
                # penalty on sigma
                LogDensity -= (self.pen_sigma * tf.reduce_sum(self.unc_sigma))

        return LogDensity

    def getParams(self):
        """Return the learnable parameters of the model
        """
        rets = self.GMM_net.variables
        return rets
